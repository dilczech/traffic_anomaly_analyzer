import os
import sys
import glob
import re
import datetime
import argparse
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import matplotlib
matplotlib.use('Agg') # Фоновый рендеринг без GUI
import matplotlib.pyplot as plt

# Настройка шрифтов для корректного отображения кириллицы
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Calibri', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. ПОТОКОВЫЙ ПАРСЕР С ЗАЩИТОЙ ОТ "ГРЯЗНЫХ" ДАННЫХ
# ==============================================================================

def parse_traffic_excel(filepath):
    """
    Потоковый парсер: находит и разделяет участки трассы, 
    распознает любые форматы дат и очищает текстовые числа.
    """
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    ws = wb.active
    
    records = []
    current_section = "Неизвестный участок"
    
    # Универсальный regex для дат (с ведущими нулями и без: 1.10.2020, 01/10/2020 и т.д.)
    date_regex = re.compile(r"^\s*\d{1,2}[\./]\d{1,2}[\./]\d{4}\s+\d{1,2}:\d{2}(:\d{2})?\s*$")
    section_regex = re.compile(r"км\s*\d+(\+\d+)?", re.IGNORECASE)
    
    for row in ws.iter_rows(values_only=True):
        if not row:
            continue
            
        # Поиск названия участка в первых 6 ячейках строки
        for cell in row[:6]:
            if cell is not None and isinstance(cell, str):
                cell_s = cell.strip()
                if section_regex.search(cell_s):
                    current_section = cell_s
                    break
                elif ("а/д" in cell_s.lower() or "р-193" in cell_s.lower()) and "отчет" not in cell_s.lower():
                    current_section = cell_s
                    break
                    
        first_val = row[0]
        if first_val is None:
            continue
            
        first_str = str(first_val).strip()
        
        # Парсинг даты (нативный datetime или строка)
        dt_obj = None
        if isinstance(first_val, (datetime.datetime, datetime.date)):
            dt_obj = pd.to_datetime(first_val)
        elif date_regex.match(first_str):
            try:
                dt_obj = pd.to_datetime(first_str, dayfirst=True)
            except Exception:
                dt_obj = None
                
        if dt_obj is not None and pd.notnull(dt_obj):
            row_vals = list(row[:29])
            if len(row_vals) < 29:
                row_vals.extend([0] * (29 - len(row_vals)))
            records.append([current_section, dt_obj] + row_vals[1:29])
            
    wb.close()
    
    col_names = [
        'road_section', 'timestamp',
        'vol_total', 'vol_direct', 'vol_reverse',
        'cars_total', 'cars_direct', 'cars_reverse',
        'small_trucks_total', 'small_trucks_direct', 'small_trucks_reverse',
        'med_trucks_total', 'med_trucks_direct', 'med_trucks_reverse',
        'large_trucks_total', 'large_trucks_direct', 'large_trucks_reverse',
        'road_trains_total', 'road_trains_direct', 'road_trains_reverse',
        'buses_total', 'buses_direct', 'buses_reverse',
        'motos_total', 'motos_direct', 'motos_reverse',
        'speed_direct', 'speed_reverse',
        'load_direct', 'load_reverse'
    ]
    
    if not records:
        return pd.DataFrame(columns=col_names)
        
    df = pd.DataFrame(records, columns=col_names)
    
    # Очистка чисел: убираем неразрывные пробелы (\xa0), пробелы и запятые
    for col in col_names[2:]:
        if df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace('\xa0', '', regex=False)
                       .str.replace(' ', '', regex=False)
                       .str.replace(',', '.', regex=False))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df


# ==============================================================================
# 2. ПОИСК АНОМАЛЬНЫХ СУТОК
# ==============================================================================

def detect_daily_anomalies(df):
    if df.empty:
        return pd.DataFrame()
        
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['timestamp'].dt.weekday.isin([5, 6]).astype(int)
    df['month_key'] = df['timestamp'].dt.to_period('M').astype(str)
    
    anomalies_list = []
    
    for section, sec_df in df.groupby('road_section'):
        
        # 1. ТЕХНИЧЕСКИЕ И РЕЖИМНЫЕ АНОМАЛИИ
        for dt_date, day_df in sec_df.groupby('date'):
            day_df = day_df.sort_values('hour')
            hours_cnt = len(day_df)
            
            # Неполные сутки
            if hours_cnt < 24:
                anomalies_list.append({
                    'Участок дороги': section, 'Дата': dt_date, 'Уровень': 'Критический',
                    'Детектор': 'Целостность', 'Тип аномалии': 'Неполные сутки',
                    'Параметр': 'Часов в сутках', 'Факт': hours_cnt, 'Норма': 24,
                    'Отклонение (%)': round((hours_cnt - 24) / 24 * 100, 1), 'Индекс (Score)': 3.0,
                    'Описание цифрами': f"В сутках сохранено лишь {hours_cnt} часов вместо 24"
                })
                
            # Залипание датчика (Flatline)
            vols = day_df['vol_total'].values
            max_consec = 1
            cur_consec = 1
            flat_val = None
            for i in range(1, len(vols)):
                if vols[i] == vols[i-1] and vols[i] > 10:
                    cur_consec += 1
                    if cur_consec > max_consec:
                        max_consec = cur_consec
                        flat_val = vols[i]
                else:
                    cur_consec = 1
            if max_consec >= 5:
                anomalies_list.append({
                    'Участок дороги': section, 'Дата': dt_date, 'Уровень': 'Критический',
                    'Детектор': 'Датчик', 'Тип аномалии': 'Залипание датчика (Flatline)',
                    'Параметр': 'Интенсивность', 'Факт': flat_val, 'Норма': 'Вариативность',
                    'Отклонение (%)': 0.0, 'Индекс (Score)': 4.0,
                    'Описание цифрами': f"Значение не менялось {max_consec} часов подряд ({flat_val} авт./ч)"
                })
                
            # Дисбаланс направлений (днем: 07:00-21:00)
            daytime_df = day_df[(day_df['hour'] >= 7) & (day_df['hour'] <= 21)]
            daytime_tot = daytime_df['vol_total'].sum()
            if daytime_tot > 500:
                dir_ratio = daytime_df['vol_direct'].sum() / daytime_tot
                if dir_ratio > 0.85 or dir_ratio < 0.15:
                    anomalies_list.append({
                        'Участок дороги': section, 'Дата': dt_date, 'Уровень': 'Высокий',
                        'Детектор': 'Направление', 'Тип аномалии': 'Дисбаланс направлений',
                        'Параметр': 'Доля прямого напр.', 'Факт': round(dir_ratio * 100, 1), 'Норма': '45-55%',
                        'Отклонение (%)': round((dir_ratio - 0.5) * 100, 1), 'Индекс (Score)': 2.8,
                        'Описание цифрами': f"Доля прямого направления составила {dir_ratio*100:.1f}% от дневного объема"
                    })
                    
            # Затор / срыв потока
            min_spd = daytime_df['speed_direct'].min() if len(daytime_df) > 0 else 99
            max_load = daytime_df['load_direct'].max() if len(daytime_df) > 0 else 0
            if min_spd < 35 and max_load > 60:
                jam_h = daytime_df.loc[daytime_df['speed_direct'] == min_spd, 'hour'].iloc[0]
                anomalies_list.append({
                    'Участок дороги': section, 'Дата': dt_date, 'Уровень': 'Критический',
                    'Детектор': 'Скорость/Загрузка', 'Тип аномалии': 'Экстремальный затор',
                    'Параметр': 'Мин. скорость (Прямое)', 'Факт': round(min_spd, 1), 'Норма': '> 70 км/ч',
                    'Отклонение (%)': round((min_spd - 80) / 80 * 100, 1), 'Индекс (Score)': 4.5,
                    'Описание цифрами': f"В {jam_h}:00 скорость упала до {min_spd} км/ч при загрузке {max_load}%"
                })

        # 2. АМПЛИТУДНЫЕ АНОМАЛИИ И ФОРМА ПРОФИЛЯ
        for (m_key, is_wknd), group_df in sec_df.groupby(['month_key', 'is_weekend']):
            norm_hourly_vol = group_df.groupby('hour')['vol_total'].median()
            
            daily_stats = group_df.groupby('date').agg(
                daily_vol=('vol_total', 'sum')
            ).reset_index()
            
            vol_med = daily_stats['daily_vol'].median()
            vol_mad = (daily_stats['daily_vol'] - vol_med).abs().median()
            vol_mad = vol_mad if vol_mad > 0 else daily_stats['daily_vol'].std()
            vol_mad = vol_mad if (vol_mad > 0 and pd.notnull(vol_mad)) else 1.0
            
            # Проверка формы профиля
            for dt_date, day_data in group_df.groupby('date'):
                day_data = day_data.sort_values('hour')
                if len(day_data) == 24 and norm_hourly_vol.std() > 0:
                    day_vector = day_data.set_index('hour')['vol_total']
                    if day_vector.std() > 0:
                        corr = np.corrcoef(norm_hourly_vol.values, day_vector.values)[0, 1]
                        if corr < 0.70 and day_vector.sum() > 1000:
                            anomalies_list.append({
                                'Участок дороги': section, 'Дата': dt_date, 'Уровень': 'Высокий',
                                'Детектор': 'Форма профиля', 'Тип аномалии': 'Искажение суточного профиля',
                                'Параметр': 'Корреляция с нормой', 'Факт': round(corr, 2), 'Норма': '> 0.85',
                                'Отклонение (%)': round((corr - 1.0) * 100, 1), 'Индекс (Score)': 2.5,
                                'Описание цифрами': f"Форма суточного хода нетипична (корреляция {corr:.2f} < 0.70)"
                            })
                            
            # Проверка суточного объема
            for _, r in daily_stats.iterrows():
                dt_date = r['date']
                d_vol = r['daily_vol']
                z_vol = 0.6745 * (d_vol - vol_med) / vol_mad if vol_mad > 0 else 0
                delta_p = round((d_vol - vol_med) / vol_med * 100, 1) if vol_med > 0 else 0
                
                # Спад
                if (z_vol < -3.0 and delta_p <= -25.0) or delta_p <= -50.0:
                    anomalies_list.append({
                        'Участок дороги': section, 'Дата': dt_date, 'Уровень': 'Критический',
                        'Детектор': 'Амплитуда', 'Тип аномалии': 'Аномальный спад интенсивности',
                        'Параметр': 'Суточный объем', 'Факт': int(d_vol), 'Норма': int(vol_med),
                        'Отклонение (%)': delta_p, 'Индекс (Score)': round(abs(z_vol), 1),
                        'Описание цифрами': f"Суточный трафик {int(d_vol)} ниже нормы {int(vol_med)} на {abs(delta_p)}% (Z={z_vol:.1f})"
                    })
                # Всплеск
                elif (z_vol > 3.0 and delta_p >= 25.0) or delta_p >= 50.0:
                    anomalies_list.append({
                        'Участок дороги': section, 'Дата': dt_date, 'Уровень': 'Высокий',
                        'Детектор': 'Амплитуда', 'Тип аномалии': 'Аномальный всплеск интенсивности',
                        'Параметр': 'Суточный объем', 'Факт': int(d_vol), 'Норма': int(vol_med),
                        'Отклонение (%)': delta_p, 'Индекс (Score)': round(abs(z_vol), 1),
                        'Описание цифрами': f"Суточный трафик {int(d_vol)} выше нормы {int(vol_med)} на +{delta_p}% (Z={z_vol:.1f})"
                    })

    res_df = pd.DataFrame(anomalies_list)
    if not res_df.empty:
        res_df = res_df.sort_values(by=['Участок дороги', 'Дата'])
    return res_df


# ==============================================================================
# 3. ПОИСК АНОМАЛЬНЫХ МЕСЯЦЕВ (МАКРО YoY)
# ==============================================================================

def detect_monthly_anomalies(all_data_df):
    if all_data_df.empty:
        return pd.DataFrame()
        
    df = all_data_df.copy()
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['year_month'] = df['timestamp'].dt.to_period('M').astype(str)
    
    monthly_stats = []
    for (sec, ym), m_df in df.groupby(['road_section', 'year_month']):
        y = m_df['year'].iloc[0]
        m = m_df['month'].iloc[0]
        
        tot_hours = len(m_df)
        days_in_m = pd.Period(ym).days_in_month
        exp_hours = days_in_m * 24
        completeness = (tot_hours / exp_hours) * 100
        
        daily_vols = m_df.groupby(m_df['timestamp'].dt.date)['vol_total'].sum()
        madt = daily_vols.mean() if len(daily_vols) > 0 else 0
        
        avg_spd_dir = m_df['speed_direct'].mean()
        tot_vol = m_df['vol_total'].sum()
        tot_heavy = m_df['large_trucks_total'].sum() + m_df['road_trains_total'].sum()
        heavy_share = (tot_heavy / tot_vol * 100) if tot_vol > 0 else 0
        
        monthly_stats.append({
            'road_section': sec, 'year_month': ym, 'year': y, 'month': m,
            'completeness_pct': round(completeness, 1),
            'madt': round(madt, 1),
            'avg_speed_dir': round(avg_spd_dir, 1),
            'heavy_share': round(heavy_share, 1)
        })
        
    m_df = pd.DataFrame(monthly_stats)
    monthly_anomalies = []
    
    for (sec, cal_month), group in m_df.groupby(['road_section', 'month']):
        if len(group) < 2:
            continue
            
        madt_med = group['madt'].median()
        spd_med = group['avg_speed_dir'].median()
        hvy_med = group['heavy_share'].median()
        
        for _, row in group.iterrows():
            ym = row['year_month']
            
            # Полнота данных
            if row['completeness_pct'] < 75.0:
                monthly_anomalies.append({
                    'Участок дороги': sec, 'Период': ym, 'Уровень': 'Критический',
                    'Детектор': 'Полнота данных', 'Параметр': 'Отработано часов (%)',
                    'Факт': f"{row['completeness_pct']}%", 'Норма': '>= 95%',
                    'Отклонение (%)': round(row['completeness_pct'] - 100, 1),
                    'Описание цифрами': f"Данные за месяц неполные (сохранено лишь {row['completeness_pct']}% часов)"
                })
                
            # Сдвиг интенсивности (ССИ / MADT)
            delta_madt = ((row['madt'] - madt_med) / madt_med) * 100 if madt_med > 0 else 0
            if abs(delta_madt) >= 25.0:
                lvl = 'Критический' if abs(delta_madt) >= 40.0 else 'Высокий'
                monthly_anomalies.append({
                    'Участок дороги': sec, 'Период': ym, 'Уровень': lvl,
                    'Детектор': 'Межгодовой YoY', 'Параметр': 'Среднесуточная интенсивность (ССИ)',
                    'Факт': int(row['madt']), 'Норма': int(madt_med),
                    'Отклонение (%)': round(delta_madt, 1),
                    'Описание цифрами': f"ССИ месяца ({int(row['madt'])} авт./сут) отклонилась на {delta_madt:+.1f}% от многолетней нормы ({int(madt_med)})"
                })
                
            # Сдвиг доли тяжелых грузовиков
            delta_hvy = row['heavy_share'] - hvy_med
            if abs(delta_hvy) >= 8.0:
                monthly_anomalies.append({
                    'Участок дороги': sec, 'Период': ym, 'Уровень': 'Высокий',
                    'Детектор': 'Состав потока YoY', 'Параметр': 'Доля тяжелых грузовиков (%)',
                    'Факт': f"{row['heavy_share']}%", 'Норма': f"{hvy_med}%",
                    'Отклонение (%)': round((delta_hvy / hvy_med) * 100, 1) if hvy_med > 0 else 0,
                    'Описание цифрами': f"Доля тяжелых грузовиков изменилась на {delta_hvy:+.1f}% п.п. (факт: {row['heavy_share']}%, норма: {hvy_med}%)"
                })
                
            # Просадка скорости
            delta_spd = row['avg_speed_dir'] - spd_med
            if delta_spd <= -15.0:
                monthly_anomalies.append({
                    'Участок дороги': sec, 'Период': ym, 'Уровень': 'Критический',
                    'Детектор': 'Скорость YoY', 'Параметр': 'Средняя скорость (км/ч)',
                    'Факт': f"{row['avg_speed_dir']}", 'Норма': f"{spd_med}",
                    'Отклонение (%)': round((delta_spd / spd_med) * 100, 1) if spd_med > 0 else 0,
                    'Описание цифрами': f"Средняя скорость за месяц упала на {abs(delta_spd):.1f} км/ч ниже многолетней нормы"
                })
                
    res_df = pd.DataFrame(monthly_anomalies)
    if not res_df.empty:
        res_df = res_df.sort_values(by=['Участок дороги', 'Период'])
    return res_df


# ==============================================================================
# 4. ПОСТРОЕНИЕ ГРАФИКОВ (БЕЗ ДУБЛИРОВАНИЯ)
# ==============================================================================

def generate_anomaly_plots(df, daily_anomalies_df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    if daily_anomalies_df.empty or df.empty:
        return
        
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = df['timestamp'].dt.weekday.isin([5, 6]).astype(int)
    df['month_key'] = df['timestamp'].dt.to_period('M').astype(str)
    
    # Группируем по уникальному дню (чтобы не плодить дубликаты файлов)
    grouped = daily_anomalies_df.groupby(['Участок дороги', 'Дата'])
    print(f"\nГенерация графиков (уникальных аномальных дней: {len(grouped)})...")
    
    hours = np.arange(24)
    for (sec, dt_d), rows in grouped:
        day_df = df[(df['road_section'] == sec) & (df['date'] == dt_d)].sort_values('hour')
        if len(day_df) == 0:
            continue
            
        m_key = day_df['month_key'].iloc[0]
        is_wknd = day_df['is_weekend'].iloc[0]
        
        # Список всех аномалий этого дня для подзаголовка
        anom_types = "; ".join(rows['Тип аномалии'].unique())
        
        ref_df = df[(df['road_section'] == sec) & (df['month_key'] == m_key) & (df['is_weekend'] == is_wknd)]
        norm_vol = ref_df.groupby('hour')['vol_total'].median().reindex(hours).fillna(0).values
        norm_spd = ref_df.groupby('hour')['speed_direct'].median().reindex(hours).fillna(0).values
        
        fact_vol = day_df.set_index('hour')['vol_total'].reindex(hours).fillna(0).values
        fact_spd = day_df.set_index('hour')['speed_direct'].reindex(hours).fillna(0).values
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Панель 1: Интенсивность
        ax1.bar(hours, norm_vol, color='#BDC3C7', alpha=0.6, width=0.6, label='Норма месяца (Медиана)', edgecolor='#95A5A6')
        ax1.plot(hours, fact_vol, color='#E74C3C', linewidth=2.5, marker='o', markersize=5, label='Факт аномального дня')
        ax1.set_ylabel('Интенсивность, авт./ч', fontsize=10, fontweight='bold')
        ax1.set_title(f"{sec} | {dt_d}\nСобытия: {anom_types}", fontsize=11, fontweight='bold', pad=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left', frameon=True)
        
        diff = np.abs(fact_vol - norm_vol)
        max_h = np.argmax(diff)
        if diff[max_h] > 100:
            ax1.annotate(f'Пик: {int(fact_vol[max_h])} vs {int(norm_vol[max_h])}',
                         xy=(max_h, fact_vol[max_h]),
                         xytext=(max_h, fact_vol[max_h] * 1.15 if fact_vol[max_h] > norm_vol[max_h] else fact_vol[max_h] * 0.7),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                         fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", fc="#FADBD8", ec="#E74C3C", lw=1))
                         
        # Панель 2: Скорость
        ax2.plot(hours, norm_spd, color='#7F8C8D', linestyle='--', linewidth=1.8, label='Норма скорости')
        ax2.plot(hours, fact_spd, color='#2980B9', linewidth=2.0, marker='s', markersize=4, label='Факт скорости (Прямое)')
        ax2.axhline(40, color='#C0392B', linestyle=':', alpha=0.7, label='Порог затора (40 км/ч)')
        ax2.set_ylabel('Скорость, км/ч', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Час суток (00:00 - 23:00)', fontsize=10, fontweight='bold')
        ax2.set_xticks(hours)
        ax2.set_xticklabels([f"{h:02d}" for h in hours])
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='lower left', frameon=True)
        ax2.set_ylim(0, max(110, np.max(fact_spd) + 10))
        
        plt.tight_layout()
        safe_sec = re.sub(r'[^a-zA-Z0-9а-яА-Я+]', '_', sec)[:20]
        fname = f"{dt_d}_{safe_sec}.png"
        fig.savefig(os.path.join(output_folder, fname), dpi=110)
        plt.close(fig)


# ==============================================================================
# 5. ЭКСПОРТ В EXCEL
# ==============================================================================

def export_anomalies_to_excel(daily_df, monthly_df, output_path):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        if not daily_df.empty:
            daily_df.to_excel(writer, sheet_name="Аномальные сутки", index=False)
        else:
            pd.DataFrame([{"Инфо": "Аномальные сутки не обнаружены"}]).to_excel(writer, sheet_name="Аномальные сутки", index=False)
            
        if not monthly_df.empty:
            monthly_df.to_excel(writer, sheet_name="Аномальные месяцы (YoY)", index=False)
        else:
            pd.DataFrame([{"Инфо": "Аномальные месяцы не обнаружены"}]).to_excel(writer, sheet_name="Аномальные месяцы (YoY)", index=False)
            
    wb = openpyxl.load_workbook(output_path)
    
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    crit_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    high_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    border_thin = Border(left=Side(style='thin', color='D9D9D9'), right=Side(style='thin', color='D9D9D9'),
                         top=Side(style='thin', color='D9D9D9'), bottom=Side(style='thin', color='D9D9D9'))
    
    for ws in wb.worksheets:
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            
        for row_idx in range(2, ws.max_row + 1):
            level_val = None
            for col_idx in range(1, ws.max_column + 1):
                c = ws.cell(row=row_idx, column=col_idx)
                c.border = border_thin
                c.alignment = Alignment(vertical="center")
                if ws.cell(row=1, column=col_idx).value == 'Уровень':
                    level_val = c.value
                    
            if level_val == 'Критический':
                for col_idx in range(1, ws.max_column + 1):
                    ws.cell(row=row_idx, column=col_idx).fill = crit_fill
            elif level_val == 'Высокий':
                for col_idx in range(1, ws.max_column + 1):
                    ws.cell(row=row_idx, column=col_idx).fill = high_fill
                    
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 48)
            
        ws.row_dimensions[1].height = 28
        
    wb.save(output_path)


# ==============================================================================
# ТОЧКА ВХОДА
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Анализатор аномалий трафика (2020-2026)")
    parser.add_argument("--input", default="./input_data", help="Папка с входными файлами .xlsx")
    parser.add_argument("--output", default="./reports", help="Папка для сохранения отчетов и графиков")
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    plots_folder = os.path.join(output_folder, "plots")
    
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    excel_files = sorted(glob.glob(os.path.join(input_folder, "*.xlsx")) + glob.glob(os.path.join(input_folder, "*.xls")))
    
    if not excel_files:
        print(f"\n[Внимание] В папке '{input_folder}' нет файлов .xlsx!")
        print("Скопируйте файлы отчетов в эту папку и запустите скрипт заново.")
        return
        
    print(f"\n=== СТАРТ ОБРАБОТКИ ДАННЫХ ===")
    print(f"Найдено файлов для анализа: {len(excel_files)}")
    
    all_dfs = []
    for idx, fpath in enumerate(excel_files, 1):
        fname = os.path.basename(fpath)
        print(f"[{idx}/{len(excel_files)}] Чтение {fname}...", end=" ", flush=True)
        try:
            df_part = parse_traffic_excel(fpath)
            print(f"OK ({len(df_part)} строк)")
            all_dfs.append(df_part)
        except Exception as e:
            print(f"ОШИБКА: {e}")
            
    if not all_dfs:
        print("Не удалось прочитать ни одного файла.")
        return
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nВсего загружено: {len(full_df)} записей.")
    print("Участки в базе:", full_df['road_section'].unique().tolist())
    
    print("\n1. Поиск аномальных суток...")
    daily_anomalies = detect_daily_anomalies(full_df)
    print(f"   Найдено аномальных суток: {len(daily_anomalies)}")
    
    print("2. Поиск аномальных месяцев (YoY)...")
    monthly_anomalies = detect_monthly_anomalies(full_df)
    print(f"   Найдено аномальных месяцев: {len(monthly_anomalies)}")
    
    report_xlsx_path = os.path.join(output_folder, "traffic_anomalies_report.xlsx")
    export_anomalies_to_excel(daily_anomalies, monthly_anomalies, report_xlsx_path)
    print(f"\n[Успешно] Реестр сохранен: {report_xlsx_path}")
    
    generate_anomaly_plots(full_df, daily_anomalies, plots_folder)
    print(f"[Успешно] Графики сохранены: {plots_folder}")
    print("\n=== АНАЛИЗ ПОЛНОСТЬЮ ЗАВЕРШЕН ===")

if __name__ == "__main__":
    main()