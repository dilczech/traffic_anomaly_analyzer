"""Quality-aware, explainable hourly, daily and period detectors."""
import numpy as np
import pandas as pd

from .baseline import AnalysisContext, build_context
from .events import TrafficEvent, event_frame, runs, EVENT_COLUMNS
from .models import AnalysisConfig
from .quality import prepare_data


def as_context(data, config=None):
    if isinstance(data,AnalysisContext):
        return data
    config=config or AnalysisConfig()
    return build_context(prepare_data(data,config=config),config)


def percentage(value, expected):
    return (value-expected)/expected*100 if pd.notna(expected) and expected > 0 else None


def detect_daily_anomalies(data, config=None):
    context = as_context(data, config)
    policy = context.config
    events = []
    merged = context.prepared.hourly.merge(context.hourly_reference, on=["road_section", "timestamp"], validate="one_to_one")
    daily = context.prepared.daily.merge(context.daily_reference, on=["road_section", "date"], validate="one_to_one")
    for (section, day), g in merged.groupby(["road_section", "date"]):
        g = g.sort_values("timestamp")
        n = int(g.reference_days.iloc[0])
        # A flagged hour is an actual hourly test, not inferred from a daily total.
        for metric, direction in (("vol_total", "Оба"), ("vol_direct", "Прямое"), ("vol_reverse", "Обратное")):
            expected = g[f"{metric}_expected"]
            residual = g[metric]-expected
            z = residual/g[f"{metric}_sigma"]
            practical = residual.abs() >= np.maximum(expected.abs()*policy.hourly_delta_pct/100, 50)
            for sign, kind in ((-1, "Часовой спад интенсивности"), (1, "Часовой всплеск интенсивности")):
                mask = g.valid_volume & z.mul(sign).ge(policy.z_threshold) & practical
                for start, end in runs(mask.to_numpy()):
                    interval = g.iloc[start:end+1]
                    value = interval[metric].sum()
                    norm = interval[f"{metric}_expected"].sum()
                    events.append(TrafficEvent(
                        section, interval.timestamp.iloc[0], interval.timestamp.iloc[-1], "Трафик", kind,
                        metric=metric, direction=direction, value=value, expected=norm,
                        delta_pct=percentage(value,norm), score=float(z.iloc[start:end+1].abs().max()), reference_days=int(interval[f"{metric}_n"].min()),
                        severity="Высокий" if end>start else "Наблюдение",
                        description="Факт и норма — сумма автомобилей за отмеченный интервал. Каждый час прошёл робастный и практический пороги.",
                    ))
        for direction, label in (("direct", "Прямое"), ("reverse", "Обратное")):
            metric = f"speed_{direction}"
            # Speed/load must coincide and have a real measured directional flow.
            mask = g.valid_volume & g[metric].lt(policy.jam_speed) & g[f"load_{direction}"].gt(policy.jam_load) & g[f"vol_{direction}"].gt(0)
            for start, end in runs(mask.to_numpy()):
                interval = g.iloc[start:end+1]
                events.append(TrafficEvent(
                    section, interval.timestamp.iloc[0], interval.timestamp.iloc[-1], "Трафик", "Низкая скорость при высокой загрузке",
                    metric=metric, direction=label, value=interval[metric].min(), expected=policy.jam_speed,
                    severity="Высокий",
                    description=f"В одни и те же часы скорость < {policy.jam_speed:g} км/ч, загрузка > {policy.jam_load:g}%. Это признак нарушения режима, причина не установлена.",
                ))
            expected = g[f"{metric}_expected"]
            mask = g.valid_volume & g[f"vol_{direction}"].gt(0) & (g[metric]-expected).le(-15) & ((g[metric]-expected)/g[f"{metric}_sigma"]).le(-policy.z_threshold)
            for start, end in runs(mask.to_numpy()):
                interval = g.iloc[start:end+1]
                events.append(TrafficEvent(
                    section, interval.timestamp.iloc[0], interval.timestamp.iloc[-1], "Трафик", "Нетипично низкая скорость",
                    metric=metric, direction=label, value=interval[metric].mean(), expected=interval[f"{metric}_expected"].mean(),
                    score=float(((interval[metric]-interval[f"{metric}_expected"])/interval[f"{metric}_sigma"]).abs().max()),
                    reference_days=int(interval[f"{metric}_n"].min()), severity="Наблюдение",
                    description="Средняя скорость интервала; каждый отмеченный час ниже своей нормы минимум на 15 км/ч и прошёл робастный порог.",
                ))
        stats = daily[daily.road_section.eq(section) & daily.date.eq(day)].iloc[0]
        if not stats.complete or pd.isna(stats.expected):
            continue
        z = (stats.daily_vol-stats.expected)/stats.sigma
        delta = percentage(stats.daily_vol, stats.expected)
        if abs(z) >= policy.z_threshold and abs(stats.daily_vol-stats.expected) >= max(stats.expected*policy.daily_delta_pct/100, 100):
            events.append(TrafficEvent(
                section, day, day+pd.Timedelta(hours=23), "Трафик", "Аномальный суточный объём",
                scope="Сутки", value=stats.daily_vol, expected=stats.expected,
                delta_pct=delta, score=abs(z), reference_days=n, severity="Высокий",
                description="Сравниваются только полные пригодные сутки с другими однотипными днями. Не означает, что каждый час аномален.",
            ))
        actual, norm = g.vol_total.to_numpy(), g.vol_total_expected.to_numpy()
        if np.isfinite(norm).all() and actual.std()>0 and norm.std()>0:
            correlation = float(np.corrcoef(actual,norm)[0,1])
            if correlation < .70 and actual.sum()>1000:
                events.append(TrafficEvent(
                    section, day, day+pd.Timedelta(hours=23), "Трафик", "Нетипичная форма суточного профиля",
                    scope="Сутки", value=correlation, expected=.70, reference_days=n,
                    description="Факт — корреляция с опорным профилем. Общая форма отличается; отдельные часы этим тестом не маркируются.",
                ))
        daytime = g[g.timestamp.dt.hour.between(7,21)]
        if daytime.vol_total.sum()>500:
            observed_ratio = daytime.vol_direct.sum()/daytime.vol_total.sum()
            norm_total = daytime.vol_total_expected.sum(min_count=15)
            norm_ratio = daytime.vol_direct_expected.sum(min_count=15)/norm_total if norm_total>0 else np.nan
            if pd.notna(norm_ratio) and abs(observed_ratio-norm_ratio)>=.25:
                events.append(TrafficEvent(
                    section, day, day+pd.Timedelta(hours=23), "Трафик", "Нетипичное распределение направлений",
                    scope="Сутки", metric="vol_direct", direction="Прямое", value=observed_ratio*100,
                    expected=norm_ratio*100, reference_days=n,
                    description=f"Доля прямого направления за 07–21 ч отличается от своей нормы на {(observed_ratio-norm_ratio)*100:+.1f} п.п.",
                ))
    traffic = event_frame(events)
    records=context.prepared.quality_events.to_dict("records")+traffic.to_dict("records")
    from .priorities import promote_correlated_events
    return promote_correlated_events(pd.DataFrame(records,columns=EVENT_COLUMNS)).sort_values(["Участок дороги", "Начало"]).reset_index(drop=True)


def detect_regime_changes(data, config=None):
    context = as_context(data, config)
    events = []
    for section, daily in context.prepared.daily.groupby("road_section"):
        daily = daily.sort_values("date").copy()
        daily["group"] = daily.date.map(lambda d: 7 if d.date() in context.config.holiday_dates else d.weekday())
        # A fixed historical comparison does not adapt away a sustained new regime.
        for i in range(len(daily)):
            if not daily.iloc[i].complete:
                continue
            target = daily.iloc[i]
            history = daily.iloc[:i]
            history = history[history.complete & history.group.eq(target.group) & (history.date >= target.date-pd.Timedelta(days=168))]
            if len(history)<context.config.min_reference_days:
                continue
            segment = daily.iloc[i:i+context.config.min_regime_days]
            if len(segment)<context.config.min_regime_days or not segment.complete.all():
                continue
            norms=[]
            sample_counts=[]
            for _, row in segment.iterrows():
                peers=daily.iloc[:i]
                peers=peers[peers.complete & peers.group.eq(row.group) & (peers.date >= target.date-pd.Timedelta(days=168))]
                norms.append(peers.daily_vol.median() if len(peers)>=context.config.min_reference_days else np.nan)
                sample_counts.append(len(peers))
            norms_array=np.asarray(norms,dtype=float)
            ratios=np.divide(segment.daily_vol.to_numpy(),norms_array,out=np.full(len(norms),np.nan),where=norms_array>0)
            if np.isfinite(ratios).all() and ((ratios<.75).all() or (ratios>1.25).all()):
                start, end=segment.date.iloc[0],segment.date.iloc[-1]
                if events and events[-1].section==section and start<=events[-1].end:
                    continue
                events.append(TrafficEvent(
                    section,start,end+pd.Timedelta(hours=23),"Изменение режима","Устойчивое изменение интенсивности",
                    scope="Период",value=segment.daily_vol.mean(),expected=float(np.mean(norms)),
                    delta_pct=percentage(segment.daily_vol.mean(),float(np.mean(norms))),reference_days=min(sample_counts),severity="Высокий",
                    description=f"Не менее {context.config.min_regime_days} полных дней подряд по одну сторону от исторической нормы на 25% и более. Факт/норма — среднесуточные объёмы; это окно наблюдения, не точная дата окончания режима.",
                ))
    return event_frame(events)


def monthly_summary(data, config=None):
    context = as_context(data, config)
    h=context.prepared.hourly.copy()
    h["month"]=h.timestamp.dt.to_period("M")
    records=[]
    for (section,month),g in h.groupby(["road_section","month"]):
        d=context.prepared.daily
        d=d[d.road_section.eq(section)&d.date.dt.to_period("M").eq(month)]
        expected=month.days_in_month*24
        full=d[d.complete]
        row={"Участок дороги":section,"Период":str(month),"Год":month.year,"Месяц":month.month,
             "Измерено часов":int(g.observed.sum()),"Ожидается часов":expected,
             "Полнота (%)":g.observed.sum()/expected*100,"Пригодно часов (%)":g.valid_volume.sum()/expected*100,
             "Полных пригодных суток":len(full),"ССИ":full.daily_vol.mean()}
        valid=g[g.date.isin(full.date)]
        total=valid.vol_total.sum()
        row["Доля тяжёлых (%)"]=(valid.large_trucks_total.sum()+valid.road_trains_total.sum())/total*100 if total>0 else np.nan
        for direction,label in (("direct","прямое"),("reverse","обратное")):
            speed=valid[f"speed_{direction}"]
            volume=valid[f"vol_{direction}"]
            mask=speed.notna() & volume.gt(0)
            row[f"Скорость, {label}"]=(speed[mask]*volume[mask]).sum()/volume[mask].sum() if mask.any() else np.nan
        records.append(row)
    result=pd.DataFrame(records)
    result["Пригоден для сравнения"]=result["Пригодно часов (%)"].ge(context.config.min_month_coverage) & result["ССИ"].notna()
    result["Статус YoY"]="Недостаточно сопоставимых лет"
    for i,row in result.iterrows():
        peers=result[result["Участок дороги"].eq(row["Участок дороги"]) & result["Месяц"].eq(row["Месяц"]) & result["Год"].ne(row["Год"]) & result["Пригоден для сравнения"]]
        result.loc[i,"Статус YoY"]="Месяц неполный или содержит сбои" if not row["Пригоден для сравнения"] else "Доступно сравнение" if len(peers) else "Недостаточно сопоставимых лет"
    return result


def detect_monthly_anomalies(data, config=None):
    context=as_context(data,config)
    summary=monthly_summary(context)
    events=[]
    for _,row in summary.iterrows():
        month=pd.Period(row["Период"])
        start,end=month.start_time,month.end_time.floor("h")
        section=row["Участок дороги"]
        if row["Пригодно часов (%)"]<context.config.min_month_coverage:
            events.append(TrafficEvent(section,start,end,"Качество данных","Неполный или непригодный месяц",scope="Месяц",
                value=row["Пригодно часов (%)"],expected=context.config.min_month_coverage,severity="Высокий",
                description=f"Измерено {row['Полнота (%)']:.1f}% календарных часов; пригодно {row['Пригодно часов (%)']:.1f}%. Проверка не зависит от наличия других лет."))
        if row["Статус YoY"]!="Доступно сравнение":
            continue
        peers=summary[summary["Участок дороги"].eq(section)&summary["Месяц"].eq(month.month)&summary["Год"].ne(month.year)&summary["Пригоден для сравнения"]]
        specs=[("ССИ",25,"vol_total","Оба"),("Доля тяжёлых (%)",8,"heavy_share","Оба"),
               ("Скорость, прямое",15,"speed_direct","Прямое"),("Скорость, обратное",15,"speed_reverse","Обратное")]
        for column,threshold,metric,direction in specs:
            norm=peers[column].median();value=row[column]
            delta=percentage(value,norm)
            trigger=(delta is not None and abs(delta)>=threshold) if metric=="vol_total" else abs(value-norm)>=threshold if metric=="heavy_share" else value-norm<=-threshold
            if trigger:
                events.append(TrafficEvent(section,start,end,"Трафик",f"Межгодовое отклонение: {column}",metric=metric,direction=direction,scope="Месяц",
                    value=value,expected=norm,delta_pct=delta,reference_years=len(peers),severity="Высокий",
                    description=f"Сравнение с {len(peers)} другими годами; проверяемый год исключён. Для доли порог в п.п., для скорости — км/ч. При одном опорном годе многолетняя устойчивость не оценена."))
    return event_frame(events)


def yearly_summary(data, config=None):
    context=as_context(data,config)
    days=context.prepared.daily.copy()
    days["year"]=days.date.dt.year
    records=[]
    for (section,year),g in days.groupby(["road_section","year"]):
        expected=366 if pd.Timestamp(year,12,31).dayofyear==366 else 365
        full=g[g.complete]
        records.append({"Участок дороги":section,"Год":year,"Начало наблюдений":g.date.min(),"Конец наблюдений":g.date.max(),
                        "Календарных дней в году":expected,"Дней в наблюдаемом периоде":len(g),"Полных пригодных суток":len(full),
                        "Среднесуточный объём":full.daily_vol.mean(),"Статус":"Полный год" if len(full)==expected else "Неполный год; показатель относится к доступным полным суткам"})
    return pd.DataFrame(records)
