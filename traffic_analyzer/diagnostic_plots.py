"""Each event type has a chart of the quantities actually tested."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .palette import PALETTE, apply_plot_theme
from .quality import CATEGORIES, PLOT_METRICS
from .balance import balance_checks,CHECK_NAMES

BLUE, GRAY, RED, PURPLE = (PALETTE[k] for k in ("fact", "reference", "anomaly", "quality"))
SLUGS = {
    "Часовой спад интенсивности": "hourly-drop",
    "Часовой всплеск интенсивности": "hourly-spike",
    "Аномальный суточный объём": "daily-volume",
    "Нетипичная форма суточного профиля": "profile-shape",
    "Нетипично низкая скорость": "speed-drop",
    "Низкая скорость при высокой загрузке": "speed-load",
    "Нетипичное распределение направлений": "direction-share",
    "Не сходятся суммы транспорта": "balance",
    "Пропуск измерений": "missing-hours",
    "Подозрение на залипание датчика": "flatline",
    "Подозрение на нулевую телеметрию": "zero-telemetry",
    "Некорректные или пустые измерения": "invalid-values",
    "Некорректные категории транспорта": "invalid-categories",
}
CATEGORY_NAMES = dict(zip(CATEGORIES, ["Легковые", "Малые грузовые", "Грузовые", "Большие грузовые", "Автопоезда", "Автобусы", "Мотоциклы"]))


def metric_label(metric):
    group, _, direction = metric.rpartition("_")
    suffix = {"direct": "прямое", "reverse": "обратное", "total": "оба направления"}.get(direction, direction)
    name = {"vol": "Интенсивность", "speed": "Скорость", "load": "Загрузка", **CATEGORY_NAMES}.get(group, group)
    return f"{name}, {suffix}"


def event_mask(events, metric=None):
    mask = np.zeros(24, dtype=bool)
    for _, event in events.iterrows():
        if metric is not None and event["_metric"] != metric:
            continue
        mask[event["Начало"].hour:event["Конец"].hour+1] = True
    return mask


def balance_components(day):
    """Only compare complete operands; missing category values never become zero."""
    return [(CHECK_NAMES[c.key],c.declared.to_numpy(float),c.calculated.to_numpy(float),c.difference.to_numpy(float)) for c in balance_checks(day)]


@dataclass
class DayDiagnostic:
    context: object
    section: str
    date: pd.Timestamp
    events: pd.DataFrame
    day: pd.DataFrame
    reference: pd.DataFrame
    raw: pd.DataFrame


def diagnostic_data(context, section, day, events):
    day = pd.Timestamp(day).normalize()
    h = context.prepared.hourly
    rows = h[h.road_section.eq(section) & h.date.eq(day)].sort_values("timestamp").reset_index(drop=True)
    refs = context.hourly_reference
    refs = refs[refs.road_section.eq(section) & refs.timestamp.dt.normalize().eq(day)].sort_values("timestamp").reset_index(drop=True)
    raw = context.prepared.raw_values
    if raw is None:
        raw = rows
    else:
        raw = raw[raw.road_section.eq(section) & raw.timestamp.dt.normalize().eq(day)].sort_values("timestamp").reset_index(drop=True)
    return DayDiagnostic(context, section, day, events, rows, refs, raw)


def make_axes(rows=1, cols=1, *, height=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_plot_theme(plt)
    return plt.subplots(rows, cols, figsize=(14, height or 2.7*rows+2), squeeze=False)


def finish(fig, diagnostic, title, *, note=""):
    fig.suptitle(f"{diagnostic.section.strip()} · {diagnostic.date:%d.%m.%Y}\n{title}",
                 fontsize=13, color=PALETTE["heading"], weight="medium")
    for ax in fig.axes:
        ax.grid(axis="y", alpha=.5)
        ax.spines[["top", "right"]].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=9, frameon=False)
    if note:
        import textwrap
        fig.text(.5,.025,"\n".join(textwrap.wrap(note,150)),ha="center",fontsize=9,color=PALETTE["text"])
    fig.tight_layout(rect=(.025,.10,.99,.90))
    return fig


def hour_axis(ax, unit):
    ax.set_xlim(-.5,23.5)
    ax.set_xticks([0,3,6,9,12,15,18,21,23])
    ax.set_xlabel("Час суток (00 = 00:00–00:59)")
    ax.set_ylabel(unit)


def measured_line(ax, values, mask=None, *, label="Факт", colour=BLUE):
    values = np.asarray(values,dtype=float)
    hours = np.arange(24)
    ax.plot(hours,values,color=colour,marker=".",lw=1.7,label=label)
    if mask is not None and mask.any():
        ax.plot(hours,np.where(mask,values,np.nan),color=RED,marker="o",lw=2,label="Сработавшие часы")
        for h in np.flatnonzero(mask):
            ax.axvspan(h-.45,h+.45,color=RED,alpha=.07,lw=0)


def reference_line(ax, diagnostic, metric, *, band=True):
    refs=diagnostic.reference
    ax.plot(range(24),refs[f"{metric}_expected"],color=GRAY,ls="--",label="Опорная медиана")
    if band:
        ax.fill_between(range(24),refs[f"{metric}_lower"],refs[f"{metric}_upper"],color=GRAY,alpha=.13,label="Диапазон детектора")


def reference_note(d):
    available=d.events["Опорных дней"].dropna()
    n=int(available.min()) if len(available) else int(d.reference.reference_days.iloc[0])
    return f"Опорных дней: {n}; требуется минимум {d.context.config.min_reference_days}. Диапазон детектора не является доверительным интервалом."


def plot_hourly_volume(d):
    metrics=list(dict.fromkeys(d.events["_metric"]))
    fig,axes=make_axes(len(metrics))
    for ax,metric in zip(axes[:,0],metrics):
        reference_line(ax,d,metric)
        measured_line(ax,d.day[metric],event_mask(d.events,metric))
        ax.set_title(metric_label(metric))
        hour_axis(ax,"авт./ч")
        ax.set_ylim(bottom=0)
    return finish(fig,d,d.events["Тип аномалии"].iloc[0],note=reference_note(d))


def plot_daily_volume(d):
    fig,axes=make_axes(1,2,height=5.5)
    reference_line(axes[0,0],d,"vol_total")
    measured_line(axes[0,0],d.day.vol_total)
    axes[0,0].set_title("Почасовой профиль — контекст, без часовых меток")
    hour_axis(axes[0,0],"авт./ч")
    event=d.events.iloc[0]
    axes[0,1].bar(["Факт за сутки","Опорная норма суток"],[event["Факт"],event["Норма"]],color=[RED,GRAY],width=.55)
    axes[0,1].set_ylabel("авт./сут")
    for i,value in enumerate([event["Факт"],event["Норма"]]):
        axes[0,1].annotate(f"{value:,.0f}".replace(","," "),(i,value),xytext=(0,6),textcoords="offset points",ha="center")
    axes[0,1].margins(y=.18)
    delta=(event["Факт"]-event["Норма"])/event["Норма"]*100 if event["Норма"]>0 else np.nan
    axes[0,1].set_title(f"Отклонение объёма: {delta:+.1f}%" if np.isfinite(delta) else "Опорный объём равен нулю")
    return finish(fig,d,"Аномальный суточный объём",note="Суточный тест не определяет аномальные часы. "+reference_note(d))


def plot_profile(d):
    fig,axes=make_axes(2)
    fact=d.day.vol_total.to_numpy(float)
    norm=d.reference.vol_total_expected.to_numpy(float)
    for ax,actual,expected,unit,title in [
        (axes[0,0],fact,norm,"авт./ч","Исходный профиль"),
        (axes[1,0],fact/fact.sum()*100,norm/norm.sum()*100,"% суточного объёма","Форма после нормирования по суточному объёму"),
    ]:
        measured_line(ax,actual)
        ax.plot(range(24),expected,color=GRAY,ls="--",label="Опорный профиль")
        ax.set_title(title)
        hour_axis(ax,unit)
    correlation=d.events.iloc[0]["Факт"]
    return finish(fig,d,f"Нетипичная форма профиля · корреляция r = {correlation:.3f}; порог 0,70",
                  note="Сравнивается форма всего дня; отдельные часы этим тестом не объявляются аномальными.")


def plot_speed(d):
    metrics=list(dict.fromkeys(d.events["_metric"]))
    fig,axes=make_axes(len(metrics),2)
    for row,metric in enumerate(metrics):
        mask=event_mask(d.events,metric)
        fact=d.day[metric].where(d.day[metric.replace("speed_","vol_")]>0).to_numpy(float)
        reference_line(axes[row,0],d,metric)
        measured_line(axes[row,0],fact,mask)
        axes[row,0].set_title(metric_label(metric))
        hour_axis(axes[row,0],"км/ч")
        difference=fact-d.reference[f"{metric}_expected"].to_numpy(float)
        axes[row,1].bar(range(24),difference,color=[RED if m else GRAY for m in mask],alpha=.8)
        axes[row,1].axhline(-15,color=RED,ls=":",label="Падение на 15 км/ч")
        axes[row,1].set_title("Разность факта и опорной скорости")
        hour_axis(axes[row,1],"км/ч к норме")
    return finish(fig,d,"Нетипично низкая скорость",note="Красные часы прошли оба порога: по разности скорости и робастному Z. "+reference_note(d))


def plot_speed_load(d):
    metrics=list(dict.fromkeys(d.events["_metric"]))
    fig,axes=make_axes(3,len(metrics),height=9)
    for col,speed_metric in enumerate(metrics):
        direction=speed_metric.split("_")[1]
        mask=event_mask(d.events,speed_metric)
        for row,(metric,unit,threshold) in enumerate([
            (speed_metric,"км/ч",d.context.config.jam_speed),
            (f"load_{direction}","%",d.context.config.jam_load),
            (f"vol_{direction}","авт./ч",None),
        ]):
            measured_line(axes[row,col],d.day[metric],mask)
            if threshold is not None:
                axes[row,col].axhline(threshold,color=RED,ls=":",label=f"Порог {threshold:g} {unit}")
            axes[row,col].set_title(metric_label(metric))
            hour_axis(axes[row,col],unit)
            axes[row,col].set_ylim(bottom=0)
    return finish(fig,d,"Низкая скорость при высокой загрузке",
                  note="Красные интервалы совпадают на трёх панелях: скорость ниже порога, загрузка выше порога, поток направления больше нуля. Причина не установлена.")


def plot_directions(d):
    fig,axes=make_axes(1,2,height=5.5)
    ratio=d.day.vol_direct.div(d.day.vol_total.where(d.day.vol_total>0))*100
    norm=d.reference.vol_direct_expected.div(d.reference.vol_total_expected.where(d.reference.vol_total_expected>0))*100
    measured_line(axes[0,0],ratio)
    axes[0,0].plot(range(24),norm,color=GRAY,ls="--",label="Опорная доля")
    axes[0,0].axvspan(6.5,21.5,color=GRAY,alpha=.10,label="Дневной интервал 07–21 ч")
    hour_axis(axes[0,0],"Доля прямого направления, %")
    axes[0,0].set_ylim(0,100)
    event=d.events.iloc[0]
    axes[0,1].bar(["Факт 07–21 ч","Норма 07–21 ч"],[event["Факт"],event["Норма"]],color=[RED,GRAY])
    axes[0,1].set_ylim(0,100)
    axes[0,1].set_title(f"Отклонение: {event['Факт']-event['Норма']:+.1f} п.п.")
    axes[0,1].set_ylabel("Доля прямого направления, %")
    return finish(fig,d,"Нетипичное распределение направлений",note="Дневные доли считаются из суммарных объёмов, а не усреднением часовых процентов.")


def plot_balance(d):
    checks=balance_checks(d.day,d.context.config)
    rows=[]
    for check in checks:
        for hour in np.flatnonzero(check.significant):
            label="Сумма направлений" if check.key=="directions" else f"Сумма категорий, {check.direction.lower()}"
            values=(float(check.declared.iloc[hour]),float(check.calculated.iloc[hour]),float(check.difference.iloc[hour]),float(check.tolerance.iloc[hour]))
            if rows and rows[-1]["key"]==check.key and rows[-1]["end"]==hour-1 and rows[-1]["values"]==values:
                rows[-1]["end"]=hour
            else:
                rows.append({"key":check.key,"start":hour,"end":hour,"label":label,"values":values})
    if not rows:
        raise ValueError("Событие баланса не подтверждается операндами графика.")
    fig,axes=make_axes(height=max(4.5,len(rows)*.43+2.8))
    ax=axes[0,0]
    ax.axis("off")
    cells=[]
    for row in rows:
        left,right,difference,tolerance=row["values"]
        cells.append([f"{row['start']:02d}:00–{row['end']:02d}:59",row["label"],f"{left:g}",f"{right:g}",f"{difference:+g}",f"{tolerance:.2f}"])
    table=ax.table(cellText=cells,colLabels=["Час / интервал","Контрольная сумма","Объявлено, авто","Сумма, авто","Разность, авто","Допуск, авто"],
                   colWidths=[.16,.32,.14,.14,.12,.12],cellLoc="center",bbox=[0,.03,1,.88])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_linewidth(.4)
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_facecolor(PALETTE["header"] if row==0 else PALETTE["stripe"] if row%2 else PALETTE["paper"])
        cell.get_text().set_color(PALETTE["header_text"] if row==0 else RED if col==4 else PALETTE["text"])
    ax.set_title(f"Допуск: max({d.context.config.balance_absolute_tolerance:g} авто; {d.context.config.balance_relative_pct:g}% большего из сравниваемых значений)")
    return finish(fig,d,"Расхождения количества транспорта выше допуска",
                  note="Показаны конкретные числа без совпадающих кривых. Разность = объявлено − сумма. Связанные проверки не суммируются как независимые ошибки; одинаковые соседние часы объединены.")


def status_matrix(d, *, zero_packet=False):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    if zero_packet:
        values=d.raw[PLOT_METRICS].to_numpy(float).T
        status=np.where(np.isfinite(values),np.where(values==0,2,1),0)
        status[:,d.day.zero_packet.to_numpy()]=3
        labels=[metric_label(m) for m in PLOT_METRICS]
        cell=lambda r,h: "—" if not np.isfinite(values[r,h]) else f"{values[r,h]:g}"
        title="Подозрение на нулевую телеметрию"
    else:
        flags=np.vstack([d.day.observed.to_numpy(),d.day.valid_volume.to_numpy()])
        status=np.where(flags,1,0)
        labels=["Измерение присутствует","Объём пригоден для анализа"]
        cell=lambda r,h: "Да" if flags[r,h] else "Нет"
        title="Пропуск измерений"
    fig,axes=make_axes(height=5.5 if zero_packet else 4.5)
    ax=axes[0,0]
    colours=[PALETTE["grid"],PALETTE["stripe"],"#ECE6F1",PURPLE]
    ax.imshow(status,aspect="auto",interpolation="nearest",vmin=0,vmax=3,cmap=ListedColormap(colours))
    ax.set_yticks(range(len(labels)),labels)
    ax.set_xticks(range(24),[f"{h:02d}" for h in range(24)])
    ax.set_xlabel("Час суток")
    ax.grid(False)
    for r in range(len(labels)):
        for h in range(24):
            ax.text(h,r,cell(r,h),ha="center",va="center",fontsize=8,color=PALETTE["paper"] if status[r,h]==3 else PALETTE["text"])
    handles=[Patch(color=colours[0],label="Нет пригодного измерения"),Patch(color=colours[1],label="Измерение присутствует")]
    if zero_packet:
        handles += [Patch(color=colours[2],label="Измеренный ноль"),Patch(color=colours[3],label="Длительная серия нулей всех каналов")]
    fig.legend(handles=handles,loc="lower center",ncol=2,fontsize=9,bbox_to_anchor=(.5,.09),frameon=False)
    result=finish(fig,d,title,note=f"Записано {int(d.day.observed.sum())} из 24 часов; пригодно по объёму {int(d.day.valid_volume.sum())}. Исходные нули отличаются от пропусков.")
    ax.grid(False)
    return result


def plot_flatline(d):
    fig,axes=make_axes(2)
    measured_line(axes[0,0],d.day.vol_total)
    mask=event_mask(d.events)
    axes[0,0].plot(range(24),d.day.vol_total.where(mask),color=PURPLE,marker="s",label="Подозрительная постоянная серия")
    axes[0,0].set_title("Измеренная интенсивность")
    hour_axis(axes[0,0],"авт./ч")
    axes[1,0].bar(range(24),d.day.vol_total.diff(),color=GRAY)
    axes[1,0].axhline(0,color=PURPLE,ls=":",label="Нет изменения от прошлого часа")
    axes[1,0].set_title("Разность соседних измерений")
    hour_axis(axes[1,0],"авт./ч к предыдущему часу")
    return finish(fig,d,"Подозрение на залипание датчика",note="Показан фрагмент серии в границах суток. Проверка последовательности выполняется и через полночь; отсутствие изменения не доказывает поломку.")


def plot_invalid(d):
    metrics=list(dict.fromkeys(d.events["_metric"]))
    if d.events["Тип аномалии"].iloc[0]=="Некорректные категории транспорта":
        metrics=[c for c in d.raw if any(c.startswith(category+"_") for category in CATEGORIES)
                 and ((~np.isfinite(d.raw[c])) | d.raw[c].lt(0) | d.raw[c].mod(1).ne(0)).any()]
    fig,axes=make_axes(len(metrics))
    for ax,metric in zip(axes[:,0],metrics):
        measured_line(ax,d.raw[metric],label="Исходные числовые значения")
        clean=d.day[metric]
        invalid=d.day.observed & clean.isna()
        for h in np.flatnonzero(invalid):
            ax.axvspan(h-.45,h+.45,color=PURPLE,alpha=.15)
            if not np.isfinite(d.raw[metric].iloc[h]):
                ax.text(h,.05,"нет\nзначения",transform=ax.get_xaxis_transform(),ha="center",fontsize=8)
        ax.set_title(metric_label(metric))
        hour_axis(ax,"км/ч" if metric.startswith("speed_") else "%" if metric.startswith("load_") else "авт.")
    return finish(fig,d,d.events["Тип аномалии"].iloc[0],note="Фиолетовым отмечены поля, исключённые при проверке диапазона или наличия измерения. Непригодные числа показаны как исходные; нечисловые значения остаются пропусками.")


RENDERERS = {
    "Часовой спад интенсивности": plot_hourly_volume,
    "Часовой всплеск интенсивности": plot_hourly_volume,
    "Аномальный суточный объём": plot_daily_volume,
    "Нетипичная форма суточного профиля": plot_profile,
    "Нетипично низкая скорость": plot_speed,
    "Низкая скорость при высокой загрузке": plot_speed_load,
    "Нетипичное распределение направлений": plot_directions,
    "Не сходятся суммы транспорта": plot_balance,
    "Пропуск измерений": status_matrix,
    "Подозрение на нулевую телеметрию": lambda d: status_matrix(d,zero_packet=True),
    "Подозрение на залипание датчика": plot_flatline,
    "Некорректные или пустые измерения": plot_invalid,
    "Некорректные категории транспорта": plot_invalid,
}


def plot_diagnostic(context, section, day, events, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds=events["Тип аномалии"].unique()
    if len(kinds)!=1:
        raise ValueError("Один диагностический график должен объяснять один тип события.")
    diagnostic=diagnostic_data(context,section,day,events)
    fig=RENDERERS[kinds[0]](diagnostic)
    try:
        output_path=Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output_path,dpi=110)
    finally:
        plt.close(fig)
