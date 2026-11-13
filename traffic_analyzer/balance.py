"""One tolerance definition for quality gates, monthly auditing and diagnostics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import AnalysisConfig

CATEGORIES = ["cars", "small_trucks", "med_trucks", "large_trucks", "road_trains", "buses", "motos"]
VOLUME_COLUMNS = ["vol_total", "vol_direct", "vol_reverse"]
CHECK_NAMES = {
    "directions": "Общий поток и сумма направлений",
    "categories_total": "Общий поток и сумма категорий",
    "categories_direct": "Прямое направление и сумма категорий",
    "categories_reverse": "Обратное направление и сумма категорий",
}


@dataclass
class BalanceCheck:
    key: str
    direction: str
    declared: pd.Series
    calculated: pd.Series
    difference: pd.Series
    tolerance: pd.Series
    valid: pd.Series
    significant: pd.Series
    minor: pd.Series


def balance_checks(frame, config=None):
    config=config or AnalysisConfig()
    operands=[("directions","Оба",frame.vol_total,frame[["vol_direct","vol_reverse"]].sum(axis=1,min_count=2))]
    for suffix,label in (("total","Оба"),("direct","Прямое"),("reverse","Обратное")):
        columns=[f"{category}_{suffix}" for category in CATEGORIES]
        if set(columns).issubset(frame.columns):
            operands.append((f"categories_{suffix}",label,frame[f"vol_{suffix}"],frame[columns].sum(axis=1,min_count=len(columns))))
    result=[]
    for key,direction,left,right in operands:
        valid=left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
        difference=(left-right).where(valid)
        scale=pd.concat([left.abs(),right.abs()],axis=1).max(axis=1)
        tolerance=(scale*config.balance_relative_pct/100).clip(lower=config.balance_absolute_tolerance).where(valid)
        significant=valid & difference.abs().gt(tolerance)
        minor=valid & difference.ne(0) & ~significant
        result.append(BalanceCheck(key,direction,left,right,difference,tolerance,valid,significant,minor))
    return result


def detail_frame(section, timestamps, checks):
    records=[]
    for check in checks:
        records.append(pd.DataFrame({
            "road_section":section,"timestamp":np.asarray(timestamps),"check":check.key,
            "direction":check.direction,"declared":check.declared.to_numpy(),"calculated":check.calculated.to_numpy(),
            "difference":check.difference.to_numpy(),"tolerance":check.tolerance.to_numpy(),
            "valid":check.valid.to_numpy(),"significant":check.significant.to_numpy(),"minor":check.minor.to_numpy(),
        }))
    return pd.concat(records,ignore_index=True)


def monthly_balance_summary(details, config):
    data=details.copy()
    data["month"]=data.timestamp.dt.to_period("M").astype(str)
    rows=[]
    for (section,month,key),all_rows in data.groupby(["road_section","month","check"]):
        checked=all_rows[all_rows.valid]
        errors=checked[checked.difference.ne(0)]
        minor=checked[checked.minor]
        significant=checked[checked.significant]
        hours=len(errors)
        days=errors.timestamp.dt.normalize().nunique()
        positive=int(errors.difference.gt(0).sum())
        negative=int(errors.difference.lt(0).sum())
        share=max(positive,negative)/hours*100 if hours else 0.
        systematic=hours>=config.systematic_min_hours and days>=config.systematic_min_days and share>=config.systematic_sign_share_pct
        rows.append({
            "Участок дороги":section,"Период":month,"Проверка":CHECK_NAMES[key],"Направление":all_rows.direction.iloc[0],
            "Проверено часов":len(checked),"Часов с расхождением":hours,"Часов в допуске":len(minor),
            "Доля мелких расхождений (%)":len(minor)/len(checked)*100 if len(checked) else np.nan,
            "Дней с мелкими расхождениями":minor.timestamp.dt.normalize().nunique(),
            "Часов выше допуска":len(significant),"Дней с расхождениями":days,
            "Макс. расхождение, авт.":errors.difference.abs().max() if hours else 0.,
            "Медианная разность, авт.":errors.difference.median() if hours else np.nan,
            "Преобладающий знак":"+" if positive>negative else "−" if negative>positive else "Нет",
            "Доля преобладающего знака (%)":share if hours else np.nan,
            "Систематическое расхождение":"Да" if systematic else "Нет",
            "Основание":f"Часов {hours}/{config.systematic_min_hours}; дней {days}/{config.systematic_min_days}; одинаковый знак {share:.1f}%/{config.systematic_sign_share_pct:g}%.",
            "_check":key,
        })
    return pd.DataFrame(rows)


def systematic_balance_events(details, summary, config):
    from .events import TrafficEvent,event_frame
    events=[]
    triggered=summary[summary["Систематическое расхождение"].eq("Да")]
    for (section,period),group in triggered.groupby(["Участок дороги","Период"]):
        month=pd.Period(period)
        rows=details[details.road_section.eq(section) & details.timestamp.dt.to_period("M").eq(month) & details["check"].isin(group["_check"]) & details.valid & details.difference.ne(0)]
        unique_hours=rows.timestamp.nunique()
        unique_days=rows.timestamp.dt.normalize().nunique()
        significant_hours=rows.loc[rows.significant,"timestamp"].nunique()
        descriptions=[f"{r['Проверка']}: {r['Часов с расхождением']} ч, {r['Дней с расхождениями']} дней, знак {r['Преобладающий знак']} в {r['Доля преобладающего знака (%)']:.1f}% случаев" for _,r in group.iterrows()]
        events.append(TrafficEvent(
            section,month.start_time,month.end_time.floor("h"),"Качество данных","Систематическое расхождение учёта",
            metric="balance_difference",scope="Месяц",value=unique_hours,expected=config.systematic_min_hours,
            severity="Высокий" if significant_hours>=config.systematic_min_hours else "Наблюдение",
            description="; ".join(descriptions)+f". Всего уникальных часов {unique_hours}, дней {unique_days}; выше допуска {significant_hours} ч. Связанные проверки одного часа не суммируются. Сигнал показывает несогласованность учёта, не доказывает ошибку конкретного датчика.",
        ))
    return event_frame(events)
