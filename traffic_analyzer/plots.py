"""Select anomalous days and dispatch one diagnostic per event type."""
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

from .detectors import as_context
from .diagnostic_plots import SLUGS, plot_diagnostic
from .monthly_plots import generate_monthly_plots

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlotBatch:
    file_count: int
    day_count: int
    manifest_path: Path


def select_plot_events(events, *, include_quality_only=False):
    """Keep all diagnostics on selected traffic days, not quality-only days."""
    if events.empty or include_quality_only:
        return events.copy()
    keys=["Участок дороги","Дата"]
    selected=events.loc[events["Категория"].eq("Трафик"),keys].drop_duplicates()
    return events.merge(selected,on=keys,how="inner",validate="many_to_one")


def plot_day(context, section, day, events, output_path):
    """Compatibility entry point, now accepts exactly one diagnostic type."""
    return plot_diagnostic(context,section,day,events,output_path)


def generate_anomaly_plots(data, daily_anomalies_df, output_folder):
    folder=Path(output_folder)
    folder.mkdir(parents=True,exist_ok=True)
    if daily_anomalies_df.empty:
        manifest=folder/"index.json"
        manifest.write_text(json.dumps({"day_count":0,"file_count":0,"charts":[]}),encoding="utf-8")
        return PlotBatch(0,0,manifest)
    context=as_context(data)
    selected=select_plot_events(daily_anomalies_df,include_quality_only=context.config.plot_quality_only_days)
    day_count=len(selected[["Участок дороги","Дата"]].drop_duplicates())
    grouped=selected.groupby(["Участок дороги","Дата","Тип аномалии"])
    logger.info("Диагностика: %s графиков для %s аномальных дней по участкам",len(grouped),day_count)
    records=[]
    for i,((section,day,kind),events) in enumerate(grouped,1):
        slug=SLUGS[kind]
        digest=hashlib.sha256(section.encode("utf-8")).hexdigest()[:10]
        relative=Path(f"{day}_{digest}")/f"{slug}.png"
        plot_diagnostic(context,section,day,events,folder/relative)
        records.append({"section":section,"date":str(day),"kind":kind,
                        "category":str(events["Категория"].iloc[0]),"file":relative.as_posix(),
                        "intervals":[{"start":str(e["Начало"]),"end":str(e["Конец"])} for _,e in events.iterrows()]})
        if i%50==0:
            logger.info("Построено диагностик: %s/%s",i,len(grouped))
    manifest=folder/"index.json"
    manifest.write_text(json.dumps({"selection":"all_signals" if context.config.plot_quality_only_days else "traffic_days",
                                    "day_count":day_count,"file_count":len(records),"charts":records},ensure_ascii=False,indent=2),encoding="utf-8")
    # Reused output directories may contain files from an earlier analysis.
    # Preserve them, but make the current set unambiguous via index.json.
    return PlotBatch(len(records),day_count,manifest)
