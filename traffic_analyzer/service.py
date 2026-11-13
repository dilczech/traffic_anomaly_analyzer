"""Application pipeline independent of command-line argument parsing."""
import logging
from collections.abc import Iterable
from pathlib import Path
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

from .detectors import detect_daily_anomalies, detect_monthly_anomalies, detect_regime_changes
from .baseline import build_context
from .quality import prepare_data
from .models import AnalysisConfig, AnalysisError, AnalysisResult, SourceIssue
from .parser import parse_traffic_excel
from .reporting import export_anomalies_to_excel
from .sources import FileSource
from .validation import InputValidationError
from .balance import monthly_balance_summary,systematic_balance_events

logger = logging.getLogger(__name__)


def run_analysis(sources: Iterable[FileSource], config: AnalysisConfig) -> AnalysisResult:
    frames = []
    issues = []
    for source in sources:
        logger.info("Чтение %s", source.name)
        try:
            with source.materialize() as path:
                frame = parse_traffic_excel(path)
        except (InputValidationError, OSError, BadZipFile, InvalidFileException, ParseError) as exc:
            issues.append(SourceIssue(source.name, str(exc)))
            logger.warning("Пропущен %s: %s", source.name, exc)
            continue
        frames.append(frame)

    if not frames:
        details = "; ".join(f"{issue.source}: {issue.message}" for issue in issues)
        raise AnalysisError("Нет пригодных данных для анализа." + (f" {details}" if details else ""))

    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["road_section", "timestamp"]).any():
        raise AnalysisError(
            "Найдены повторяющиеся записи «участок + время». "
            "Уберите перекрывающиеся выгрузки или разрешите конфликты перед анализом."
        )
    try:
        prepared = prepare_data(data, config=config)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from exc
    logger.info("Расчёт общей нормы по сопоставимым дням")
    context = build_context(prepared, config)
    daily = detect_daily_anomalies(context)
    monthly = detect_monthly_anomalies(context)
    audit=monthly_balance_summary(prepared.balance_details,config)
    persistent=systematic_balance_events(prepared.balance_details,audit,config)
    if not persistent.empty:
        monthly=pd.DataFrame(monthly.to_dict("records")+persistent.to_dict("records"),columns=monthly.columns)
    regimes = detect_regime_changes(context)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "traffic_anomalies_report.xlsx"
    export_anomalies_to_excel(daily, monthly, report_path, context=context, regimes=regimes, source_issues=issues)
    plots_dir = None
    plot_count = plotted_day_count = 0
    if config.generate_plots:
        from .plots import generate_anomaly_plots, generate_monthly_plots
        plots_dir = output_dir / "plots"
        batch = generate_anomaly_plots(context, daily, plots_dir)
        plot_count, plotted_day_count = batch.file_count, batch.day_count
        from .systematic_plots import generate_systematic_plots
        periods=generate_systematic_plots(context,plots_dir/"systematic")
        if periods:
            import json
            manifest=json.loads(batch.manifest_path.read_text(encoding="utf-8"))
            manifest["daily_file_count"]=batch.file_count
            manifest["period_charts"]=[dict(row,file="systematic/"+row["file"]) for row in periods]
            manifest["file_count"]=batch.file_count+len(periods)
            batch.manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
            plot_count+=len(periods)
        if config.plot_monthly_overview:
            generate_monthly_plots(context, monthly, plots_dir / "months")

    return AnalysisResult(
        record_count=len(data),
        section_count=data["road_section"].nunique(),
        daily_event_count=len(daily),
        anomalous_day_count=0 if daily.empty else len(daily[["Участок дороги", "Дата"]].drop_duplicates()),
        monthly_event_count=len(monthly),
        anomalous_month_count=0 if monthly.empty else len(monthly.assign(month=monthly["Начало"].dt.to_period("M"))[["Участок дороги", "month"]].drop_duplicates()),
        report_path=report_path,
        plots_dir=plots_dir,
        source_issues=tuple(issues),
        quality_event_count=int(daily["Категория"].eq("Качество данных").sum()),
        regime_event_count=len(regimes),
        plot_count=plot_count,
        plotted_day_count=plotted_day_count,
        systematic_event_count=len(persistent),
    )
