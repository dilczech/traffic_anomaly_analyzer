"""Command-line adapter for the application service."""
import argparse
import logging
from pathlib import Path

from .models import AnalysisConfig, AnalysisError
from .service import run_analysis
from .sources import discover_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Анализатор аномалий дорожного трафика")
    parser.add_argument("--input", type=Path, default=Path("input_data"), help="Папка исходных .xlsx")
    parser.add_argument("--output", type=Path, default=Path("reports"), help="Папка результатов")
    parser.add_argument("--no-plots", action="store_true", help="Сформировать только Excel-отчёт")
    parser.add_argument("--include-quality-plots", action="store_true", help="Также рисовать дни только с проблемами качества данных")
    parser.add_argument("--monthly-plots", action="store_true", help="Дополнительно построить месячные обзоры")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        sources = discover_sources(args.input)
        result = run_analysis(sources, AnalysisConfig(args.output, not args.no_plots, plot_quality_only_days=args.include_quality_plots, plot_monthly_overview=args.monthly_plots))
    except (AnalysisError, OSError) as exc:
        logging.error("%s", exc)
        return 1

    print(f"Загружено записей: {result.record_count}; участков: {result.section_count}")
    print(f"Срабатываний по суткам: {result.daily_event_count}; уникальных дней по участкам: {result.anomalous_day_count}")
    print(f"Срабатываний по месяцам: {result.monthly_event_count}; уникальных месяцев по участкам: {result.anomalous_month_count}")
    print(f"Событий качества данных: {result.quality_event_count}; устойчивых изменений: {result.regime_event_count}")
    print(f"Месячных систематических расхождений учёта: {result.systematic_event_count}")
    print(f"Отчёт: {result.report_path}")
    if result.plots_dir is not None:
        print(f"Графики: {result.plots_dir}")
        print(f"Диагностических графиков: {result.plot_count}; дней по участкам: {result.plotted_day_count}")
    if result.source_issues:
        print(f"Внимание: пропущено файлов: {len(result.source_issues)}. Результат частичный.")
        return 2
    return 0
