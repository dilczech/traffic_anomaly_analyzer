"""Export purpose-specific sheets without changing analysis data."""
from pathlib import Path
import math

import openpyxl
import pandas as pd
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill

from .legend import legend_rows
from .models import AnalysisConfig
from .events import EVENT_COLUMNS
from .priorities import CHECK,FIRST
from .palette import excel_colour
from .report_views import (
    MISSING, coverage_table, display_missing, monthly_events_view,
    monthly_indicators_view, quality_view, regime_view, traffic_view,
)


def build_report_sheets(daily, monthly, *, context=None, regimes=None, source_issues=()):
    from .detectors import monthly_summary, yearly_summary

    config = context.config if context is not None else AnalysisConfig()
    if daily.empty:
        daily = daily.reindex(columns=EVENT_COLUMNS)
    quality = daily[daily["Категория"].eq("Качество данных")]
    traffic = daily[daily["Категория"].eq("Трафик")]
    regimes = regimes if regimes is not None else pd.DataFrame()
    summary = [
        ("Событий качества данных по часам", len(quality)),
        ("Событий трафика", len(traffic)),
        ("Устойчивых изменений", len(regimes)),
        ("Событий по месяцам", len(monthly)),
        ("Пропущено источников", len(source_issues)),
    ]
    if context is not None:
        hourly = context.prepared.hourly
        summary.extend([
            ("Начало анализируемого периода", hourly.timestamp.min()),
            ("Конец анализируемого периода", hourly.timestamp.max()),
            ("Участков", hourly.road_section.nunique()),
            ("Исходных почасовых записей", int(hourly.observed.sum())),
            ("Полных пригодных суток (участок + день)", int(context.prepared.daily.complete.sum())),
            ("Уникальных дней с событиями (участок + день)", len(daily[["Участок дороги", "Дата"]].drop_duplicates())),
            ("Дней с аномалиями трафика (участок + день)", len(traffic[["Участок дороги", "Дата"]].drop_duplicates())),
            ("Дней только с проблемами качества", len(daily[["Участок дороги", "Дата"]].drop_duplicates()) - len(traffic[["Участок дороги", "Дата"]].drop_duplicates())),
            ("Норма", f"Тот же день недели; ±{config.reference_window_days} дней; минимум {config.min_reference_days} опорных дней"),
            ("Календарь праздников", f"Задано дат: {len(config.holiday_dates)}; автоматический календарь не подключён"),
            ("Отбор дней для графиков", "Все дни с сигналами" if config.plot_quality_only_days else "Только дни с аномалиями трафика; качество — на этих же днях"),
            ("Месячные обзорные графики", "Включены" if config.plot_monthly_overview else "Отключены"),
            ("Допуск сведения сумм", f"max({config.balance_absolute_tolerance:g} авто; {config.balance_relative_pct:g}% большего из двух сравниваемых чисел)"),
            ("Систематичность учёта", f"≥{config.systematic_min_hours} часов, ≥{config.systematic_min_days} дней, ≥{config.systematic_sign_share_pct:g}% расхождений одного знака"),
        ])
    summary.append(("Результат обработки", "Частичный: некоторые источники пропущены" if source_issues else "Все переданные источники прочитаны"))
    sheets = {
        "Сводка": pd.DataFrame(summary, columns=["Показатель", "Значение"]),
        "Качество данных": quality_view(quality),
        "Аномалии трафика": traffic_view(traffic),
        "Изменения режима": regime_view(regimes),
    }
    if context is not None:
        sheets["Месячные показатели"] = monthly_indicators_view(monthly_summary(context))
        from .balance import monthly_balance_summary
        if context.prepared.balance_details is not None:
            audit=monthly_balance_summary(context.prepared.balance_details,config)
            sheets["Учёт по месяцам"]=audit.drop(columns=["_check"])
    sheets["События по месяцам"] = monthly_events_view(monthly)
    if context is not None:
        sheets["Годовые показатели"] = yearly_summary(context)
        sheets["Покрытие и норма"] = coverage_table(context)
    if source_issues:
        sheets["Ошибки источников"] = pd.DataFrame([
            {"Источник": issue.source, "Ошибка": issue.message,
             "Последствие": "Источник пропущен; отчёт построен по оставшимся данным"}
            for issue in source_issues
        ])
    sheets["Легенда"] = pd.DataFrame(legend_rows(config), columns=["Раздел", "Обозначение", "Пояснение"])
    return sheets


def column_width(sheet, header, index):
    if sheet.title == "Легенда":
        return {1: 18, 2: 35, 3: 112}[index]
    if sheet.title == "Сводка":
        return 57 if index == 1 else 85
    if header in {"Пояснение", "Ошибка", "Последствие", "Основание"}:
        return 66
    if header == "Участок дороги":
        return 36
    if header in {"Проблема", "Тип аномалии", "Проверка", "Статус нормы", "Статус объёма", "Межгодовое сравнение", "Статус"}:
        return 42
    if header in {"Затронутые измерения", "Показатель"}:
        return 34
    if header in {"Начало", "Конец"}:
        return 24
    if header in {"Направление", "Приоритет", "Единица", "Единица отклонения", "Период", "Год", "Дата"}:
        return 29 if header=="Приоритет" else 17
    return 24


def style_workbook(workbook):
    for sheet in workbook:
        sheet.sheet_view.showGridLines = False
        sheet.freeze_panes = "B2" if sheet.max_column > 3 else "A2"
        sheet.auto_filter.ref = sheet.dimensions
        headers = {cell.column: str(cell.value) for cell in sheet[1]}
        unit_column = next((i for i, label in headers.items() if label == "Единица"), None)
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor=excel_colour("header"))
            cell.font = Font(name="Calibri", size=11, bold=True, color=excel_colour("header_text"))
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            sheet.column_dimensions[cell.column_letter].width = column_width(sheet, headers[cell.column], cell.column)
        sheet.row_dimensions[1].height = 44
        for row in sheet.iter_rows(min_row=2):
            coefficient = unit_column is not None and sheet.cell(row[0].row, unit_column).value == "коэффициент"
            for cell in row:
                is_missing = cell.value == MISSING
                cell.font = Font(name="Calibri", size=11, color=excel_colour("reference" if is_missing else "text"))
                cell.fill = PatternFill("solid", fgColor=excel_colour("stripe" if cell.row % 2 == 0 else "paper"))
                if headers[cell.column] == "Приоритет" and cell.value in {FIRST,CHECK}:
                    # Cache the visible colour for viewers that discard Excel CF.
                    role = "high_priority" if cell.value == FIRST else "observation"
                    cell.fill = PatternFill("solid", fgColor=excel_colour(role))
                horizontal = "right" if cell.data_type == "n" or is_missing else "left"
                cell.alignment = Alignment(horizontal=horizontal, vertical="center", wrap_text=True, indent=1)
                if cell.is_date:
                    cell.number_format = "dd.mm.yyyy hh:mm:ss" if sheet.title == "Сводка" or headers[cell.column] in {"Начало", "Конец"} else "dd.mm.yyyy"
                elif cell.data_type == "n":
                    header = headers[cell.column]
                    integer = header in {"Год", "Опорных дней", "Опорных лет", "Минимум опорных дней", "Затронуто часов", "Измерено часов", "Пригодно часов", "Ожидается часов", "Длительность, дней", "Полных пригодных суток"}
                    cell.number_format = "0.000" if coefficient and header in {"Факт", "Ожидаемое значение / порог"} else "0.00" if header == "Робастный Z" else "0" if integer or isinstance(cell.value, int) else "0.0"
            lines = max(max(1, math.ceil(len(str(cell.value)) / max(8, sheet.column_dimensions[cell.column_letter].width - 4))) for cell in row)
            sheet.row_dimensions[row[0].row].height = max(32, lines * 16 + 8)
        for index, header in headers.items():
            if header != "Приоритет" or sheet.max_row < 2:
                continue
            letter = sheet.cell(1, index).column_letter
            target = f"{letter}2:{letter}{sheet.max_row}"
            for value, role in ((FIRST, "high_priority"), (CHECK, "observation")):
                sheet.conditional_formatting.add(target, CellIsRule(
                    operator="equal", formula=[f'"{value}"'],
                    fill=PatternFill("solid", fgColor=excel_colour(role)),
                ))
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        sheet.print_title_rows = "1:1"


def export_anomalies_to_excel(daily_df, monthly_df, output_path, *, context=None, regimes=None, source_issues=()):
    sheets = build_report_sheets(daily_df, monthly_df, context=context, regimes=regimes, source_issues=source_issues)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            if frame.empty:
                frame = pd.DataFrame([{"Статус": "Событий не обнаружено. Доступность сравнения указана на листах показателей и покрытия."}])
            display_missing(frame).to_excel(writer, sheet_name=name, index=False)
    workbook = openpyxl.load_workbook(output_path)
    try:
        style_workbook(workbook)
        workbook.save(output_path)
    finally:
        workbook.close()
