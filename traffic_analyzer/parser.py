"""Read the supported hourly traffic Excel export."""
import datetime
import re

import openpyxl
import pandas as pd

from .validation import InputValidationError, validate_input_file, validate_worksheet


def parse_traffic_excel(filepath):
    """
    Потоковый парсер: находит и разделяет участки трассы,
    распознает любые форматы дат и очищает текстовые числа.
    """
    validate_input_file(filepath)
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)

    try:
        ws = wb.active
        validate_worksheet(ws)
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
                    row_vals.extend([None] * (29 - len(row_vals)))
                records.append([current_section, dt_obj] + row_vals[1:29])

    finally:
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
        raise InputValidationError("В файле нет почасовых записей трафика.")

    df = pd.DataFrame(records, columns=col_names)

    # Очистка чисел: убираем неразрывные пробелы (\xa0), пробелы и запятые
    for col in col_names[2:]:
        if df[col].dtype == object:
            df[col] = (df[col].astype(str)
                       .str.replace('\xa0', '', regex=False)
                       .str.replace(' ', '', regex=False)
                       .str.replace(',', '.', regex=False))
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
