"""Validate the supported Excel format independently of an API framework."""
from pathlib import Path
from zipfile import BadZipFile, ZipFile


class InputValidationError(ValueError):
    """Input is not a supported hourly traffic workbook."""


def validate_input_file(filepath: str | Path) -> None:
    path = Path(filepath)
    if path.suffix.lower() != ".xlsx":
        raise InputValidationError("Поддерживается .xlsx. Файл .xls нужно сохранить как .xlsx.")
    if not path.is_file():
        raise InputValidationError(f"Файл не найден: {path}")
    try:
        with ZipFile(path) as archive:
            required = {"[Content_Types].xml", "xl/workbook.xml"}
            if not required.issubset(archive.namelist()):
                raise InputValidationError("Файл не содержит структуру книги Excel .xlsx.")
    except BadZipFile as exc:
        raise InputValidationError("Файл повреждён или не является .xlsx.") from exc


def validate_worksheet(worksheet) -> None:
    # Column positions are part of the supported export contract.
    expected = {
        0: "Дата", 1: "Общая интенсивность автомобилей",
        4: "Легковые (до 6 м)", 7: "Малые груз. (6-9 м)",
        10: "Грузовые (9-13 м)", 13: "Груз. большие (13-22 м)",
        16: "Автопоезда (22-30 м)", 19: "Автобусы", 22: "Мотоциклы",
        25: "Скорость, км/ч", 27: "Загрузка, %",
    }

    def normalize(value):
        return " ".join(str(value or "").split()).casefold()

    for row in worksheet.iter_rows(min_row=1, max_row=20, max_col=29, values_only=True):
        if all(normalize(row[index]) == normalize(label) for index, label in expected.items()):
            return
    raise InputValidationError(
        "Не найдена поддерживаемая шапка почасового отчёта: дата, интенсивность, "
        "категории ТС, скорость и загрузка должны находиться в ожидаемых столбцах."
    )
