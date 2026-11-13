"""Sheet-specific presentation; never replace missing values in analysis data."""
import numpy as np
import pandas as pd

MISSING = "—"
METRICS = {
    "vol_total": "Интенсивность и категории, оба направления",
    "vol_direct": "Интенсивность, прямое",
    "vol_reverse": "Интенсивность, обратное",
    "speed_direct": "Скорость, прямое",
    "speed_reverse": "Скорость, обратное",
    "load_direct": "Загрузка, прямое",
    "load_reverse": "Загрузка, обратное",
    "all": "Все измерения",
}


def deviation(row):
    """Return a signed deviation with its physical unit, not a universal %."""
    value, expected = row.get("Факт"), row.get("Норма")
    if pd.isna(value) or pd.isna(expected):
        return np.nan, None
    kind = str(row.get("Тип аномалии", ""))
    metric = str(row.get("_metric", ""))
    if metric=="balance_difference":
        return np.nan,None
    if "форма" in kind.lower():
        return np.nan, None
    if metric.startswith("speed_"):
        return value - expected, "км/ч"
    if metric == "heavy_share" or "распределение" in kind.lower() or row.get("Категория") == "Качество данных":
        return value - expected, "п.п."
    return ((value - expected) / expected * 100, "%") if expected > 0 else (np.nan, None)


def quality_view(frame):
    columns = ["Участок дороги", "Начало", "Конец", "Проблема", "Затронутые измерения",
               "Направление", "Затронуто часов", "Приоритет", "Пояснение"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.rename(columns={"Тип аномалии": "Проблема", "Факт": "Затронуто часов", "Уровень": "Приоритет"}).copy()
    result["Затронутые измерения"] = result["_metric"].map(METRICS)
    return result[columns]


def traffic_view(frame):
    columns = ["Участок дороги", "Начало", "Конец", "Масштаб", "Тип аномалии", "Направление",
               "Показатель", "Единица", "Факт", "Ожидаемое значение / порог", "Основание сравнения",
               "Отклонение", "Единица отклонения", "Робастный Z", "Опорных дней", "Приоритет", "Пояснение"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    differences = [deviation(row) for _, row in result.iterrows()]
    result["Отклонение"] = [value for value, _ in differences]
    result["Единица отклонения"] = [unit for _, unit in differences]
    result["Основание сравнения"] = [
        "Порог скорости" if row["Тип аномалии"] == "Низкая скорость при высокой загрузке"
        else "Порог корреляции" if "форма" in row["Тип аномалии"].lower()
        else "Часы, дни и устойчивость знака" if row["_metric"]=="balance_difference"
        else "Порог пригодности" if row["Категория"] == "Качество данных"
        else "Другие годы" if row["Масштаб"] == "Месяц"
        else "Опорная медиана" for _, row in result.iterrows()
    ]
    return result.rename(columns={"Норма": "Ожидаемое значение / порог", "Уровень": "Приоритет"})[columns]


def regime_view(frame):
    columns = ["Участок дороги", "Начало", "Конец", "Длительность, дней", "Показатель", "Единица",
               "Среднесуточный факт", "Среднесуточная норма", "Отклонение (%)", "Минимум опорных дней", "Приоритет", "Пояснение"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    result["Длительность, дней"] = (result["Конец"].dt.normalize() - result["Начало"].dt.normalize()).dt.days + 1
    return result.rename(columns={"Факт": "Среднесуточный факт", "Норма": "Среднесуточная норма",
                                  "Опорных дней": "Минимум опорных дней", "Уровень": "Приоритет"})[columns]


def monthly_indicators_view(frame):
    result = frame.drop(columns=["Год", "Месяц"]).copy()
    result["Пригоден для сравнения"] = result["Пригоден для сравнения"].map({True: "Пригоден", False: "Не пригоден"})
    return result.rename(columns={"Пригоден для сравнения": "Статус качества", "Статус YoY": "Межгодовое сравнение", "ССИ": "ССИ, авт./сут"})


def monthly_events_view(frame):
    if frame.empty:
        return pd.DataFrame()
    periods = frame["Начало"].dt.to_period("M").astype(str)
    if frame["Тип аномалии"].eq("Неполный или непригодный месяц").all():
        return pd.DataFrame({
            "Участок дороги": frame["Участок дороги"], "Период": periods, "Проблема": frame["Тип аномалии"],
            "Пригодно часов (%)": frame["Факт"], "Требуется (%)": frame["Норма"],
            "Недостаёт (п.п.)": (frame["Норма"] - frame["Факт"]).clip(lower=0),
            "Приоритет": frame["Уровень"], "Пояснение": frame["Пояснение"],
        })
    result = traffic_view(frame).drop(columns=["Начало", "Конец", "Масштаб", "Робастный Z", "Опорных дней"])
    result.insert(1, "Период", periods)
    if frame["Категория"].nunique() > 1:
        result.insert(2, "Категория", frame["Категория"])
    if frame["Опорных лет"].notna().any():
        result.insert(len(result.columns)-2, "Опорных лет", frame["Опорных лет"])
    return result


def coverage_table(context):
    result = context.prepared.daily.merge(context.daily_reference, on=["road_section", "date"], validate="one_to_one").drop(columns=["sigma"])
    result["volume_status"] = ["Рассчитан" if row.complete else f"Не рассчитан: пригодно {row.valid_hours} из 24 ч" for row in result.itertuples()]
    result["complete"] = result["complete"].map({True: "Пригодны", False: "Не пригодны"})
    return result.rename(columns={"road_section": "Участок дороги", "date": "Дата", "observed_hours": "Измерено часов",
                                  "valid_hours": "Пригодно часов", "complete": "Пригодные сутки", "daily_vol": "Суточный объём",
                                  "reference_days": "Опорных дней", "expected": "Норма суточного объёма",
                                  "reference_status": "Статус нормы", "volume_status": "Статус объёма"})


def display_missing(frame):
    """Use dashes only in the exported view; preserve numbers and real zeros."""
    return frame.astype(object).map(lambda value: MISSING if pd.isna(value) or (isinstance(value, str) and not value.strip()) else value)
