"""Shared, inspectable event contract for detectors, plots and export."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrafficEvent:
    section: str
    start: pd.Timestamp
    end: pd.Timestamp
    category: str
    kind: str
    metric: str = "vol_total"
    direction: str = "Оба"
    scope: str = "Часы"
    value: float | None = None
    expected: float | None = None
    delta_pct: float | None = None
    score: float | None = None
    reference_days: int = 0
    reference_years: int = 0
    severity: str = "Наблюдение"
    description: str = ""

    def record(self):
        metric_labels = {
            "vol_total": "Интенсивность, оба направления", "vol_direct": "Интенсивность, прямое",
            "vol_reverse": "Интенсивность, обратное", "speed_direct": "Скорость, прямое",
            "speed_reverse": "Скорость, обратное", "heavy_share": "Доля тяжёлых ТС", "all": "Измерения",
        }
        unit = "км/ч" if self.metric.startswith("speed_") else "%" if self.metric == "heavy_share" else "авт."
        label = metric_labels.get(self.metric, self.metric)
        description = self.description
        direction = self.direction
        from .priorities import event_priority
        priority,reason=event_priority(self)
        if self.category == "Качество данных":
            label, unit = ("Пригодность месяца", "%") if self.scope == "Месяц" else ("Затронуто часов", "ч")
            if self.metric=="balance_difference":
                label,unit="Часы с систематическим расхождением","ч"
            if self.scope != "Месяц" and self.metric != "all":
                description=metric_labels.get(self.metric,self.metric)+". "+description
                if self.metric.endswith("_direct"):
                    direction="Прямое"
                elif self.metric.endswith("_reverse"):
                    direction="Обратное"
        elif "форма" in self.kind.lower():
            label, unit = "Корреляция профилей", "коэффициент"
        elif "распределение" in self.kind.lower():
            label, unit = "Доля прямого направления", "%"
        elif self.scope in {"Сутки", "Месяц", "Период"} and self.metric.startswith("vol_"):
            unit = "авт./сут"
        return {
            "Участок дороги": self.section, "Дата": self.start.date(),
            "Начало": self.start, "Конец": self.end + pd.Timedelta(minutes=59, seconds=59),
            "Категория": self.category, "Тип аномалии": self.kind,
            "Масштаб": self.scope, "Направление": direction, "Уровень": priority,
            "Показатель": label, "Единица": unit,
            "Факт": self.value, "Норма": self.expected,
            "Отклонение (%)": self.delta_pct, "Робастный Z": self.score,
            "Опорных дней": self.reference_days if self.scope != "Месяц" and self.reference_days else None,
            "Опорных лет": self.reference_years if self.scope == "Месяц" and self.reference_years else None,
            "Пояснение": description+" Приоритет: "+reason,
            "_metric": self.metric,
        }


EVENT_COLUMNS = list(TrafficEvent("", pd.Timestamp("2000-01-01"), pd.Timestamp("2000-01-01"), "", "").record())


def event_frame(events):
    frame = pd.DataFrame([event.record() for event in events], columns=EVENT_COLUMNS)
    return frame.sort_values(["Участок дороги", "Начало", "Тип аномалии"]).reset_index(drop=True)


def runs(mask):
    """Inclusive positional intervals; a false or missing observation breaks a run."""
    start = None
    for i, active in enumerate(list(mask) + [False]):
        if bool(active) and start is None:
            start = i
        elif not bool(active) and start is not None:
            yield start, i - 1
            start = None
