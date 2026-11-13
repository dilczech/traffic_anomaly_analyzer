"""Preserve missing measurements and construct an explicit hourly calendar."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .events import TrafficEvent, event_frame, runs
from .models import AnalysisConfig
from .balance import CATEGORIES, VOLUME_COLUMNS, balance_checks, detail_frame

PLOT_METRICS = VOLUME_COLUMNS + ["speed_direct", "speed_reverse", "load_direct", "load_reverse"]


@dataclass
class PreparedData:
    hourly: pd.DataFrame
    daily: pd.DataFrame
    quality_events: pd.DataFrame
    raw_values: pd.DataFrame | None = None
    balance_details: pd.DataFrame | None = None


def prepare_data(data: pd.DataFrame, config=None) -> PreparedData:
    config=config or AnalysisConfig()
    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.floor("h")
    if data.duplicated(["road_section", "timestamp"]).any():
        raise ValueError("Повторяющиеся записи «участок + час».")
    calendar = pd.date_range(data.timestamp.min().normalize(), data.timestamp.max().normalize() + pd.Timedelta(hours=23), freq="h")
    hourly_frames, raw_frames, balance_frames, days, events = [], [], [], [], []
    for section, raw in data.groupby("road_section"):
        raw = raw.set_index("timestamp")
        h = raw.reindex(calendar)
        h.index.name = "timestamp"
        h["road_section"] = section
        h["observed"] = h.index.isin(raw.index)
        numeric = [c for c in h if c not in {"road_section", "observed"}]
        for col in numeric:
            h[col] = pd.to_numeric(h[col], errors="coerce")
        raw_frame=h[numeric].copy()
        raw_frame["road_section"]=section
        raw_frames.append(raw_frame.reset_index())
        invalid = pd.DataFrame(False, index=h.index, columns=numeric)
        for col in numeric:
            invalid[col] = ~np.isfinite(h[col]) | (h[col] < 0)
            if col.startswith("load_"):
                invalid[col] |= h[col] > 100
            elif col.startswith("speed_"):
                invalid[col] |= h[col] > 250
            else:
                invalid[col] |= h[col].mod(1).ne(0)
        # Do not turn invalid data into zero; plots show gaps for these fields.
        h[numeric] = h[numeric].mask(invalid)
        balance_bad = pd.Series(False, index=h.index)
        checks=balance_checks(h,config)
        for check in checks:
            balance_bad |= check.significant
        balance_frames.append(detail_frame(section,h.index,checks))
        h["balance_bad"] = balance_bad
        h["flatline"] = False
        equal = h.vol_total.eq(h.vol_total.shift()) & h.vol_total.gt(10)
        for start, end in runs(equal.to_numpy()):
            if end - start + 2 >= 5:
                h.iloc[start - 1:end + 1, h.columns.get_loc("flatline")] = True
        h["zero_packet"] = False
        packet_zero = h[PLOT_METRICS].eq(0).all(axis=1) & h.observed
        for start,end in runs(packet_zero.to_numpy()):
            if end-start+1 >= config.zero_packet_hours:
                h.iloc[start:end+1,h.columns.get_loc("zero_packet")] = True
        count_columns = [c for c in numeric if not c.startswith(("speed_", "load_"))]
        h["valid_volume"] = h[count_columns].notna().all(axis=1) & ~balance_bad & ~h.flatline & ~h.zero_packet
        h["date"] = h.index.normalize()
        for day, g in h.groupby("date"):
            observed = int(g.observed.sum())
            valid = int(g.valid_volume.sum())
            complete = valid == 24
            days.append({
                "road_section": section, "date": day, "observed_hours": observed,
                "valid_hours": valid, "complete": complete,
                "daily_vol": g.vol_total.sum(min_count=24) if complete else np.nan,
            })
            masks = [
                (~g.observed, "Пропуск измерений", "vol_total"),
                (g.balance_bad, "Не сходятся суммы транспорта", "vol_total"),
                (g.flatline, "Подозрение на залипание датчика", "vol_total"),
                (g.zero_packet, "Подозрение на нулевую телеметрию", "all"),
            ]
            for metric in PLOT_METRICS:
                masks.append((invalid.loc[g.index,metric] & g.observed, "Некорректные или пустые измерения", metric))
            extra_counts=[c for c in count_columns if c not in VOLUME_COLUMNS]
            if extra_counts:
                masks.append((invalid.loc[g.index,extra_counts].any(axis=1) & g.observed, "Некорректные категории транспорта", "vol_total"))
            for mask, kind, metric in masks:
                for start, end in runs(mask.to_numpy()):
                    events.append(TrafficEvent(
                        section, g.index[start], g.index[end], "Качество данных", kind,
                        metric=metric, value=end - start + 1,
                        severity="Высокий" if end - start + 1 >= 5 else "Наблюдение",
                        description=f"Затронуто часов: {end - start + 1}. Не интерпретировать как доказанное изменение трафика.",
                    ))
        hourly_frames.append(h.reset_index())
    return PreparedData(pd.concat(hourly_frames, ignore_index=True), pd.DataFrame(days), event_frame(events), pd.concat(raw_frames,ignore_index=True),pd.concat(balance_frames,ignore_index=True))
