"""A single retrospective reference shared by detection and plotting."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import warnings

from .models import AnalysisConfig
from .quality import PreparedData, PLOT_METRICS


@dataclass
class AnalysisContext:
    prepared: PreparedData
    hourly_reference: pd.DataFrame
    daily_reference: pd.DataFrame
    config: AnalysisConfig


def robust_stats(values, relative_floor=.10, absolute_floor=20.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    median = float(np.median(values))
    sigma = max(1.4826 * float(np.median(np.abs(values - median))), abs(median) * relative_floor, absolute_floor)
    return median, sigma


def build_context(prepared: PreparedData, config: AnalysisConfig) -> AnalysisContext:
    hourly_refs, daily_refs = [], []
    holidays = set(config.holiday_dates)
    for section, h in prepared.hourly.groupby("road_section"):
        daily = prepared.daily[prepared.daily.road_section.eq(section)].set_index("date")
        h = h.copy()
        h["hour"] = h.timestamp.dt.hour
        available = daily.index[daily.complete]
        day_group = lambda d: 7 if d.date() in holidays else d.weekday()
        matrices = {metric: h.pivot(index="date", columns="hour", values=metric).reindex(columns=range(24)) for metric in PLOT_METRICS}
        for day in daily.index:
            candidates = [d for d in available if d != day and abs((d-day).days) <= config.reference_window_days and day_group(d) == day_group(day)]
            n = len(candidates)
            daily_med, daily_sigma = robust_stats(daily.loc[candidates, "daily_vol"], absolute_floor=100)
            if n < config.min_reference_days:
                daily_med = daily_sigma = np.nan
            daily_refs.append({"road_section": section, "date": day, "reference_days": n,
                               "expected": daily_med, "sigma": daily_sigma,
                               "reference_status": "Достаточно данных" if n >= config.min_reference_days else "Недостаточно опорных дней"})
            records = [{"road_section": section, "timestamp": day + pd.Timedelta(hours=hour), "reference_days": n} for hour in range(24)]
            for metric, matrix in matrices.items():
                values = matrix.loc[candidates].to_numpy(dtype=float)
                if metric.startswith("speed_"):
                    direction = metric.split("_")[1]
                    values = np.where(matrices[f"vol_{direction}"].loc[candidates].to_numpy() > 0, values, np.nan)
                counts = np.isfinite(values).sum(axis=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    medians = np.nanmedian(values, axis=0)
                    mads = np.nanmedian(np.abs(values-medians), axis=0)
                floor = 20 if metric.startswith("vol_") else 3 if metric.startswith("speed_") else 2
                scales = np.maximum.reduce([1.4826*mads, np.abs(medians)*.10, np.full(24, floor)])
                medians = np.where(counts >= config.min_reference_days, medians, np.nan)
                scales = np.where(counts >= config.min_reference_days, scales, np.nan)
                for hour, record in enumerate(records):
                    med, sigma = medians[hour], scales[hour]
                    lower = np.maximum(0, med-config.z_threshold*sigma)
                    upper = med+config.z_threshold*sigma
                    if metric.startswith("load_"):
                        upper = np.minimum(100, upper)
                    record.update({f"{metric}_expected": med, f"{metric}_sigma": sigma,
                                   f"{metric}_lower": lower, f"{metric}_upper": upper,
                                   f"{metric}_n": int(counts[hour])})
            hourly_refs.extend(records)
    return AnalysisContext(prepared, pd.DataFrame(hourly_refs), pd.DataFrame(daily_refs), config)
