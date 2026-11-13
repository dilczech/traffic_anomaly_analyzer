"""Configuration and results shared by CLI and future API adapters."""
from dataclasses import dataclass
from pathlib import Path
from datetime import date


@dataclass(frozen=True)
class AnalysisConfig:
    output_dir: Path = Path("reports")
    generate_plots: bool = True
    reference_window_days: int = 84
    min_reference_days: int = 6
    z_threshold: float = 3.5
    daily_delta_pct: float = 25.0
    hourly_delta_pct: float = 35.0
    jam_speed: float = 35.0
    jam_load: float = 60.0
    min_month_coverage: float = 95.0
    min_regime_days: int = 7
    zero_packet_hours: int = 12
    holiday_dates: tuple[date, ...] = ()
    plot_quality_only_days: bool = False
    plot_monthly_overview: bool = False
    balance_absolute_tolerance: float = 10.0
    balance_relative_pct: float = 2.0
    systematic_min_hours: int = 30
    systematic_min_days: int = 7
    systematic_sign_share_pct: float = 80.0

    def __post_init__(self):
        if self.reference_window_days < 7 or self.min_reference_days < 2:
            raise ValueError("Окно нормы должно быть >= 7 дней, опорных дней >= 2.")
        if self.z_threshold <= 0 or min(self.daily_delta_pct, self.hourly_delta_pct) <= 0:
            raise ValueError("Пороги отклонений должны быть положительными.")
        if not 0 < self.min_month_coverage <= 100 or self.min_regime_days < 2:
            raise ValueError("Некорректные пороги полноты или длительности режима.")
        if not 0 < self.jam_speed <= 250 or not 0 <= self.jam_load < 100:
            raise ValueError("Порог скорости должен быть в (0, 250], загрузки — в [0, 100).")
        if self.zero_packet_hours < 2:
            raise ValueError("Для подозрения на нулевую телеметрию нужны минимум 2 часа.")
        import math
        if any(not math.isfinite(v) or v < 0 for v in (self.balance_absolute_tolerance,self.balance_relative_pct)):
            raise ValueError("Допуски баланса должны быть конечными неотрицательными числами.")
        if self.systematic_min_hours < 1 or self.systematic_min_days < 1 or not 50 < self.systematic_sign_share_pct <= 100:
            raise ValueError("Некорректные пороги систематических расхождений.")


@dataclass(frozen=True)
class SourceIssue:
    source: str
    message: str


@dataclass(frozen=True)
class AnalysisResult:
    record_count: int
    section_count: int
    daily_event_count: int
    anomalous_day_count: int
    monthly_event_count: int
    anomalous_month_count: int
    report_path: Path
    plots_dir: Path | None
    source_issues: tuple[SourceIssue, ...]
    quality_event_count: int = 0
    regime_event_count: int = 0
    plot_count: int = 0
    plotted_day_count: int = 0
    systematic_event_count: int = 0


class AnalysisError(Exception):
    """Analysis cannot produce a report."""
