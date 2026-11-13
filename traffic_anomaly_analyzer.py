"""Compatibility entry point for the traffic_analyzer package."""
from traffic_analyzer.cli import main
from traffic_analyzer.detectors import detect_daily_anomalies, detect_monthly_anomalies
from traffic_analyzer.parser import parse_traffic_excel
from traffic_analyzer.plots import generate_anomaly_plots
from traffic_analyzer.reporting import export_anomalies_to_excel

if __name__ == "__main__":
    raise SystemExit(main())
