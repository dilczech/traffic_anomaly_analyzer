"""Boundary tests for file validation and application orchestration."""
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import openpyxl
import pandas as pd

from traffic_analyzer import AnalysisConfig, AnalysisError, LocalFileSource, run_analysis
from traffic_analyzer.cli import main
from traffic_analyzer.parser import parse_traffic_excel
from traffic_analyzer.sources import discover_sources
from traffic_analyzer.validation import InputValidationError


HEADERS = {
    1: "Дата", 2: "Общая интенсивность автомобилей", 5: "Легковые (до 6 м)",
    8: "Малые груз. (6-9 м)", 11: "Грузовые (9-13 м)",
    14: "Груз. большие (13-22 м)", 17: "Автопоезда (22-30 м)",
    20: "Автобусы", 23: "Мотоциклы", 26: "Скорость, км/ч", 28: "Загрузка, %",
}


def write_fixture(path, *, header=True, records=True):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    if header:
        for column, label in HEADERS.items():
            sheet.cell(3, column, label)
    sheet.cell(6, 1, "км 13+650 а/д Р-193 Воронеж-Тамбов")
    if records:
        for hour in range(24):
            row = [f"01.01.2020 {hour:02d}:59:59", "1 200", 600, 600]
            row += [1200, 600, 600] + [0] * 18 + [80, 80, 1, 1]
            for column, value in enumerate(row, 1):
                sheet.cell(7 + hour, column, value)
    workbook.save(path)
    workbook.close()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = AnalysisConfig(self.root / "out", generate_plots=False)

    def test_valid_export_parses_text_numbers_and_timestamps(self):
        path = self.root / "traffic.xlsx"
        write_fixture(path)
        frame = parse_traffic_excel(path)
        self.assertEqual(len(frame), 24)
        self.assertEqual(frame.vol_total.tolist(), [1200] * 24)
        self.assertEqual(frame.timestamp.iloc[-1], pd.Timestamp("2020-01-01 23:59:59"))

    def test_wrong_layout_is_rejected_even_with_valid_dates(self):
        path = self.root / "other.xlsx"
        write_fixture(path, header=False)
        with self.assertRaises(InputValidationError):
            parse_traffic_excel(path)

    def test_empty_measurement_remains_unknown_after_parsing(self):
        path=self.root/"blank.xlsx"
        write_fixture(path)
        workbook=openpyxl.load_workbook(path)
        workbook.active["B7"]=None
        workbook.save(path)
        workbook.close()
        frame=parse_traffic_excel(path)
        self.assertTrue(pd.isna(frame.vol_total.iloc[0]))

    def test_header_without_records_is_rejected(self):
        path = self.root / "empty.xlsx"
        write_fixture(path, records=False)
        with self.assertRaises(InputValidationError):
            parse_traffic_excel(path)

    def test_renamed_text_file_is_rejected(self):
        path = self.root / "fake.xlsx"
        path.write_text("not an Excel workbook")
        with self.assertRaises(InputValidationError):
            parse_traffic_excel(path)

    def test_legacy_xls_has_explicit_error(self):
        path = self.root / "old.xls"
        path.write_bytes(b"legacy")
        with self.assertRaisesRegex(InputValidationError, "сохранить как .xlsx"):
            parse_traffic_excel(path)

    def test_discovery_ignores_lock_files_and_accepts_uppercase_extension(self):
        for name in ("report.XLSX", "~$report.xlsx", "notes.txt"):
            (self.root / name).touch()
        self.assertEqual([source.path.name for source in discover_sources(self.root)], ["report.XLSX"])

    def test_missing_directory_is_not_created(self):
        path = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            discover_sources(path)
        self.assertFalse(path.exists())

    def test_empty_batch_produces_no_report(self):
        with self.assertRaises(AnalysisError):
            run_analysis([], self.config)
        self.assertFalse(self.config.output_dir.exists())

    def test_overlapping_exports_fail_before_export(self):
        path = self.root / "traffic.xlsx"
        write_fixture(path)
        source = LocalFileSource(path)
        with self.assertRaisesRegex(AnalysisError, "повторяющиеся"):
            run_analysis([source, source], self.config)
        self.assertFalse(self.config.output_dir.exists())

    def test_partial_batch_reports_issues_and_counts_unique_days(self):
        good = self.root / "traffic.xlsx"
        bad = self.root / "bad.xlsx"
        write_fixture(good)
        bad.write_bytes(b"invalid")
        with patch("traffic_analyzer.service.export_anomalies_to_excel") as export:
            result = run_analysis([LocalFileSource(good), LocalFileSource(bad)], self.config)
        self.assertEqual(result.record_count, 24)
        self.assertEqual(result.daily_event_count, 1)
        self.assertEqual(result.anomalous_day_count, 1)
        self.assertEqual(len(result.source_issues), 1)
        self.assertIsNone(result.plots_dir)
        export.assert_called_once()

    def test_source_context_is_released_on_validation_error(self):
        events = []
        class TemporarySource:
            name = "downloaded report"

            @contextmanager
            def materialize(self):
                try:
                    events.append("opened")
                    yield source_path
                finally:
                    events.append("closed")

        source_path = self.root / "missing.xlsx"
        with self.assertRaises(AnalysisError):
            run_analysis([TemporarySource()], self.config)
        self.assertEqual(events, ["opened", "closed"])

    def test_cli_returns_failure_for_missing_input(self):
        self.assertEqual(main(["--input", str(self.root / "missing")]), 1)


if __name__ == "__main__":
    unittest.main()
