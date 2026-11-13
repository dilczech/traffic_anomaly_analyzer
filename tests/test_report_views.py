"""Report contracts: units, relevant columns, missing data and styling."""
import tempfile
import unittest
from pathlib import Path

import openpyxl
import pandas as pd

from traffic_analyzer.events import TrafficEvent, event_frame
from traffic_analyzer.palette import PALETTE
from traffic_analyzer.report_views import (
    MISSING, display_missing, quality_view, traffic_view, monthly_events_view, regime_view,
)
from traffic_analyzer.reporting import export_anomalies_to_excel


def event(**overrides):
    values=dict(section="test",start=pd.Timestamp("2020-01-01"),end=pd.Timestamp("2020-01-01"),
                category="Трафик",kind="Часовой спад интенсивности",value=100.,expected=200.)
    values.update(overrides)
    return TrafficEvent(**values)


class ReportViewsTests(unittest.TestCase):
    def test_quality_has_no_unused_statistical_columns(self):
        frame=quality_view(event_frame([event(category="Качество данных",value=4)]))
        self.assertEqual(frame["Затронуто часов"].tolist(),[4])
        self.assertTrue({"Норма","Отклонение (%)","Робастный Z","Опорных дней","Опорных лет","Дата"}.isdisjoint(frame.columns))

    def test_month_quality_expresses_shortfall_in_percentage_points(self):
        frame=monthly_events_view(event_frame([event(category="Качество данных",kind="Неполный или непригодный месяц",scope="Месяц",value=34.6,expected=95)]))
        self.assertAlmostEqual(frame.iloc[0]["Недостаёт (п.п.)"],60.4)
        self.assertTrue({"Робастный Z","Опорных дней","Опорных лет"}.isdisjoint(frame.columns))

    def test_month_yoy_adds_reference_years_only_for_actual_comparisons(self):
        frame=monthly_events_view(event_frame([event(scope="Месяц",reference_years=2)]))
        self.assertEqual(frame["Опорных лет"].tolist(),[2])
        self.assertNotIn("Опорных дней",frame)
        self.assertNotIn("Робастный Z",frame)

    def test_mixed_months_keep_comparison_basis_and_signed_deviation(self):
        frame=monthly_events_view(event_frame([
            event(category="Качество данных",scope="Месяц",value=50,expected=95),
            event(scope="Месяц",reference_years=1),
        ]))
        self.assertEqual(set(frame["Основание сравнения"]),{"Порог пригодности","Другие годы"})
        self.assertIn("Опорных лет",frame)

    def test_speed_deviation_has_physical_units(self):
        frame=traffic_view(event_frame([event(metric="speed_direct",value=40,expected=80,score=5)]))
        self.assertEqual(frame.iloc[0]["Отклонение"],-40)
        self.assertEqual(frame.iloc[0]["Единица отклонения"],"км/ч")

    def test_volume_and_share_use_different_units(self):
        frame=traffic_view(event_frame([event(),event(metric="heavy_share",value=45,expected=50)]))
        self.assertEqual(frame["Отклонение"].tolist(),[-50.,-5.])
        self.assertEqual(frame["Единица отклонения"].tolist(),["%","п.п."])

    def test_correlation_has_no_percent_deviation(self):
        frame=traffic_view(event_frame([event(kind="Нетипичная форма суточного профиля",value=.651,expected=.7)]))
        self.assertTrue(pd.isna(frame.iloc[0]["Отклонение"]))
        self.assertEqual(frame.iloc[0]["Основание сравнения"],"Порог корреляции")

    def test_dash_does_not_change_zeros_or_analysis_frame(self):
        source=pd.DataFrame({"value":[0.,float("nan"),2.]})
        view=display_missing(source)
        self.assertEqual(view.value.tolist(),[0.,MISSING,2.])
        self.assertTrue(pd.isna(source.value.iloc[1]))
        self.assertEqual(display_missing(pd.DataFrame({"text":["","  ","ok"]})).text.tolist(),[MISSING,MISSING,"ok"])

    def test_regime_duration_and_relevant_fields(self):
        frame=regime_view(event_frame([event(scope="Период",end=pd.Timestamp("2020-01-07 23:00"))]))
        self.assertEqual(frame.iloc[0]["Длительность, дней"],7)
        self.assertNotIn("Робастный Z",frame)
        self.assertNotIn("Опорных лет",frame)

    def test_export_stores_dash_and_numeric_zero_and_uses_new_colours(self):
        frame=event_frame([event(value=0,score=None,severity="Высокий"),event(value=150,severity="Наблюдение")])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"styled.xlsx"
            export_anomalies_to_excel(frame,event_frame([]),path)
            book=openpyxl.load_workbook(path)
            try:
                sheet=book["Аномалии трафика"]
                headers={cell.value:cell.column for cell in sheet[1]}
                self.assertEqual(sheet.cell(2,headers["Факт"]).value,0)
                self.assertEqual(sheet.cell(2,headers["Робастный Z"]).value,MISSING)
                self.assertEqual(sheet["A1"].fill.fgColor.rgb[-6:],PALETTE["header"].lstrip("#"))
                self.assertEqual(sheet["A1"].font.color.rgb[-6:],PALETTE["header_text"].lstrip("#"))
                self.assertNotEqual(sheet["A2"].fill.fgColor.rgb,sheet["A3"].fill.fgColor.rgb)
                self.assertEqual(sheet.cell(2,headers["Приоритет"]).fill.fgColor.rgb[-6:],PALETTE["high_priority"].lstrip("#"))
                self.assertEqual(sheet.cell(3,headers["Приоритет"]).fill.fgColor.rgb[-6:],PALETTE["observation"].lstrip("#"))
                self.assertEqual(sheet.cell(2,headers["Начало"]).data_type,"d")
                self.assertEqual(sheet.cell(2,headers["Начало"]).number_format,"dd.mm.yyyy hh:mm:ss")
                rules=list(sheet.conditional_formatting)
                self.assertEqual(len(rules),1)
                self.assertEqual(len(sheet.conditional_formatting[rules[0]]),2)
                self.assertEqual(book.sheetnames[-1],"Легенда")
                self.assertFalse(any(cell.value is None for row in sheet for cell in row))
            finally:
                book.close()

    def test_legacy_empty_export_is_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"empty.xlsx"
            export_anomalies_to_excel(pd.DataFrame(),pd.DataFrame(),path)
            book=openpyxl.load_workbook(path,read_only=True)
            try:
                self.assertEqual(book["Качество данных"]["A1"].value,"Статус")
                self.assertTrue(book["Качество данных"]["A2"].value)
            finally:
                book.close()


if __name__=="__main__":
    unittest.main()
