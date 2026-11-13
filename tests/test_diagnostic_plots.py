"""Check selection and that diagnostics expose the operands behind a signal."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_analysis import sample
from traffic_analyzer.baseline import build_context
from traffic_analyzer.events import TrafficEvent, event_frame
from traffic_analyzer.models import AnalysisConfig
from traffic_analyzer.quality import prepare_data
from traffic_analyzer.plots import select_plot_events, generate_anomaly_plots
from traffic_analyzer.diagnostic_plots import balance_components, diagnostic_data, RENDERERS, SLUGS, plot_balance
from traffic_analyzer.palette import PALETTE


def event(day="2020-02-15",kind="Часовой спад интенсивности",category="Трафик",metric="vol_total",**kwargs):
    return TrafficEvent("test",pd.Timestamp(day)+pd.Timedelta(hours=7),pd.Timestamp(day)+pd.Timedelta(hours=8),category,kind,metric=metric,**kwargs)


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=sample(days=84)
        time=pd.Timestamp("2020-02-15 07:00")
        data.loc[data.timestamp.eq(time),"cars_direct"]+=30
        data.loc[data.timestamp.eq(time+pd.Timedelta(hours=1)),"speed_direct"]=-1
        cls.context=build_context(prepare_data(data),AnalysisConfig())

    def test_quality_only_days_are_excluded_and_same_day_quality_is_kept(self):
        events=event_frame([event(),event(category="Качество данных",kind="Не сходятся суммы транспорта"),
                            event(day="2020-02-16",category="Качество данных",kind="Пропуск измерений")])
        selected=select_plot_events(events)
        self.assertEqual(len(selected),2)
        self.assertEqual(len(select_plot_events(events,include_quality_only=True)),3)
        self.assertEqual(len(events),3)

    def test_no_traffic_produces_no_default_charts(self):
        events=event_frame([event(category="Качество данных",kind="Пропуск измерений")])
        self.assertTrue(select_plot_events(events).empty)
        with tempfile.TemporaryDirectory() as tmp:
            batch=generate_anomaly_plots(self.context,events,tmp)
            self.assertEqual(batch.file_count,0)
            self.assertEqual(json.loads(batch.manifest_path.read_text(encoding="utf-8"))["charts"],[])

    def test_balance_chart_exposes_the_exact_discrepancy(self):
        events=event_frame([event(category="Качество данных",kind="Не сходятся суммы транспорта")])
        d=diagnostic_data(self.context,"test","2020-02-15",events)
        checks=balance_components(d.day)
        self.assertEqual(checks[2][3][7],-30)
        self.assertEqual(checks[0][3][7],0)
        fig=plot_balance(d)
        try:
            self.assertEqual(len(fig.axes),1)
            self.assertFalse(fig.axes[0].lines)
            texts=[cell.get_text().get_text() for table in fig.axes[0].tables for cell in table.get_celld().values()]
            self.assertIn("Объявлено, авто",texts)
            self.assertIn("-30",texts)
            self.assertEqual(fig._suptitle.get_color(),PALETTE["heading"])
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_missing_category_is_not_treated_as_zero_in_balance(self):
        h=self.context.prepared.hourly.iloc[:24].copy()
        h.loc[h.index[0],"cars_direct"]=np.nan
        self.assertTrue(np.isnan(balance_components(h)[2][3][0]))

    def test_raw_invalid_speed_is_retained_for_diagnostics(self):
        raw=self.context.prepared.raw_values
        time=pd.Timestamp("2020-02-15 08:00")
        self.assertEqual(raw.loc[raw.timestamp.eq(time),"speed_direct"].iloc[0],-1)
        h=self.context.prepared.hourly
        self.assertTrue(pd.isna(h.loc[h.timestamp.eq(time),"speed_direct"].iloc[0]))

    def test_each_known_daily_event_type_has_a_renderer(self):
        self.assertEqual(set(SLUGS),set(RENDERERS))
        self.assertEqual(len(set(SLUGS.values())),len(SLUGS))

    def test_renderers_handle_their_required_measurements(self):
        import matplotlib.pyplot as plt
        for kind,renderer in RENDERERS.items():
            if kind=="Некорректные категории транспорта":
                continue  # needs a missing category fixture, checked separately
            with self.subTest(kind=kind):
                metric="speed_direct" if "скорость" in kind.lower() else "vol_total"
                events=event_frame([event(kind=kind,metric=metric,value=.5 if "форма" in kind.lower() else 100,expected=200)])
                fig=renderer(diagnostic_data(self.context,"test","2020-02-15",events))
                try:
                    self.assertGreater(len(fig.axes),0)
                    self.assertTrue(fig._suptitle.get_text())
                finally:
                    plt.close(fig)

    def test_grouping_emits_one_chart_per_type_not_per_interval(self):
        from unittest.mock import patch
        events=event_frame([event(),event(),event(category="Качество данных",kind="Не сходятся суммы транспорта")])
        with tempfile.TemporaryDirectory() as tmp,patch("traffic_analyzer.plots.plot_diagnostic") as render:
            batch=generate_anomaly_plots(self.context,events,tmp)
            self.assertEqual(batch.file_count,2)
            self.assertEqual(batch.day_count,1)
            self.assertEqual(render.call_count,2)
            manifest=json.loads(Path(batch.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual({record["kind"] for record in manifest["charts"]},set(events["Тип аномалии"]))

    def test_invalid_category_chart_identifies_the_missing_field(self):
        from traffic_analyzer.diagnostic_plots import plot_invalid
        import matplotlib.pyplot as plt
        events=event_frame([event(kind="Некорректные категории транспорта",category="Качество данных")])
        d=diagnostic_data(self.context,"test","2020-02-15",events)
        d.raw=d.raw.copy()
        d.day=d.day.copy()
        d.raw.loc[4,"cars_reverse"]=np.nan
        d.day.loc[4,"cars_reverse"]=np.nan
        fig=plot_invalid(d)
        try:
            self.assertEqual(len(fig.axes),1)
            self.assertIn("Легковые, обратное",fig.axes[0].get_title())
            self.assertTrue(np.isnan(fig.axes[0].lines[0].get_ydata()[4]))
        finally:
            plt.close(fig)


if __name__=="__main__":
    unittest.main()
