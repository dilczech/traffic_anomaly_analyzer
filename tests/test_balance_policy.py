"""Boundary tests for tolerances, repeated bias and review priorities."""
import unittest

import numpy as np
import pandas as pd

from test_analysis import sample
from traffic_analyzer.balance import balance_checks,monthly_balance_summary,systematic_balance_events
from traffic_analyzer.models import AnalysisConfig
from traffic_analyzer.quality import prepare_data
from traffic_analyzer.events import TrafficEvent,event_frame
from traffic_analyzer.priorities import CHECK,FIRST,promote_correlated_events
from traffic_analyzer.report_views import monthly_events_view


class BalancePolicyTests(unittest.TestCase):
    def test_absolute_and_relative_boundaries_are_inclusive(self):
        frame=sample(days=1)
        frame.loc[0,["vol_total","vol_direct","vol_reverse"]]=[1000,490,490]
        check=balance_checks(frame)[0]
        self.assertEqual(check.tolerance.iloc[0],20)
        self.assertFalse(check.significant.iloc[0])
        frame.loc[0,"vol_reverse"]=489
        self.assertTrue(balance_checks(frame)[0].significant.iloc[0])
        frame.loc[0,["vol_total","vol_direct","vol_reverse"]]=[100,45,45]
        self.assertFalse(balance_checks(frame)[0].significant.iloc[0])
        frame.loc[0,"vol_reverse"]=44
        self.assertTrue(balance_checks(frame)[0].significant.iloc[0])

    def test_small_discrepancy_does_not_exclude_an_hour_or_day(self):
        frame=sample(days=1)
        frame.loc[3,"cars_total"]+=1
        prepared=prepare_data(frame)
        self.assertTrue(prepared.daily.iloc[0].complete)
        self.assertFalse(prepared.hourly.balance_bad.any())
        self.assertFalse(prepared.quality_events["Тип аномалии"].eq("Не сходятся суммы транспорта").any())
        stats=monthly_balance_summary(prepared.balance_details,AnalysisConfig())
        total=stats[stats["_check"].eq("categories_total")].iloc[0]
        self.assertEqual(total["Часов в допуске"],1)
        self.assertEqual(total["Часов выше допуска"],0)

    def test_large_discrepancy_excludes_volume_but_not_raw_evidence(self):
        frame=sample(days=1)
        frame.loc[3,"cars_total"]+=100
        prepared=prepare_data(frame)
        self.assertFalse(prepared.daily.iloc[0].complete)
        self.assertTrue(prepared.hourly.loc[3,"balance_bad"])
        self.assertEqual(prepared.raw_values.loc[3,"cars_total"],frame.loc[3,"cars_total"])

    def test_missing_operand_is_not_a_small_or_large_error(self):
        frame=sample(days=1)
        frame.loc[0,"cars_direct"]=np.nan
        check=balance_checks(frame)[2]
        self.assertFalse(check.valid.iloc[0])
        self.assertFalse(check.minor.iloc[0])
        self.assertFalse(check.significant.iloc[0])

    def make_persistent(self, *, days=7, hours=5, alternating=False):
        data=sample(days=days)
        mask=data.timestamp.dt.hour.lt(hours)
        indices=data.index[mask]
        changes=(np.where(np.arange(len(indices))%2,1,-1) if alternating else np.ones(len(indices),dtype=int)).astype(data["cars_total"].dtype)
        data.loc[indices,"cars_total"]+=changes
        data.loc[indices,"cars_direct"]+=changes
        return prepare_data(data)

    def test_repeated_minor_errors_are_a_monthly_anomaly(self):
        prepared=self.make_persistent()
        config=AnalysisConfig()
        summary=monthly_balance_summary(prepared.balance_details,config)
        triggered=summary[summary["Систематическое расхождение"].eq("Да")]
        self.assertEqual(len(triggered),2)
        self.assertFalse(prepared.hourly.balance_bad.any())
        events=systematic_balance_events(prepared.balance_details,summary,config)
        self.assertEqual(len(events),1)
        self.assertEqual(events.iloc[0]["Факт"],35)  # 35 hours, not 70 related checks
        self.assertEqual(events.iloc[0]["Уровень"],CHECK)
        view=monthly_events_view(events)
        self.assertEqual(view.iloc[0]["Единица"],"ч")
        self.assertNotIn("Пригодно часов (%)",view)

    def test_same_sign_and_distinct_days_are_required(self):
        config=AnalysisConfig()
        for prepared in (self.make_persistent(alternating=True),self.make_persistent(days=2,hours=20)):
            summary=monthly_balance_summary(prepared.balance_details,config)
            self.assertTrue(systematic_balance_events(prepared.balance_details,summary,config).empty)

    def test_systematic_thresholds_include_exact_hour_day_and_sign_boundaries(self):
        data=sample(days=7)
        indices=data.index[data.timestamp.dt.hour.lt(5)]
        changes=np.r_[np.ones(28,dtype=np.int32),-np.ones(7,dtype=np.int32)]
        data.loc[indices,"cars_total"]+=changes
        prepared=prepare_data(data)
        exact=AnalysisConfig(systematic_min_hours=35,systematic_min_days=7,systematic_sign_share_pct=80)
        stats=monthly_balance_summary(prepared.balance_details,exact)
        self.assertEqual(len(systematic_balance_events(prepared.balance_details,stats,exact)),1)
        for config in (AnalysisConfig(systematic_min_hours=36),AnalysisConfig(systematic_sign_share_pct=80.1)):
            stats=monthly_balance_summary(prepared.balance_details,config)
            self.assertTrue(systematic_balance_events(prepared.balance_details,stats,config).empty)

    def test_event_priority_follows_strength_duration_and_independent_evidence(self):
        def event(metric,start,end,value,expected,score):
            return TrafficEvent("test",pd.Timestamp(start),pd.Timestamp(end),"Трафик","Нетипично низкая скорость" if metric.startswith("speed") else "Часовой спад интенсивности",metric=metric,direction="Прямое",value=value,expected=expected,score=score,reference_days=8)
        weak=event("vol_direct","2020-01-01 07:00","2020-01-01 07:00",60,100,3.6)
        self.assertEqual(weak.record()["Уровень"],CHECK)
        strong=event("speed_direct","2020-01-01 07:00","2020-01-01 07:00",30,80,6.2)
        self.assertEqual(strong.record()["Уровень"],FIRST)
        mild_speed=event("speed_direct","2020-01-01 07:00","2020-01-01 07:00",60,80,3.6)
        promoted=promote_correlated_events(event_frame([weak,mild_speed]))
        self.assertEqual(set(promoted["Уровень"]),{FIRST})
        same_family=promote_correlated_events(event_frame([weak,weak]))
        self.assertEqual(set(same_family["Уровень"]),{CHECK})

    def test_promoting_requires_time_and_direction_overlap(self):
        base=dict(section="test",category="Трафик",value=60,expected=100,score=3.6,reference_days=8)
        a=TrafficEvent(start=pd.Timestamp("2020-01-01 07:00"),end=pd.Timestamp("2020-01-01 07:00"),kind="Часовой спад интенсивности",metric="vol_direct",direction="Прямое",**base)
        b=TrafficEvent(start=pd.Timestamp("2020-01-01 09:00"),end=pd.Timestamp("2020-01-01 09:00"),kind="Нетипично низкая скорость",metric="speed_direct",direction="Прямое",**base)
        frame=promote_correlated_events(event_frame([a,b]))
        self.assertEqual(set(frame["Уровень"]),{CHECK})


if __name__=="__main__":
    unittest.main()
