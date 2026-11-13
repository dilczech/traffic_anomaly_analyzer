"""Controlled failures and traffic changes; no real incident labels assumed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from traffic_analyzer.baseline import build_context, AnalysisContext
from traffic_analyzer.detectors import detect_daily_anomalies, detect_monthly_anomalies, detect_regime_changes, monthly_summary
from traffic_analyzer.models import AnalysisConfig
from traffic_analyzer.quality import prepare_data, CATEGORIES
from traffic_analyzer.reporting import export_anomalies_to_excel


def sample(days=112,start="2020-01-01"):
    timestamps=pd.date_range(start,periods=days*24,freq="h")
    total=100+timestamps.hour.to_numpy()*20+timestamps.weekday.to_numpy()*5
    direct=np.floor(total*.55)
    frame=pd.DataFrame({"road_section":"test","timestamp":timestamps,
                        "vol_total":total,"vol_direct":direct,"vol_reverse":total-direct,
                        "speed_direct":80.,"speed_reverse":80.,"load_direct":10.,"load_reverse":10.})
    for category in CATEGORIES:
        for suffix in ("total","direct","reverse"):
            frame[f"{category}_{suffix}"]=frame[f"vol_{suffix}"] if category=="cars" else 0
    return frame


def scale_flow(frame,mask,factor):
    for column in [c for c in frame if c.startswith("vol_") or c.startswith("cars_")]:
        frame.loc[mask,column]*=factor


class AnalysisTests(unittest.TestCase):
    def test_missing_whole_day_is_quality_not_traffic_drop(self):
        data=sample()
        date=pd.Timestamp("2020-02-20")
        data=data[data.timestamp.dt.normalize().ne(date)]
        context=build_context(prepare_data(data),AnalysisConfig())
        day=context.prepared.daily[context.prepared.daily.date.eq(date)].iloc[0]
        self.assertEqual(day.observed_hours,0)
        self.assertTrue(pd.isna(day.daily_vol))
        events=detect_daily_anomalies(context)
        affected=events[pd.to_datetime(events["Дата"]).eq(date)]
        self.assertEqual(len(affected),1)
        self.assertEqual(affected.iloc[0]["Категория"],"Качество данных")
        self.assertEqual(affected.iloc[0]["Факт"],24)

    def test_partial_day_is_excluded_from_reference(self):
        data=sample()
        date=pd.Timestamp("2020-02-20")
        data=data[~(data.timestamp.dt.normalize().eq(date)&data.timestamp.dt.hour.eq(12))]
        context=build_context(prepare_data(data),AnalysisConfig())
        row=context.prepared.daily[context.prepared.daily.date.eq(date)].iloc[0]
        self.assertFalse(row.complete)
        same_day=context.daily_reference[context.daily_reference.date.eq(date)]
        self.assertTrue(same_day.expected.notna().all())

    def test_self_is_excluded_and_zero_mad_is_finite(self):
        data=sample()
        day=pd.Timestamp("2020-02-20")
        original=data[data.timestamp.dt.normalize().eq(day)].vol_total.sum()
        scale_flow(data,data.timestamp.dt.normalize().eq(day),3)
        context=build_context(prepare_data(data),AnalysisConfig())
        reference=context.daily_reference[context.daily_reference.date.eq(day)].iloc[0]
        self.assertEqual(reference.expected,original)
        self.assertTrue(np.isfinite(reference.sigma))
        events=detect_daily_anomalies(context)
        daily=events[events["Тип аномалии"].eq("Аномальный суточный объём")]
        self.assertTrue(pd.to_datetime(daily["Дата"]).eq(day).any())

    def test_single_hour_injection_is_located_exactly(self):
        data=sample()
        time=pd.Timestamp("2020-02-20 12:00")
        scale_flow(data,data.timestamp.eq(time),3)
        events=detect_daily_anomalies(build_context(prepare_data(data),AnalysisConfig()))
        hourly=events[events["Тип аномалии"].eq("Часовой всплеск интенсивности") & events["_metric"].eq("vol_total")]
        self.assertEqual(hourly["Начало"].tolist(),[time])
        self.assertEqual(hourly.iloc[0]["Конец"],time+pd.Timedelta(minutes=59,seconds=59))

    def test_normal_weekly_pattern_has_no_traffic_events(self):
        events=detect_daily_anomalies(build_context(prepare_data(sample()),AnalysisConfig()))
        self.assertTrue(events.empty)

    def test_jam_requires_same_hour_and_handles_reverse(self):
        data=sample(days=1)
        data.loc[data.timestamp.dt.hour.eq(12),"speed_direct"]=20
        data.loc[data.timestamp.dt.hour.eq(14),"load_direct"]=90
        data.loc[data.timestamp.dt.hour.eq(16),["speed_reverse","load_reverse"]]=[20,90]
        events=detect_daily_anomalies(build_context(prepare_data(data),AnalysisConfig()))
        jams=events[events["Тип аномалии"].eq("Низкая скорость при высокой загрузке")]
        self.assertEqual(jams["Направление"].tolist(),["Обратное"])
        self.assertEqual(jams.iloc[0]["Начало"].hour,16)

    def test_no_flow_has_no_speed_jam(self):
        data=sample(days=1)
        mask=data.timestamp.dt.hour.eq(12)
        scale_flow(data,mask,0)
        data.loc[mask,["speed_direct","load_direct"]]=[0,90]
        events=detect_daily_anomalies(build_context(prepare_data(data),AnalysisConfig()))
        self.assertFalse(events["Тип аномалии"].eq("Низкая скорость при высокой загрузке").any())

    def test_missing_and_zero_are_distinct(self):
        data=sample(days=1)
        data.loc[0,"vol_total"]=np.nan
        scale_flow(data,data.index==1,0)
        h=prepare_data(data).hourly
        self.assertTrue(pd.isna(h.loc[0,"vol_total"]))
        self.assertEqual(h.loc[1,"vol_total"],0)
        self.assertFalse(h.loc[0,"valid_volume"])
        self.assertTrue(h.loc[1,"valid_volume"])

    def test_flatline_crosses_midnight_and_breaks_at_missing_hour(self):
        data=sample(days=3)
        hours=data.timestamp.between("2020-01-01 22:00","2020-01-02 02:00")
        data.loc[hours,"vol_total"]=200
        prepared=prepare_data(data)
        self.assertEqual(int(prepared.hourly.flatline.sum()),5)
        data=data[data.timestamp.ne(pd.Timestamp("2020-01-02"))]
        self.assertEqual(int(prepare_data(data).hourly.flatline.sum()),0)

    def test_month_coverage_works_without_second_year(self):
        data=sample(days=31)
        data=data[~data.timestamp.dt.day.between(10,20)]
        context=build_context(prepare_data(data),AnalysisConfig())
        events=detect_monthly_anomalies(context)
        self.assertEqual(events["Тип аномалии"].tolist(),["Неполный или непригодный месяц"])
        self.assertNotEqual(monthly_summary(context).iloc[0]["Статус YoY"],"Доступно сравнение")

    def test_insufficient_reference_has_no_zero_baseline(self):
        context=build_context(prepare_data(sample(days=3)),AnalysisConfig())
        self.assertTrue(context.hourly_reference.vol_total_expected.isna().all())
        self.assertTrue(context.hourly_reference.vol_total_lower.isna().all())

    def test_sustained_change_is_detected_after_historical_reference(self):
        data=sample(days=112)
        scale_flow(data,data.timestamp.ge(pd.Timestamp("2020-03-15")),2)
        context=build_context(prepare_data(data),AnalysisConfig())
        events=detect_regime_changes(context)
        self.assertFalse(events.empty)
        self.assertTrue((events["Отклонение (%)"]>25).all())
        for _,event in events.iterrows():
            self.assertAlmostEqual(event["Отклонение (%)"],(event["Факт"]-event["Норма"])/event["Норма"]*100)

    def test_report_ends_with_legend_and_explains_insufficient_years(self):
        context=build_context(prepare_data(sample(days=3)),AnalysisConfig())
        events=detect_daily_anomalies(context)
        months=detect_monthly_anomalies(context)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"report.xlsx"
            export_anomalies_to_excel(events,months,path,context=context)
            book=openpyxl.load_workbook(path,data_only=True)
            try:
                self.assertEqual(book.sheetnames[-1],"Легенда")
                self.assertGreater(book["Легенда"].max_row,20)
                values=list(book["Месячные показатели"].values)
                self.assertIn("Межгодовое сравнение",values[0])
                self.assertFalse(any(str(cell.value).startswith("_") for cell in book["Качество данных"][1]))
            finally:
                book.close()

    def test_yoy_excludes_checked_year(self):
        first=sample(days=31,start="2020-01-01")
        second=sample(days=31,start="2021-01-01")
        scale_flow(second,second.index>=0,2)
        context=AnalysisContext(prepare_data(pd.concat([first,second])),pd.DataFrame(),pd.DataFrame(),AnalysisConfig())
        stats=monthly_summary(context)
        expected=stats.loc[stats["Период"].eq("2020-01"),"ССИ"].iloc[0]
        events=detect_monthly_anomalies(context)
        signal=events[events["Начало"].eq(pd.Timestamp("2021-01-01")) & events["_metric"].eq("vol_total") & events["Категория"].eq("Трафик")]
        self.assertEqual(len(signal),1)
        self.assertEqual(signal.iloc[0]["Норма"],expected)
        self.assertEqual(signal.iloc[0]["Опорных лет"],1)
        self.assertTrue(pd.isna(signal.iloc[0]["Опорных дней"]))

    def test_invalid_speed_does_not_erase_correct_volume(self):
        data=sample(days=1)
        data.loc[3,"speed_direct"]=-1
        prepared=prepare_data(data)
        self.assertTrue(prepared.daily.iloc[0].complete)
        self.assertTrue(pd.isna(prepared.hourly.loc[3,"speed_direct"]))
        invalid=prepared.quality_events[prepared.quality_events["Тип аномалии"].eq("Некорректные или пустые измерения")]
        self.assertEqual(invalid["_metric"].tolist(),["speed_direct"])
        self.assertEqual(invalid["Направление"].tolist(),["Прямое"])
        self.assertIn("Скорость, прямое",invalid.iloc[0]["Пояснение"])

    def test_different_timestamps_within_one_hour_are_duplicates(self):
        data=sample(days=1)
        extra=data.iloc[[0]].copy()
        extra["timestamp"]+=pd.Timedelta(minutes=59,seconds=59)
        with self.assertRaisesRegex(ValueError,"участок"):
            prepare_data(pd.concat([data,extra]))

    def test_configured_holiday_is_not_pooled_with_regular_weekday(self):
        date=pd.Timestamp("2020-02-20")
        config=AnalysisConfig(holiday_dates=(date.date(),))
        context=build_context(prepare_data(sample()),config)
        reference=context.daily_reference[context.daily_reference.date.eq(date)].iloc[0]
        self.assertEqual(reference.reference_days,0)
        self.assertTrue(pd.isna(reference.expected))

    def test_long_zero_packet_is_suspect_but_values_are_preserved(self):
        data=sample(days=2)
        mask=data.timestamp.between("2020-01-01 18:00","2020-01-02 07:00")
        numeric=data.select_dtypes("number").columns
        data.loc[mask,numeric]=0
        prepared=prepare_data(data)
        self.assertEqual(int(prepared.hourly.zero_packet.sum()),14)
        self.assertEqual(prepared.hourly.loc[mask,"vol_total"].sum(),0)
        self.assertFalse(prepared.hourly.loc[mask,"valid_volume"].any())

    def test_short_zero_packet_is_a_real_observation_not_missing(self):
        data=sample(days=1)
        mask=data.timestamp.dt.hour.eq(12)
        data.loc[mask,data.select_dtypes("number").columns]=0
        prepared=prepare_data(data)
        self.assertFalse(prepared.hourly.zero_packet.any())
        self.assertTrue(prepared.hourly.loc[mask,"valid_volume"].all())
        self.assertTrue(prepared.hourly.loc[mask,"observed"].all())

    def test_statistical_speed_signal_preserves_computed_z(self):
        data=sample()
        time=pd.Timestamp("2020-02-20 12:00")
        data.loc[data.timestamp.eq(time),"speed_direct"]=20
        events=detect_daily_anomalies(build_context(prepare_data(data),AnalysisConfig()))
        speed=events[events["Тип аномалии"].eq("Нетипично низкая скорость")]
        self.assertEqual(len(speed),1)
        self.assertAlmostEqual(speed.iloc[0]["Робастный Z"],7.5)


if __name__=="__main__":
    unittest.main()
