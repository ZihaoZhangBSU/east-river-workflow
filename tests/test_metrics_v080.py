import numpy as np
import pandas as pd
import pytest

from east_river_workflow.metrics import pearson_mean_bias, paired_event_series, select_case_years, cumulative_half_date


def test_pearson_and_mean_bias_use_model_minus_reference():
    s=pearson_mean_bias([1,2,3],[2,3,4])
    assert s['pearson_r'] == 1.0
    assert s['mean_bias'] == 1.0


def test_event_filter_uses_either_member_threshold():
    dates=pd.date_range('2020-01-01',periods=4)
    ref=pd.Series([0,0.5,0.5,0.5],index=dates)
    mod=pd.Series([0,2.0,2.0,2.0],index=dates)
    x,y=paired_event_series(ref,mod,1.0,'accumulation')
    assert len(x)==1
    assert x[0] == 0.5
    assert y[0] == 2.0


def test_case_year_selection_uses_absolute_magnitude_but_retains_sign():
    f=pd.DataFrame({'water_year':[2018,2019,2020],'metric':[-2.0,8.0,-0.5]})
    c=select_case_years(f,'metric')
    assert c['largest_year']==2019 and c['largest_signed_value']==8.0
    assert c['smallest_year']==2020 and c['smallest_signed_value']==-0.5


def test_cumulative_half_date():
    dates=pd.date_range('2020-03-01',periods=4)
    result=cumulative_half_date(dates,[1,1,6,2])
    assert result == pd.Timestamp('2020-03-03')


def test_h1_decomposition_closes_and_shares_sum_to_100(tmp_path):
    import copy
    from pathlib import Path
    from east_river_workflow.config import DEFAULTS, WorkflowConfig
    from east_river_workflow.metrics import calculate_h1_h2_watershed_metrics

    data = copy.deepcopy(DEFAULTS)
    data["project"]["water_years"] = [2018]
    data["paths"] = {
        "watershed_shapefile": "x", "isnobal_root_template": "x", "isnobal_topo_file": "x",
        "noahmp_output_file": "x", "noahmp_geo_file": "x", "snotel_csv": "x",
        "aso_directory": "x", "output_dir": str(tmp_path),
    }
    cfg = WorkflowConfig(data, Path(tmp_path / "cfg.yaml")); cfg.create_output_tree()
    dates = pd.to_datetime(["2017-11-01", "2018-03-15"])
    i = pd.DataFrame({"date": dates, "water_year": [2018, 2018], "swe_mm": [20.0, 120.0], "snowfall_mm": [50.0, 150.0], "precip_mm": [60.0, 160.0]})
    n = pd.DataFrame({"date": dates, "water_year": [2018, 2018], "swe_mm": [15.0, 90.0], "snowfall_mm": [45.0, 120.0], "precip_mm": [55.0, 130.0]})
    table, _, _ = calculate_h1_h2_watershed_metrics(cfg, i, n, np.ones((1, 1)))
    row = table.iloc[0]
    assert row["decomposition_closure_error_mm"] == pytest.approx(0.0)
    assert row[["share_initial_pct", "share_snowfall_pct", "share_retention_pct"]].sum() == pytest.approx(100.0)


def test_snotel_f_swe_precip_uses_own_peak_and_complete_oct1_to_peak(tmp_path):
    import copy
    from pathlib import Path
    from east_river_workflow.config import DEFAULTS, WorkflowConfig
    from east_river_workflow.metrics import calculate_snotel_metrics

    data = copy.deepcopy(DEFAULTS)
    data["project"]["water_years"] = [2018]
    data["snotel"]["stations"] = [{"id": 1, "name": "Test", "latitude": 0.0, "longitude": 0.0, "elevation_ft": 0}]
    data["paths"] = {
        "watershed_shapefile": "x", "isnobal_root_template": "x", "isnobal_topo_file": "x",
        "noahmp_output_file": "x", "noahmp_geo_file": "x", "snotel_csv": "x",
        "aso_directory": "x", "output_dir": str(tmp_path),
    }
    cfg = WorkflowConfig(data, Path(tmp_path / "cfg.yaml")); cfg.create_output_tree()
    dates = pd.date_range("2017-10-01", "2017-10-04", freq="D")
    s = pd.DataFrame({"date": dates, "water_year": 2018, "station_id": 1, "swe_mm": [0, 10, 30, 20], "snow_depth_mm": [0, 30, 90, 60], "precip_gapfilled_mm": [2, 3, 5, 7]})
    i = pd.DataFrame({"date": dates, "water_year": 2018, "station_id": 1, "swe_nearest_mm": [0, 8, 24, 20], "snow_depth_nearest_mm": [0, 25, 75, 60], "precip_nearest_mm": [1, 2, 3, 4]})
    n = pd.DataFrame({"date": dates, "water_year": 2018, "station_id": 1, "swe_mm": [0, 7, 21, 18], "snow_depth_mm": [0, 20, 65, 55], "precip_mm": [2, 2, 2, 2]})
    row = calculate_snotel_metrics(cfg, s, i, n).iloc[0]
    # All three peak on Oct 3, so precipitation through peak includes Oct 1-3 only.
    assert row["precip_to_peak_SNOTEL_mm"] == pytest.approx(10.0)
    assert row["f_SWE_precip_SNOTEL"] == pytest.approx(3.0)
    assert row["precip_to_peak_I_mm"] == pytest.approx(6.0)
    assert row["f_SWE_precip_I"] == pytest.approx(4.0)
    assert row["precip_to_peak_N_mm"] == pytest.approx(6.0)
    assert row["f_SWE_precip_N"] == pytest.approx(3.5)
