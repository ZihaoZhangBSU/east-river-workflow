import numpy as np
import pandas as pd
import pytest
import xarray as xr

from east_river_workflow.data_access import decode_wrf_times, integrate_depth_like, integrate_shortwave


def _times(strings):
    values=np.asarray([list(s) for s in strings],dtype='S1')
    return xr.Dataset({'Times':(('Time','DateStrLen'),values)})


def test_internal_wrf_times_are_authoritative_and_hourly():
    ds=_times(['2017-10-01_00:00:00','2017-10-01_01:00:00','2017-10-01_02:00:00'])
    result=decode_wrf_times(ds,{'time_coordinate_mode':'wrf_times','time_variable':'Times'})
    assert result.equals(pd.DatetimeIndex(['2017-10-01 00:00','2017-10-01 01:00','2017-10-01 02:00']))


def test_no_filename_or_numeric_index_fallback_when_wrf_times_required():
    ds=xr.Dataset(coords={'Time':np.arange(3)})
    with pytest.raises(ValueError,match='No parseable Noah-MP time coordinate'):
        decode_wrf_times(ds,{'time_coordinate_mode':'wrf_times','time_variable':'Times'})


def test_rainrate_mm_per_timestep_sums_directly():
    values=np.ones((24,2,2))*2.0
    out=integrate_depth_like(values,'mm/timestep',3600,'RAINRATE')
    assert np.allclose(out,48.0)


def test_rate_mm_per_second_is_integrated_by_timestep():
    values=np.ones((24,1,1))*0.001
    out=integrate_depth_like(values,'mm/s',3600,'QSNOW')
    assert out[0,0] == pytest.approx(86.4)


def test_shortwave_returns_daily_mean_and_energy():
    values=np.ones((24,1,1))*100.0
    mean,energy=integrate_shortwave(values,'W m-2',3600,'SWFORC')
    assert mean[0,0] == pytest.approx(100.0)
    assert energy[0,0] == pytest.approx(8.64)


def test_numeric_time_index_decodes_from_configurable_time_zero():
    ds = xr.Dataset(coords={'Time': np.arange(3, dtype=np.int32)})
    result = decode_wrf_times(ds, {
        'time_coordinate_mode': 'numeric_index',
        'time_variable': 'Time',
        'output_start_datetime': '2016-10-01 01:00:00',
        'expected_timestep_hours': 1,
    })
    assert result.equals(pd.DatetimeIndex([
        '2016-10-01 01:00:00',
        '2016-10-01 02:00:00',
        '2016-10-01 03:00:00',
    ]))


def test_numeric_time_index_full_production_range_ends_2026063023():
    ds = xr.Dataset(coords={'Time': np.arange(85439, dtype=np.int32)})
    result = decode_wrf_times(ds, {
        'time_coordinate_mode': 'numeric_index',
        'time_variable': 'Time',
        'output_start_datetime': '2016-10-01 01:00:00',
        'expected_timestep_hours': 1,
    })
    assert len(result) == 85439
    assert result[0] == pd.Timestamp('2016-10-01 01:00:00')
    assert result[-1] == pd.Timestamp('2026-06-30 23:00:00')


def test_numeric_time_index_rejects_missing_record_index():
    ds = xr.Dataset(coords={'Time': np.array([0, 1, 3], dtype=np.int32)})
    with pytest.raises(ValueError, match='increment by 1'):
        decode_wrf_times(ds, {
            'time_coordinate_mode': 'numeric_index',
            'time_variable': 'Time',
            'output_start_datetime': '2016-10-01 01:00:00',
            'expected_timestep_hours': 1,
        })


def test_numeric_time_index_requires_reference_datetime():
    ds = xr.Dataset(coords={'Time': np.arange(3, dtype=np.int32)})
    with pytest.raises(ValueError, match='output_start_datetime'):
        decode_wrf_times(ds, {
            'time_coordinate_mode': 'numeric_index',
            'time_variable': 'Time',
            'output_start_datetime': None,
            'expected_timestep_hours': 1,
        })
