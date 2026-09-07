import numpy as np
import pandas as pd

from east_river_workflow.data_access import time_step_seconds
from east_river_workflow.processing import _check_noah_time_axis


def _hourly_index(unit: str) -> pd.DatetimeIndex:
    values = np.array([
        "2016-10-01T01:00:00",
        "2016-10-01T02:00:00",
        "2016-10-01T03:00:00",
    ], dtype=f"datetime64[{unit}]")
    return pd.DatetimeIndex(values)


def test_time_step_seconds_is_resolution_independent():
    for unit in ["s", "ms", "us", "ns"]:
        assert time_step_seconds(_hourly_index(unit), 1.0) == 3600.0


def test_noah_gap_check_is_resolution_independent():
    section = {
        "expected_timestep_hours": 1,
        "expected_record_count": None,
        "expected_first_time": None,
        "expected_last_time": None,
    }
    for unit in ["s", "ms", "us", "ns"]:
        result = _check_noah_time_axis(_hourly_index(unit), section)
        assert result["unexpected_gap_count"] == 0
