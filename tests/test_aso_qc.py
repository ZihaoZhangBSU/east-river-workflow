import numpy as np
from affine import Affine
from pyproj import CRS

from east_river_workflow.aso_qc import apply_aso_quality_control
from east_river_workflow.grids import GridSpec


def test_apply_aso_quality_control_masks_only_hard_invalid_values():
    raw = np.array([[0.0, -0.5, -5.0], [100.0, 5000.0, 6000.0]])
    grid = GridSpec(
        "test",
        CRS.from_epsg(32613),
        Affine(50, 0, 0, 0, -50, 100),
        3,
        2,
        np.array([25.0, 75.0, 125.0]),
        np.array([75.0, 25.0]),
    )
    qc = {
        "negative_clip_tolerance_mm": 1.0,
        "swe_hard_max_mm": 5000.0,
        "snow_depth_hard_max_mm": 15000.0,
    }
    result = apply_aso_quality_control(
        raw,
        date="20180331",
        variable="SWE",
        source_path="test.tif",
        grid=grid,
        metadata={},
        inside_watershed=np.ones_like(raw, dtype=bool),
        qc=qc,
    )
    assert result.cleaned_mm[0, 1] == 0.0
    assert np.isnan(result.cleaned_mm[0, 2])
    assert result.cleaned_mm[1, 1] == 5000.0
    assert np.isnan(result.cleaned_mm[1, 2])
    assert result.summary["clipped_to_zero_count"] == 1
    assert result.summary["masked_negative_count"] == 1
    assert result.summary["masked_above_hard_max_count"] == 1
    assert result.summary["adjusted_count"] == 3


def test_adjusted_cell_log_has_no_duplicate_small_negative_records():
    from east_river_workflow.aso_qc import _flagged_cells_frame

    raw = np.array([[-0.5, -5.0, 6000.0]])
    grid = GridSpec(
        "test",
        CRS.from_epsg(32613),
        Affine(50, 0, 0, 0, -50, 50),
        3,
        1,
        np.array([25.0, 75.0, 125.0]),
        np.array([25.0]),
    )
    qc = {
        "negative_clip_tolerance_mm": 1.0,
        "swe_hard_max_mm": 5000.0,
        "snow_depth_hard_max_mm": 15000.0,
    }
    result = apply_aso_quality_control(
        raw,
        date="20180331",
        variable="SWE",
        source_path="test.tif",
        grid=grid,
        metadata={},
        inside_watershed=np.ones_like(raw, dtype=bool),
        qc=qc,
    )
    logged = _flagged_cells_frame(result, maximum_records=100)
    assert len(logged) == 3
    assert logged[["row", "column"]].duplicated().sum() == 0
