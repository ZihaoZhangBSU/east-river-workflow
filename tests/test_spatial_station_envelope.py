import numpy as np
from pyproj import CRS

from east_river_workflow.grids import grid_from_centers, source_centers_within_target_cell


def test_isnobal_envelope_selects_all_centers_inside_noah_cell():
    src=grid_from_centers('iSnobal',np.arange(0.5,4.0,1.0),np.arange(0.5,4.0,1.0),CRS.from_epsg(32613))
    dst=grid_from_centers('Noah',np.array([1.0,3.0]),np.array([1.0,3.0]),CRS.from_epsg(32613))
    # Noah row 0/col 0 footprint = x 0..2, y 0..2 => four source centers.
    result=source_centers_within_target_cell(src,dst,0,0)
    assert len(result['rows'])==4
    pairs=set(zip(result['rows'].tolist(),result['cols'].tolist()))
    assert pairs == {(0,0),(0,1),(1,0),(1,1)}
