"""Grid definitions, station lookup, fractional masks, and conservative averaging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import mapping, box
from shapely.ops import transform as shapely_transform

from .utils import save_json


@dataclass(frozen=True)
class GridSpec:
    name: str
    crs: CRS
    transform: Affine
    width: int
    height: int
    x: np.ndarray
    y: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def cell_area_m2(self) -> float:
        return abs(float(self.transform.a * self.transform.e))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.width, self.height)
        return min(left, right), min(bottom, top), max(left, right), max(bottom, top)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        left, bottom, right, top = self.bounds
        return left, right, bottom, top

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "crs": self.crs.to_wkt(),
            "transform": tuple(self.transform),
            "width": self.width,
            "height": self.height,
            "cell_area_m2": self.cell_area_m2,
            "bounds": self.bounds,
        }


def grid_from_centers(name: str, x: np.ndarray, y: np.ndarray, crs: CRS | str) -> GridSpec:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Rectilinear grid coordinates must be one-dimensional.")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if not np.allclose(np.diff(x), dx, rtol=0, atol=max(abs(dx), 1.0) * 1e-5):
        raise ValueError(f"{name} x coordinates are not regularly spaced.")
    if not np.allclose(np.diff(y), dy, rtol=0, atol=max(abs(dy), 1.0) * 1e-5):
        raise ValueError(f"{name} y coordinates are not regularly spaced.")
    transform = Affine(dx, 0.0, x[0] - dx / 2.0, 0.0, dy, y[0] - dy / 2.0)
    return GridSpec(name, CRS.from_user_input(crs), transform, len(x), len(y), x, y)


def wrf_lambert_crs(attrs: dict[str, object]) -> CRS:
    def scalar(name: str) -> float:
        value = attrs[name]
        array = np.asarray(value).reshape(-1)
        return float(array[0])

    return CRS.from_proj4(
        "+proj=lcc "
        f"+lat_1={scalar('TRUELAT1')} +lat_2={scalar('TRUELAT2')} "
        f"+lat_0={scalar('MOAD_CEN_LAT') if 'MOAD_CEN_LAT' in attrs else scalar('CEN_LAT')} "
        f"+lon_0={scalar('STAND_LON')} +a=6370000 +b=6370000 +units=m +no_defs"
    )


def grid_from_wrf_latlon(
    name: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    attrs: dict[str, object],
    residual_tolerance_m: float = 25.0,
) -> tuple[GridSpec, dict[str, float]]:
    lat = np.asarray(latitude, dtype=float).squeeze()
    lon = np.asarray(longitude, dtype=float).squeeze()
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("WRF latitude and longitude arrays must be two-dimensional after squeeze().")
    crs = wrf_lambert_crs(attrs)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected_x, projected_y = transformer.transform(lon, lat)
    dx = float(np.asarray(attrs["DX"]).reshape(-1)[0])
    dy = float(np.asarray(attrs["DY"]).reshape(-1)[0])
    x = projected_x[0, 0] + np.arange(lon.shape[1]) * dx
    y = projected_y[0, 0] + np.arange(lat.shape[0]) * dy
    expected_x = np.broadcast_to(x, projected_x.shape)
    expected_y = np.broadcast_to(y[:, None], projected_y.shape)
    residual = np.hypot(projected_x - expected_x, projected_y - expected_y)
    diagnostics = {
        "maximum_center_residual_m": float(np.nanmax(residual)),
        "median_center_residual_m": float(np.nanmedian(residual)),
        "dx_m": dx,
        "dy_m": dy,
    }
    if diagnostics["maximum_center_residual_m"] > residual_tolerance_m:
        raise ValueError(
            "The WRF projection reconstructed from geo_em metadata does not align with XLAT_M/XLONG_M. "
            f"Maximum residual is {diagnostics['maximum_center_residual_m']:.2f} m."
        )
    return grid_from_centers(name, x, y, crs), diagnostics


def select_watershed(path: str | Path, name_field: str, contains: str) -> gpd.GeoDataFrame:
    """Select the East River feature from the full watershed shapefile.

    The full WBDHU10 layer is read, but scientific calculations use only the
    intended East River feature.  If a case-insensitive exact name match exists
    it is preferred.  Otherwise a contains-match is allowed only when unique.
    """
    watersheds = gpd.read_file(path)
    if name_field not in watersheds.columns:
        candidates = {column.lower(): column for column in watersheds.columns}
        if name_field.lower() in candidates:
            name_field = candidates[name_field.lower()]
        else:
            raise KeyError(
                f"Watershed field {name_field!r} was not found. Available fields: {list(watersheds.columns)}"
            )
    names = watersheds[name_field].astype(str).str.strip()
    exact = watersheds[names.str.casefold().eq(str(contains).strip().casefold())].copy()
    if len(exact) == 1:
        selected = exact
    else:
        selected = watersheds[names.str.contains(contains, case=False, na=False)].copy()
        if len(selected) != 1:
            matches = selected[name_field].astype(str).tolist()
            raise ValueError(
                f"Expected exactly one watershed matching {contains!r} in {name_field!r}; "
                f"found {len(selected)}: {matches}"
            )
    selected["geometry"] = selected.geometry.make_valid()
    return selected[[name_field, "geometry"]].rename(columns={name_field: "name"}).reset_index(drop=True)


def grid_cell_polygon(grid: GridSpec, row: int, col: int):
    """Return one grid-cell polygon in the grid CRS."""
    x0, y0 = grid.transform * (int(col), int(row))
    x1, y1 = grid.transform * (int(col) + 1, int(row) + 1)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def source_centers_within_target_cell(
    source_grid: GridSpec,
    target_grid: GridSpec,
    target_row: int,
    target_col: int,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Find all source-grid pixel centers inside one target-grid cell.

    This implements the SNOTEL comparison requested for East River: the nearest
    Noah-MP cell is the coarse footprint, while the iSnobal envelope uses every
    valid iSnobal pixel whose center falls inside that Noah-MP cell.
    """
    target_poly = grid_cell_polygon(target_grid, target_row, target_col)
    if target_grid.crs != source_grid.crs:
        transformer = Transformer.from_crs(target_grid.crs, source_grid.crs, always_xy=True)
        target_poly = shapely_transform(transformer.transform, target_poly)

    minx, miny, maxx, maxy = target_poly.bounds
    candidate_cols = np.flatnonzero((source_grid.x >= minx) & (source_grid.x <= maxx))
    candidate_rows = np.flatnonzero((source_grid.y >= miny) & (source_grid.y <= maxy))
    if len(candidate_rows) == 0 or len(candidate_cols) == 0:
        return {"rows": np.array([], dtype=int), "cols": np.array([], dtype=int)}
    rr, cc = np.meshgrid(candidate_rows, candidate_cols, indexing="ij")
    rows = rr.ravel().astype(int)
    cols = cc.ravel().astype(int)
    points = shapely.points(source_grid.x[cols], source_grid.y[rows])
    inside = np.asarray(shapely.covers(target_poly, points), dtype=bool)
    rows, cols = rows[inside], cols[inside]
    if valid_mask is not None and len(rows):
        valid = np.asarray(valid_mask, dtype=bool)[rows, cols]
        rows, cols = rows[valid], cols[valid]
    return {"rows": rows, "cols": cols}


def fractional_polygon_weights(
    grid: GridSpec,
    polygon: gpd.GeoDataFrame,
    *,
    chunk_size: int = 50000,
) -> np.ndarray:
    """Calculate exact polygon/cell overlap fractions using vectorized Shapely operations."""
    local = polygon.to_crs(grid.crs)
    geom = local.geometry.union_all()
    candidate = rasterize(
        [(mapping(geom), 1)],
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    rows, cols = np.nonzero(candidate)
    result = np.zeros(grid.shape, dtype=np.float32)
    pixel_area = grid.cell_area_m2
    for start in range(0, len(rows), chunk_size):
        r = rows[start : start + chunk_size]
        c = cols[start : start + chunk_size]
        x0 = grid.transform.c + c * grid.transform.a
        x1 = x0 + grid.transform.a
        y0 = grid.transform.f + r * grid.transform.e
        y1 = y0 + grid.transform.e
        boxes = shapely.box(np.minimum(x0, x1), np.minimum(y0, y1), np.maximum(x0, x1), np.maximum(y0, y1))
        covered = shapely.covers(geom, boxes)
        fractions = np.ones(len(boxes), dtype=float)
        partial = ~covered
        if np.any(partial):
            fractions[partial] = shapely.area(shapely.intersection(boxes[partial], geom)) / pixel_area
        result[r, c] = np.clip(fractions, 0.0, 1.0).astype(np.float32)
    return result


def save_weights(path: str | Path, weights: np.ndarray, grid: GridSpec) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, weights=weights, x=grid.x, y=grid.y, crs=grid.crs.to_wkt(), transform=np.array(tuple(grid.transform)))


def load_weights(path: str | Path, expected_grid: GridSpec) -> np.ndarray:
    """Load cached fractional weights and verify the complete grid definition.

    Shape-only checks can silently reuse weights from a different grid that
    happens to have the same number of rows and columns.  The cache therefore
    stores and validates coordinates, transform, and CRS before reuse.
    """
    with np.load(path, allow_pickle=False) as archive:
        weights = archive["weights"]
        cached_x = archive["x"]
        cached_y = archive["y"]
        cached_crs = CRS.from_wkt(str(archive["crs"].item()))
        cached_transform = Affine(*np.asarray(archive["transform"], dtype=float).tolist())
    if weights.shape != expected_grid.shape:
        raise ValueError(
            f"Cached weights shape {weights.shape} does not match expected grid {expected_grid.shape}."
        )
    if not np.allclose(cached_x, expected_grid.x, rtol=0.0, atol=1.0e-6):
        raise ValueError("Cached weight x coordinates do not match the current grid.")
    if not np.allclose(cached_y, expected_grid.y, rtol=0.0, atol=1.0e-6):
        raise ValueError("Cached weight y coordinates do not match the current grid.")
    if cached_crs != expected_grid.crs:
        raise ValueError("Cached weight CRS does not match the current grid.")
    if not np.allclose(tuple(cached_transform), tuple(expected_grid.transform), rtol=0.0, atol=1.0e-9):
        raise ValueError("Cached weight transform does not match the current grid.")
    return weights


def nearest_grid_cells(
    grid: GridSpec,
    station_longitudes: Iterable[float],
    station_latitudes: Iterable[float],
    count: int,
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    xx, yy = np.meshgrid(grid.x, grid.y)
    if valid_mask is None:
        valid = np.ones(grid.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
    rows, cols = np.nonzero(valid)
    points = np.column_stack([xx[valid], yy[valid]])
    tree = cKDTree(points)
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    output: list[dict[str, np.ndarray]] = []
    for lon, lat in zip(station_longitudes, station_latitudes):
        sx, sy = transformer.transform(float(lon), float(lat))
        distance, index = tree.query([sx, sy], k=min(count, len(points)))
        index = np.atleast_1d(index)
        output.append(
            {
                "rows": rows[index].astype(int),
                "cols": cols[index].astype(int),
                "distance_m": np.atleast_1d(distance).astype(float),
                "station_x": np.array([sx]),
                "station_y": np.array([sy]),
            }
        )
    return output


def regrid_area_average(
    source: np.ndarray,
    source_grid: GridSpec,
    destination_grid: GridSpec,
    *,
    minimum_valid_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-average a depth-like raster and return destination values and valid coverage."""
    data = np.asarray(source, dtype=np.float32)
    valid = np.isfinite(data)
    sentinel = np.float32(-3.4e38)
    prepared = np.where(valid, data, sentinel).astype(np.float32)
    destination = np.full(destination_grid.shape, sentinel, dtype=np.float32)
    coverage = np.zeros(destination_grid.shape, dtype=np.float32)
    reproject(
        source=prepared,
        destination=destination,
        src_transform=source_grid.transform,
        src_crs=source_grid.crs,
        src_nodata=sentinel,
        dst_transform=destination_grid.transform,
        dst_crs=destination_grid.crs,
        dst_nodata=sentinel,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    reproject(
        source=valid.astype(np.float32),
        destination=coverage,
        src_transform=source_grid.transform,
        src_crs=source_grid.crs,
        src_nodata=None,
        dst_transform=destination_grid.transform,
        dst_crs=destination_grid.crs,
        dst_nodata=0.0,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    output = destination.astype(float)
    output[(destination == sentinel) | (coverage < minimum_valid_coverage)] = np.nan
    return output, coverage


def write_geotiff(path: str | Path, data: np.ndarray, grid: GridSpec, nodata: float = -9999.0) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=np.float32)
    with rasterio.open(
        target,
        "w",
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
    ) as dst:
        dst.write(np.where(np.isfinite(array), array, nodata).astype(np.float32), 1)


def save_grid_diagnostics(output_dir: Path, grids: Iterable[GridSpec], extra: dict[str, object] | None = None) -> None:
    report = {grid.name: grid.to_metadata() for grid in grids}
    if extra:
        report["extra"] = extra
    save_json(report, output_dir / "diagnostics" / "grid_diagnostics.json")
