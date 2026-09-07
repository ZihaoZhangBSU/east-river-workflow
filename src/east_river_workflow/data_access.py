"""Readers, time decoding, and unit conversion for East River raw products."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS

from .constants import MJ_PER_J
from .grids import GridSpec, grid_from_centers, grid_from_wrf_latlon
from .utils import open_dataset_robust


def _decode_attr(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def variable_units(variable: xr.DataArray) -> str:
    return _decode_attr(variable.attrs.get("units", "")).strip()


def normalized_units(value: str | None) -> str:
    return " ".join((value or "").strip().lower().replace("**", "^").split())


def read_isnobal_grid(topo_file: str | Path, dem_variable: str = "dem") -> tuple[GridSpec, np.ndarray, np.ndarray]:
    with open_dataset_robust(topo_file, decode_times=False) as ds:
        if dem_variable not in ds:
            raise KeyError(f"{dem_variable!r} not found in {topo_file}")
        dem = np.asarray(ds[dem_variable].values, dtype=float).squeeze()
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        crs = None
        for candidate in ["projection", "crs"]:
            if candidate in ds:
                attrs = ds[candidate].attrs
                crs_text = attrs.get("spatial_ref") or attrs.get("crs_wkt")
                if crs_text:
                    crs = CRS.from_wkt(_decode_attr(crs_text))
                    break
        if crs is None:
            crs = CRS.from_epsg(32613)
    grid = grid_from_centers("iSnobal", x, y, crs)
    valid = np.isfinite(dem)
    return grid, dem, valid


def read_noah_grid(
    geo_file: str | Path, cfg: dict[str, Any]
) -> tuple[GridSpec, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    with open_dataset_robust(geo_file, decode_times=False) as ds:
        lat_name = cfg["latitude_variable"]
        lon_name = cfg["longitude_variable"]
        elev_name = cfg["elevation_variable"]
        lat = np.asarray(ds[lat_name].values, dtype=float).squeeze()
        lon = np.asarray(ds[lon_name].values, dtype=float).squeeze()
        elevation = np.asarray(ds[elev_name].values, dtype=float).squeeze()
        grid, diagnostics = grid_from_wrf_latlon("Noah-MP", lat, lon, dict(ds.attrs))
    return grid, elevation, lat, lon, diagnostics


def _state_2d(variable: xr.DataArray, time_index: int) -> np.ndarray:
    indexer = {"time": time_index} if "time" in variable.dims else ({"Time": time_index} if "Time" in variable.dims else {})
    return np.asarray(variable.isel(indexer).values, dtype=float).squeeze()


def read_isnobal_snow(path: str | Path, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    with open_dataset_robust(path, decode_times=True) as ds:
        swe_v = ds[cfg["swe_variable"]]
        depth_v = ds[cfg["snow_depth_variable"]]
        swe = _state_2d(swe_v, int(cfg["time_index"]))
        depth = _state_2d(depth_v, int(cfg["time_index"])) * 1000.0
        time = pd.NaT
        if "time" in ds.coords:
            time = pd.Timestamp(ds["time"].isel(time=int(cfg["time_index"])).values)
    return swe, depth, time


def _read_hourly_array(path: str | Path, variable_name: str) -> tuple[np.ndarray, str, pd.DatetimeIndex | None]:
    with open_dataset_robust(path, decode_times=True) as ds:
        if variable_name not in ds:
            raise KeyError(f"{variable_name!r} not found in {path}")
        variable = ds[variable_name]
        array = np.asarray(variable.values, dtype=float)
        units = variable_units(variable)
        times = None
        if "time" in ds.coords:
            times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    return array, units, times


def read_isnobal_precip_snowfall(
    precip_path: str | Path,
    percent_snow_path: str | Path,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    precip, p_units, p_times = _read_hourly_array(precip_path, cfg["precip_variable"])
    fraction, f_units, f_times = _read_hourly_array(percent_snow_path, cfg["percent_snow_variable"])
    if precip.ndim == 2:
        precip = precip[None, ...]
    if fraction.ndim == 2:
        fraction = fraction[None, ...]
    if precip.shape != fraction.shape:
        raise ValueError(f"iSnobal precip/percent_snow shape mismatch: {precip.shape} vs {fraction.shape}")
    expected = int(cfg.get("expected_hourly_timesteps", 24))
    if precip.shape[0] != expected:
        raise ValueError(f"Expected {expected} iSnobal precip hours, found {precip.shape[0]} in {precip_path}")
    p_norm = normalized_units(p_units)
    if p_norm not in {"mm", "kg m-2", "kg m^-2"}:
        raise ValueError(f"Unexpected iSnobal precip units {p_units!r}")
    finite_fraction = fraction[np.isfinite(fraction)]
    scale = "fraction"
    if finite_fraction.size:
        lo = float(np.nanmin(finite_fraction)); hi = float(np.nanmax(finite_fraction))
        if lo < -1e-6:
            raise ValueError(f"iSnobal percent_snow contains negative values; min={lo}")
        if hi <= 1.5:
            snow_fraction = fraction
        elif hi <= 100.5:
            snow_fraction = fraction / 100.0
            scale = "percent"
        else:
            raise ValueError(f"iSnobal percent_snow exceeds expected range; max={hi}")
    else:
        snow_fraction = fraction
    total_precip = np.sum(precip, axis=0)
    snowfall = np.sum(precip * snow_fraction, axis=0)
    return total_precip, snowfall, {
        "precip_units": p_units,
        "percent_snow_units": f_units,
        "percent_snow_numeric_scale": scale,
        "precip_time_count": precip.shape[0],
    }


def read_isnobal_shortwave(
    incoming_path: str | Path,
    absorbed_path: str | Path,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    incoming, in_units, times = _read_hourly_array(incoming_path, cfg["incoming_sw_variable"])
    absorbed, abs_units, times2 = _read_hourly_array(absorbed_path, cfg["absorbed_sw_variable"])
    if incoming.ndim == 2: incoming = incoming[None, ...]
    if absorbed.ndim == 2: absorbed = absorbed[None, ...]
    if incoming.shape != absorbed.shape:
        raise ValueError(f"iSnobal incoming/absorbed shortwave shape mismatch: {incoming.shape} vs {absorbed.shape}")
    accepted = {"w m-2", "w/m2", "watt/m2", "watt m-2", "w m^-2"}
    if normalized_units(in_units) not in accepted or normalized_units(abs_units) not in accepted:
        raise ValueError(f"Unexpected iSnobal shortwave units: incoming={in_units!r}, absorbed={abs_units!r}")
    expected = int(cfg.get("expected_hourly_timesteps", 24))
    if incoming.shape[0] != expected:
        raise ValueError(f"Expected {expected} iSnobal shortwave hours, found {incoming.shape[0]}")
    dt_seconds = 3600.0
    if times is not None and len(times) > 1:
        diffs = np.asarray((times[1:] - times[:-1]).total_seconds(), dtype=float)
        dt_seconds = float(np.median(diffs))
    in_energy = np.sum(incoming * dt_seconds * MJ_PER_J, axis=0)
    abs_energy = np.sum(absorbed * dt_seconds * MJ_PER_J, axis=0)
    in_mean = np.mean(incoming, axis=0)
    abs_mean = np.mean(absorbed, axis=0)
    albedo = np.full_like(in_energy, np.nan, dtype=float)
    valid = np.isfinite(in_energy) & np.isfinite(abs_energy) & (in_energy > 0)
    albedo[valid] = 1.0 - abs_energy[valid] / in_energy[valid]
    return in_mean, abs_mean, in_energy, abs_energy, albedo


def read_isnobal_energy(path: str | Path, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    with open_dataset_robust(path, decode_times=True) as ds:
        swi = _state_2d(ds[cfg["swi_variable"]], int(cfg["time_index"]))
        melt = _state_2d(ds[cfg["snowmelt_variable"]], int(cfg["time_index"]))
        time = pd.NaT
        if "time" in ds.coords:
            time = pd.Timestamp(ds["time"].isel(time=int(cfg["time_index"])).values)
    return swi, melt, time


_WRF_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}[_ T]\d{2}:\d{2}:\d{2}")


def _normalise_wrf_timestamp(value: str, index: int) -> str:
    cleaned = value.replace("\x00", "").strip()
    match = _WRF_TIMESTAMP_PATTERN.search(cleaned)
    if match is None:
        raise ValueError(f"Could not parse Noah-MP Times[{index}]={value!r}")
    return match.group(0).replace("_", " ").replace("T", " ")


def _decode_wrf_character_times(ds: xr.Dataset, variable_name: str = "Times") -> pd.DatetimeIndex:
    values = np.asarray(ds[variable_name].values)
    if values.ndim == 2:
        raw = []
        for row in values:
            raw.append("".join(_decode_attr(item) for item in row))
    elif values.ndim == 1:
        raw = [_decode_attr(item) for item in values]
    else:
        raise ValueError(f"Unexpected Noah-MP Times dimensions: {values.shape}")
    text = [_normalise_wrf_timestamp(value, i) for i, value in enumerate(raw)]
    return pd.DatetimeIndex(pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="raise"))


def read_noah_numeric_time_index(ds: xr.Dataset, section: dict[str, Any] | None = None) -> np.ndarray:
    """Return and validate the raw numeric Noah-MP time index.

    In the production East River LDASOUT file, ``Time`` is not a datetime
    coordinate.  It is a record index such as ``0, 1, 2, ..., 85438``.
    ``output_start_datetime`` defines the datetime represented by Time=0 and
    ``expected_timestep_hours`` defines the duration of one index increment.
    """
    section = section or {}
    variable_name = str(section.get("time_variable", "Time"))
    if variable_name in ds.variables:
        values = np.asarray(ds[variable_name].values)
    elif variable_name in ds.sizes:
        # Support files where Time exists only as a dimension and xarray does
        # not expose an explicit coordinate variable.
        values = np.arange(int(ds.sizes[variable_name]), dtype=float)
    else:
        raise ValueError(f"Noah-MP numeric time index {variable_name!r} was not found in the LDASOUT file.")

    values = np.asarray(values).squeeze()
    if values.ndim != 1:
        raise ValueError(f"Noah-MP numeric time index {variable_name!r} must be one-dimensional; shape={values.shape}.")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Noah-MP numeric time index {variable_name!r} must be numeric; dtype={values.dtype}.")

    numeric = values.astype(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"Noah-MP numeric time index {variable_name!r} contains non-finite values.")
    if len(numeric) > 1:
        increments = np.diff(numeric)
        bad = np.flatnonzero(~np.isclose(increments, 1.0, rtol=0.0, atol=1e-9))
        if len(bad):
            preview = bad[:5].tolist()
            raise ValueError(
                f"Noah-MP numeric time index {variable_name!r} is expected to increment by 1; "
                f"found {len(bad)} unexpected increments (first positions: {preview})."
            )
    return numeric


def decode_noah_times(ds: xr.Dataset, section: dict[str, Any] | None = None) -> pd.DatetimeIndex:
    """Decode the Noah-MP record axis to real datetimes.

    Supported modes
    ---------------
    ``numeric_index``
        Production East River format.  ``Time`` contains numeric record
        indices (0, 1, 2, ...).  Datetimes are reconstructed as::

            output_start_datetime + Time * expected_timestep_hours

        Here ``output_start_datetime`` is the datetime represented by Time=0.

    ``wrf_times`` / ``times``
        Backward-compatible support for a WRF character ``Times`` variable.

    ``auto``
        Prefer the configured WRF character variable when present, otherwise
        use a true datetime coordinate.  Numeric indices are intentionally not
        guessed in auto mode because they require an explicit reference time.
    """
    section = section or {}
    mode = str(section.get("time_coordinate_mode", "numeric_index")).strip().lower()
    time_variable = str(section.get("time_variable", "Time"))

    if mode in {"numeric_index", "numeric", "index"}:
        base = section.get("output_start_datetime")
        if base in {None, ""}:
            raise ValueError(
                "noahmp.output_start_datetime is required when time_coordinate_mode='numeric_index'. "
                "It must be the datetime represented by Time=0."
            )
        base_time = pd.Timestamp(base)
        if pd.isna(base_time):
            raise ValueError(f"Invalid noahmp.output_start_datetime: {base!r}")
        timestep_hours = float(section.get("expected_timestep_hours", 1.0))
        if timestep_hours <= 0:
            raise ValueError("noahmp.expected_timestep_hours must be positive.")
        numeric = read_noah_numeric_time_index(ds, section)
        offsets = pd.to_timedelta(numeric * timestep_hours, unit="h")
        return pd.DatetimeIndex(base_time + offsets)

    if mode in {"wrf_times", "times", "auto"} and time_variable in ds:
        return _decode_wrf_character_times(ds, time_variable)

    if mode == "auto":
        for candidate in ["time", "Time"]:
            if candidate in ds.coords and np.issubdtype(ds[candidate].dtype, np.datetime64):
                return pd.DatetimeIndex(pd.to_datetime(ds[candidate].values))

    raise ValueError(
        "No parseable Noah-MP time coordinate was found for "
        f"time_coordinate_mode={mode!r} and time_variable={time_variable!r}."
    )


# Backward-compatible function name retained for older imports/workbench code.
def decode_wrf_times(ds: xr.Dataset, section: dict[str, Any] | None = None) -> pd.DatetimeIndex:
    return decode_noah_times(ds, section)


def time_step_seconds(times: pd.DatetimeIndex, expected_hours: float = 1.0) -> float:
    if len(times) < 2:
        return float(expected_hours) * 3600.0
    seconds = np.asarray((times[1:] - times[:-1]).total_seconds(), dtype=float)
    finite = seconds[np.isfinite(seconds)]
    if not len(finite):
        return float(expected_hours) * 3600.0
    return float(np.median(finite))


def integrate_depth_like(array: np.ndarray, units: str, timestep_seconds: float, context: str) -> np.ndarray:
    """Convert a stack of per-record depths or rates to total depth in mm."""
    u = normalized_units(units)
    values = np.asarray(array, dtype=float)
    direct = {"mm/timestep", "mm / timestep", "mm", "kg m-2", "kg m^-2"}
    rates = {"mm/s", "mm s-1", "mm s^-1", "mm sec-1", "mm second-1"}
    if u in direct:
        return np.sum(values, axis=0)
    if u in rates:
        return np.sum(values * float(timestep_seconds), axis=0)
    raise ValueError(f"Unexpected units for {context}: {units!r}")


def integrate_shortwave(array: np.ndarray, units: str, timestep_seconds: float, context: str) -> tuple[np.ndarray, np.ndarray]:
    u = normalized_units(units)
    accepted = {"w/m2", "w m-2", "w m^-2", "watt/m2", "watt m-2"}
    if u not in accepted:
        raise ValueError(f"Unexpected units for {context}: {units!r}")
    values = np.asarray(array, dtype=float)
    mean_w_m2 = np.mean(values, axis=0)
    energy_mj_m2 = np.sum(values * float(timestep_seconds) * MJ_PER_J, axis=0)
    return mean_w_m2, energy_mj_m2


def read_aso_raster(path: str | Path) -> tuple[np.ndarray, GridSpec, dict[str, str]]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype(float).filled(np.nan)
        tags = src.tags()
        description = " ".join(tags.values()).lower()
        if "[m]" not in description and " meter" not in description and " metre" not in description:
            raise ValueError(f"ASO raster {path} does not explicitly indicate meter units in metadata: {tags}")
        data *= 1000.0
        x = src.transform.c + (np.arange(src.width) + 0.5) * src.transform.a
        y = src.transform.f + (np.arange(src.height) + 0.5) * src.transform.e
        grid = grid_from_centers(f"ASO:{Path(path).stem}", x, y, src.crs)
        metadata = {"tags": str(tags), "nodata": str(src.nodata), "crs": str(src.crs)}
    return data, grid, metadata
