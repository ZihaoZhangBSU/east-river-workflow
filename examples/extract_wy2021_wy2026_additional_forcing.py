#!/usr/bin/env python
"""Extract additional WY2021/WY2026 forcing and energy diagnostics.

This is a standalone companion to ``wy2021_wy2026_watershed_diagnostic.py``.
It does not modify the East River Workflow v0.8.2 package or H1-H4 metrics.

Purpose
-------
Create compact daily watershed-average files for the common Oct 1-Jun 30
periods of WY2021 and WY2026, focusing on variables needed to diagnose the
low-SWE / early-ablation mechanism.

Source separation is explicit:

* iSnobal atmospheric forcing comes from each daily iSnobal run directory:
  ``air_temp.nc``, ``thermal.nc``, and ``wind_speed.nc``.
* Noah-MP atmospheric forcing comes from the hourly HRLDAS input files
  ``YYYYMMDDHH.LDASIN_DOMAIN1``.  The required variables are ``T2D``
  (air temperature), ``U2D``/``V2D`` (wind components), and ``LWDOWN``
  (incoming longwave radiation).  Each file represents one hour and its
  timestamp is taken from the filename.
* Noah-MP model diagnostics come from the production LDASOUT, including
  T2MV/T2MB, HFX, LH, FIRA, GRDFLX, SAG, native FSNO when available, and a
  direct sublimation variable only when one is genuinely present.

Important scientific guardrails
-------------------------------
1. Noah atmospheric temperature, wind, and incoming longwave are read from
   LDASIN, not inferred from LDASOUT diagnostics.
2. Noah wind speed is computed at each grid cell as sqrt(U2D**2 + V2D**2)
   before the fractional watershed mean is calculated.
3. T2MV/T2MB remain Noah model diagnostics and are never substituted for T2D.
4. No direct Noah snow sublimation is invented from LH or generic evaporation.
5. Native Noah FSNO, when present, is sampled at the configured daily state
   hour (23:00 in the current v0.8.2 configuration), matching the Noah SWE
   daily-state convention.  FSNO is never derived from SWE.
6. iSnobal ``em.nc:evaporation`` is retained under its source name and source
   metadata; it is not silently relabeled as pure sublimation.
7. Watershed means use the existing exact fractional East River weights and
   finite-value renormalization, consistent with v0.8.2.

Outputs
-------
By default, files are written under::

    <output_dir>/diagnostics/wy2021_wy2026_watershed/additional_forcing/

including::

    isnobal_additional_forcing_daily.csv
    noahmp_additional_forcing_daily.csv
    wy2021_wy2026_additional_forcing_daily.csv
    noahmp_ldasin_variable_inventory.csv
    noahmp_ldasout_variable_inventory.csv
    noahmp_fsno_daily.csv                    [only when FSNO exists]
    additional_forcing_metadata.json

Run from the v0.8.2 project root after installing the package::

    python examples/extract_wy2021_wy2026_additional_forcing.py \
        --config config/east_river_config.yaml

The default LDASIN directory is the production path supplied for this study;
use ``--noah-ldasin-root`` to override it on another system.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from east_river_workflow.config import WorkflowConfig, load_config
from east_river_workflow.data_access import decode_noah_times, normalized_units
from east_river_workflow.utils import open_dataset_robust


YEARS = (2021, 2026)
SECONDS_PER_DAY = 86400.0
MJ_PER_J = 1.0e-6

# Noah atmospheric forcing is taken from HRLDAS LDASIN, not LDASOUT.
NOAH_LDASIN_REQUIRED = {
    "T2D": "air_temperature_forcing",
    "U2D": "u_wind_forcing",
    "V2D": "v_wind_forcing",
    "LWDOWN": "incoming_longwave_forcing",
}

NOAH_SUBLIMATION_CANDIDATES = (
    "SUBLIM",
    "SUBLIMATION",
    "QSUB",
    "QSUBLIM",
    "QSUBLIMATION",
    "QSNOWSUB",
    "QSNOWSUBL",
    "QSNOWSUBLIM",
    "ESNOW",
    "SNOW_SUBLIM",
    "SNOW_SUBLIMATION",
)

# These are useful Noah LDASOUT diagnostics, not atmospheric forcing variables.
NOAH_DIAGNOSTICS = {
    "T2MV": "noah_t2mv_diag",
    "T2MB": "noah_t2mb_diag",
    "HFX": "noah_sensible_heat_to_atmosphere",
    "LH": "noah_latent_heat_to_atmosphere",
    "FIRA": "noah_net_longwave_to_atmosphere",
    "GRDFLX": "noah_ground_heat_flux_into_soil",
    "SAG": "noah_solar_absorbed_by_ground",
}

FSNO_FULL_TOLERANCE = 1.0e-6
DEFAULT_NOAH_LDASIN_ROOT = Path(
    "/bsuhome/zihaozhang/scratch/NOAH-MP/hrldas/hrldas/run/cases/HRRR/erw_ultimate/out"
)
DEFAULT_NOAH_LDASIN_PATTERN = "{timestamp:%Y%m%d%H}.LDASIN_DOMAIN1"

ISNOBAL_EM_VARIABLES = {
    "evaporation": "isnobal_em_evaporation",
    "net_rad": "isnobal_em_net_radiation",
    "latent_heat": "isnobal_em_latent_heat",
    "sensible_heat": "isnobal_em_sensible_heat",
    "snow_soil": "isnobal_em_snow_soil_heat",
    "precip_advected": "isnobal_em_precip_advected_heat",
    "sum_EB": "isnobal_em_sum_energy_balance",
}


# -----------------------------------------------------------------------------
# CLI and basic helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/east_river_config.yaml"),
        help="Path to East River Workflow v0.8.2 configuration YAML.",
    )
    parser.add_argument(
        "--output-subdir",
        default="wy2021_wy2026_watershed/additional_forcing",
        help="Subdirectory created below output/diagnostics/.",
    )
    parser.add_argument(
        "--missing-file-policy",
        choices=("warn", "error"),
        default="warn",
        help=(
            "How to handle missing optional iSnobal files. 'warn' preserves the date "
            "with NaNs and records the gap; 'error' stops immediately."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print iSnobal/Noah LDASIN progress every N processed days (default: 25).",
    )
    parser.add_argument(
        "--skip-isnobal",
        action="store_true",
        help="Skip raw iSnobal extraction.",
    )
    parser.add_argument(
        "--skip-noah",
        action="store_true",
        help="Skip Noah-MP extraction.",
    )
    parser.add_argument(
        "--isnobal-air-temp-filename",
        default="air_temp.nc",
        help="Daily iSnobal air-temperature forcing filename.",
    )
    parser.add_argument(
        "--isnobal-air-temp-variable",
        default="air_temp",
        help="iSnobal air-temperature variable name.",
    )
    parser.add_argument(
        "--isnobal-longwave-filename",
        default="thermal.nc",
        help="Daily iSnobal incoming thermal/longwave forcing filename.",
    )
    parser.add_argument(
        "--isnobal-longwave-variable",
        default="thermal",
        help="iSnobal incoming thermal/longwave variable name.",
    )
    parser.add_argument(
        "--isnobal-wind-filename",
        default="wind_speed.nc",
        help="Daily iSnobal wind-speed forcing filename.",
    )
    parser.add_argument(
        "--isnobal-wind-variable",
        default="wind_speed",
        help="iSnobal wind-speed variable name.",
    )
    parser.add_argument(
        "--noah-ldasin-root",
        type=Path,
        default=DEFAULT_NOAH_LDASIN_ROOT,
        help=(
            "Directory containing hourly YYYYMMDDHH.LDASIN_DOMAIN1 files. "
            f"Default: {DEFAULT_NOAH_LDASIN_ROOT}"
        ),
    )
    parser.add_argument(
        "--noah-ldasin-pattern",
        default=DEFAULT_NOAH_LDASIN_PATTERN,
        help=(
            "Python datetime-format filename pattern for hourly LDASIN files "
            "(default: '{timestamp:%%Y%%m%%d%%H}.LDASIN_DOMAIN1')."
        ),
    )
    parser.add_argument(
        "--noah-ldasin-missing-file-policy",
        choices=("warn", "error"),
        default="error",
        help=(
            "How to handle missing Noah LDASIN hourly files. 'error' is the "
            "default because complete 24-hour forcing is preferred; 'warn' "
            "preserves the day with incomplete-hour diagnostics."
        ),
    )
    parser.add_argument(
        "--fsno-variable",
        default="FSNO",
        help="Native Noah-MP ground snow-cover-fraction variable in LDASOUT (default: FSNO).",
    )
    return parser.parse_args()


def analysis_bounds(wy: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(wy - 1, 10, 1), pd.Timestamp(wy, 6, 30)


def expected_dates() -> list[tuple[int, pd.Timestamp]]:
    rows: list[tuple[int, pd.Timestamp]] = []
    for wy in YEARS:
        start, end = analysis_bounds(wy)
        rows.extend((wy, d) for d in pd.date_range(start, end, freq="D"))
    return rows


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2, sort_keys=True)


def load_fractional_weights(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Watershed-weight cache not found: {path}\n"
            "Run the v0.8.2 workflow first so exact East River fractional weights exist."
        )
    with np.load(path, allow_pickle=False) as archive:
        if "weights" not in archive:
            raise KeyError(f"Weight cache {path} does not contain a 'weights' array.")
        weights = np.asarray(archive["weights"], dtype=float)
    if weights.ndim != 2 or not np.any(np.isfinite(weights) & (weights > 0)):
        raise ValueError(f"Invalid/empty fractional weights in {path}.")
    return weights


def positive_weight_bbox(weights: np.ndarray, pad: int = 0) -> tuple[slice, slice, np.ndarray]:
    valid = np.isfinite(weights) & (weights > 0)
    rows, cols = np.nonzero(valid)
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(weights.shape[0], int(rows.max()) + 1 + pad)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(weights.shape[1], int(cols.max()) + 1 + pad)
    return slice(r0, r1), slice(c0, c1), np.asarray(weights[r0:r1, c0:c1], dtype=float)


def weighted_mean_timeseries(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean for [time,row,col] values plus valid-area fraction per time."""
    data = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3 or data.shape[1:] != w.shape:
        raise ValueError(f"Expected [time,{w.shape[0]},{w.shape[1]}], got {data.shape}.")
    base = np.isfinite(w) & (w > 0)
    total_weight = float(np.sum(w[base]))
    if total_weight <= 0:
        return np.full(data.shape[0], np.nan), np.zeros(data.shape[0])
    valid = np.isfinite(data) & base[None, :, :]
    numerator = np.sum(np.where(valid, data * w[None, :, :], 0.0), axis=(1, 2))
    denominator = np.sum(np.where(valid, w[None, :, :], 0.0), axis=(1, 2))
    mean = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    return mean, denominator / total_weight


def weighted_mean_2d(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean, valid = weighted_mean_timeseries(np.asarray(values)[None, ...], weights)
    return float(mean[0]), float(valid[0])


def _pick_spatial_dims(da: Any, weights_shape: tuple[int, int]) -> tuple[str, str]:
    dims = list(da.dims)
    sizes = dict(da.sizes)
    preferred = [
        ("south_north", "west_east"),
        ("y", "x"),
        ("lat", "lon"),
    ]
    for rdim, cdim in preferred:
        if rdim in dims and cdim in dims and sizes[rdim] == weights_shape[0] and sizes[cdim] == weights_shape[1]:
            return rdim, cdim

    row_candidates = [d for d in dims if sizes[d] == weights_shape[0]]
    col_candidates = [d for d in dims if sizes[d] == weights_shape[1]]
    for rdim in row_candidates:
        for cdim in col_candidates:
            if rdim != cdim:
                return rdim, cdim
    raise ValueError(
        f"Could not identify spatial dimensions for variable {getattr(da, 'name', '<unnamed>')}; "
        f"dims={dims}, sizes={sizes}, expected grid={weights_shape}."
    )


def _pick_time_dim(da: Any) -> str | None:
    for name in ("Time", "time"):
        if name in da.dims:
            return name
    nonspatial = [d for d in da.dims if d.lower() not in {"x", "y", "south_north", "west_east", "lat", "lon"}]
    return nonspatial[0] if len(nonspatial) == 1 else None


def read_subset_array(
    da: Any,
    full_weights_shape: tuple[int, int],
    row_slice: slice,
    col_slice: slice,
    *,
    time_selection: Any = slice(None),
    require_time: bool = True,
) -> np.ndarray:
    rdim, cdim = _pick_spatial_dims(da, full_weights_shape)
    tdim = _pick_time_dim(da)
    indexers: dict[str, Any] = {rdim: row_slice, cdim: col_slice}
    if tdim is not None:
        indexers[tdim] = time_selection
    elif require_time:
        raise ValueError(f"Variable {da.name!r} has no identifiable time dimension; dims={da.dims}.")

    sub = da.isel(indexers)
    # Scalar time indexing drops the time dimension (used for iSnobal em.nc
    # time_index=-1); vector/slice indexing preserves it.
    if tdim is not None and tdim in sub.dims:
        sub = sub.transpose(tdim, rdim, cdim)
    else:
        sub = sub.transpose(rdim, cdim)
    return np.asarray(sub.values, dtype=float)


def clean_units_text(units: Any) -> str:
    return "" if units is None else str(units).strip()


def temperature_to_c(values: np.ndarray, units: str, context: str) -> np.ndarray:
    u = normalized_units(units)
    arr = np.asarray(values, dtype=float)
    celsius = {
        "c", "degc", "degree_celsius", "degrees_celsius", "degree celsius",
        "degrees celsius", "celsius", "°c",
    }
    kelvin = {"k", "kelvin", "degree_kelvin", "degrees_kelvin"}
    if u in celsius or "celsius" in u:
        return arr
    if u in kelvin or "kelvin" in u:
        return arr - 273.15
    raise ValueError(f"Unrecognized temperature units for {context}: {units!r}")


def require_radiation_units(units: str, context: str) -> None:
    u = normalized_units(units)
    accepted = {"w/m2", "w/m^2", "w m-2", "w m^-2", "w m^(-2)", "watt/m2", "watt/m^2", "watt m-2", "watt m^-2"}
    if u not in accepted:
        raise ValueError(f"Unrecognized radiation/energy-flux units for {context}: {units!r}")


def require_wind_units(units: str, context: str) -> None:
    u = normalized_units(units)
    accepted = {"m/s", "m s-1", "m s^-1", "meter/s", "meters/s", "metre/s", "metres/s"}
    if u not in accepted:
        raise ValueError(f"Unrecognized wind-speed units for {context}: {units!r}")


def radiation_daily_stats(hourly_w_m2: np.ndarray, timestep_seconds: float) -> dict[str, float]:
    values = np.asarray(hourly_w_m2, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"mean_w_m2": np.nan, "energy_mj_m2": np.nan, "finite_hours": 0}
    # Do not undercount daily energy if an hourly watershed mean is missing.
    energy = np.sum(values * timestep_seconds * MJ_PER_J) if finite.all() else np.nan
    return {
        "mean_w_m2": float(np.nanmean(values)),
        "energy_mj_m2": float(energy) if np.isfinite(energy) else np.nan,
        "finite_hours": int(finite.sum()),
    }


def scalar_daily_stats(hourly: np.ndarray) -> dict[str, float]:
    values = np.asarray(hourly, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "finite_hours": 0}
    return {
        "mean": float(np.nanmean(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "finite_hours": int(finite.sum()),
    }


def water_depth_daily_from_hourly(hourly: np.ndarray, units: str, timestep_seconds: float, context: str) -> float:
    values = np.asarray(hourly, dtype=float)
    if not np.isfinite(values).all():
        return np.nan
    u = normalized_units(units)
    direct = {"mm/timestep", "mm / timestep", "mm", "kg m-2", "kg m^-2"}
    rates = {
        "mm/s", "mm s-1", "mm s^-1", "mm sec-1", "mm second-1",
        "kg m-2 s-1", "kg m^-2 s^-1", "kg/m2/s",
    }
    if u in direct:
        return float(np.sum(values))
    if u in rates:
        return float(np.sum(values * timestep_seconds))
    raise ValueError(f"Unrecognized depth-like units for {context}: {units!r}")


def handle_missing(path: Path, policy: str, missing_log: list[str]) -> bool:
    if path.exists():
        return False
    message = f"Missing optional raw file: {path}"
    missing_log.append(str(path))
    if policy == "error":
        raise FileNotFoundError(message)
    warnings.warn(message, stacklevel=2)
    return True


# -----------------------------------------------------------------------------
# iSnobal extraction
# -----------------------------------------------------------------------------


def _daily_isnobal_dir(cfg: WorkflowConfig, date: pd.Timestamp, wy: int) -> Path:
    template = str(cfg.paths["isnobal_root_template"])
    root = Path(template.format(wy=wy))
    pattern = str(cfg.section("isnobal")["daily_folder_pattern"])
    return root / pattern.format(date=pd.Timestamp(date).to_pydatetime())


def _read_isnobal_hourly_watershed(
    path: Path,
    variable_name: str,
    full_weights: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    subset_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    with open_dataset_robust(path, decode_times=False) as ds:
        if variable_name not in ds:
            raise KeyError(f"{path} does not contain variable {variable_name!r}; variables={list(ds.data_vars)}")
        da = ds[variable_name]
        units = clean_units_text(da.attrs.get("units"))
        description = str(da.attrs.get("long_name", da.attrs.get("description", "")))
        arr = read_subset_array(
            da,
            full_weights.shape,
            row_slice,
            col_slice,
            time_selection=slice(None),
            require_time=True,
        )
    hourly_mean, hourly_valid = weighted_mean_timeseries(arr, subset_weights)
    return hourly_mean, hourly_valid, units, description


def _read_isnobal_em(
    path: Path,
    full_weights: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    subset_weights: np.ndarray,
    time_index: int,
) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    values: dict[str, float] = {}
    metadata: dict[str, dict[str, str]] = {}
    with open_dataset_robust(path, decode_times=False) as ds:
        for raw_name, stem in ISNOBAL_EM_VARIABLES.items():
            if raw_name not in ds:
                continue
            da = ds[raw_name]
            units = clean_units_text(da.attrs.get("units"))
            description = str(da.attrs.get("description", da.attrs.get("long_name", "")))
            tdim = _pick_time_dim(da)
            selection = time_index if tdim is not None else slice(None)
            arr = read_subset_array(
                da,
                full_weights.shape,
                row_slice,
                col_slice,
                time_selection=selection,
                require_time=False,
            )
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"Expected 2-D iSnobal em field after time selection for {raw_name}, got {arr.shape}.")
            mean, valid_area = weighted_mean_2d(arr, subset_weights)
            if raw_name == "evaporation" and normalized_units(units) in {"kg m-2", "kg m^-2", "mm"}:
                # 1 kg m^-2 of water mass is numerically 1 mm water equivalent.
                values[f"{stem}_mm_water_equiv"] = mean
                values[f"{stem}_valid_area_fraction"] = valid_area
            elif "w" in normalized_units(units) and "m" in normalized_units(units):
                values[f"{stem}_mean_w_m2"] = mean
                values[f"{stem}_valid_area_fraction"] = valid_area
            else:
                values[f"{stem}_raw_mean"] = mean
                values[f"{stem}_valid_area_fraction"] = valid_area
            metadata[raw_name] = {"units": units, "description": description}
    return values, metadata


def extract_isnobal(
    cfg: WorkflowConfig,
    args: argparse.Namespace,
    full_weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_slice, col_slice, weights = positive_weight_bbox(full_weights)
    isnobal = cfg.section("isnobal")
    expected_hours = int(isnobal.get("expected_hourly_timesteps", 24))
    time_index = int(isnobal.get("time_index", -1))
    timestep_seconds = SECONDS_PER_DAY / expected_hours
    energy_filename = str(isnobal.get("energy_filename", "em.nc"))

    rows: list[dict[str, Any]] = []
    missing_log: list[str] = []
    variable_metadata: dict[str, Any] = {}
    all_dates = expected_dates()

    print(
        f"[iSnobal] extracting {len(all_dates)} dates using watershed bbox "
        f"rows {row_slice.start}:{row_slice.stop}, cols {col_slice.start}:{col_slice.stop}"
    )

    for n, (wy, date) in enumerate(all_dates, start=1):
        folder = _daily_isnobal_dir(cfg, date, wy)
        row: dict[str, Any] = {"date": date, "water_year": wy, "source": "iSnobal"}

        specs = [
            (
                "air_temp",
                folder / args.isnobal_air_temp_filename,
                args.isnobal_air_temp_variable,
            ),
            (
                "incoming_longwave",
                folder / args.isnobal_longwave_filename,
                args.isnobal_longwave_variable,
            ),
            (
                "wind_speed",
                folder / args.isnobal_wind_filename,
                args.isnobal_wind_variable,
            ),
        ]

        for kind, path, var_name in specs:
            if handle_missing(path, args.missing_file_policy, missing_log):
                continue
            hourly, valid_area, units, description = _read_isnobal_hourly_watershed(
                path, var_name, full_weights, row_slice, col_slice, weights
            )
            variable_metadata.setdefault(
                var_name,
                {"file": str(path.name), "units": units, "description": description},
            )
            if len(hourly) != expected_hours:
                message = f"{path}: expected {expected_hours} hourly values, found {len(hourly)}"
                if args.missing_file_policy == "error":
                    raise ValueError(message)
                warnings.warn(message, stacklevel=2)

            row[f"{kind}_minimum_hourly_valid_area_fraction"] = (
                float(np.nanmin(valid_area)) if len(valid_area) else np.nan
            )

            if kind == "air_temp":
                hourly_c = temperature_to_c(hourly, units, f"iSnobal {var_name}")
                stats = scalar_daily_stats(hourly_c)
                row["air_temp_forcing_mean_C"] = stats["mean"]
                row["air_temp_forcing_min_hourly_watershed_mean_C"] = stats["min"]
                row["air_temp_forcing_max_hourly_watershed_mean_C"] = stats["max"]
                row["air_temp_forcing_finite_hours"] = stats["finite_hours"]
            elif kind == "incoming_longwave":
                require_radiation_units(units, f"iSnobal {var_name}")
                stats = radiation_daily_stats(hourly, timestep_seconds)
                row["incoming_longwave_mean_w_m2"] = stats["mean_w_m2"]
                row["incoming_longwave_energy_mj_m2"] = stats["energy_mj_m2"]
                row["incoming_longwave_finite_hours"] = stats["finite_hours"]
            elif kind == "wind_speed":
                require_wind_units(units, f"iSnobal {var_name}")
                stats = scalar_daily_stats(hourly)
                row["wind_speed_forcing_mean_m_s"] = stats["mean"]
                row["wind_speed_forcing_max_hourly_watershed_mean_m_s"] = stats["max"]
                row["wind_speed_forcing_finite_hours"] = stats["finite_hours"]

        em_path = folder / energy_filename
        if not handle_missing(em_path, args.missing_file_policy, missing_log):
            em_values, em_meta = _read_isnobal_em(
                em_path,
                full_weights,
                row_slice,
                col_slice,
                weights,
                time_index=time_index,
            )
            row.update(em_values)
            if em_meta:
                variable_metadata.setdefault("em.nc", {}).update(em_meta)

        rows.append(row)
        if args.progress_every > 0 and (n % args.progress_every == 0 or n == len(all_dates)):
            print(f"[iSnobal] {n}/{len(all_dates)} dates complete ({date.date()})")

    frame = pd.DataFrame(rows).sort_values(["water_year", "date"]).reset_index(drop=True)
    metadata = {
        "record_count": len(frame),
        "weights_shape": list(full_weights.shape),
        "watershed_bbox": {
            "row_start": row_slice.start,
            "row_stop": row_slice.stop,
            "col_start": col_slice.start,
            "col_stop": col_slice.stop,
        },
        "expected_hourly_timesteps": expected_hours,
        "assumed_timestep_seconds": timestep_seconds,
        "time_index_for_em": time_index,
        "missing_file_policy": args.missing_file_policy,
        "missing_file_count": len(missing_log),
        "missing_files": missing_log,
        "variables": variable_metadata,
        "evaporation_note": (
            "em.nc:evaporation is preserved according to the source variable/metadata. "
            "When its units are kg m-2, values are reported numerically as mm water equivalent. "
            "Do not interpret it as pure sublimation without confirming the iSnobal sign/process convention."
        ),
    }
    return frame, metadata


# -----------------------------------------------------------------------------
# Noah-MP LDASIN forcing + LDASOUT diagnostic extraction
# -----------------------------------------------------------------------------


def variable_inventory(ds: Any) -> pd.DataFrame:
    rows = []
    for name in ds.data_vars:
        da = ds[name]
        rows.append(
            {
                "variable": name,
                "dims": ",".join(da.dims),
                "units": clean_units_text(da.attrs.get("units")),
                "description": str(da.attrs.get("description", da.attrs.get("long_name", ""))),
            }
        )
    return pd.DataFrame(rows)


def _first_present(ds: Any, candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in ds:
            return name
    return None


def _description_match(ds: Any, required_words: Sequence[str], forbidden_words: Sequence[str] = ()) -> str | None:
    required = [w.lower() for w in required_words]
    forbidden = [w.lower() for w in forbidden_words]
    matches: list[str] = []
    for name in ds.data_vars:
        da = ds[name]
        text = " ".join(
            [
                name,
                str(da.attrs.get("description", "")),
                str(da.attrs.get("long_name", "")),
                str(da.attrs.get("standard_name", "")),
            ]
        ).lower()
        if all(w in text for w in required) and not any(w in text for w in forbidden):
            matches.append(name)
    return matches[0] if len(matches) == 1 else None


def _selected_noah_time_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    tf = pd.DataFrame({"index": np.arange(len(times), dtype=int), "time": pd.DatetimeIndex(times)})
    tf["date"] = pd.to_datetime(tf["time"]).dt.normalize()
    tf["hour"] = pd.to_datetime(tf["time"]).dt.hour
    tf["water_year"] = np.where(tf["date"].dt.month >= 10, tf["date"].dt.year + 1, tf["date"].dt.year).astype(int)
    chosen = []
    for wy in YEARS:
        start, end = analysis_bounds(wy)
        chosen.append(tf[(tf["water_year"].eq(wy)) & tf["date"].between(start, end)].copy())
    return pd.concat(chosen, ignore_index=True).sort_values("index").reset_index(drop=True)


def _validate_noah_daily_completeness(chosen: pd.DataFrame, expected_hours: int) -> None:
    counts = chosen.groupby(["water_year", "date"]).size()
    bad = counts[counts.ne(expected_hours)]
    if len(bad):
        raise ValueError(
            "Noah-MP selected LDASOUT hourly records are incomplete. First bad days: "
            f"{bad.head(10).to_dict()}"
        )


def _extract_noah_hourly_weighted(
    ds: Any,
    var_name: str,
    chosen: pd.DataFrame,
    full_weights: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    subset_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    da = ds[var_name]
    if "Time" not in da.dims:
        raise ValueError(f"Noah variable {var_name!r} does not contain Time; dims={da.dims}.")
    indices = chosen["index"].to_numpy(dtype=int)
    arr = read_subset_array(
        da,
        full_weights.shape,
        row_slice,
        col_slice,
        time_selection=indices,
        require_time=True,
    )
    means, valid = weighted_mean_timeseries(arr, subset_weights)
    return (
        means,
        valid,
        clean_units_text(da.attrs.get("units")),
        str(da.attrs.get("description", da.attrs.get("long_name", ""))),
    )


def _group_hourly_to_daily(
    chosen: pd.DataFrame,
    hourly: np.ndarray,
    valid_area: np.ndarray,
    *,
    kind: str,
    units: str,
    timestep_seconds: float,
    prefix: str,
) -> pd.DataFrame:
    work = chosen[["date", "water_year"]].copy()
    work["value"] = np.asarray(hourly, dtype=float)
    work["valid_area_fraction"] = np.asarray(valid_area, dtype=float)
    rows = []
    for (wy, date), sub in work.groupby(["water_year", "date"], sort=True):
        values = sub["value"].to_numpy(dtype=float)
        valid_values = sub["valid_area_fraction"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "date": pd.Timestamp(date),
            "water_year": int(wy),
            f"{prefix}_minimum_hourly_valid_area_fraction": (
                float(np.nanmin(valid_values)) if np.isfinite(valid_values).any() else np.nan
            ),
        }
        if kind == "temperature":
            c = temperature_to_c(values, units, prefix)
            stats = scalar_daily_stats(c)
            row[f"{prefix}_mean_C"] = stats["mean"]
            row[f"{prefix}_min_hourly_watershed_mean_C"] = stats["min"]
            row[f"{prefix}_max_hourly_watershed_mean_C"] = stats["max"]
            row[f"{prefix}_finite_hours"] = stats["finite_hours"]
        elif kind == "wind":
            require_wind_units(units, prefix)
            stats = scalar_daily_stats(values)
            row[f"{prefix}_mean_m_s"] = stats["mean"]
            row[f"{prefix}_max_hourly_watershed_mean_m_s"] = stats["max"]
            row[f"{prefix}_finite_hours"] = stats["finite_hours"]
        elif kind == "radiation":
            require_radiation_units(units, prefix)
            stats = radiation_daily_stats(values, timestep_seconds)
            row[f"{prefix}_mean_w_m2"] = stats["mean_w_m2"]
            row[f"{prefix}_energy_mj_m2"] = stats["energy_mj_m2"]
            row[f"{prefix}_finite_hours"] = stats["finite_hours"]
        elif kind == "depth":
            row[f"{prefix}_mm_day"] = water_depth_daily_from_hourly(values, units, timestep_seconds, prefix)
            row[f"{prefix}_finite_hours"] = int(np.isfinite(values).sum())
        else:
            raise ValueError(f"Unknown daily aggregation kind {kind!r}.")
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_daily(base: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
    return base.merge(addition, on=["date", "water_year"], how="left", validate="one_to_one")


def _classify_sublimation_kind(units: str) -> str:
    u = normalized_units(units)
    depth_tokens = {
        "mm/timestep", "mm / timestep", "mm", "kg m-2", "kg m^-2",
        "mm/s", "mm s-1", "mm s^-1", "mm sec-1", "mm second-1",
        "kg m-2 s-1", "kg m^-2 s^-1", "kg/m2/s",
    }
    if u in depth_tokens:
        return "depth"
    radiation_tokens = {
        "w/m2", "w/m^2", "w m-2", "w m^-2", "w m^(-2)",
        "watt/m2", "watt/m^2", "watt m-2", "watt m^-2",
    }
    if u in radiation_tokens:
        return "radiation"
    return "unsupported"


def weighted_fraction(mask: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted fraction among finite/valid cells plus valid watershed-area fraction."""
    w = np.asarray(weights, dtype=float)
    base = np.isfinite(w) & (w > 0)
    valid_mask = np.asarray(valid, dtype=bool) & base
    total_weight = float(np.sum(w[base]))
    valid_weight = float(np.sum(w[valid_mask]))
    if valid_weight <= 0:
        return np.nan, 0.0
    numerator = float(np.sum(w[valid_mask & np.asarray(mask, dtype=bool)]))
    return numerator / valid_weight, valid_weight / total_weight if total_weight > 0 else 0.0


# ----- LDASIN atmospheric forcing ------------------------------------------------


def _ldasin_path(root: Path, pattern: str, timestamp: pd.Timestamp) -> Path:
    return Path(root) / pattern.format(timestamp=pd.Timestamp(timestamp).to_pydatetime())


def _read_ldasin_hour(
    path: Path,
    full_weights: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    subset_weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, str]], pd.DataFrame]:
    """Read one hourly LDASIN file and reduce required forcing to the watershed."""
    with open_dataset_robust(path, decode_times=False) as ds:
        missing = [name for name in NOAH_LDASIN_REQUIRED if name not in ds]
        if missing:
            raise KeyError(f"{path} is missing required LDASIN variables: {missing}; variables={list(ds.data_vars)}")

        inventory = variable_inventory(ds)
        arrays: dict[str, np.ndarray] = {}
        metadata: dict[str, dict[str, str]] = {}
        for name in NOAH_LDASIN_REQUIRED:
            da = ds[name]
            arr = read_subset_array(
                da,
                full_weights.shape,
                row_slice,
                col_slice,
                time_selection=0,
                require_time=True,
            )
            if arr.ndim != 2:
                raise ValueError(f"Expected one 2-D field from {path}:{name}, got {arr.shape}.")
            arrays[name] = arr
            metadata[name] = {
                "units": clean_units_text(da.attrs.get("units")),
                "description": str(da.attrs.get("description", da.attrs.get("long_name", ""))),
            }

    require_wind_units(metadata["U2D"]["units"], f"{path.name}:U2D")
    require_wind_units(metadata["V2D"]["units"], f"{path.name}:V2D")
    require_radiation_units(metadata["LWDOWN"]["units"], f"{path.name}:LWDOWN")
    # Explicitly validate temperature units now; conversion to C occurs after
    # the watershed mean, which is equivalent for the linear K->C transform.
    _ = temperature_to_c(np.asarray([273.15]), metadata["T2D"]["units"], f"{path.name}:T2D")

    # Crucial ordering: derive wind speed on the grid first, then spatially average.
    wind_speed = np.sqrt(arrays["U2D"] ** 2 + arrays["V2D"] ** 2)

    values: dict[str, float] = {}
    valid_area: dict[str, float] = {}
    for name, field in (
        ("T2D", arrays["T2D"]),
        ("LWDOWN", arrays["LWDOWN"]),
        ("WIND_SPEED_FROM_U2D_V2D", wind_speed),
    ):
        mean, valid = weighted_mean_2d(field, subset_weights)
        values[name] = mean
        valid_area[name] = valid

    metadata["WIND_SPEED_FROM_U2D_V2D"] = {
        "units": metadata["U2D"]["units"],
        "description": "Grid-cell wind speed computed as sqrt(U2D^2 + V2D^2) before watershed averaging.",
    }
    return values, valid_area, metadata, inventory


def extract_noah_ldasin_forcing(
    args: argparse.Namespace,
    full_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(args.noah_ldasin_root)
    if not root.exists():
        raise FileNotFoundError(f"Noah-MP LDASIN forcing directory does not exist: {root}")

    row_slice, col_slice, weights = positive_weight_bbox(full_weights)
    timestep_seconds = 3600.0
    rows: list[dict[str, Any]] = []
    missing_files: list[str] = []
    first_inventory = pd.DataFrame()
    variable_metadata: dict[str, Any] = {}
    expected_units: dict[str, str] = {}
    all_dates = expected_dates()

    print(
        f"[Noah-MP LDASIN] extracting {len(all_dates)} dates from {root}; "
        f"watershed bbox rows {row_slice.start}:{row_slice.stop}, cols {col_slice.start}:{col_slice.stop}"
    )

    for n, (wy, date) in enumerate(all_dates, start=1):
        hourly_t2d: list[float] = []
        hourly_lw: list[float] = []
        hourly_wind: list[float] = []
        valid_t2d: list[float] = []
        valid_lw: list[float] = []
        valid_wind: list[float] = []
        day_missing = 0

        for hour in range(24):
            timestamp = pd.Timestamp(date) + pd.Timedelta(hours=hour)
            path = _ldasin_path(root, args.noah_ldasin_pattern, timestamp)
            if not path.exists():
                missing_files.append(str(path))
                day_missing += 1
                if args.noah_ldasin_missing_file_policy == "error":
                    raise FileNotFoundError(f"Missing required Noah-MP hourly LDASIN file: {path}")
                warnings.warn(f"Missing Noah-MP hourly LDASIN file: {path}", stacklevel=2)
                hourly_t2d.append(np.nan); valid_t2d.append(np.nan)
                hourly_lw.append(np.nan); valid_lw.append(np.nan)
                hourly_wind.append(np.nan); valid_wind.append(np.nan)
                continue

            values, valid, metadata, inventory = _read_ldasin_hour(
                path, full_weights, row_slice, col_slice, weights
            )
            if first_inventory.empty:
                first_inventory = inventory

            # Unit consistency is checked across every file actually read.
            for name, info in metadata.items():
                units = info["units"]
                if name not in expected_units:
                    expected_units[name] = units
                    variable_metadata[name] = {
                        **info,
                        "source": "LDASIN",
                    }
                elif normalized_units(units) != normalized_units(expected_units[name]):
                    raise ValueError(
                        f"LDASIN units changed for {name}: expected {expected_units[name]!r}, "
                        f"found {units!r} in {path}."
                    )

            hourly_t2d.append(values["T2D"]); valid_t2d.append(valid["T2D"])
            hourly_lw.append(values["LWDOWN"]); valid_lw.append(valid["LWDOWN"])
            hourly_wind.append(values["WIND_SPEED_FROM_U2D_V2D"]); valid_wind.append(valid["WIND_SPEED_FROM_U2D_V2D"])

        t2d_units = expected_units.get("T2D", "K")
        lw_units = expected_units.get("LWDOWN", "W/m^2")
        wind_units = expected_units.get("WIND_SPEED_FROM_U2D_V2D", expected_units.get("U2D", "m/s"))

        t2d_c = temperature_to_c(np.asarray(hourly_t2d, dtype=float), t2d_units, "Noah LDASIN T2D")
        t_stats = scalar_daily_stats(t2d_c)
        lw_stats = radiation_daily_stats(np.asarray(hourly_lw, dtype=float), timestep_seconds)
        w_stats = scalar_daily_stats(np.asarray(hourly_wind, dtype=float))

        row = {
            "date": pd.Timestamp(date),
            "water_year": int(wy),
            "source": "Noah-MP",
            "air_temp_forcing_mean_C": t_stats["mean"],
            "air_temp_forcing_min_hourly_watershed_mean_C": t_stats["min"],
            "air_temp_forcing_max_hourly_watershed_mean_C": t_stats["max"],
            "air_temp_forcing_finite_hours": t_stats["finite_hours"],
            "air_temp_minimum_hourly_valid_area_fraction": (
                float(np.nanmin(valid_t2d)) if np.isfinite(valid_t2d).any() else np.nan
            ),
            "incoming_longwave_mean_w_m2": lw_stats["mean_w_m2"],
            "incoming_longwave_energy_mj_m2": lw_stats["energy_mj_m2"],
            "incoming_longwave_finite_hours": lw_stats["finite_hours"],
            "incoming_longwave_minimum_hourly_valid_area_fraction": (
                float(np.nanmin(valid_lw)) if np.isfinite(valid_lw).any() else np.nan
            ),
            "wind_speed_forcing_mean_m_s": w_stats["mean"],
            "wind_speed_forcing_max_hourly_watershed_mean_m_s": w_stats["max"],
            "wind_speed_forcing_finite_hours": w_stats["finite_hours"],
            "wind_speed_minimum_hourly_valid_area_fraction": (
                float(np.nanmin(valid_wind)) if np.isfinite(valid_wind).any() else np.nan
            ),
            "ldasin_missing_hour_count": int(day_missing),
        }
        rows.append(row)

        if args.progress_every > 0 and (n % args.progress_every == 0 or n == len(all_dates)):
            print(f"[Noah-MP LDASIN] {n}/{len(all_dates)} dates complete ({date.date()})")

    frame = pd.DataFrame(rows).sort_values(["water_year", "date"]).reset_index(drop=True)
    metadata = {
        "source_directory": str(root),
        "filename_pattern": args.noah_ldasin_pattern,
        "timestamp_rule": "The YYYYMMDDHH timestamp in each LDASIN filename is treated as the forcing hour.",
        "required_variables": NOAH_LDASIN_REQUIRED,
        "wind_rule": "sqrt(U2D^2 + V2D^2) is computed at each grid cell before the watershed mean.",
        "record_count": len(frame),
        "expected_hours_per_day": 24,
        "timestep_seconds": timestep_seconds,
        "missing_file_policy": args.noah_ldasin_missing_file_policy,
        "missing_file_count": len(missing_files),
        "missing_files": missing_files,
        "weights_shape": list(full_weights.shape),
        "watershed_bbox": {
            "row_start": row_slice.start,
            "row_stop": row_slice.stop,
            "col_start": col_slice.start,
            "col_stop": col_slice.stop,
        },
        "variables": variable_metadata,
    }
    return frame, first_inventory, metadata


# ----- LDASOUT diagnostics -------------------------------------------------------


def discover_noah_ldasout_variables(ds: Any, fsno_variable: str) -> dict[str, Any]:
    sublim = _first_present(ds, NOAH_SUBLIMATION_CANDIDATES)
    if sublim is None:
        sublim = _description_match(ds, ["sublim"])
    diagnostics = {raw: stem for raw, stem in NOAH_DIAGNOSTICS.items() if raw in ds}
    return {
        "direct_sublimation": sublim,
        "diagnostics": diagnostics,
        "fsno_variable": fsno_variable if fsno_variable in ds else None,
    }


def _extract_noah_fsno_daily(
    ds: Any,
    variable_name: str,
    chosen: pd.DataFrame,
    state_hour: int,
    full_weights: np.ndarray,
    row_slice: slice,
    col_slice: slice,
    subset_weights: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    var = ds[variable_name]
    if "Time" not in var.dims:
        raise ValueError(f"{variable_name!r} does not contain Time; dims={var.dims}.")

    state = chosen[chosen["hour"].eq(state_hour)].copy().sort_values(["water_year", "date"])
    counts = state.groupby(["water_year", "date"]).size()
    bad = counts[counts.ne(1)]
    if len(bad):
        raise ValueError(f"FSNO extraction found non-unique/missing state-hour records: {bad.head(10).to_dict()}")

    indices = state["index"].to_numpy(dtype=int)
    arrays = read_subset_array(
        var,
        full_weights.shape,
        row_slice,
        col_slice,
        time_selection=indices,
        require_time=True,
    )
    if arrays.ndim == 2:
        arrays = arrays[None, ...]

    rows: list[dict[str, Any]] = []
    out_of_range_count = 0
    for (_, rec), arr in zip(state.iterrows(), arrays):
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        bad_range = finite & ((arr < -FSNO_FULL_TOLERANCE) | (arr > 1.0 + FSNO_FULL_TOLERANCE))
        out_of_range_count += int(np.count_nonzero(bad_range))
        if np.any(bad_range):
            arr = arr.copy()
            arr[bad_range] = np.nan
        arr = np.where(np.isfinite(arr), np.clip(arr, 0.0, 1.0), np.nan)
        valid = np.isfinite(arr)
        mean_fsno, valid_area = weighted_mean_2d(arr, subset_weights)
        classes = {
            "snowfree_0": arr <= FSNO_FULL_TOLERANCE,
            "very_low_gt0_0p1": (arr > FSNO_FULL_TOLERANCE) & (arr < 0.1),
            "partial_0p1_0p9": (arr >= 0.1) & (arr < 0.9),
            "nearly_full_0p9_lt1": (arr >= 0.9) & (arr < 1.0 - FSNO_FULL_TOLERANCE),
            "full_1": arr >= 1.0 - FSNO_FULL_TOLERANCE,
            "any_partial_gt0_lt1": (arr > FSNO_FULL_TOLERANCE) & (arr < 1.0 - FSNO_FULL_TOLERANCE),
        }
        row: dict[str, Any] = {
            "date": pd.Timestamp(rec["date"]),
            "water_year": int(rec["water_year"]),
            "fsno_mean": mean_fsno,
            "fsno_valid_area_fraction": valid_area,
        }
        for name, mask in classes.items():
            frac, _ = weighted_fraction(mask, valid, subset_weights)
            row[f"fsno_area_fraction_{name}"] = frac
        rows.append(row)

    metadata = {
        "available": True,
        "variable": variable_name,
        "units": clean_units_text(var.attrs.get("units")),
        "description": str(var.attrs.get("description", var.attrs.get("long_name", ""))),
        "state_hour": state_hour,
        "temporal_rule": f"Daily {state_hour:02d}:00 state, matching the Noah SWE/SNOWH daily-state convention.",
        "record_count": len(rows),
        "masked_out_of_range_cell_records": out_of_range_count,
        "range_rule": "Values outside [0,1] beyond numerical tolerance are masked; boundary roundoff is clipped.",
    }
    if out_of_range_count:
        warnings.warn(
            f"Masked {out_of_range_count} {variable_name} cell-records outside [0,1] beyond numerical tolerance.",
            stacklevel=2,
        )
    return pd.DataFrame(rows), metadata


def extract_noah_ldasout_diagnostics(
    cfg: WorkflowConfig,
    args: argparse.Namespace,
    full_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = cfg.paths["noahmp_output_file"]
    if not source.exists():
        raise FileNotFoundError(f"Configured Noah-MP LDASOUT does not exist: {source}")

    row_slice, col_slice, weights = positive_weight_bbox(full_weights)
    noah_cfg = cfg.section("noahmp")
    timestep_seconds = float(noah_cfg.get("expected_timestep_hours", 1.0)) * 3600.0
    expected_hours = int(round(SECONDS_PER_DAY / timestep_seconds))
    state_hour = int(noah_cfg.get("daily_state_hour", 23))

    print(f"[Noah-MP LDASOUT] opening production diagnostics: {source}")
    with open_dataset_robust(source, decode_times=False) as ds:
        inventory = variable_inventory(ds)
        discovered = discover_noah_ldasout_variables(ds, args.fsno_variable)
        times = decode_noah_times(ds, noah_cfg)
        chosen = _selected_noah_time_frame(pd.DatetimeIndex(times))
        _validate_noah_daily_completeness(chosen, expected_hours)

        print("[Noah-MP LDASOUT] variable discovery:")
        print(f"    direct_sublimation: {discovered['direct_sublimation']}")
        print(f"    FSNO: {discovered['fsno_variable']}")
        print(f"    diagnostics: {sorted(discovered['diagnostics'])}")

        base = chosen[["date", "water_year"]].drop_duplicates().sort_values(["water_year", "date"]).reset_index(drop=True)
        variable_metadata: dict[str, Any] = {}

        def extract_one(raw_name: str, kind: str, prefix: str) -> None:
            nonlocal base
            print(f"[Noah-MP LDASOUT] reading {raw_name} -> {prefix}")
            hourly, valid, units, desc = _extract_noah_hourly_weighted(
                ds, raw_name, chosen, full_weights, row_slice, col_slice, weights
            )
            daily = _group_hourly_to_daily(
                chosen,
                hourly,
                valid,
                kind=kind,
                units=units,
                timestep_seconds=timestep_seconds,
                prefix=prefix,
            )
            base = _merge_daily(base, daily)
            variable_metadata[raw_name] = {
                "source": "LDASOUT",
                "role": prefix,
                "units": units,
                "description": desc,
                "aggregation": kind,
            }

        sublim = discovered["direct_sublimation"]
        sublimation_status: dict[str, Any] = {"available": False, "variable": sublim, "reason": None}
        if sublim is not None:
            units = clean_units_text(ds[sublim].attrs.get("units"))
            kind = _classify_sublimation_kind(units)
            if kind == "unsupported":
                sublimation_status["reason"] = (
                    f"Found {sublim} but did not recognize units {units!r}; raw conversion was not attempted."
                )
                warnings.warn(sublimation_status["reason"], stacklevel=2)
            else:
                extract_one(sublim, kind, "direct_snow_sublimation")
                sublimation_status.update({"available": True, "reason": None, "aggregation": kind, "units": units})
        else:
            sublimation_status["reason"] = (
                "No direct Noah-MP variable with a recognized sublimation name/description was present."
            )

        # Model diagnostics are intentionally kept separate from LDASIN forcing.
        for raw_name, stem in discovered["diagnostics"].items():
            kind = "temperature" if raw_name in {"T2MV", "T2MB"} else "radiation"
            extract_one(raw_name, kind, stem)

        fsno_name = discovered["fsno_variable"]
        if fsno_name is not None:
            fsno_daily, fsno_status = _extract_noah_fsno_daily(
                ds,
                fsno_name,
                chosen,
                state_hour,
                full_weights,
                row_slice,
                col_slice,
                weights,
            )
            base = _merge_daily(base, fsno_daily)
            variable_metadata[fsno_name] = {
                "source": "LDASOUT",
                "role": "native ground snow-cover fraction",
                "units": clean_units_text(ds[fsno_name].attrs.get("units")),
                "description": str(ds[fsno_name].attrs.get("description", ds[fsno_name].attrs.get("long_name", ""))),
                "aggregation": f"state at {state_hour:02d}:00",
            }
        else:
            fsno_status = {
                "available": False,
                "variable": args.fsno_variable,
                "reason": f"{args.fsno_variable!r} is not present in the production Noah-MP LDASOUT.",
                "state_hour": state_hour,
            }

        metadata = {
            "source": str(source),
            "record_count": len(base),
            "selected_hourly_record_count": len(chosen),
            "expected_hours_per_day": expected_hours,
            "timestep_seconds": timestep_seconds,
            "daily_state_hour": state_hour,
            "weights_shape": list(full_weights.shape),
            "watershed_bbox": {
                "row_start": row_slice.start,
                "row_stop": row_slice.stop,
                "col_start": col_slice.start,
                "col_stop": col_slice.stop,
            },
            "discovered": discovered,
            "variables": variable_metadata,
            "direct_sublimation": sublimation_status,
            "fsno": fsno_status,
            "guardrails": {
                "forcing_source": "Atmospheric T2D/U2D/V2D/LWDOWN are read from LDASIN, not this LDASOUT.",
                "T2MV_T2MB": "Extracted only as Noah model diagnostics; never substituted for atmospheric T2D forcing.",
                "latent_heat": "Total Noah LH is retained as an energy diagnostic only; it is not converted to snow sublimation.",
                "FSNO": f"Native FSNO is sampled at {state_hour:02d}:00 to match the Noah SWE daily-state convention and is never inferred from SWE.",
            },
        }
        return base, inventory, metadata


def extract_noah(
    cfg: WorkflowConfig,
    args: argparse.Namespace,
    full_weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Combine LDASIN atmospheric forcing and LDASOUT model diagnostics by day."""
    forcing, ldasin_inventory, ldasin_meta = extract_noah_ldasin_forcing(args, full_weights)
    diagnostics, ldasout_inventory, ldasout_meta = extract_noah_ldasout_diagnostics(cfg, args, full_weights)

    merged = forcing.merge(
        diagnostics,
        on=["date", "water_year"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_ldasout"),
    )
    if "source_ldasout" in merged:
        merged = merged.drop(columns=["source_ldasout"])
    merged["source"] = "Noah-MP"

    # Requested periods should be represented one-to-one in both sources.
    forcing_keys = set(zip(forcing["water_year"], pd.to_datetime(forcing["date"])))
    diagnostic_keys = set(zip(diagnostics["water_year"], pd.to_datetime(diagnostics["date"])))
    missing_diag = sorted(forcing_keys - diagnostic_keys)
    extra_diag = sorted(diagnostic_keys - forcing_keys)
    if missing_diag or extra_diag:
        raise ValueError(
            "LDASIN/LDASOUT daily date coverage does not match for the requested periods. "
            f"Missing LDASOUT days={missing_diag[:5]}, extra LDASOUT days={extra_diag[:5]}"
        )

    metadata = {
        "record_count": len(merged),
        "forcing": ldasin_meta,
        "diagnostics": ldasout_meta,
        "source_separation": {
            "atmospheric_forcing": "Hourly HRLDAS LDASIN files: T2D, U2D, V2D, LWDOWN.",
            "model_diagnostics": "Production LDASOUT: T2MV/T2MB, energy terms, native FSNO, and direct sublimation only if present.",
        },
    }
    return merged.sort_values(["water_year", "date"]).reset_index(drop=True), ldasin_inventory, ldasout_inventory, metadata


# -----------------------------------------------------------------------------
# Output assembly
# -----------------------------------------------------------------------------


def combine_source_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    valid = [f.copy() for f in frames if f is not None and len(f)]
    if not valid:
        return pd.DataFrame()
    all_columns = sorted(set().union(*(set(f.columns) for f in valid)))
    preferred = ["date", "water_year", "source"]
    ordered = preferred + [c for c in all_columns if c not in preferred]
    for f in valid:
        for c in ordered:
            if c not in f:
                f[c] = np.nan
        f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    return pd.concat([f[ordered] for f in valid], ignore_index=True).sort_values(["source", "water_year", "date"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.output_dir / "diagnostics" / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    isnobal_weights_path = cfg.output_dir / "cache" / "east_river_fraction_isnobal.npz"
    noah_weights_path = cfg.output_dir / "cache" / "east_river_fraction_noahmp.npz"

    source_frames: list[pd.DataFrame] = []
    manifest: dict[str, Any] = {
        "analysis": "WY2021 versus WY2026 additional forcing and energy extraction",
        "periods": {f"WY{wy}": [str(analysis_bounds(wy)[0].date()), str(analysis_bounds(wy)[1].date())] for wy in YEARS},
        "output_directory": str(output_dir),
        "scientific_guardrails": [
            "Noah atmospheric T2D/U2D/V2D/LWDOWN are read from hourly LDASIN files; LDASOUT diagnostics are not forcing substitutes.",
            "Noah wind speed is computed as sqrt(U2D^2 + V2D^2) at each grid cell before watershed averaging.",
            "No snow sublimation is derived from total latent heat or generic evaporation.",
            "Native Noah FSNO is sampled at the configured daily state hour (23:00 in the current v0.8.2 configuration) and is never derived from SWE.",
            "iSnobal em.nc:evaporation is preserved under its source meaning and not silently relabeled as pure sublimation.",
        ],
    }

    if not args.skip_isnobal:
        isnobal_weights = load_fractional_weights(isnobal_weights_path)
        isnobal_frame, isnobal_meta = extract_isnobal(cfg, args, isnobal_weights)
        isnobal_path = output_dir / "isnobal_additional_forcing_daily.csv"
        isnobal_frame.to_csv(isnobal_path, index=False)
        source_frames.append(isnobal_frame)
        manifest["isnobal"] = {"output": str(isnobal_path), **isnobal_meta}
        print(f"[iSnobal] wrote {isnobal_path}")

    if not args.skip_noah:
        noah_weights = load_fractional_weights(noah_weights_path)
        noah_frame, ldasin_inventory, ldasout_inventory, noah_meta = extract_noah(cfg, args, noah_weights)
        noah_path = output_dir / "noahmp_additional_forcing_daily.csv"
        ldasin_inventory_path = output_dir / "noahmp_ldasin_variable_inventory.csv"
        ldasout_inventory_path = output_dir / "noahmp_ldasout_variable_inventory.csv"
        noah_frame.to_csv(noah_path, index=False)
        ldasin_inventory.to_csv(ldasin_inventory_path, index=False)
        ldasout_inventory.to_csv(ldasout_inventory_path, index=False)

        fsno_columns = [c for c in noah_frame.columns if c == "fsno_mean" or c.startswith("fsno_")]
        fsno_path = None
        if "fsno_mean" in noah_frame.columns and pd.to_numeric(noah_frame["fsno_mean"], errors="coerce").notna().any():
            fsno_path = output_dir / "noahmp_fsno_daily.csv"
            noah_frame[["date", "water_year"] + fsno_columns].to_csv(fsno_path, index=False)

        source_frames.append(noah_frame)
        manifest["noahmp"] = {
            "output": str(noah_path),
            "ldasin_variable_inventory": str(ldasin_inventory_path),
            "ldasout_variable_inventory": str(ldasout_inventory_path),
            "fsno_output": str(fsno_path) if fsno_path is not None else None,
            **noah_meta,
        }
        print(f"[Noah-MP] wrote {noah_path}")
        print(f"[Noah-MP] wrote {ldasin_inventory_path}")
        print(f"[Noah-MP] wrote {ldasout_inventory_path}")
        if fsno_path is not None:
            print(f"[Noah-MP] wrote {fsno_path}")

    combined = combine_source_frames(source_frames)
    if len(combined):
        combined_path = output_dir / "wy2021_wy2026_additional_forcing_daily.csv"
        combined.to_csv(combined_path, index=False)
        manifest["combined_output"] = str(combined_path)
        print(f"[combined] wrote {combined_path}")

    metadata_path = output_dir / "additional_forcing_metadata.json"
    save_json(manifest, metadata_path)
    print(f"[metadata] wrote {metadata_path}")
    print("Done. Upload the additional_forcing folder for the next analysis step.")


if __name__ == "__main__":
    main()
