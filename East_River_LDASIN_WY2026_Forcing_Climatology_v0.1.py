#!/usr/bin/env python3
"""East River LDASIN forcing climatology focused on WY2026.

Purpose
-------
This companion diagnostic asks how unusual WY2026 was relative to WY2018-WY2025
using ONLY Noah-MP/HRLDAS LDASIN atmospheric forcing over the East River
watershed. It does not use SNOTEL and does not modify the East River Workflow
v0.8.2 H1-H4 products.

The common comparison window is fixed to October 1 through June 30 for every
water year WY2018-WY2026.

Authoritative LDASIN variables
------------------------------
    T2D      : atmospheric 2-m air temperature forcing [K]
    RAINRATE : precipitation-rate forcing [mm/s]
    SWDOWN   : incoming downward shortwave radiation [W/m^2]
    LWDOWN   : incoming downward longwave radiation [W/m^2]

Daily construction
------------------
For each YYYYMMDDHH.LDASIN_DOMAIN1 file, the timestamp is taken from the
filename. No timezone conversion is applied: calendar-day grouping follows the
filename clock exactly.

Each hourly field is first reduced to an East River watershed mean using the
same exact Noah-grid fractional watershed weights as East River Workflow v0.8.2.
All positive-weight watershed cells must be finite; otherwise processing stops.
This deliberately avoids silently renormalizing a partially missing forcing
field and therefore does not need a per-variable valid-area-fraction product.

For a complete 24-hour day:
    precipitation_mm_day = sum(hourly watershed RAINRATE * 3600 s)
    temperature_mean_C   = mean(hourly watershed T2D) - 273.15
    incoming_SW_mean_Wm2 = mean(hourly watershed SWDOWN)
    incoming_LW_mean_Wm2 = mean(hourly watershed LWDOWN)
    incoming_SW_MJm2_day = sum(hourly watershed SWDOWN * 3600 * 1e-6)
    incoming_LW_MJm2_day = sum(hourly watershed LWDOWN * 3600 * 1e-6)

Radiation is retained both as daily mean W/m^2 (daily forcing intensity) and
integrated MJ/m^2/day so physically meaningful seasonal cumulative radiation
energy can be compared across water years.

QC policy
---------
* Exactly 24 hourly files are required for every calendar day in every target
  Oct 1-Jun 30 window.
* Duplicate files for the same filename timestamp are a hard error; the script
  never chooses one silently.
* Required variables and expected units are checked in every hourly file.
* Every positive-weight East River grid cell must be finite for all four fields.
* Negative precipitation or materially negative incoming radiation is rejected.

Outputs
-------
Created below <main output_dir>/ldasin_wy2026_forcing_v0.1 by default:

    tables/
        ldasin_watershed_daily_oct1_jun30.csv
        ldasin_window_metrics_all_years.csv
        ldasin_wy2026_specialness_summary.csv
        ldasin_wy2026_daily_anomalies.csv
        ldasin_wy2026_extreme_anomaly_periods.csv

    diagnostics/
        ldasin_hourly_inventory_qc.csv
        ldasin_duplicate_timestamps.csv              [only when duplicates exist]
        ldasin_manifest.json

    figures/
        figure01_wy2026_vs_history_daily_forcing.<format>
        figure02_oct1_jun30_metrics_all_years.<format>
        figure03_wy2026_daily_anomalies.<format>

Run
---
From the East_River_Workflow_Code_v0.8.2 repository/environment:

    python East_River_LDASIN_WY2026_Forcing_Climatology_v0.1.py \
        --config config/east_river_config.yaml \
        --ldasin-root /path/to/hourly/LDASIN/archive

The archive may be flat or nested. The default search is recursive and accepts
files whose basename is exactly YYYYMMDDHH.LDASIN_DOMAIN1.

Use --overwrite to rebuild the expensive daily table after it already exists.
Use --no-plots to generate tables/diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from netCDF4 import Dataset as NetCDFDataset
except Exception:  # pragma: no cover - fallback depends on environment
    NetCDFDataset = None


# -----------------------------------------------------------------------------
# Allow this companion script to live in the repository root, examples/, or a
# nearby folder without requiring an editable install.
# -----------------------------------------------------------------------------
def _bootstrap_east_river_import() -> None:
    try:
        import east_river_workflow  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in [here.parent, *here.parents]:
        candidates.extend(
            [
                parent / "src",
                parent / "East_River_Workflow_Code_v0.8.2" / "src",
                parent / "erw_code" / "East_River_Workflow_Code_v0.8.2" / "src",
            ]
        )
    for candidate in candidates:
        if (candidate / "east_river_workflow").is_dir():
            sys.path.insert(0, str(candidate))
            return


_bootstrap_east_river_import()

from east_river_workflow import __version__ as ERW_VERSION  # noqa: E402
from east_river_workflow.config import WorkflowConfig, load_config  # noqa: E402
from east_river_workflow.data_access import normalized_units, read_noah_grid  # noqa: E402
from east_river_workflow.grids import (  # noqa: E402
    fractional_polygon_weights,
    load_weights,
    select_watershed,
)
from east_river_workflow.plotting import configure_matplotlib  # noqa: E402
from east_river_workflow.utils import save_json  # noqa: E402


SCRIPT_VERSION = "0.1.0"
DEFAULT_OUTPUT_SUBDIR = "ldasin_wy2026_forcing_v0.1"
DEFAULT_YEARS = list(range(2018, 2027))
FOCUS_WY = 2026
SECONDS_PER_HOUR = 3600.0
MJ_PER_J = 1.0e-6
KELVIN_OFFSET = 273.15
EXPECTED_HOURS_PER_DAY = 24
ROLLING_DAYS = 7

LDASIN_BASENAME_RE = re.compile(r"^(?P<stamp>\d{10})\.LDASIN_DOMAIN1$")

VAR_T2D = "T2D"
VAR_RAIN = "RAINRATE"
VAR_SW = "SWDOWN"
VAR_LW = "LWDOWN"
REQUIRED_VARIABLES = (VAR_T2D, VAR_RAIN, VAR_SW, VAR_LW)

DAILY_TABLE = "ldasin_watershed_daily_oct1_jun30.csv"
WINDOW_TABLE = "ldasin_window_metrics_all_years.csv"
SPECIAL_TABLE = "ldasin_wy2026_specialness_summary.csv"
ANOMALY_TABLE = "ldasin_wy2026_daily_anomalies.csv"
EXTREME_TABLE = "ldasin_wy2026_extreme_anomaly_periods.csv"
INVENTORY_QC = "ldasin_hourly_inventory_qc.csv"
DUPLICATE_QC = "ldasin_duplicate_timestamps.csv"
MANIFEST_JSON = "ldasin_manifest.json"


@dataclass(frozen=True)
class Settings:
    ldasin_root: Path
    output_root: Path
    years: tuple[int, ...]
    overwrite: bool
    recursive: bool
    make_plots: bool
    drop_feb29: bool

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def diagnostics_dir(self) -> Path:
        return self.output_root / "diagnostics"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    def create_tree(self) -> None:
        for path in [self.tables_dir, self.diagnostics_dir, self.figures_dir]:
            path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Time/window helpers
# =============================================================================
def _wy_start(wy: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(wy) - 1, month=10, day=1)


def _wy_common_end(wy: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(wy), month=6, day=30)


def _window_bounds(wy: int, window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if window == "oct1_jun30":
        return _wy_start(wy), _wy_common_end(wy)
    if window == "nov1_mar15":
        return pd.Timestamp(wy - 1, 11, 1), pd.Timestamp(wy, 3, 15)
    if window == "apr1_jun30":
        return pd.Timestamp(wy, 4, 1), pd.Timestamp(wy, 6, 30)
    raise KeyError(window)


WINDOWS: dict[str, str] = {
    "oct1_jun30": "Oct 1-Jun 30",
    "nov1_mar15": "Nov 1-Mar 15",
    "apr1_jun30": "Apr 1-Jun 30",
}


def _target_days(years: Iterable[int], drop_feb29: bool) -> pd.DatetimeIndex:
    pieces: list[pd.DatetimeIndex] = []
    for wy in years:
        days = pd.date_range(_wy_start(wy), _wy_common_end(wy), freq="D")
        if drop_feb29:
            days = days[~((days.month == 2) & (days.day == 29))]
        pieces.append(days)
    if not pieces:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([p.to_numpy() for p in pieces]))


def _expected_hourly_timestamps(years: Iterable[int], drop_feb29: bool) -> pd.DatetimeIndex:
    pieces: list[pd.DatetimeIndex] = []
    for day in _target_days(years, drop_feb29):
        pieces.append(pd.date_range(day, day + pd.Timedelta(hours=23), freq="h"))
    if not pieces:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([p.to_numpy() for p in pieces]))


def _water_year_from_date(date: pd.Timestamp) -> int:
    return int(date.year + 1 if date.month >= 10 else date.year)


def _month_day(date: pd.Timestamp) -> str:
    return f"{date.month:02d}-{date.day:02d}"


def _map_to_focus_wy_date(month_day: str, focus_wy: int = FOCUS_WY) -> pd.Timestamp | pd.NaT:
    month, day = [int(x) for x in month_day.split("-")]
    year = focus_wy - 1 if month >= 10 else focus_wy
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        # e.g. Feb 29 when focus WY is not a leap-year spring.
        return pd.NaT


# =============================================================================
# Inventory / QC
# =============================================================================
def _iter_ldasin_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            for name in filenames:
                if name.endswith(".LDASIN_DOMAIN1"):
                    yield base / name
    else:
        for path in root.iterdir():
            if path.is_file() and path.name.endswith(".LDASIN_DOMAIN1"):
                yield path


def _inventory_archive(
    settings: Settings,
    logger: logging.Logger,
) -> tuple[dict[pd.Timestamp, Path], pd.DataFrame, pd.DataFrame]:
    expected = _expected_hourly_timestamps(settings.years, settings.drop_feb29)
    expected_set = set(expected.to_pydatetime())

    primary: dict[pd.Timestamp, Path] = {}
    duplicate_rows: list[dict[str, Any]] = []
    candidate_count = 0

    logger.info("Scanning LDASIN archive: %s", settings.ldasin_root)
    for path in _iter_ldasin_files(settings.ldasin_root, settings.recursive):
        match = LDASIN_BASENAME_RE.match(path.name)
        if match is None:
            continue
        candidate_count += 1
        ts = pd.Timestamp(datetime.strptime(match.group("stamp"), "%Y%m%d%H"))
        if ts.to_pydatetime() not in expected_set:
            continue
        if ts in primary:
            duplicate_rows.append(
                {
                    "timestamp": ts,
                    "first_path": str(primary[ts]),
                    "duplicate_path": str(path),
                }
            )
        else:
            primary[ts] = path

    duplicate_df = pd.DataFrame(duplicate_rows)

    qc_rows: list[dict[str, Any]] = []
    for day in _target_days(settings.years, settings.drop_feb29):
        expected_day = pd.date_range(day, day + pd.Timedelta(hours=23), freq="h")
        found = [ts for ts in expected_day if ts in primary]
        missing = [ts for ts in expected_day if ts not in primary]
        qc_rows.append(
            {
                "date": day,
                "water_year": _water_year_from_date(day),
                "expected_hour_count": EXPECTED_HOURS_PER_DAY,
                "found_hour_count": len(found),
                "complete_00_23": len(found) == EXPECTED_HOURS_PER_DAY,
                "missing_hours": ",".join(f"{ts.hour:02d}" for ts in missing),
            }
        )
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(settings.diagnostics_dir / INVENTORY_QC, index=False)
    if len(duplicate_df):
        duplicate_df.to_csv(settings.diagnostics_dir / DUPLICATE_QC, index=False)

    missing_hours = int((EXPECTED_HOURS_PER_DAY - qc["found_hour_count"]).clip(lower=0).sum())
    logger.info(
        "LDASIN inventory: scanned %s filename candidates; target hourly files found=%s/%s; missing=%s; duplicates=%s",
        f"{candidate_count:,}",
        f"{len(primary):,}",
        f"{len(expected):,}",
        f"{missing_hours:,}",
        f"{len(duplicate_df):,}",
    )

    if len(duplicate_df):
        preview = duplicate_df.head(10).to_dict(orient="records")
        raise RuntimeError(
            "Duplicate LDASIN files were found for the same filename timestamp. "
            "The script will not choose silently. See diagnostics/"
            f"{DUPLICATE_QC}. First conflicts: {preview}"
        )

    incomplete = qc[~qc["complete_00_23"]]
    if len(incomplete):
        preview = incomplete[["date", "found_hour_count", "missing_hours"]].head(10).to_dict(orient="records")
        raise RuntimeError(
            f"LDASIN hourly coverage is incomplete on {len(incomplete)} target day(s). "
            f"See diagnostics/{INVENTORY_QC}. First incomplete days: {preview}"
        )

    return primary, qc, duplicate_df


# =============================================================================
# Spatial support
# =============================================================================
def _load_noah_weights(
    cfg: WorkflowConfig,
    settings: Settings,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    noah_grid, _dem, _lat, _lon, grid_diag = read_noah_grid(
        cfg.data["paths"]["noahmp_geo_file"], cfg.section("noahmp")
    )

    main_cache = cfg.output_dir / "cache" / "east_river_fraction_noahmp.npz"
    if main_cache.exists():
        weights = load_weights(main_cache, noah_grid)
        source = str(main_cache)
    else:
        logger.warning(
            "Main Noah-grid watershed-weight cache is absent; rebuilding exact East River fractional weights in memory."
        )
        watershed = select_watershed(
            cfg.data["paths"]["watershed_shapefile"],
            cfg.section("watershed")["name_field"],
            cfg.section("watershed")["name_contains"],
        )
        weights = fractional_polygon_weights(
            noah_grid,
            watershed,
            chunk_size=int(cfg.section("watershed")["fractional_chunk_size"]),
        )
        source = "rebuilt in memory by LDASIN companion script"

    positive = np.asarray(weights, dtype=float) > 0
    if not np.any(positive):
        raise ValueError("No positive East River fractional watershed weights were found on the Noah grid.")
    normalized = np.asarray(weights[positive], dtype=float)
    normalized /= float(np.sum(normalized))

    diag = {
        "weight_source": source,
        "noah_grid_shape": list(noah_grid.shape),
        "positive_weight_cell_count": int(np.count_nonzero(positive)),
        "fractional_weight_sum_cells": float(np.sum(weights)),
        "fractional_weight_area_m2": float(np.sum(weights) * noah_grid.cell_area_m2),
        "noah_grid_diagnostics": grid_diag,
    }
    return positive, normalized, diag


# =============================================================================
# Hourly NetCDF reading
# =============================================================================
def _as_finite_array(value: Any) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        return np.asarray(np.ma.filled(value, np.nan), dtype=float)
    return np.asarray(value, dtype=float)


def _squeeze_hourly_field(values: np.ndarray, variable: str, path: Path) -> np.ndarray:
    arr = _as_finite_array(values).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"{variable} in {path} must reduce to a 2-D Noah grid; got shape {arr.shape}.")
    return arr


def _check_units(variable: str, units: str, path: Path) -> None:
    norm = normalized_units(units)
    accepted: dict[str, set[str]] = {
        VAR_T2D: {"k", "kelvin"},
        VAR_RAIN: {"mm/s", "mm s-1", "mm s^-1", "mm s^(-1)", "kg m-2 s-1", "kg m^-2 s^-1"},
        VAR_SW: {"w/m^2", "w/m2", "w m-2", "w m^-2", "watt/m2", "watt m-2"},
        VAR_LW: {"w/m^2", "w/m2", "w m-2", "w m^-2", "watt/m2", "watt m-2"},
    }
    if norm not in accepted[variable]:
        raise ValueError(
            f"Unexpected units for LDASIN {variable} in {path}: {units!r} (normalized={norm!r}); "
            f"accepted={sorted(accepted[variable])}."
        )


def _weighted_hourly_means_netcdf4(
    path: Path,
    positive_mask: np.ndarray,
    normalized_weights: np.ndarray,
) -> dict[str, float]:
    if NetCDFDataset is None:
        raise RuntimeError("netCDF4 is unavailable")

    with NetCDFDataset(path, mode="r") as ds:
        missing = [name for name in REQUIRED_VARIABLES if name not in ds.variables]
        if missing:
            raise KeyError(f"Required LDASIN variables missing in {path}: {missing}")

        result: dict[str, float] = {}
        for name in REQUIRED_VARIABLES:
            var = ds.variables[name]
            units = str(getattr(var, "units", ""))
            _check_units(name, units, path)
            field = _squeeze_hourly_field(var[:], name, path)
            if field.shape != positive_mask.shape:
                raise ValueError(
                    f"LDASIN {name} grid shape {field.shape} does not match Noah weight grid {positive_mask.shape} in {path}."
                )
            basin_values = field[positive_mask]
            if not np.all(np.isfinite(basin_values)):
                bad = int(np.count_nonzero(~np.isfinite(basin_values)))
                raise ValueError(
                    f"LDASIN {name} contains {bad} non-finite East River watershed cell(s) in {path}. "
                    "Partial-area renormalization is intentionally disabled for this diagnostic."
                )
            result[name] = float(np.dot(basin_values, normalized_weights))
    return result


def _weighted_hourly_means_xarray(
    path: Path,
    positive_mask: np.ndarray,
    normalized_weights: np.ndarray,
) -> dict[str, float]:
    # Fallback path: keep imports local because netCDF4 is expected in the project environment.
    from east_river_workflow.utils import open_dataset_robust

    with open_dataset_robust(path, decode_times=False) as ds:
        missing = [name for name in REQUIRED_VARIABLES if name not in ds]
        if missing:
            raise KeyError(f"Required LDASIN variables missing in {path}: {missing}")
        result: dict[str, float] = {}
        for name in REQUIRED_VARIABLES:
            var = ds[name]
            units = str(var.attrs.get("units", ""))
            _check_units(name, units, path)
            field = _squeeze_hourly_field(var.values, name, path)
            if field.shape != positive_mask.shape:
                raise ValueError(
                    f"LDASIN {name} grid shape {field.shape} does not match Noah weight grid {positive_mask.shape} in {path}."
                )
            basin_values = field[positive_mask]
            if not np.all(np.isfinite(basin_values)):
                bad = int(np.count_nonzero(~np.isfinite(basin_values)))
                raise ValueError(
                    f"LDASIN {name} contains {bad} non-finite East River watershed cell(s) in {path}."
                )
            result[name] = float(np.dot(basin_values, normalized_weights))
    return result


def _weighted_hourly_means(
    path: Path,
    positive_mask: np.ndarray,
    normalized_weights: np.ndarray,
) -> dict[str, float]:
    if NetCDFDataset is not None:
        return _weighted_hourly_means_netcdf4(path, positive_mask, normalized_weights)
    return _weighted_hourly_means_xarray(path, positive_mask, normalized_weights)


# =============================================================================
# Daily processing
# =============================================================================
def _validate_hourly_physics(values: dict[str, float], path: Path) -> None:
    if values[VAR_RAIN] < -1.0e-12:
        raise ValueError(f"Negative watershed-mean RAINRATE={values[VAR_RAIN]} mm/s in {path}")
    if values[VAR_SW] < -1.0e-6:
        raise ValueError(f"Materially negative watershed-mean SWDOWN={values[VAR_SW]} W/m^2 in {path}")
    if values[VAR_LW] < -1.0e-6:
        raise ValueError(f"Materially negative watershed-mean LWDOWN={values[VAR_LW]} W/m^2 in {path}")
    if not (150.0 <= values[VAR_T2D] <= 350.0):
        raise ValueError(f"Implausible watershed-mean T2D={values[VAR_T2D]} K in {path}")


def _process_daily(
    inventory: dict[pd.Timestamp, Path],
    settings: Settings,
    positive_mask: np.ndarray,
    normalized_weights: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_days = _target_days(settings.years, settings.drop_feb29)

    for day_index, day in enumerate(target_days, start=1):
        t2: list[float] = []
        rain: list[float] = []
        sw: list[float] = []
        lw: list[float] = []

        for hour in range(EXPECTED_HOURS_PER_DAY):
            ts = day + pd.Timedelta(hours=hour)
            path = inventory[ts]
            values = _weighted_hourly_means(path, positive_mask, normalized_weights)
            _validate_hourly_physics(values, path)
            t2.append(values[VAR_T2D])
            rain.append(max(0.0, values[VAR_RAIN]))
            sw.append(max(0.0, values[VAR_SW]))
            lw.append(max(0.0, values[VAR_LW]))

        t2_arr = np.asarray(t2, dtype=float)
        rain_arr = np.asarray(rain, dtype=float)
        sw_arr = np.asarray(sw, dtype=float)
        lw_arr = np.asarray(lw, dtype=float)

        rows.append(
            {
                "date": day,
                "water_year": _water_year_from_date(day),
                "month_day": _month_day(day),
                "hour_count": EXPECTED_HOURS_PER_DAY,
                "precip_mm_day": float(np.sum(rain_arr * SECONDS_PER_HOUR)),
                "temperature_mean_c": float(np.mean(t2_arr) - KELVIN_OFFSET),
                "incoming_sw_mean_w_m2": float(np.mean(sw_arr)),
                "incoming_sw_energy_mj_m2_day": float(np.sum(sw_arr * SECONDS_PER_HOUR * MJ_PER_J)),
                "incoming_lw_mean_w_m2": float(np.mean(lw_arr)),
                "incoming_lw_energy_mj_m2_day": float(np.sum(lw_arr * SECONDS_PER_HOUR * MJ_PER_J)),
            }
        )

        if day_index % 30 == 0 or day_index == len(target_days):
            logger.info("Processed %s/%s target days through %s", day_index, len(target_days), day.date())

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["precip_accum_oct1_mm"] = daily.groupby("water_year", sort=False)["precip_mm_day"].cumsum()
    return daily


def _validate_daily_table(daily: pd.DataFrame, settings: Settings) -> None:
    required = {
        "date",
        "water_year",
        "month_day",
        "precip_mm_day",
        "temperature_mean_c",
        "incoming_sw_mean_w_m2",
        "incoming_sw_energy_mj_m2_day",
        "incoming_lw_mean_w_m2",
        "incoming_lw_energy_mj_m2_day",
        "precip_accum_oct1_mm",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Cached daily LDASIN table is missing columns: {missing}")

    dates = pd.DatetimeIndex(pd.to_datetime(daily["date"]))
    expected = _target_days(settings.years, settings.drop_feb29)
    if len(dates) != len(expected) or not dates.equals(expected):
        raise ValueError(
            "Cached daily LDASIN table does not exactly match the requested WY/date configuration. "
            "Re-run with --overwrite."
        )


# =============================================================================
# Scientific summaries
# =============================================================================
def _window_metrics(daily: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for wy in settings.years:
        wy_frame = daily[daily["water_year"].eq(wy)].sort_values("date")
        for window, label in WINDOWS.items():
            start, end = _window_bounds(wy, window)
            sub = wy_frame[(wy_frame["date"] >= start) & (wy_frame["date"] <= end)].copy()
            if settings.drop_feb29:
                sub = sub[~((sub["date"].dt.month == 2) & (sub["date"].dt.day == 29))]
            if sub.empty:
                raise ValueError(f"No daily LDASIN data for WY{wy} window {label}.")

            expected_days = pd.date_range(start, end, freq="D")
            if settings.drop_feb29:
                expected_days = expected_days[~((expected_days.month == 2) & (expected_days.day == 29))]
            if len(sub) != len(expected_days):
                raise ValueError(
                    f"WY{wy} {label} has {len(sub)} days, expected {len(expected_days)}; daily cache is incomplete."
                )

            rows.append(
                {
                    "water_year": int(wy),
                    "window": window,
                    "window_label": label,
                    "start_date": start,
                    "end_date": end,
                    "n_days": int(len(sub)),
                    "precip_total_mm": float(sub["precip_mm_day"].sum()),
                    "temperature_mean_c": float(sub["temperature_mean_c"].mean()),
                    "incoming_sw_mean_w_m2": float(sub["incoming_sw_mean_w_m2"].mean()),
                    "incoming_sw_energy_mj_m2": float(sub["incoming_sw_energy_mj_m2_day"].sum()),
                    "incoming_lw_mean_w_m2": float(sub["incoming_lw_mean_w_m2"].mean()),
                    "incoming_lw_energy_mj_m2": float(sub["incoming_lw_energy_mj_m2_day"].sum()),
                }
            )
    return pd.DataFrame(rows)


SPECIAL_METRICS: dict[str, dict[str, Any]] = {
    "precip_total_mm": {
        "label": "Accumulated precipitation",
        "unit": "mm",
        "percent": True,
    },
    "temperature_mean_c": {
        "label": "Average daily temperature",
        "unit": "°C",
        "percent": False,
    },
    "incoming_sw_energy_mj_m2": {
        "label": "Incoming shortwave energy",
        "unit": "MJ/m²",
        "percent": True,
    },
    "incoming_lw_energy_mj_m2": {
        "label": "Incoming longwave energy",
        "unit": "MJ/m²",
        "percent": True,
    },
}


def _specialness_summary(window_metrics: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if FOCUS_WY not in settings.years:
        raise ValueError(f"Focus WY{FOCUS_WY} is not present in requested years {settings.years}.")

    rows: list[dict[str, Any]] = []
    for window, label in WINDOWS.items():
        sub = window_metrics[window_metrics["window"].eq(window)].copy()
        hist = sub[sub["water_year"].lt(FOCUS_WY)]
        focus = sub[sub["water_year"].eq(FOCUS_WY)]
        if len(focus) != 1:
            raise ValueError(f"Expected one WY{FOCUS_WY} row for {label}, found {len(focus)}.")

        for column, meta in SPECIAL_METRICS.items():
            all_values = sub[["water_year", column]].dropna().copy()
            focus_value = float(focus.iloc[0][column])
            hist_values = hist[column].dropna().to_numpy(dtype=float)
            hist_mean = float(np.mean(hist_values))
            hist_median = float(np.median(hist_values))
            hist_std = float(np.std(hist_values, ddof=1)) if len(hist_values) >= 2 else np.nan
            anomaly = focus_value - hist_mean
            pct = (anomaly / hist_mean * 100.0) if meta["percent"] and hist_mean != 0 else np.nan
            z = anomaly / hist_std if np.isfinite(hist_std) and hist_std > 0 else np.nan

            ranked = all_values.sort_values([column, "water_year"], ascending=[False, True]).reset_index(drop=True)
            rank = int(ranked.index[ranked["water_year"].eq(FOCUS_WY)][0] + 1)
            n = int(len(ranked))

            rows.append(
                {
                    "window": window,
                    "window_label": label,
                    "metric": column,
                    "metric_label": meta["label"],
                    "unit": meta["unit"],
                    "wy2026_value": focus_value,
                    "historical_years": f"WY{min(settings.years)}-WY{FOCUS_WY - 1}",
                    "historical_mean": hist_mean,
                    "historical_median": hist_median,
                    "historical_std": hist_std,
                    "wy2026_anomaly_vs_hist_mean": anomaly,
                    "wy2026_percent_anomaly_vs_hist_mean": pct,
                    "wy2026_zscore_vs_hist_mean": z,
                    "wy2026_rank_highest_1": rank,
                    "rank_n_years": n,
                }
            )
    return pd.DataFrame(rows)


def _add_rolling_columns(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values(["water_year", "date"])
    for col in ["temperature_mean_c", "incoming_sw_mean_w_m2", "incoming_lw_mean_w_m2", "precip_mm_day"]:
        out[f"{col}_7day"] = out.groupby("water_year", sort=False)[col].transform(
            lambda s: s.rolling(ROLLING_DAYS, center=True, min_periods=4).mean()
        )
    return out


def _daily_climatology_and_anomalies(daily: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    work = _add_rolling_columns(daily)
    hist = work[work["water_year"].lt(FOCUS_WY)].copy()
    focus = work[work["water_year"].eq(FOCUS_WY)].copy()

    value_columns = [
        "precip_accum_oct1_mm",
        "precip_mm_day_7day",
        "temperature_mean_c_7day",
        "incoming_sw_mean_w_m2_7day",
        "incoming_lw_mean_w_m2_7day",
    ]

    records: list[pd.DataFrame] = []
    grouped = hist.groupby("month_day", sort=False)
    for col in value_columns:
        stats = grouped[col].agg(
            hist_mean="mean",
            hist_median="median",
            hist_min="min",
            hist_max="max",
            hist_std="std",
            hist_n="count",
        ).reset_index()
        q = grouped[col].quantile([0.10, 0.90]).unstack(level=-1).reset_index()
        q = q.rename(columns={0.10: "hist_p10", 0.90: "hist_p90"})
        stats = stats.merge(q, on="month_day", how="left")

        f = focus[["date", "water_year", "month_day", col]].copy().rename(columns={col: "wy2026_value"})
        f = f.merge(stats, on="month_day", how="left")
        f["variable"] = col
        f["anomaly_vs_hist_mean"] = f["wy2026_value"] - f["hist_mean"]
        f["zscore_vs_hist_mean"] = np.where(
            np.isfinite(f["hist_std"]) & (f["hist_std"] > 0),
            f["anomaly_vs_hist_mean"] / f["hist_std"],
            np.nan,
        )
        records.append(f)

    result = pd.concat(records, ignore_index=True)
    return result[
        [
            "date",
            "water_year",
            "month_day",
            "variable",
            "wy2026_value",
            "hist_mean",
            "hist_median",
            "hist_min",
            "hist_max",
            "hist_p10",
            "hist_p90",
            "hist_std",
            "hist_n",
            "anomaly_vs_hist_mean",
            "zscore_vs_hist_mean",
        ]
    ].sort_values(["variable", "date"])


def _extreme_anomaly_periods(anomalies: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "precip_accum_oct1_mm": ("Cumulative precipitation", "mm"),
        "temperature_mean_c_7day": ("7-day mean temperature", "°C"),
        "incoming_sw_mean_w_m2_7day": ("7-day mean incoming shortwave", "W/m²"),
        "incoming_lw_mean_w_m2_7day": ("7-day mean incoming longwave", "W/m²"),
    }
    rows: list[dict[str, Any]] = []
    for variable, (label, unit) in specs.items():
        sub = anomalies[anomalies["variable"].eq(variable)].dropna(subset=["anomaly_vs_hist_mean"]).copy()
        if sub.empty:
            continue
        for direction, idx in [
            ("largest_positive_anomaly", sub["anomaly_vs_hist_mean"].idxmax()),
            ("largest_negative_anomaly", sub["anomaly_vs_hist_mean"].idxmin()),
        ]:
            row = sub.loc[idx]
            rows.append(
                {
                    "variable": variable,
                    "variable_label": label,
                    "unit": unit,
                    "extreme_type": direction,
                    "date": row["date"],
                    "wy2026_value": row["wy2026_value"],
                    "historical_mean_same_calendar_day": row["hist_mean"],
                    "anomaly": row["anomaly_vs_hist_mean"],
                    "zscore": row["zscore_vs_hist_mean"],
                }
            )

        zsub = sub.dropna(subset=["zscore_vs_hist_mean"]).copy()
        if len(zsub):
            idx = zsub["zscore_vs_hist_mean"].abs().idxmax()
            row = zsub.loc[idx]
            rows.append(
                {
                    "variable": variable,
                    "variable_label": label,
                    "unit": unit,
                    "extreme_type": "largest_absolute_zscore",
                    "date": row["date"],
                    "wy2026_value": row["wy2026_value"],
                    "historical_mean_same_calendar_day": row["hist_mean"],
                    "anomaly": row["anomaly_vs_hist_mean"],
                    "zscore": row["zscore_vs_hist_mean"],
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Plotting helpers
# =============================================================================
def _save_figure(fig: plt.Figure, cfg: WorkflowConfig, settings: Settings, stem: str) -> Path:
    ext = str(cfg.section("plotting")["figure_format"])
    path = settings.figures_dir / f"{stem}.{ext}"
    fig.savefig(path, dpi=int(cfg.section("plotting")["dpi"]), bbox_inches="tight")
    if bool(cfg.section("plotting").get("close_after_save", True)):
        plt.close(fig)
    return path


def _format_focus_xaxis(ax: plt.Axes) -> None:
    ax.set_xlim(_wy_start(FOCUS_WY), _wy_common_end(FOCUS_WY))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.spines[["top", "right"]].set_visible(False)


def _plot_date_series_for_hist(hist: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = hist[["month_day", value_col]].copy()
    out["plot_date"] = out["month_day"].map(_map_to_focus_wy_date)
    return out.dropna(subset=["plot_date", value_col]).sort_values("plot_date")


def _climatology_stats_for_plot(work: pd.DataFrame, value_col: str) -> pd.DataFrame:
    hist = work[work["water_year"].lt(FOCUS_WY)].copy()
    grouped = hist.groupby("month_day", sort=False)[value_col]
    stats = grouped.agg(hist_mean="mean", hist_min="min", hist_max="max").reset_index()
    stats["plot_date"] = stats["month_day"].map(_map_to_focus_wy_date)
    return stats.dropna(subset=["plot_date"]).sort_values("plot_date")


def _plot_wy2026_vs_history(
    cfg: WorkflowConfig,
    settings: Settings,
    daily: pd.DataFrame,
) -> Path:
    work = _add_rolling_columns(daily)
    focus = work[work["water_year"].eq(FOCUS_WY)].sort_values("date")

    specs = [
        ("precip_accum_oct1_mm", "Accumulated precipitation", "Accumulated precipitation (mm)"),
        ("temperature_mean_c_7day", "Air temperature (7-day running mean)", "Daily mean temperature (°C)"),
        ("incoming_sw_mean_w_m2_7day", "Incoming shortwave (7-day running mean)", "Incoming shortwave (W/m²)"),
        ("incoming_lw_mean_w_m2_7day", "Incoming longwave (7-day running mean)", "Incoming longwave (W/m²)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for ax, (column, panel_title, ylabel) in zip(axes, specs):
        stats = _climatology_stats_for_plot(work, column)
        ax.fill_between(
            stats["plot_date"],
            stats["hist_min"],
            stats["hist_max"],
            color="0.82",
            alpha=0.55,
            linewidth=0,
            label="WY2018-WY2025 range",
        )
        ax.plot(stats["plot_date"], stats["hist_mean"], color="0.40", lw=1.5, ls="--", label="WY2018-WY2025 mean")
        ax.plot(focus["date"], focus[column], color="black", lw=1.8, label="WY2026")
        ax.set_title(panel_title, fontsize=20, fontweight="normal", pad=8)
        ax.set_ylabel(ylabel)
        _format_focus_xaxis(ax)

    handles = [
        Line2D([], [], color="black", lw=1.8, label="WY2026"),
        Line2D([], [], color="0.40", lw=1.5, ls="--", label="WY2018-WY2025 mean"),
        Line2D([], [], color="0.82", lw=8, alpha=0.55, label="WY2018-WY2025 range"),
    ]
    fig.supxlabel("Month", y=0.055)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.98, bottom=0.10, left=0.11, right=0.98, hspace=0.26)
    return _save_figure(fig, cfg, settings, "figure01_wy2026_vs_history_daily_forcing")


def _plot_full_window_metrics(
    cfg: WorkflowConfig,
    settings: Settings,
    window_metrics: pd.DataFrame,
    special: pd.DataFrame,
) -> Path:
    full = window_metrics[window_metrics["window"].eq("oct1_jun30")].sort_values("water_year")
    sp = special[special["window"].eq("oct1_jun30")].set_index("metric")

    specs = [
        ("precip_total_mm", "Oct 1-Jun 30 precipitation", "Precipitation (mm)"),
        ("temperature_mean_c", "Oct 1-Jun 30 mean temperature", "Temperature (°C)"),
        ("incoming_sw_energy_mj_m2", "Oct 1-Jun 30 incoming shortwave", "Cumulative energy (MJ/m²)"),
        ("incoming_lw_energy_mj_m2", "Oct 1-Jun 30 incoming longwave", "Cumulative energy (MJ/m²)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), squeeze=False)
    for ax, (column, panel_title, ylabel) in zip(axes.flat, specs):
        years = full["water_year"].to_numpy(dtype=int)
        values = full[column].to_numpy(dtype=float)
        hist_mean = float(sp.loc[column, "historical_mean"])
        rank = int(sp.loc[column, "wy2026_rank_highest_1"])
        n = int(sp.loc[column, "rank_n_years"])
        anomaly = float(sp.loc[column, "wy2026_anomaly_vs_hist_mean"])

        ax.plot(years, values, color="0.55", lw=1.2, marker="o", ms=5)
        focus_mask = years == FOCUS_WY
        ax.scatter(years[focus_mask], values[focus_mask], color="black", s=70, zorder=4)
        ax.axhline(hist_mean, color="0.35", lw=1.2, ls="--")
        unit = SPECIAL_METRICS[column]["unit"]
        ax.text(
            0.97,
            0.95,
            f"WY2026 rank = {rank}/{n}\nAnomaly = {anomaly:+.2f} {unit}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=max(18, int(cfg.section("plotting")["font_size"]) - 4),
        )
        ax.set_title(panel_title, fontsize=20, fontweight="normal", pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(years)
        ax.set_xlabel("Water year")
        ax.spines[["top", "right"]].set_visible(False)

    fig.legend(
        handles=[
            Line2D([], [], color="0.55", lw=1.2, marker="o", label="WY2018-WY2026"),
            Line2D([], [], color="black", lw=0, marker="o", markersize=8, label="WY2026"),
            Line2D([], [], color="0.35", lw=1.2, ls="--", label="WY2018-WY2025 mean"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(top=0.97, bottom=0.12, left=0.10, right=0.98, hspace=0.30, wspace=0.22)
    return _save_figure(fig, cfg, settings, "figure02_oct1_jun30_metrics_all_years")


def _plot_wy2026_anomalies(
    cfg: WorkflowConfig,
    settings: Settings,
    anomalies: pd.DataFrame,
) -> Path:
    specs = [
        ("precip_accum_oct1_mm", "Cumulative precipitation anomaly", "WY2026 - historical mean (mm)"),
        ("temperature_mean_c_7day", "Temperature anomaly (7-day running mean)", "WY2026 - historical mean (°C)"),
        ("incoming_sw_mean_w_m2_7day", "Incoming shortwave anomaly (7-day running mean)", "WY2026 - historical mean (W/m²)"),
        ("incoming_lw_mean_w_m2_7day", "Incoming longwave anomaly (7-day running mean)", "WY2026 - historical mean (W/m²)"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (variable, panel_title, ylabel) in zip(axes, specs):
        sub = anomalies[anomalies["variable"].eq(variable)].sort_values("date")
        ax.plot(sub["date"], sub["anomaly_vs_hist_mean"], color="black", lw=1.6)
        ax.axhline(0.0, color="0.65", lw=0.9)
        ax.set_title(panel_title, fontsize=20, fontweight="normal", pad=8)
        ax.set_ylabel(ylabel)
        _format_focus_xaxis(ax)

    fig.supxlabel("Month", y=0.055)
    fig.subplots_adjust(top=0.98, bottom=0.09, left=0.12, right=0.98, hspace=0.26)
    return _save_figure(fig, cfg, settings, "figure03_wy2026_daily_anomalies")


# =============================================================================
# Manifest / orchestration
# =============================================================================
def _source_metadata(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    return {
        "path": str(resolved),
        "exists": path.exists(),
    }


def _write_manifest(
    cfg: WorkflowConfig,
    settings: Settings,
    weight_diag: dict[str, Any],
    daily_path: Path,
    window_path: Path,
    special_path: Path,
    anomaly_path: Path,
    extreme_path: Path,
    figure_paths: list[Path],
    daily_source: str,
) -> None:
    manifest = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "east_river_workflow_version": ERW_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": "LDASIN-only East River forcing comparison focused on WY2026",
        "focus_water_year": FOCUS_WY,
        "comparison_years": list(settings.years),
        "comparison_window": "Oct 1-Jun 30 for every water year",
        "secondary_windows": list(WINDOWS.values()),
        "drop_feb29": settings.drop_feb29,
        "timestamp_rule": "YYYYMMDDHH from LDASIN basename; no timezone conversion",
        "expected_hours_per_day": EXPECTED_HOURS_PER_DAY,
        "daily_source": daily_source,
        "ldasin_root": _source_metadata(settings.ldasin_root),
        "main_config": _source_metadata(cfg.source_path),
        "noahmp_geo_file": _source_metadata(Path(cfg.data["paths"]["noahmp_geo_file"]).expanduser()),
        "watershed_shapefile": _source_metadata(Path(cfg.data["paths"]["watershed_shapefile"]).expanduser()),
        "variables": {
            "RAINRATE": "mm/s -> daily total mm",
            "T2D": "K -> daily mean degC",
            "SWDOWN": "W/m2 -> daily mean W/m2 and daily integrated MJ/m2",
            "LWDOWN": "W/m2 -> daily mean W/m2 and daily integrated MJ/m2",
        },
        "spatial_policy": (
            "Exact v0.8.2 Noah-grid East River fractional weights. All positive-weight cells must be finite; "
            "no partial-area renormalization and no valid-area-fraction output."
        ),
        "radiation_integration": "sum(hourly W/m2 * 3600 s * 1e-6) -> MJ/m2/day",
        "historical_baseline_for_wy2026": "WY2018-WY2025",
        "rank_definition": "rank_highest_1: 1 means the highest value among WY2018-WY2026",
        "weight_diagnostics": weight_diag,
        "tables": [str(p) for p in [daily_path, window_path, special_path, anomaly_path, extreme_path]],
        "diagnostics": [
            str(settings.diagnostics_dir / INVENTORY_QC),
            str(settings.diagnostics_dir / MANIFEST_JSON),
        ],
        "figures": [str(p) for p in figure_paths],
    }
    save_json(manifest, settings.diagnostics_dir / MANIFEST_JSON)


def run(cfg: WorkflowConfig, settings: Settings) -> dict[str, Any]:
    settings.create_tree()
    logging.basicConfig(
        level=getattr(logging, str(cfg.section("project").get("log_level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("east_river_ldasin_wy2026")

    daily_path = settings.tables_dir / DAILY_TABLE

    positive_mask, normalized_weights, weight_diag = _load_noah_weights(cfg, settings, logger)

    if daily_path.exists() and not settings.overwrite:
        logger.info("Reusing existing daily LDASIN table: %s", daily_path)
        daily = pd.read_csv(daily_path, parse_dates=["date"])
        _validate_daily_table(daily, settings)
        daily_source = "reused daily cache; hourly archive was not rescanned"
    else:
        inventory, _inventory_qc, _duplicates = _inventory_archive(settings, logger)
        daily = _process_daily(inventory, settings, positive_mask, normalized_weights, logger)
        _validate_daily_table(daily, settings)
        daily.to_csv(daily_path, index=False)
        daily_source = "rebuilt from hourly LDASIN archive"
        logger.info("Wrote %s", daily_path)

    window_metrics = _window_metrics(daily, settings)
    window_path = settings.tables_dir / WINDOW_TABLE
    window_metrics.to_csv(window_path, index=False)

    special = _specialness_summary(window_metrics, settings)
    special_path = settings.tables_dir / SPECIAL_TABLE
    special.to_csv(special_path, index=False)

    anomalies = _daily_climatology_and_anomalies(daily, settings)
    anomaly_path = settings.tables_dir / ANOMALY_TABLE
    anomalies.to_csv(anomaly_path, index=False)

    extremes = _extreme_anomaly_periods(anomalies)
    extreme_path = settings.tables_dir / EXTREME_TABLE
    extremes.to_csv(extreme_path, index=False)

    figure_paths: list[Path] = []
    if settings.make_plots:
        configure_matplotlib(cfg)
        figure_paths.extend(
            [
                _plot_wy2026_vs_history(cfg, settings, daily),
                _plot_full_window_metrics(cfg, settings, window_metrics, special),
                _plot_wy2026_anomalies(cfg, settings, anomalies),
            ]
        )

    _write_manifest(
        cfg,
        settings,
        weight_diag,
        daily_path,
        window_path,
        special_path,
        anomaly_path,
        extreme_path,
        figure_paths,
        daily_source,
    )

    logger.info("LDASIN WY2026 forcing diagnostic complete: %s", settings.output_root)
    return {
        "daily": daily_path,
        "window_metrics": window_path,
        "specialness": special_path,
        "daily_anomalies": anomaly_path,
        "extremes": extreme_path,
        "figures": figure_paths,
        "manifest": settings.diagnostics_dir / MANIFEST_JSON,
    }


def _parse_years(text: str) -> tuple[int, ...]:
    years = tuple(sorted({int(part.strip()) for part in text.split(",") if part.strip()}))
    if not years:
        raise argparse.ArgumentTypeError("At least one water year is required.")
    return years


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="East River Workflow v0.8.2 YAML configuration.")
    parser.add_argument(
        "--ldasin-root",
        required=True,
        help="Root directory containing hourly YYYYMMDDHH.LDASIN_DOMAIN1 files (flat or nested).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional diagnostic output directory. Default: "
            "<main workflow output_dir>/ldasin_wy2026_forcing_v0.1"
        ),
    )
    parser.add_argument(
        "--years",
        type=_parse_years,
        default=tuple(DEFAULT_YEARS),
        help="Comma-separated water years. Default: 2018,2019,...,2026",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the expensive daily LDASIN table.")
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Search only the top level of --ldasin-root rather than nested directories.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Create tables/diagnostics but skip figures.")
    parser.add_argument(
        "--drop-feb29",
        action="store_true",
        help=(
            "Exclude Feb 29 from leap-year windows. Default is to retain all calendar days exactly as requested "
            "for Oct 1-Jun 30; this option makes every WY have the same day count."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    root = Path(args.ldasin_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LDASIN root is not a directory: {root}")

    years = tuple(args.years)
    required = set(DEFAULT_YEARS)
    if set(years) != required:
        print(
            "WARNING: the requested scientific design is WY2018-WY2026. "
            f"You supplied years={years}; outputs/ranks will reflect only those years.",
            file=sys.stderr,
        )
    if FOCUS_WY not in years:
        raise ValueError(f"WY{FOCUS_WY} must be included because this diagnostic is explicitly focused on WY{FOCUS_WY}.")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (cfg.output_dir / DEFAULT_OUTPUT_SUBDIR)
    )
    settings = Settings(
        ldasin_root=root,
        output_root=output_root,
        years=years,
        overwrite=bool(args.overwrite),
        recursive=not bool(args.non_recursive),
        make_plots=not bool(args.no_plots),
        drop_feb29=bool(args.drop_feb29),
    )
    run(cfg, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
