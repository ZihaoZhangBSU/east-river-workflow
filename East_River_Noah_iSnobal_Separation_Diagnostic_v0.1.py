#!/usr/bin/env python3
"""Standalone Noah-MP / iSnobal separation diagnostic for East River v0.8.2.

Purpose
-------
This companion script does NOT modify H1-H4.  It reuses the validated v0.8.2
common-Noah-grid daily spatial caches, adds native Noah-MP FSNO sampled at the
same 23:00 daily state as SNEQV/SNOWH, and builds reusable full-period derived
products for WY2018-WY2026 (or the water years configured in the main YAML).

The expensive/derived products are calculated for each complete available water
year (WY2026 is naturally clipped by project.analysis_end_datetime).  The
mechanism analysis and figures are restricted to a configurable focus window,
defaulting to Feb 1-Apr 30.

Scientific guardrails
---------------------
* iSnobal is compared to Noah-MP on the existing common Noah-MP grid.
* East River basin means use the exact v0.8.2 Noah-grid fractional weights.
* Bulk snow density is calculated cell-by-cell first:
      density = 1000 * SWE_mm / snow_depth_mm
  and is only considered valid where SWE >= 10 mm and snow depth > 0.
* Paired density comparisons use the SAME valid cell support for both models.
* Noah FSNO is read directly from LDASOUT; it is never reconstructed from SWE.
* Noah state variables use the configured daily_state_hour (23:00 in v0.8.2).
* H2-style ablation events retain a cell-day when EITHER model has signed SWE
  loss > 1 mm/day.  Event statistics are unweighted over eligible watershed
  cells, consistent with the v0.8.2 paired cell-day event convention.
* For an ablation interval t-1 -> t, FSNO_start is FSNO at t-1 (the state before
  the loss).  FSNO_end is also saved.
* Whole-basin and paired-snow-covered shortwave diagnostics are both retained so
  exposed bare-ground absorption is not confused with energy over remaining snow.
* This script does not use latent heat as a sublimation proxy and does not add
  atmospheric forcing diagnostics; forcing/preprocessing is intentionally left
  for a separate companion script.

Expected main-workflow inputs
-----------------------------
The v0.8.2 main workflow must already have produced, for each configured WY:
    output_dir/cache/isnobal_wyYYYY_noahgrid_daily.npz
    output_dir/cache/noahmp_wyYYYY_noahgrid_daily.npz
and normally:
    output_dir/cache/east_river_fraction_noahmp.npz

If the Noah fractional-weight cache is absent, this script rebuilds the exact
weights from the configured watershed shapefile and Noah grid.  It does NOT
rebuild missing model daily caches; run the main v0.8.2 preprocessing first.

Outputs
-------
Created below:
    <main output_dir>/separation_diagnostic_v0.1/

cache/
    separation_static_noahgrid.npz
    fsno_wyYYYY_23h.npz
    separation_derived_wyYYYY.npz

tables/
    separation_watershed_daily_full_period.csv
    separation_watershed_daily_focus_window.csv
    separation_focus_annual_summary.csv
    separation_fsno_conditioned_ablation.csv
    separation_fsno_relationship_summary.csv

diagnostics/
    separation_manifest.json
    date_alignment.csv
    density_qc_by_year.csv

figures/
    figure01_density_yearly_comparison.<format>
    figure02_density_fsno_transition_yearly.<format>
    figure03a_fsno_vs_swe_yearly.<format>
    figure03b_fsno_vs_snow_depth_yearly.<format>
    figure03c_fsno_vs_density_yearly.<format>
    figure04_fsno_conditioned_ablation_yearly.<format>
    figure05_divergence_clock_yearly.<format>
    figure06_feedback_chain_wy2021_wy2026.<format>
    figure07_focus_metrics_all_years.<format>

Run
---
From the East_River_Workflow_Code_v0.8.2 repository/environment:

    python noah_isnobal_separation_diagnostic_v0.1.py \
        --config config/east_river_config.yaml

Changing --focus-start/--focus-end only changes analysis tables/figures; the
full-period derived caches remain reusable.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Allow the script to live in the repository root, examples/, or a nearby folder
# without requiring an editable install.  An installed package is preferred.
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
        candidates.extend([
            parent / "src",
            parent / "East_River_Workflow_Code_v0.8.2" / "src",
            parent / "erw_code" / "East_River_Workflow_Code_v0.8.2" / "src",
        ])
    for candidate in candidates:
        if (candidate / "east_river_workflow").is_dir():
            sys.path.insert(0, str(candidate))
            return


_bootstrap_east_river_import()

from east_river_workflow import __version__ as ERW_VERSION  # noqa: E402
from east_river_workflow.config import WorkflowConfig, load_config  # noqa: E402
from east_river_workflow.constants import MODEL_COLORS  # noqa: E402
from east_river_workflow.data_access import decode_noah_times, read_noah_grid  # noqa: E402
from east_river_workflow.grids import (  # noqa: E402
    fractional_polygon_weights,
    load_weights,
    select_watershed,
)
from east_river_workflow.plotting import configure_matplotlib  # noqa: E402
from east_river_workflow.processing import load_spatial_cache, spatial_cache_path  # noqa: E402
from east_river_workflow.utils import (  # noqa: E402
    open_dataset_robust,
    save_json,
    weighted_nanmean,
    water_year_bounds,
)


SCRIPT_VERSION = "0.1.0"
DEFAULT_SUBDIR = "separation_diagnostic_v0.1"
DEFAULT_FOCUS_START = "02-01"
DEFAULT_FOCUS_END = "04-30"
DEFAULT_SNOW_THRESHOLD_MM = 10.0
DEFAULT_EVENT_THRESHOLD_MM_DAY = 1.0
FSNO_FULL_TOLERANCE = 1.0e-6
MAX_SCATTER_POINTS_PER_PANEL = 40000

# Exact user-approved bins.  FSNO==0 is intentionally outside the partial-cover
# categories because it is a no-snow-covered-fraction state rather than partial
# snow cover.  The cache keeps FSNO==0; only the conditioned event bins exclude it.
FSNO_BIN_LABELS = [
    "FSNO = 1",
    "0.9 <= FSNO < 1",
    "0.5 <= FSNO < 0.9",
    "0.1 <= FSNO < 0.5",
    "0 < FSNO < 0.1",
]


@dataclass(frozen=True)
class SeparationSettings:
    output_root: Path
    focus_start_month_day: str
    focus_end_month_day: str
    snow_threshold_mm: float
    event_threshold_mm_day: float
    fsno_variable: str
    overwrite: bool

    @property
    def cache_dir(self) -> Path:
        return self.output_root / "cache"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def diagnostics_dir(self) -> Path:
        return self.output_root / "diagnostics"

    def create_tree(self) -> None:
        for path in [self.cache_dir, self.tables_dir, self.figures_dir, self.diagnostics_dir]:
            path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Generic helpers
# =============================================================================
def _season_date(wy: int, month_day: str) -> pd.Timestamp:
    """Return a date inside a water year from MM-DD."""
    month, day = [int(x) for x in month_day.split("-")]
    year = wy - 1 if month >= 10 else wy
    return pd.Timestamp(year=year, month=month, day=day)


def _focus_bounds(wy: int, settings: SeparationSettings, cfg: WorkflowConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _season_date(wy, settings.focus_start_month_day)
    end = _season_date(wy, settings.focus_end_month_day)
    wy_start, wy_end = water_year_bounds(int(wy))
    start = max(start, wy_start, cfg.analysis_start.normalize())
    end = min(end, wy_end, cfg.analysis_end.normalize())
    return start, end


def _save_figure(fig: plt.Figure, cfg: WorkflowConfig, settings: SeparationSettings, stem: str) -> Path:
    ext = str(cfg.section("plotting")["figure_format"])
    path = settings.figures_dir / f"{stem}.{ext}"
    fig.savefig(path, dpi=int(cfg.section("plotting")["dpi"]), bbox_inches="tight")
    if bool(cfg.section("plotting").get("close_after_save", True)):
        plt.close(fig)
    return path


def _format_focus_axis(ax: plt.Axes, wy: int, settings: SeparationSettings, cfg: WorkflowConfig) -> None:
    start, end = _focus_bounds(wy, settings, cfg)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.spines[["top", "right"]].set_visible(False)


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    m = np.asarray(mask, dtype=bool)
    valid_w = np.isfinite(w) & (w > 0)
    denom = float(np.sum(w[valid_w]))
    if denom <= 0:
        return np.nan
    return float(np.sum(w[valid_w & m]) / denom)


def _weighted_mean_masked(values: np.ndarray, weights: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = np.where(np.asarray(mask, dtype=bool), arr, np.nan)
    return weighted_nanmean(arr, weights)[0]


def _paired_weighted_means(a: np.ndarray, b: np.ndarray, weights: np.ndarray, support: np.ndarray | None = None) -> tuple[float, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if support is not None:
        valid &= np.asarray(support, dtype=bool)
    return (
        _weighted_mean_masked(aa, weights, valid),
        _weighted_mean_masked(bb, weights, valid),
    )


def _bulk_density_kg_m3(swe_mm: np.ndarray, snow_depth_mm: np.ndarray, threshold_mm: float) -> tuple[np.ndarray, np.ndarray]:
    swe = np.asarray(swe_mm, dtype=float)
    depth = np.asarray(snow_depth_mm, dtype=float)
    valid = np.isfinite(swe) & np.isfinite(depth) & (swe >= threshold_mm) & (depth > 0.0)
    density = np.full(swe.shape, np.nan, dtype=float)
    density[valid] = 1000.0 * swe[valid] / depth[valid]
    return density, valid


def _signed_ablation(swe_mm: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Return t-1 minus t SWE; positive is ablation, negative is accumulation."""
    swe = np.asarray(swe_mm, dtype=float)
    out = np.full(swe.shape, np.nan, dtype=float)
    if swe.shape[0] < 2:
        return out
    dt_days = np.asarray((dates[1:] - dates[:-1]).days, dtype=int)
    consecutive = dt_days == 1
    diff = swe[:-1] - swe[1:]
    valid = np.isfinite(swe[:-1]) & np.isfinite(swe[1:])
    valid &= consecutive.reshape((-1,) + (1,) * (swe.ndim - 1))
    out[1:] = np.where(valid, diff, np.nan)
    return out


def _shift_start_state(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if arr.shape[0] < 2:
        return out
    dt_days = np.asarray((dates[1:] - dates[:-1]).days, dtype=int)
    consecutive = dt_days == 1
    shape = (-1,) + (1,) * (arr.ndim - 1)
    valid_time = consecutive.reshape(shape)
    out[1:] = np.where(valid_time, arr[:-1], np.nan)
    return out


def _fsno_bin_codes(fsno: np.ndarray) -> np.ndarray:
    values = np.asarray(fsno, dtype=float)
    code = np.full(values.shape, -1, dtype=np.int8)
    finite = np.isfinite(values)
    full = finite & np.isclose(values, 1.0, rtol=0.0, atol=FSNO_FULL_TOLERANCE)
    code[full] = 0
    code[finite & ~full & (values >= 0.9) & (values < 1.0)] = 1
    code[finite & (values >= 0.5) & (values < 0.9)] = 2
    code[finite & (values >= 0.1) & (values < 0.5)] = 3
    code[finite & (values > 0.0) & (values < 0.1)] = 4
    return code


def _ratio_albedo(incoming_mj_m2: float, absorbed_mj_m2: float, minimum_incoming: float) -> float:
    if not np.isfinite(incoming_mj_m2) or incoming_mj_m2 <= minimum_incoming or not np.isfinite(absorbed_mj_m2):
        return np.nan
    return float(1.0 - absorbed_mj_m2 / incoming_mj_m2)


def _source_fingerprint(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
    }


# =============================================================================
# Noah static grid / weights and native FSNO extraction
# =============================================================================
def _load_noah_grid_and_weights(cfg: WorkflowConfig, settings: SeparationSettings, logger: logging.Logger):
    noah_grid, noah_dem, noah_lat, noah_lon, noah_diag = read_noah_grid(
        cfg.data["paths"]["noahmp_geo_file"], cfg.section("noahmp")
    )
    main_weight_path = cfg.output_dir / "cache" / "east_river_fraction_noahmp.npz"
    if main_weight_path.exists():
        weights = load_weights(main_weight_path, noah_grid)
        weight_source = str(main_weight_path)
    else:
        logger.warning("Main Noah watershed-weight cache is absent; rebuilding exact fractional weights.")
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
        weight_source = "rebuilt by companion script"

    static_path = settings.cache_dir / "separation_static_noahgrid.npz"
    np.savez_compressed(
        static_path,
        watershed_fraction=np.asarray(weights, dtype=np.float32),
        noah_dem_m=np.asarray(noah_dem, dtype=np.float32),
        noah_lat=np.asarray(noah_lat, dtype=np.float32),
        noah_lon=np.asarray(noah_lon, dtype=np.float32),
        x=np.asarray(noah_grid.x, dtype=float),
        y=np.asarray(noah_grid.y, dtype=float),
    )
    return noah_grid, weights, {
        "weight_source": weight_source,
        "static_cache": str(static_path),
        "noah_grid_shape": list(noah_grid.shape),
        "noah_grid_diagnostics": noah_diag,
    }


def _extract_fsno_all_years(
    cfg: WorkflowConfig,
    settings: SeparationSettings,
    years: list[int],
    noah_cache_dates: dict[int, pd.DatetimeIndex],
    noah_grid_shape: tuple[int, int],
    logger: logging.Logger,
) -> dict[int, dict[str, Any]]:
    """Extract/cached FSNO at the configured daily state hour for all years."""
    result: dict[int, dict[str, Any]] = {}
    missing_years: list[int] = []
    for wy in years:
        path = settings.cache_dir / f"fsno_wy{wy}_23h.npz"
        if path.exists() and not settings.overwrite:
            with np.load(path, allow_pickle=False) as z:
                dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
                fsno = np.asarray(z["fsno"], dtype=float)
            result[wy] = {"dates": dates, "fsno": fsno, "path": path, "source": "cache"}
        else:
            missing_years.append(wy)

    if not missing_years:
        return result

    source = Path(cfg.data["paths"]["noahmp_output_file"]).expanduser()
    logger.info("Extracting native Noah-MP %s from %s", settings.fsno_variable, source)
    with open_dataset_robust(source, decode_times=False) as ds:
        if settings.fsno_variable not in ds:
            raise KeyError(
                f"Required Noah-MP FSNO variable {settings.fsno_variable!r} is absent from {source}. "
                "Set --fsno-variable if the production file uses a different name."
            )
        fsno_var = ds[settings.fsno_variable]
        if "Time" not in fsno_var.dims:
            raise ValueError(
                f"{settings.fsno_variable!r} must have a Time dimension in production LDASOUT; dims={fsno_var.dims}."
            )
        times = decode_noah_times(ds, cfg.section("noahmp"))
        frame = pd.DataFrame({"index": np.arange(len(times), dtype=int), "time": times})
        frame["date"] = frame["time"].dt.normalize()
        frame["hour"] = frame["time"].dt.hour
        state_hour = int(cfg.section("noahmp")["daily_state_hour"])
        state = frame[frame["hour"].eq(state_hour)].copy()
        duplicate = state["date"].duplicated(keep=False)
        if duplicate.any():
            dup_dates = state.loc[duplicate, "date"].dt.strftime("%Y-%m-%d").unique().tolist()[:10]
            raise ValueError(f"More than one Noah state record at hour {state_hour} for dates such as {dup_dates}")
        state_index = state.set_index("date")["index"]

        for wy in missing_years:
            target_dates = pd.DatetimeIndex(noah_cache_dates[wy]).normalize()
            missing = target_dates.difference(state_index.index)
            if len(missing):
                raise ValueError(
                    f"No {state_hour}:00 FSNO state record for {len(missing)} Noah cache dates in WY{wy}; "
                    f"first missing={missing[0]}."
                )
            indices = state_index.reindex(target_dates).to_numpy(dtype=int)
            fsno = np.asarray(fsno_var.isel(Time=indices).values, dtype=float)
            if fsno.shape != (len(target_dates), *noah_grid_shape):
                raise ValueError(
                    f"FSNO shape mismatch for WY{wy}: {fsno.shape}, expected {(len(target_dates), *noah_grid_shape)}"
                )
            finite = fsno[np.isfinite(fsno)]
            if finite.size and (float(np.min(finite)) < -1.0e-6 or float(np.max(finite)) > 1.0 + 1.0e-6):
                raise ValueError(
                    f"FSNO outside [0,1] in WY{wy}: min={float(np.min(finite))}, max={float(np.max(finite))}"
                )
            # Only remove tiny floating-point excursions after explicit validation.
            fsno = np.where(np.isfinite(fsno), np.clip(fsno, 0.0, 1.0), np.nan)
            path = settings.cache_dir / f"fsno_wy{wy}_23h.npz"
            np.savez_compressed(
                path,
                dates=np.asarray(target_dates.strftime("%Y-%m-%d"), dtype="U10"),
                fsno=np.asarray(fsno, dtype=np.float32),
            )
            result[wy] = {"dates": target_dates, "fsno": fsno, "path": path, "source": "LDASOUT"}

    return result


# =============================================================================
# Per-water-year derivation and full-period intermediate products
# =============================================================================
def _check_required_cache_variables(cache: dict[str, Any], source: str, wy: int) -> None:
    required = {
        "swe_mm",
        "snow_depth_mm",
        "melt_mm",
        "incoming_sw_energy_mj_m2",
        "absorbed_sw_energy_mj_m2",
        "effective_albedo",
    }
    if source == "iSnobal":
        required.add("swi_mm")
    else:
        required.add("qsnobot_mm")
    missing = sorted(required - set(cache))
    if missing:
        raise KeyError(f"{source} WY{wy} main cache is missing required variables: {missing}")


def _align_main_caches(ic: dict[str, Any], nc: dict[str, Any]) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, dict[str, Any]]:
    idates = pd.DatetimeIndex(ic["dates"]).normalize()
    ndates = pd.DatetimeIndex(nc["dates"]).normalize()
    common = idates.intersection(ndates).sort_values()
    if not len(common):
        raise ValueError("iSnobal and Noah caches contain no common dates.")
    ii = idates.get_indexer(common)
    ni = ndates.get_indexer(common)
    diag = {
        "isnobal_date_count": int(len(idates)),
        "noah_date_count": int(len(ndates)),
        "common_date_count": int(len(common)),
        "isnobal_only_count": int(len(idates.difference(ndates))),
        "noah_only_count": int(len(ndates.difference(idates))),
        "first_common_date": str(common.min().date()),
        "last_common_date": str(common.max().date()),
    }
    return common, ii, ni, diag


def _save_derived_cache(path: Path, dates: pd.DatetimeIndex, arrays: dict[str, np.ndarray]) -> None:
    payload: dict[str, np.ndarray] = {
        "dates": np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"),
    }
    for key, value in arrays.items():
        arr = np.asarray(value)
        if arr.dtype == bool:
            payload[key] = arr.astype(np.uint8)
        elif np.issubdtype(arr.dtype, np.integer):
            payload[key] = arr
        else:
            payload[key] = arr.astype(np.float32)
    np.savez_compressed(path, **payload)


def _load_derived_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out: dict[str, Any] = {key: z[key] for key in z.files}
    out["dates"] = pd.DatetimeIndex(pd.to_datetime(out["dates"]))
    for key in ["density_pair_mask", "snow_mask_i", "snow_mask_n", "paired_snow_mask", "ablation_event_mask"]:
        if key in out:
            out[key] = np.asarray(out[key], dtype=bool)
    return out


def _derive_one_year(
    wy: int,
    cfg: WorkflowConfig,
    settings: SeparationSettings,
    weights: np.ndarray,
    fsno_info: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    ipath = spatial_cache_path(cfg, "isnobal", wy)
    npath = spatial_cache_path(cfg, "noahmp", wy)
    if not ipath.exists() or not npath.exists():
        missing = [str(p) for p in [ipath, npath] if not p.exists()]
        raise FileNotFoundError(
            "Required v0.8.2 spatial cache(s) are missing: " + ", ".join(missing) + ". "
            "Run the main v0.8.2 preprocessing before this companion diagnostic."
        )
    ic = load_spatial_cache(ipath)
    nc = load_spatial_cache(npath)
    _check_required_cache_variables(ic, "iSnobal", wy)
    _check_required_cache_variables(nc, "Noah-MP", wy)
    dates, ii, ni, align_diag = _align_main_caches(ic, nc)

    # Align FSNO to the common cache dates.
    fdates = pd.DatetimeIndex(fsno_info["dates"]).normalize()
    fi = fdates.get_indexer(dates)
    if np.any(fi < 0):
        raise ValueError(f"FSNO cache does not cover all aligned model dates for WY{wy}.")
    fsno = np.asarray(fsno_info["fsno"], dtype=float)[fi]

    arrays: dict[str, np.ndarray] = {}
    arrays["swe_i_mm"] = np.asarray(ic["swe_mm"][ii], dtype=float)
    arrays["swe_n_mm"] = np.asarray(nc["swe_mm"][ni], dtype=float)
    arrays["snow_depth_i_mm"] = np.asarray(ic["snow_depth_mm"][ii], dtype=float)
    arrays["snow_depth_n_mm"] = np.asarray(nc["snow_depth_mm"][ni], dtype=float)
    arrays["melt_i_mm"] = np.asarray(ic["melt_mm"][ii], dtype=float)
    arrays["melt_n_mm"] = np.asarray(nc["melt_mm"][ni], dtype=float)
    arrays["swi_i_mm"] = np.asarray(ic["swi_mm"][ii], dtype=float)
    arrays["qsnobot_n_mm"] = np.asarray(nc["qsnobot_mm"][ni], dtype=float)
    arrays["incoming_sw_i_mj_m2"] = np.asarray(ic["incoming_sw_energy_mj_m2"][ii], dtype=float)
    arrays["incoming_sw_n_mj_m2"] = np.asarray(nc["incoming_sw_energy_mj_m2"][ni], dtype=float)
    arrays["absorbed_sw_i_mj_m2"] = np.asarray(ic["absorbed_sw_energy_mj_m2"][ii], dtype=float)
    arrays["absorbed_sw_n_mj_m2"] = np.asarray(nc["absorbed_sw_energy_mj_m2"][ni], dtype=float)
    arrays["effective_albedo_i"] = np.asarray(ic["effective_albedo"][ii], dtype=float)
    arrays["effective_albedo_n"] = np.asarray(nc["effective_albedo"][ni], dtype=float)
    arrays["fsno_n"] = fsno
    arrays["delta_swe_n_minus_i_mm"] = arrays["swe_n_mm"] - arrays["swe_i_mm"]
    arrays["delta_snow_depth_n_minus_i_mm"] = arrays["snow_depth_n_mm"] - arrays["snow_depth_i_mm"]
    arrays["delta_incoming_sw_n_minus_i_mj_m2"] = arrays["incoming_sw_n_mj_m2"] - arrays["incoming_sw_i_mj_m2"]
    arrays["delta_absorbed_sw_n_minus_i_mj_m2"] = arrays["absorbed_sw_n_mj_m2"] - arrays["absorbed_sw_i_mj_m2"]
    arrays["delta_effective_albedo_n_minus_i"] = arrays["effective_albedo_n"] - arrays["effective_albedo_i"]

    density_i, valid_density_i = _bulk_density_kg_m3(
        arrays["swe_i_mm"], arrays["snow_depth_i_mm"], settings.snow_threshold_mm
    )
    density_n, valid_density_n = _bulk_density_kg_m3(
        arrays["swe_n_mm"], arrays["snow_depth_n_mm"], settings.snow_threshold_mm
    )
    arrays["density_i_kg_m3"] = density_i
    arrays["density_n_kg_m3"] = density_n
    arrays["delta_density_n_minus_i_kg_m3"] = density_n - density_i
    arrays["snow_mask_i"] = np.isfinite(arrays["swe_i_mm"]) & (arrays["swe_i_mm"] >= settings.snow_threshold_mm)
    arrays["snow_mask_n"] = np.isfinite(arrays["swe_n_mm"]) & (arrays["swe_n_mm"] >= settings.snow_threshold_mm)
    arrays["paired_snow_mask"] = arrays["snow_mask_i"] & arrays["snow_mask_n"]
    arrays["density_pair_mask"] = valid_density_i & valid_density_n

    arrays["ablation_i_mm_day"] = _signed_ablation(arrays["swe_i_mm"], dates)
    arrays["ablation_n_mm_day"] = _signed_ablation(arrays["swe_n_mm"], dates)
    arrays["extra_noah_ablation_mm_day"] = arrays["ablation_n_mm_day"] - arrays["ablation_i_mm_day"]
    arrays["fsno_start_n"] = _shift_start_state(arrays["fsno_n"], dates)
    arrays["fsno_end_n"] = arrays["fsno_n"].copy()
    arrays["fsno_bin_start"] = _fsno_bin_codes(arrays["fsno_start_n"])

    watershed_mask = np.asarray(weights, dtype=float) > 0
    event = (
        np.isfinite(arrays["ablation_i_mm_day"])
        & np.isfinite(arrays["ablation_n_mm_day"])
        & (
            (arrays["ablation_i_mm_day"] > settings.event_threshold_mm_day)
            | (arrays["ablation_n_mm_day"] > settings.event_threshold_mm_day)
        )
        & np.broadcast_to(watershed_mask, arrays["ablation_i_mm_day"].shape)
    )
    arrays["ablation_event_mask"] = event

    min_incoming = float(cfg.section("h3")["incoming_sw_minimum_mj_m2_day"])
    rows: list[dict[str, Any]] = []
    for t, date in enumerate(dates):
        pair_density = arrays["density_pair_mask"][t]
        pair_snow = arrays["paired_snow_mask"][t]

        swe_i_b, swe_n_b = _paired_weighted_means(arrays["swe_i_mm"][t], arrays["swe_n_mm"][t], weights)
        depth_i_b, depth_n_b = _paired_weighted_means(
            arrays["snow_depth_i_mm"][t], arrays["snow_depth_n_mm"][t], weights
        )
        density_i_pair, density_n_pair = _paired_weighted_means(
            arrays["density_i_kg_m3"][t], arrays["density_n_kg_m3"][t], weights, pair_density
        )
        density_i_own = _weighted_mean_masked(arrays["density_i_kg_m3"][t], weights, valid_density_i[t])
        density_n_own = _weighted_mean_masked(arrays["density_n_kg_m3"][t], weights, valid_density_n[t])

        fsno_b = _weighted_mean_masked(arrays["fsno_n"][t], weights)
        fsno_on_noah_snow = _weighted_mean_masked(arrays["fsno_n"][t], weights, arrays["snow_mask_n"][t])
        sca_i = _weighted_fraction(arrays["snow_mask_i"][t], weights)
        sca_n = _weighted_fraction(arrays["snow_mask_n"][t], weights)

        ein_i = _weighted_mean_masked(arrays["incoming_sw_i_mj_m2"][t], weights)
        ein_n = _weighted_mean_masked(arrays["incoming_sw_n_mj_m2"][t], weights)
        eabs_i = _weighted_mean_masked(arrays["absorbed_sw_i_mj_m2"][t], weights)
        eabs_n = _weighted_mean_masked(arrays["absorbed_sw_n_mj_m2"][t], weights)
        alb_i = _ratio_albedo(ein_i, eabs_i, min_incoming)
        alb_n = _ratio_albedo(ein_n, eabs_n, min_incoming)

        ein_i_s, ein_n_s = _paired_weighted_means(
            arrays["incoming_sw_i_mj_m2"][t], arrays["incoming_sw_n_mj_m2"][t], weights, pair_snow
        )
        eabs_i_s, eabs_n_s = _paired_weighted_means(
            arrays["absorbed_sw_i_mj_m2"][t], arrays["absorbed_sw_n_mj_m2"][t], weights, pair_snow
        )
        alb_i_s = _ratio_albedo(ein_i_s, eabs_i_s, min_incoming)
        alb_n_s = _ratio_albedo(ein_n_s, eabs_n_s, min_incoming)

        abl_i, abl_n = _paired_weighted_means(
            arrays["ablation_i_mm_day"][t], arrays["ablation_n_mm_day"][t], weights
        )
        melt_i, melt_n = _paired_weighted_means(arrays["melt_i_mm"][t], arrays["melt_n_mm"][t], weights)
        swi_i = _weighted_mean_masked(arrays["swi_i_mm"][t], weights)
        qsnobot_n = _weighted_mean_masked(arrays["qsnobot_n_mm"][t], weights)

        row = {
            "date": date,
            "water_year": int(wy),
            "swe_i_common_noahgrid_mm": swe_i_b,
            "swe_n_common_noahgrid_mm": swe_n_b,
            "delta_swe_n_minus_i_mm": swe_n_b - swe_i_b if np.all(np.isfinite([swe_i_b, swe_n_b])) else np.nan,
            "snow_depth_i_common_noahgrid_mm": depth_i_b,
            "snow_depth_n_common_noahgrid_mm": depth_n_b,
            "delta_snow_depth_n_minus_i_mm": depth_n_b - depth_i_b if np.all(np.isfinite([depth_i_b, depth_n_b])) else np.nan,
            "density_i_pairmask_kg_m3": density_i_pair,
            "density_n_pairmask_kg_m3": density_n_pair,
            "delta_density_n_minus_i_pairmask_kg_m3": (
                density_n_pair - density_i_pair if np.all(np.isfinite([density_i_pair, density_n_pair])) else np.nan
            ),
            "density_i_ownmask_kg_m3": density_i_own,
            "density_n_ownmask_kg_m3": density_n_own,
            "fsno_n_watershed_mean": fsno_b,
            "fsno_n_mean_where_noah_swe_ge_threshold": fsno_on_noah_snow,
            "sca10_i_fraction": sca_i,
            "sca10_n_fraction": sca_n,
            "incoming_sw_i_mj_m2_day": ein_i,
            "incoming_sw_n_mj_m2_day": ein_n,
            "absorbed_sw_i_mj_m2_day": eabs_i,
            "absorbed_sw_n_mj_m2_day": eabs_n,
            "effective_albedo_i": alb_i,
            "effective_albedo_n": alb_n,
            "paired_snow_incoming_sw_i_mj_m2_day": ein_i_s,
            "paired_snow_incoming_sw_n_mj_m2_day": ein_n_s,
            "paired_snow_absorbed_sw_i_mj_m2_day": eabs_i_s,
            "paired_snow_absorbed_sw_n_mj_m2_day": eabs_n_s,
            "paired_snow_effective_albedo_i": alb_i_s,
            "paired_snow_effective_albedo_n": alb_n_s,
            "ablation_i_signed_mm_day": abl_i,
            "ablation_n_signed_mm_day": abl_n,
            "extra_noah_ablation_basin_mm_day": abl_n - abl_i if np.all(np.isfinite([abl_i, abl_n])) else np.nan,
            "melt_i_mm_day": melt_i,
            "melt_n_mm_day": melt_n,
            "swi_i_mm_day": swi_i,
            "qsnobot_n_mm_day": qsnobot_n,
            "paired_density_area_fraction": _weighted_fraction(pair_density, weights),
            "paired_snow_area_fraction": _weighted_fraction(pair_snow, weights),
            "ablation_event_cell_count": int(np.count_nonzero(event[t])),
        }
        rows.append(row)

    daily = pd.DataFrame(rows).sort_values("date")
    daily["delta_swe_change_mm_day"] = daily["delta_swe_n_minus_i_mm"].diff()
    daily["delta_absorbed_sw_n_minus_i_mj_m2_day"] = (
        daily["absorbed_sw_n_mj_m2_day"] - daily["absorbed_sw_i_mj_m2_day"]
    )
    daily["delta_paired_snow_absorbed_sw_n_minus_i_mj_m2_day"] = (
        daily["paired_snow_absorbed_sw_n_mj_m2_day"] - daily["paired_snow_absorbed_sw_i_mj_m2_day"]
    )
    daily["delta_effective_albedo_n_minus_i"] = daily["effective_albedo_n"] - daily["effective_albedo_i"]
    daily["delta_paired_snow_effective_albedo_n_minus_i"] = (
        daily["paired_snow_effective_albedo_n"] - daily["paired_snow_effective_albedo_i"]
    )
    start, end = _focus_bounds(wy, settings, cfg)
    daily["in_focus_window"] = (daily["date"] >= start) & (daily["date"] <= end)

    derived_path = settings.cache_dir / f"separation_derived_wy{wy}.npz"
    _save_derived_cache(derived_path, dates, arrays)

    density_qc = {
        "water_year": int(wy),
        "density_i_valid_cell_days": int(np.count_nonzero(np.isfinite(density_i))),
        "density_n_valid_cell_days": int(np.count_nonzero(np.isfinite(density_n))),
        "paired_density_cell_days": int(np.count_nonzero(arrays["density_pair_mask"])),
        "density_i_gt_1000_cell_days": int(np.count_nonzero(np.isfinite(density_i) & (density_i > 1000.0))),
        "density_n_gt_1000_cell_days": int(np.count_nonzero(np.isfinite(density_n) & (density_n > 1000.0))),
        "density_i_max_kg_m3": float(np.nanmax(density_i)) if np.isfinite(density_i).any() else np.nan,
        "density_n_max_kg_m3": float(np.nanmax(density_n)) if np.isfinite(density_n).any() else np.nan,
        "note": "Values >1000 kg/m3 are reported, not clipped; primary validity mask is SWE threshold + SD>0.",
    }
    align_diag.update({"water_year": int(wy), "derived_cache": str(derived_path)})
    logger.info("Derived WY%s separation cache: %s", wy, derived_path)
    return daily, arrays, align_diag, density_qc


# =============================================================================
# Focus-window summaries
# =============================================================================
def _focus_time_mask(dates: pd.DatetimeIndex, wy: int, settings: SeparationSettings, cfg: WorkflowConfig) -> np.ndarray:
    start, end = _focus_bounds(wy, settings, cfg)
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


def _summarize_fsno_conditioned_ablation(
    years: list[int],
    derived: dict[int, dict[str, Any]],
    settings: SeparationSettings,
    cfg: WorkflowConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for wy in years:
        d = derived[wy]
        dates = pd.DatetimeIndex(d["dates"])
        time_mask = _focus_time_mask(dates, wy, settings, cfg)
        event = np.asarray(d["ablation_event_mask"], dtype=bool) & time_mask[:, None, None]
        bin_code = np.asarray(d["fsno_bin_start"], dtype=np.int8)
        ai = np.asarray(d["ablation_i_mm_day"], dtype=float)
        an = np.asarray(d["ablation_n_mm_day"], dtype=float)
        extra = an - ai
        for code, label in enumerate(FSNO_BIN_LABELS):
            keep = event & (bin_code == code) & np.isfinite(ai) & np.isfinite(an)
            x = ai[keep]
            y = an[keep]
            e = extra[keep]
            if len(e) >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                r = float(np.corrcoef(x, y)[0, 1])
            else:
                r = np.nan
            rows.append({
                "water_year": int(wy),
                "fsno_bin_code": int(code),
                "fsno_bin": label,
                "event_n": int(len(e)),
                "mean_ablation_i_mm_day": float(np.mean(x)) if len(x) else np.nan,
                "mean_ablation_n_mm_day": float(np.mean(y)) if len(y) else np.nan,
                "mean_extra_noah_ablation_mm_day": float(np.mean(e)) if len(e) else np.nan,
                "median_extra_noah_ablation_mm_day": float(np.median(e)) if len(e) else np.nan,
                "q25_extra_noah_ablation_mm_day": float(np.quantile(e, 0.25)) if len(e) else np.nan,
                "q75_extra_noah_ablation_mm_day": float(np.quantile(e, 0.75)) if len(e) else np.nan,
                "fraction_events_extra_noah_positive": float(np.mean(e > 0.0)) if len(e) else np.nan,
                "pearson_r_ablation_i_vs_n": r,
            })
    return pd.DataFrame(rows)


def _relationship_stats(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[valid]
    yy = yy[valid]
    if len(xx) >= 2 and np.nanstd(xx) > 0 and np.nanstd(yy) > 0:
        r = float(np.corrcoef(xx, yy)[0, 1])
    else:
        r = np.nan
    return {
        "n": int(len(xx)),
        "pearson_r": r,
        "x_mean": float(np.mean(xx)) if len(xx) else np.nan,
        "y_mean": float(np.mean(yy)) if len(yy) else np.nan,
    }


def _summarize_fsno_relationships(
    years: list[int],
    derived: dict[int, dict[str, Any]],
    settings: SeparationSettings,
    cfg: WorkflowConfig,
    weights: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spatial = np.asarray(weights, dtype=float) > 0
    for wy in years:
        d = derived[wy]
        dates = pd.DatetimeIndex(d["dates"])
        tm = _focus_time_mask(dates, wy, settings, cfg)
        fsno = np.asarray(d["fsno_n"], dtype=float)[tm]
        spatial3 = np.broadcast_to(spatial, fsno.shape)
        for variable, key in [
            ("SWE", "swe_n_mm"),
            ("Snow depth", "snow_depth_n_mm"),
            ("Bulk density", "density_n_kg_m3"),
        ]:
            x = np.asarray(d[key], dtype=float)[tm]
            # Exclude the trivial snow-free state (FSNO==0); the relationship
            # diagnostic is intended to examine Noah snow-present states.
            valid = spatial3 & np.isfinite(fsno) & (fsno > 0.0) & np.isfinite(x)
            stats = _relationship_stats(x[valid], fsno[valid])
            rows.append({"water_year": int(wy), "variable": variable, **stats})
    return pd.DataFrame(rows)


def _first_below_date(frame: pd.DataFrame, column: str, threshold: float) -> pd.Timestamp:
    sub = frame[np.isfinite(frame[column]) & (frame[column] < threshold)]
    return pd.Timestamp(sub.iloc[0]["date"]) if len(sub) else pd.NaT


def _focus_annual_summary(
    years: list[int],
    daily: pd.DataFrame,
    event_summary: pd.DataFrame,
    settings: SeparationSettings,
    cfg: WorkflowConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    min_incoming = float(cfg.section("h3")["incoming_sw_minimum_mj_m2_day"])
    for wy in years:
        sub = daily[(daily["water_year"] == wy) & daily["in_focus_window"]].sort_values("date").copy()
        if sub.empty:
            continue
        ein_i = float(sub["incoming_sw_i_mj_m2_day"].sum(min_count=1))
        ein_n = float(sub["incoming_sw_n_mj_m2_day"].sum(min_count=1))
        eabs_i = float(sub["absorbed_sw_i_mj_m2_day"].sum(min_count=1))
        eabs_n = float(sub["absorbed_sw_n_mj_m2_day"].sum(min_count=1))
        ein_i_s = float(sub["paired_snow_incoming_sw_i_mj_m2_day"].sum(min_count=1))
        ein_n_s = float(sub["paired_snow_incoming_sw_n_mj_m2_day"].sum(min_count=1))
        eabs_i_s = float(sub["paired_snow_absorbed_sw_i_mj_m2_day"].sum(min_count=1))
        eabs_n_s = float(sub["paired_snow_absorbed_sw_n_mj_m2_day"].sum(min_count=1))
        ev = event_summary[event_summary["water_year"] == wy]
        total_n = int(ev["event_n"].sum()) if len(ev) else 0
        weighted_extra = (
            float(np.average(ev.loc[ev["event_n"] > 0, "mean_extra_noah_ablation_mm_day"], weights=ev.loc[ev["event_n"] > 0, "event_n"]))
            if len(ev[ev["event_n"] > 0]) else np.nan
        )
        fsno90 = _first_below_date(sub, "fsno_n_watershed_mean", 0.90)
        fsno75 = _first_below_date(sub, "fsno_n_watershed_mean", 0.75)
        fsno50 = _first_below_date(sub, "fsno_n_watershed_mean", 0.50)
        wy_start, _ = water_year_bounds(wy)
        rows.append({
            "water_year": int(wy),
            "focus_start": str(sub["date"].min().date()),
            "focus_end": str(sub["date"].max().date()),
            "mean_delta_swe_n_minus_i_mm": float(sub["delta_swe_n_minus_i_mm"].mean()),
            "minimum_delta_swe_n_minus_i_mm": float(sub["delta_swe_n_minus_i_mm"].min()),
            "mean_delta_snow_depth_n_minus_i_mm": float(sub["delta_snow_depth_n_minus_i_mm"].mean()),
            "mean_delta_density_pairmask_kg_m3": float(sub["delta_density_n_minus_i_pairmask_kg_m3"].mean()),
            "mean_fsno_n": float(sub["fsno_n_watershed_mean"].mean()),
            "first_fsno_mean_below_0p90": fsno90,
            "first_fsno_mean_below_0p75": fsno75,
            "first_fsno_mean_below_0p50": fsno50,
            "first_fsno_mean_below_0p90_dowy": int((fsno90 - wy_start).days + 1) if pd.notna(fsno90) else np.nan,
            "first_fsno_mean_below_0p75_dowy": int((fsno75 - wy_start).days + 1) if pd.notna(fsno75) else np.nan,
            "first_fsno_mean_below_0p50_dowy": int((fsno50 - wy_start).days + 1) if pd.notna(fsno50) else np.nan,
            "focus_effective_albedo_i": _ratio_albedo(ein_i, eabs_i, min_incoming),
            "focus_effective_albedo_n": _ratio_albedo(ein_n, eabs_n, min_incoming),
            "focus_delta_effective_albedo_n_minus_i": (
                _ratio_albedo(ein_n, eabs_n, min_incoming) - _ratio_albedo(ein_i, eabs_i, min_incoming)
            ),
            "focus_cumulative_absorbed_sw_i_mj_m2": eabs_i,
            "focus_cumulative_absorbed_sw_n_mj_m2": eabs_n,
            "focus_delta_absorbed_sw_n_minus_i_mj_m2": eabs_n - eabs_i,
            "focus_paired_snow_effective_albedo_i": _ratio_albedo(ein_i_s, eabs_i_s, min_incoming),
            "focus_paired_snow_effective_albedo_n": _ratio_albedo(ein_n_s, eabs_n_s, min_incoming),
            "focus_paired_snow_delta_absorbed_sw_n_minus_i_mj_m2": eabs_n_s - eabs_i_s,
            "focus_cumulative_melt_i_mm": float(sub["melt_i_mm_day"].sum(min_count=1)),
            "focus_cumulative_melt_n_mm": float(sub["melt_n_mm_day"].sum(min_count=1)),
            "focus_cumulative_swi_i_mm": float(sub["swi_i_mm_day"].sum(min_count=1)),
            "focus_cumulative_qsnobot_n_mm": float(sub["qsnobot_n_mm_day"].sum(min_count=1)),
            "fsno_conditioned_event_n": total_n,
            "mean_extra_noah_ablation_across_conditioned_events_mm_day": weighted_extra,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================
def _subplot_grid(years: list[int], *, sharex: bool = False, sharey: bool = True):
    ncols = 3
    nrows = int(math.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 5.5 * nrows), sharex=sharex, sharey=sharey, squeeze=False)
    for ax in axes.flat[len(years):]:
        ax.set_visible(False)
    return fig, axes


def _apply_template_spacing(fig: plt.Figure, *, legend_x: float = 0.56) -> None:
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.07, hspace=0.30, wspace=0.16)
    # legend_x is passed to callers for consistency with the preferred project template.
    _ = legend_x


def _plot_density_yearly(cfg, settings, daily, years) -> Path:
    fig, axes = _subplot_grid(years, sharey=True)
    focus = daily[daily.in_focus_window]
    values = pd.concat([
        focus["density_i_pairmask_kg_m3"],
        focus["density_n_pairmask_kg_m3"],
    ]).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    ymax = max(100.0, float(np.ceil(np.nanpercentile(finite, 99.0) / 50.0) * 50.0)) if len(finite) else 500.0
    for ax, wy in zip(axes.flat, years):
        sub = daily[(daily.water_year == wy) & daily.in_focus_window].sort_values("date")
        ax.plot(sub.date, sub.density_i_pairmask_kg_m3, color=MODEL_COLORS["iSnobal"], lw=1.5)
        ax.plot(sub.date, sub.density_n_pairmask_kg_m3, color=MODEL_COLORS["Noah-MP"], lw=1.5)
        ax.set_title(f"WY {wy}", fontsize=20, fontweight="normal", pad=8)
        ax.set_ylim(0, ymax)
        _format_focus_axis(ax, wy, settings, cfg)
    fig.supylabel("Paired-support bulk snow density (kg m$^{-3}$)", x=0.025)
    fig.supxlabel("Month", y=0.095)
    fig.legend(
        handles=[
            Line2D([], [], color=MODEL_COLORS["iSnobal"], lw=1.5, label="iSnobal"),
            Line2D([], [], color=MODEL_COLORS["Noah-MP"], lw=1.5, label="Noah-MP"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.005),
    )
    _apply_template_spacing(fig)
    return _save_figure(fig, cfg, settings, "figure01_density_yearly_comparison")


def _plot_density_fsno_transition(cfg, settings, daily, years) -> Path:
    fig, axes = _subplot_grid(years, sharey=True)
    left_values = daily.loc[daily.in_focus_window, "delta_density_n_minus_i_pairmask_kg_m3"].to_numpy(dtype=float)
    finite = np.abs(left_values[np.isfinite(left_values)])
    lim = max(25.0, float(np.ceil(np.nanpercentile(finite, 99.0) / 25.0) * 25.0)) if len(finite) else 100.0
    for idx, (ax, wy) in enumerate(zip(axes.flat, years)):
        sub = daily[(daily.water_year == wy) & daily.in_focus_window].sort_values("date")
        ax.plot(sub.date, sub.delta_density_n_minus_i_pairmask_kg_m3, color="black", lw=1.5)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"WY {wy}", fontsize=20, fontweight="normal", pad=8)
        _format_focus_axis(ax, wy, settings, cfg)
        ax2 = ax.twinx()
        ax2.plot(sub.date, sub.fsno_n_watershed_mean, color=MODEL_COLORS["Noah-MP"], lw=1.5, ls="--")
        ax2.set_ylim(0, 1.05)
        if idx % 3 != 2:
            ax2.tick_params(labelright=False)
        ax2.spines["top"].set_visible(False)
    fig.supylabel("Noah-MP - iSnobal density (kg m$^{-3}$)", x=0.025)
    fig.supxlabel("Month", y=0.095)
    fig.text(0.995, 0.52, "Noah-MP FSNO", rotation=90, va="center", ha="right", fontsize=cfg.section("plotting")["font_size"])
    fig.legend(
        handles=[
            Line2D([], [], color="black", lw=1.5, label="Density difference"),
            Line2D([], [], color=MODEL_COLORS["Noah-MP"], lw=1.5, ls="--", label="Noah-MP FSNO"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.005),
    )
    _apply_template_spacing(fig)
    return _save_figure(fig, cfg, settings, "figure02_density_fsno_transition_yearly")


def _sample_xy(x: np.ndarray, y: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    xx = np.asarray(x, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[valid]
    yy = yy[valid]
    n_full = len(xx)
    if n_full <= max_points:
        return xx, yy, n_full
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_full, size=max_points, replace=False)
    return xx[idx], yy[idx], n_full


def _plot_fsno_relationship(
    cfg,
    settings,
    years,
    derived,
    weights,
    relation_summary,
    key: str,
    xlabel: str,
    stem: str,
) -> Path:
    fig, axes = _subplot_grid(years, sharey=True)
    spatial = np.asarray(weights, dtype=float) > 0
    all_x: list[np.ndarray] = []
    per_year: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for wy in years:
        d = derived[wy]
        dates = pd.DatetimeIndex(d["dates"])
        tm = _focus_time_mask(dates, wy, settings, cfg)
        x = np.asarray(d[key], dtype=float)[tm]
        y = np.asarray(d["fsno_n"], dtype=float)[tm]
        mask = np.broadcast_to(spatial, x.shape) & np.isfinite(y) & (y > 0.0)
        x = np.where(mask, x, np.nan)
        y = np.where(mask, y, np.nan)
        xs, ys, n_full = _sample_xy(x, y, MAX_SCATTER_POINTS_PER_PANEL, seed=wy)
        per_year[wy] = (xs, ys, n_full)
        all_x.append(xs)
    finite_x = np.concatenate([x[np.isfinite(x)] for x in all_x if len(x)]) if any(len(x) for x in all_x) else np.array([])
    xmax = float(np.nanpercentile(finite_x, 99.5)) if len(finite_x) else 1.0
    if "density" in key:
        xmax = max(100.0, math.ceil(xmax / 50.0) * 50.0)
    else:
        xmax = max(10.0, math.ceil(xmax / 50.0) * 50.0)

    variable_name = {"swe_n_mm": "SWE", "snow_depth_n_mm": "Snow depth", "density_n_kg_m3": "Bulk density"}[key]
    for ax, wy in zip(axes.flat, years):
        x, y, n_full = per_year[wy]
        ax.scatter(x, y, s=8, alpha=0.12, color=MODEL_COLORS["Noah-MP"], linewidths=0)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.05)
        stat = relation_summary[(relation_summary.water_year == wy) & (relation_summary.variable == variable_name)]
        r = float(stat.pearson_r.iloc[0]) if len(stat) else np.nan
        rtxt = "NA" if not np.isfinite(r) else f"{r:.2f}"
        ax.text(0.97, 0.95, f"r = {rtxt}\nn = {n_full:,}", transform=ax.transAxes, ha="right", va="top", fontsize=max(18, int(cfg.section("plotting")["font_size"]) - 4))
        ax.set_title(f"WY {wy}", fontsize=20, fontweight="normal", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supylabel("Noah-MP FSNO", x=0.025)
    fig.supxlabel(xlabel, y=0.095)
    _apply_template_spacing(fig)
    return _save_figure(fig, cfg, settings, stem)


def _plot_fsno_conditioned_ablation(cfg, settings, years, event_summary) -> Path:
    fig, axes = _subplot_grid(years, sharey=True)
    finite = event_summary["mean_extra_noah_ablation_mm_day"].to_numpy(dtype=float)
    finite = np.abs(finite[np.isfinite(finite)])
    lim = max(1.0, float(np.ceil(np.nanpercentile(finite, 99.0)))) if len(finite) else 5.0
    x = np.arange(len(FSNO_BIN_LABELS))
    short_labels = ["=1", ".9-1", ".5-.9", ".1-.5", "0-.1"]
    for ax, wy in zip(axes.flat, years):
        sub = event_summary[event_summary.water_year == wy].set_index("fsno_bin_code").reindex(range(5))
        y = sub["mean_extra_noah_ablation_mm_day"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", lw=1.4, color="black")
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_ylim(-lim, lim)
        ax.set_xticks(x, short_labels)
        ax.set_title(f"WY {wy}", fontsize=20, fontweight="normal", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        counts = sub["event_n"].fillna(0).astype(int).to_numpy()
        ax.text(0.97, 0.95, "n: " + ", ".join(str(v) for v in counts), transform=ax.transAxes, ha="right", va="top", fontsize=max(18, int(cfg.section("plotting")["font_size"]) - 5))
    fig.supylabel("Extra Noah-MP ablation (mm day$^{-1}$)", x=0.025)
    fig.supxlabel("FSNO at start of daily ablation interval", y=0.095)
    _apply_template_spacing(fig)
    return _save_figure(fig, cfg, settings, "figure04_fsno_conditioned_ablation_yearly")


def _plot_divergence_clock(cfg, settings, daily, years) -> Path:
    fig, axes = _subplot_grid(years, sharey=True)
    delta_vals = daily.loc[daily.in_focus_window, "delta_swe_n_minus_i_mm"].to_numpy(dtype=float)
    finite = np.abs(delta_vals[np.isfinite(delta_vals)])
    lim = max(20.0, float(np.ceil(np.nanpercentile(finite, 99.0) / 20.0) * 20.0)) if len(finite) else 100.0
    for idx, (ax, wy) in enumerate(zip(axes.flat, years)):
        sub = daily[(daily.water_year == wy) & daily.in_focus_window].sort_values("date")
        ax.plot(sub.date, sub.delta_swe_n_minus_i_mm, color="black", lw=1.6)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"WY {wy}", fontsize=20, fontweight="normal", pad=8)
        _format_focus_axis(ax, wy, settings, cfg)
        ax2 = ax.twinx()
        ax2.plot(sub.date, sub.delta_swe_change_mm_day, color="0.5", lw=1.0, ls="--")
        dfinite = sub.delta_swe_change_mm_day.to_numpy(dtype=float)
        dfinite = np.abs(dfinite[np.isfinite(dfinite)])
        dlim = max(5.0, float(np.ceil(np.nanpercentile(dfinite, 99.0) / 5.0) * 5.0)) if len(dfinite) else 10.0
        ax2.set_ylim(-dlim, dlim)
        ax2.spines["top"].set_visible(False)
        if idx % 3 != 2:
            ax2.tick_params(labelright=False)
    fig.supylabel("Noah-MP - iSnobal SWE (mm)", x=0.025)
    fig.supxlabel("Month", y=0.095)
    fig.text(0.995, 0.52, "Daily change in SWE difference (mm day$^{-1}$)", rotation=90, va="center", ha="right", fontsize=cfg.section("plotting")["font_size"])
    fig.legend(
        handles=[
            Line2D([], [], color="black", lw=1.6, label="SWE difference"),
            Line2D([], [], color="0.5", lw=1.0, ls="--", label="Daily change in SWE difference"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.005),
    )
    _apply_template_spacing(fig)
    return _save_figure(fig, cfg, settings, "figure05_divergence_clock_yearly")


def _plot_feedback_chain_2021_2026(cfg, settings, daily) -> Path | None:
    years = [wy for wy in [2021, 2026] if wy in set(daily.water_year.unique())]
    if len(years) < 2:
        return None
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex="col", squeeze=False)
    for c, wy in enumerate(years):
        sub = daily[(daily.water_year == wy) & daily.in_focus_window].sort_values("date").copy()
        # Row 1: modeled subgrid FSNO and threshold-based SCA remain distinct.
        axes[0, c].plot(sub.date, sub.fsno_n_watershed_mean, color=MODEL_COLORS["Noah-MP"], lw=1.5, label="Noah-MP FSNO")
        axes[0, c].plot(sub.date, sub.sca10_n_fraction, color=MODEL_COLORS["Noah-MP"], lw=1.3, ls="--", label="Noah-MP SCA >10 mm")
        axes[0, c].plot(sub.date, sub.sca10_i_fraction, color=MODEL_COLORS["iSnobal"], lw=1.3, ls="--", label="iSnobal SCA >10 mm")
        axes[0, c].set_ylim(0, 1.05)

        # Row 2: whole-basin effective albedo from ratio of integrated energies.
        axes[1, c].plot(sub.date, sub.effective_albedo_i, color=MODEL_COLORS["iSnobal"], lw=1.5)
        axes[1, c].plot(sub.date, sub.effective_albedo_n, color=MODEL_COLORS["Noah-MP"], lw=1.5)
        axes[1, c].set_ylim(0, 1.0)

        # Row 3: cumulative N-I absorbed SW, both whole basin and paired-snow support.
        whole_cum = sub.delta_absorbed_sw_n_minus_i_mj_m2_day.cumsum()
        snow_cum = sub.delta_paired_snow_absorbed_sw_n_minus_i_mj_m2_day.fillna(0.0).cumsum()
        axes[2, c].plot(sub.date, whole_cum, color="black", lw=1.5, label="Whole watershed")
        axes[2, c].plot(sub.date, snow_cum, color="0.45", lw=1.3, ls="--", label="Paired snow-covered support")
        axes[2, c].axhline(0.0, color="0.7", lw=0.8)

        # Row 4: divergence clock.
        axes[3, c].plot(sub.date, sub.delta_swe_n_minus_i_mm, color="black", lw=1.6)
        axr = axes[3, c].twinx()
        axr.plot(sub.date, sub.delta_swe_change_mm_day, color="0.5", lw=1.0, ls="--")
        if c == len(years) - 1:
            axr.set_ylabel("Daily Δ(SWE diff.)\n(mm day$^{-1}$)")
        else:
            axr.tick_params(labelright=False)
        axr.spines["top"].set_visible(False)

        axes[0, c].set_title(f"WY {wy}", fontsize=20, fontweight="normal", pad=8)
        for r in range(4):
            _format_focus_axis(axes[r, c], wy, settings, cfg)

    axes[0, 0].set_ylabel("Fraction")
    axes[1, 0].set_ylabel("Effective albedo")
    axes[2, 0].set_ylabel("Cumulative N-I absorbed SW\n(MJ m$^{-2}$)")
    axes[3, 0].set_ylabel("N-I SWE (mm)")
    fig.supxlabel("Month", y=0.075)
    fig.legend(
        handles=[
            Line2D([], [], color=MODEL_COLORS["Noah-MP"], lw=1.5, label="Noah-MP FSNO"),
            Line2D([], [], color=MODEL_COLORS["Noah-MP"], lw=1.3, ls="--", label="Noah-MP SCA >10 mm"),
            Line2D([], [], color=MODEL_COLORS["iSnobal"], lw=1.3, ls="--", label="iSnobal SCA >10 mm"),
            Line2D([], [], color=MODEL_COLORS["iSnobal"], lw=1.5, label="iSnobal albedo"),
            Line2D([], [], color=MODEL_COLORS["Noah-MP"], lw=1.5, label="Noah-MP albedo"),
            Line2D([], [], color="black", lw=1.5, label="Whole-watershed Δ absorbed SW / SWE difference"),
            Line2D([], [], color="0.45", lw=1.3, ls="--", label="Paired-snow Δ absorbed SW / daily SWE-difference change"),
        ],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005), fontsize=max(14, int(cfg.section("plotting")["font_size"]) - 2),
    )
    fig.subplots_adjust(top=0.96, bottom=0.13, left=0.10, right=0.91, hspace=0.18, wspace=0.20)
    return _save_figure(fig, cfg, settings, "figure06_feedback_chain_wy2021_wy2026")


def _plot_focus_metrics_all_years(cfg, settings, annual) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), squeeze=False)
    wy = annual.water_year.to_numpy(dtype=int)

    axes[0, 0].plot(wy, annual.mean_delta_density_pairmask_kg_m3, marker="o", color="black")
    axes[0, 0].axhline(0, color="0.7", lw=0.8)
    axes[0, 0].set_ylabel("Mean density difference\n(kg m$^{-3}$)")

    axes[0, 1].plot(wy, annual.mean_fsno_n, marker="o", color=MODEL_COLORS["Noah-MP"])
    axes[0, 1].set_ylabel("Mean Noah-MP FSNO")
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].plot(wy, annual.focus_delta_absorbed_sw_n_minus_i_mj_m2, marker="o", color="black", label="Whole watershed")
    axes[1, 0].plot(wy, annual.focus_paired_snow_delta_absorbed_sw_n_minus_i_mj_m2, marker="s", ls="--", color="0.45", label="Paired snow-covered")
    axes[1, 0].axhline(0, color="0.7", lw=0.8)
    axes[1, 0].set_ylabel("Cumulative N-I absorbed SW\n(MJ m$^{-2}$)")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(wy, annual.mean_extra_noah_ablation_across_conditioned_events_mm_day, marker="o", color="black")
    axes[1, 1].axhline(0, color="0.7", lw=0.8)
    axes[1, 1].set_ylabel("FSNO-conditioned extra Noah-MP\nablation (mm day$^{-1}$)")

    for ax in axes.flat:
        ax.set_xticks(wy)
        ax.set_xlabel("Water year")
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=0.97, bottom=0.10, left=0.11, right=0.98, hspace=0.28, wspace=0.22)
    return _save_figure(fig, cfg, settings, "figure07_focus_metrics_all_years")


def _generate_plots(
    cfg: WorkflowConfig,
    settings: SeparationSettings,
    daily: pd.DataFrame,
    years: list[int],
    derived: dict[int, dict[str, Any]],
    weights: np.ndarray,
    event_summary: pd.DataFrame,
    relation_summary: pd.DataFrame,
    annual: pd.DataFrame,
) -> list[Path]:
    configure_matplotlib(cfg)
    paths: list[Path] = []
    paths.append(_plot_density_yearly(cfg, settings, daily, years))
    paths.append(_plot_density_fsno_transition(cfg, settings, daily, years))
    paths.append(_plot_fsno_relationship(
        cfg, settings, years, derived, weights, relation_summary,
        "swe_n_mm", "Noah-MP SWE (mm)", "figure03a_fsno_vs_swe_yearly",
    ))
    paths.append(_plot_fsno_relationship(
        cfg, settings, years, derived, weights, relation_summary,
        "snow_depth_n_mm", "Noah-MP snow depth (mm)", "figure03b_fsno_vs_snow_depth_yearly",
    ))
    paths.append(_plot_fsno_relationship(
        cfg, settings, years, derived, weights, relation_summary,
        "density_n_kg_m3", "Noah-MP bulk snow density (kg m$^{-3}$)", "figure03c_fsno_vs_density_yearly",
    ))
    paths.append(_plot_fsno_conditioned_ablation(cfg, settings, years, event_summary))
    paths.append(_plot_divergence_clock(cfg, settings, daily, years))
    pair_path = _plot_feedback_chain_2021_2026(cfg, settings, daily)
    if pair_path is not None:
        paths.append(pair_path)
    paths.append(_plot_focus_metrics_all_years(cfg, settings, annual))
    return paths


# =============================================================================
# Main workflow
# =============================================================================
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-period East River Noah-MP/iSnobal separation intermediates + focused Feb-Apr mechanism analysis."
    )
    parser.add_argument("--config", required=True, help="Path to the v0.8.2 east_river_config.yaml")
    parser.add_argument("--output-subdir", default=DEFAULT_SUBDIR, help=f"Subdirectory below main output_dir (default: {DEFAULT_SUBDIR})")
    parser.add_argument("--focus-start", default=DEFAULT_FOCUS_START, help="Focus-window start MM-DD (default: 02-01)")
    parser.add_argument("--focus-end", default=DEFAULT_FOCUS_END, help="Focus-window end MM-DD (default: 04-30)")
    parser.add_argument("--snow-threshold-mm", type=float, default=DEFAULT_SNOW_THRESHOLD_MM, help="Minimum SWE for density/snow-covered support (default: 10 mm)")
    parser.add_argument("--event-threshold-mm-day", type=float, default=DEFAULT_EVENT_THRESHOLD_MM_DAY, help="H2-style ablation event threshold (default: 1 mm/day)")
    parser.add_argument("--fsno-variable", default="FSNO", help="Native Noah-MP snow-cover-fraction variable in LDASOUT (default: FSNO)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract FSNO/rewrite derived companion caches even if present")
    parser.add_argument("--no-plots", action="store_true", help="Build/save intermediate products and summaries without figures")
    return parser.parse_args(argv)


def _validate_month_day(text: str) -> None:
    try:
        month, day = [int(x) for x in text.split("-")]
        pd.Timestamp(year=2000, month=month, day=day)
    except Exception as exc:
        raise ValueError(f"Invalid MM-DD value: {text!r}") from exc


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    _validate_month_day(args.focus_start)
    _validate_month_day(args.focus_end)
    if args.snow_threshold_mm <= 0:
        raise ValueError("--snow-threshold-mm must be >0")
    if args.event_threshold_mm_day < 0:
        raise ValueError("--event-threshold-mm-day must be >=0")

    settings = SeparationSettings(
        output_root=cfg.output_dir / args.output_subdir,
        focus_start_month_day=args.focus_start,
        focus_end_month_day=args.focus_end,
        snow_threshold_mm=float(args.snow_threshold_mm),
        event_threshold_mm_day=float(args.event_threshold_mm_day),
        fsno_variable=str(args.fsno_variable),
        overwrite=bool(args.overwrite),
    )
    settings.create_tree()

    logger = logging.getLogger("east_river_separation")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    years = [int(y) for y in cfg.section("project")["water_years"]]

    noah_grid, weights, static_diag = _load_noah_grid_and_weights(cfg, settings, logger)

    # Load main Noah cache dates first so FSNO extraction exactly follows the
    # state dates already used by the validated v0.8.2 processing.
    noah_cache_dates: dict[int, pd.DatetimeIndex] = {}
    for wy in years:
        npath = spatial_cache_path(cfg, "noahmp", wy)
        ipath = spatial_cache_path(cfg, "isnobal", wy)
        if not npath.exists() or not ipath.exists():
            missing = [str(p) for p in [ipath, npath] if not p.exists()]
            raise FileNotFoundError(
                f"WY{wy}: missing main v0.8.2 cache(s): {missing}. Run the main preprocessing first."
            )
        nc = load_spatial_cache(npath)
        noah_cache_dates[wy] = pd.DatetimeIndex(nc["dates"]).normalize()

    fsno_by_year = _extract_fsno_all_years(
        cfg, settings, years, noah_cache_dates, noah_grid.shape, logger
    )

    full_daily_parts: list[pd.DataFrame] = []
    align_rows: list[dict[str, Any]] = []
    density_qc_rows: list[dict[str, Any]] = []
    for wy in years:
        daily, _arrays, align, density_qc = _derive_one_year(
            wy, cfg, settings, weights, fsno_by_year[wy], logger
        )
        full_daily_parts.append(daily)
        align_rows.append(align)
        density_qc_rows.append(density_qc)

    full_daily = pd.concat(full_daily_parts, ignore_index=True).sort_values(["water_year", "date"])
    full_daily_path = settings.tables_dir / "separation_watershed_daily_full_period.csv"
    full_daily.to_csv(full_daily_path, index=False)

    focus_daily = full_daily[full_daily["in_focus_window"]].copy()
    focus_daily_path = settings.tables_dir / "separation_watershed_daily_focus_window.csv"
    focus_daily.to_csv(focus_daily_path, index=False)

    # Reopen the saved derived caches as the source for all subsequent
    # diagnostics, proving plots/summaries consume saved intermediates.
    derived = {
        wy: _load_derived_cache(settings.cache_dir / f"separation_derived_wy{wy}.npz")
        for wy in years
    }

    event_summary = _summarize_fsno_conditioned_ablation(years, derived, settings, cfg)
    event_summary_path = settings.tables_dir / "separation_fsno_conditioned_ablation.csv"
    event_summary.to_csv(event_summary_path, index=False)

    relationship_summary = _summarize_fsno_relationships(years, derived, settings, cfg, weights)
    relationship_summary_path = settings.tables_dir / "separation_fsno_relationship_summary.csv"
    relationship_summary.to_csv(relationship_summary_path, index=False)

    annual = _focus_annual_summary(years, full_daily, event_summary, settings, cfg)
    annual_path = settings.tables_dir / "separation_focus_annual_summary.csv"
    annual.to_csv(annual_path, index=False)

    alignment_path = settings.diagnostics_dir / "date_alignment.csv"
    pd.DataFrame(align_rows).to_csv(alignment_path, index=False)
    density_qc_path = settings.diagnostics_dir / "density_qc_by_year.csv"
    pd.DataFrame(density_qc_rows).to_csv(density_qc_path, index=False)

    figure_paths: list[Path] = []
    if not args.no_plots:
        figure_paths = _generate_plots(
            cfg, settings, full_daily, years, derived, weights,
            event_summary, relationship_summary, annual,
        )

    manifest = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "east_river_workflow_version_imported": ERW_VERSION,
        "main_config": str(Path(args.config).expanduser().resolve()),
        "main_output_dir": str(cfg.output_dir),
        "companion_output_root": str(settings.output_root),
        "water_years": years,
        "project_analysis_start": str(cfg.analysis_start),
        "project_analysis_end": str(cfg.analysis_end),
        "focus_window_month_day": [settings.focus_start_month_day, settings.focus_end_month_day],
        "density_swe_threshold_mm": settings.snow_threshold_mm,
        "ablation_event_threshold_mm_day": settings.event_threshold_mm_day,
        "fsno_variable": settings.fsno_variable,
        "fsno_daily_state_hour": int(cfg.section("noahmp")["daily_state_hour"]),
        "fsno_bins": FSNO_BIN_LABELS,
        "ablation_interval_fsno_definition": "FSNO_start = native Noah-MP FSNO at t-1 for SWE loss t-1 -> t",
        "event_weighting": "unweighted eligible watershed cell-days, consistent with v0.8.2 H1/H2 event convention",
        "basin_mean_weighting": "exact Noah-grid fractional East River overlap weights",
        "density_definition": "cellwise 1000*SWE_mm/snow_depth_mm; valid only SWE>=threshold and snow_depth>0; paired comparisons use common valid support",
        "paired_snow_energy_definition": "same cells where both iSnobal and Noah-MP SWE >= threshold on common Noah grid",
        "static_grid": static_diag,
        "source_files": {
            "noah_ldasout": _source_fingerprint(cfg.data["paths"]["noahmp_output_file"]),
            "noah_geo": _source_fingerprint(cfg.data["paths"]["noahmp_geo_file"]),
            "watershed_shapefile": _source_fingerprint(cfg.data["paths"]["watershed_shapefile"]),
        },
        "tables": [
            str(full_daily_path), str(focus_daily_path), str(annual_path),
            str(event_summary_path), str(relationship_summary_path),
        ],
        "diagnostics": [str(alignment_path), str(density_qc_path)],
        "figures": [str(p) for p in figure_paths],
        "notes": [
            "Full-period intermediates are calculated/saved for every configured WY; focus-window restriction is applied only to mechanism summaries/plots.",
            "WY2026 is clipped by the main project analysis end date (2026-06-30 in v0.8.2).",
            "FSNO is native Noah-MP output, not an observation and not reconstructed from SWE.",
            "FSNO and SCA>10 mm are retained as distinct quantities.",
            "FSNO-vs-state relationship summaries/plots exclude the trivial FSNO==0 snow-free state; full FSNO remains saved in the caches.",
            "Whole-watershed and paired-snow-covered absorbed-shortwave diagnostics are both saved to diagnose bare-ground confounding.",
            "Wind/atmospheric-forcing preprocessing diagnostics are intentionally excluded from this script and belong in a second companion analysis.",
        ],
    }
    manifest_path = settings.diagnostics_dir / "separation_manifest.json"
    save_json(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = run(args)
    print(json.dumps({
        "status": "complete",
        "companion_output_root": manifest["companion_output_root"],
        "manifest": manifest["manifest_path"],
        "figure_count": len(manifest["figures"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
