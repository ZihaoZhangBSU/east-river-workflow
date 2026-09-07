#!/usr/bin/env python
"""WY2021 versus WY2026 East River watershed diagnostic workbench.

This script is intentionally standalone.  It does not change the v0.8.2 H1-H4
metric definitions or the main ``east-river run-all`` workflow.  It consumes
existing v0.8.2 watershed tables and common-Noah-grid daily caches, and only
reopens the production Noah-MP LDASOUT when the native ``FSNO`` variable is
available.

Scientific questions
--------------------
1. What is different in WY2026 relative to WY2021?
2. When does the difference emerge?
3. Is the anomaly primarily accumulation or ablation?
4. Which forcing/mechanism is consistent with the difference?
5. Does low SWE lead to early loss of snow-covered area, lower albedo,
   greater absorbed shortwave, and accelerated melt?

The comparison is restricted to the common period Oct 1 through Jun 30 for
both water years.  No single algorithmic "divergence date" is declared; daily
and 7-day-smoothed anomalies are written for visual/diagnostic interpretation.

Primary outputs
---------------
<workflow output>/diagnostics/wy2021_wy2026_watershed/
    tables/daily_diagnostics.csv
    tables/daily_anomalies_2026_minus_2021.csv
    tables/summary_metrics_by_model_year.csv
    tables/summary_metric_anomalies.csv
    tables/noahmp_fsno_daily.csv                 [only when FSNO exists]
    figures/figure01_seasonal_state.png
    figures/figure02_interannual_anomaly.png
    figures/figure03_accumulation_input.png
    figures/figure04_accumulation_ablation_events.png
    figures/figure05_watershed_snow_coverage.png
    figures/figure06_noahmp_fsno.png             [only when FSNO exists]
    figures/figure07_feedback_chain.png
    figures/figure08_energy_anomalies.png
    figures/figure09_ablation_sensitivity.png
    figures/figure10_optional_streamflow.png      [only with --include-usgs]
    analysis_manifest.json

Example
-------
From the project root after ``pip install -e .``::

    python examples/wy2021_wy2026_watershed_diagnostic.py \
        --config config/east_river_config.yaml

Use ``--include-usgs`` only when the downstream streamflow consequence is
needed.  USGS is not used to diagnose the cause of the SWE anomaly.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from east_river_workflow.config import WorkflowConfig, load_config
from east_river_workflow.constants import MODEL_COLORS
from east_river_workflow.data_access import decode_noah_times
from east_river_workflow.plotting import configure_matplotlib
from east_river_workflow.processing import load_spatial_cache, spatial_cache_path
from east_river_workflow.utils import open_dataset_robust, weighted_nanmean


YEARS = (2021, 2026)
REFERENCE_START = pd.Timestamp("2000-10-01")
REFERENCE_END = pd.Timestamp("2001-06-30")
DEFAULT_SCA_THRESHOLDS_MM = (1.0, 10.0, 25.0)
DEFAULT_EVENT_THRESHOLD_MM = 1.0
DEFAULT_FRACTIONAL_LOSS_MIN_SWE_MM = 10.0
DEFAULT_SNOW_ENERGY_THRESHOLD_MM = 10.0
FSNO_FULL_TOLERANCE = 1.0e-6

YEAR_LINESTYLE = {2021: "-", 2026: "--"}
YEAR_MARKER = {2021: "o", 2026: "^"}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/east_river_config.yaml"),
        help="Path to the v0.8.2 YAML configuration.",
    )
    parser.add_argument(
        "--analysis-subdir",
        default="wy2021_wy2026_watershed",
        help="Subdirectory created under diagnostics/ for this standalone analysis.",
    )
    parser.add_argument(
        "--event-threshold-mm",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD_MM,
        help="Nontrivial daily |dSWE| threshold used for event summaries (default: 1 mm/day).",
    )
    parser.add_argument(
        "--fractional-loss-min-swe-mm",
        type=float,
        default=DEFAULT_FRACTIONAL_LOSS_MIN_SWE_MM,
        help="Minimum previous-day SWE used for fractional ablation calculations (default: 10 mm).",
    )
    parser.add_argument(
        "--snow-energy-threshold-mm",
        type=float,
        default=DEFAULT_SNOW_ENERGY_THRESHOLD_MM,
        help="SWE threshold defining snow-covered cells for snow-only energy diagnostics (default: 10 mm).",
    )
    parser.add_argument(
        "--fsno-variable",
        default="FSNO",
        help="Native Noah-MP ground snow-cover-fraction variable name (default: FSNO).",
    )
    parser.add_argument(
        "--include-usgs",
        action="store_true",
        help="Add an optional downstream streamflow figure when streamflow_daily_products.csv exists.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Keep figures open and show them interactively after saving.",
    )
    return parser.parse_args()


def analysis_bounds(wy: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(wy - 1, 10, 1), pd.Timestamp(wy, 6, 30)


def season_day(date: pd.Timestamp, wy: int) -> int:
    start, _ = analysis_bounds(wy)
    return int((pd.Timestamp(date).normalize() - start).days)


def reference_date_from_season_day(day: int) -> pd.Timestamp:
    return REFERENCE_START + pd.Timedelta(days=int(day))


def add_analysis_axis(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["water_year"] = pd.to_numeric(out["water_year"], errors="raise").astype(int)
    out["season_day"] = [season_day(d, wy) for d, wy in zip(out["date"], out["water_year"])]
    out["plot_date"] = [reference_date_from_season_day(v) for v in out["season_day"]]
    return out


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else str(pd.Timestamp(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2, sort_keys=True)


def load_table(path: Path, years: tuple[int, int]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required processed table not found: {path}\n"
            "Run the v0.8.2 workflow first so the watershed daily tables and caches exist."
        )
    frame = pd.read_csv(path)
    require_columns(frame, ["date", "water_year"], str(path))
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["water_year"] = pd.to_numeric(frame["water_year"], errors="raise").astype(int)
    frame = frame[frame["water_year"].isin(years)].copy()
    pieces = []
    for wy in years:
        start, end = analysis_bounds(wy)
        sub = frame[(frame["water_year"].eq(wy)) & frame["date"].between(start, end)].copy()
        pieces.append(sub)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return add_analysis_axis(result.sort_values(["water_year", "date"]))


def load_noah_weights(cfg: WorkflowConfig) -> np.ndarray:
    path = cfg.output_dir / "cache" / "east_river_fraction_noahmp.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Noah-MP fractional watershed-weight cache not found: {path}\n"
            "Run prepare-spatial or run-all before this diagnostic."
        )
    with np.load(path, allow_pickle=False) as archive:
        if "weights" not in archive:
            raise KeyError(f"Weight cache {path} does not contain 'weights'.")
        weights = np.asarray(archive["weights"], dtype=float)
    if weights.ndim != 2 or not np.any(np.isfinite(weights) & (weights > 0)):
        raise ValueError(f"Invalid Noah-MP watershed weights in {path}; shape={weights.shape}")
    return weights


def weighted_fraction(mask: np.ndarray, valid: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return weighted fraction among valid cells plus valid watershed-area fraction.

    Missing cells are not treated as snow-free.  The denominator is the finite
    subset of the positive watershed weights, consistent with the workflow's
    finite-value renormalization convention.  ``valid_area_fraction`` is also
    returned so incomplete spatial coverage remains auditable.
    """
    w = np.asarray(weights, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(w) & (w > 0)
    total = float(np.nansum(w[np.isfinite(w) & (w > 0)]))
    valid_weight = float(np.nansum(w[valid]))
    if valid_weight <= 0:
        return np.nan, 0.0
    numerator = float(np.nansum(w[valid & np.asarray(mask, dtype=bool)]))
    return numerator / valid_weight, valid_weight / total if total > 0 else 0.0


def weighted_energy_albedo(
    incoming: np.ndarray,
    absorbed: np.ndarray,
    weights: np.ndarray,
    selection: np.ndarray,
    minimum_incoming: float,
) -> tuple[float, float, float, float]:
    """Weighted incoming, absorbed, ratio-derived albedo, valid area fraction."""
    incoming = np.asarray(incoming, dtype=float)
    absorbed = np.asarray(absorbed, dtype=float)
    selection = np.asarray(selection, dtype=bool)
    valid = selection & np.isfinite(incoming) & np.isfinite(absorbed)
    masked_weights = np.where(valid, weights, 0.0)
    in_mean, in_valid = weighted_nanmean(incoming, masked_weights)
    abs_mean, abs_valid = weighted_nanmean(absorbed, masked_weights)
    albedo = np.nan
    if np.isfinite(in_mean) and in_mean > minimum_incoming and np.isfinite(abs_mean):
        albedo = 1.0 - abs_mean / in_mean
    return in_mean, abs_mean, albedo, min(in_valid, abs_valid)


# -----------------------------------------------------------------------------
# Spatial SWE / snow-covered-area diagnostics
# -----------------------------------------------------------------------------


def spatial_diagnostics_for_cache(
    cfg: WorkflowConfig,
    source: str,
    wy: int,
    weights: np.ndarray,
    sca_thresholds_mm: tuple[float, ...],
    snow_energy_threshold_mm: float,
) -> pd.DataFrame:
    path = spatial_cache_path(cfg, source, wy)
    if not path.exists():
        raise FileNotFoundError(
            f"Required common-Noah-grid cache not found: {path}\n"
            "Run the v0.8.2 workflow first."
        )
    cache = load_spatial_cache(path)
    required = ["swe_mm", "incoming_sw_energy_mj_m2", "absorbed_sw_energy_mj_m2"]
    missing = [key for key in required if key not in cache]
    if missing:
        raise KeyError(f"Spatial cache {path} is missing required arrays: {missing}")

    dates = pd.DatetimeIndex(cache["dates"]).normalize()
    start, end = analysis_bounds(wy)
    keep = np.flatnonzero((dates >= start) & (dates <= end))
    min_incoming = float(cfg.section("h3")["incoming_sw_minimum_mj_m2_day"])

    rows: list[dict[str, Any]] = []
    for idx in keep:
        swe = np.asarray(cache["swe_mm"][idx], dtype=float)
        incoming = np.asarray(cache["incoming_sw_energy_mj_m2"][idx], dtype=float)
        absorbed = np.asarray(cache["absorbed_sw_energy_mj_m2"][idx], dtype=float)
        if swe.shape != weights.shape:
            raise ValueError(
                f"{source} WY{wy} SWE cache shape {swe.shape} does not match Noah watershed weights {weights.shape}."
            )
        valid_swe = np.isfinite(swe)
        _, valid_area_fraction = weighted_fraction(np.ones_like(valid_swe, dtype=bool), valid_swe, weights)
        row: dict[str, Any] = {
            "date": dates[idx],
            "water_year": wy,
            "source": "iSnobal" if source == "isnobal" else "Noah-MP",
            "swe_spatial_valid_area_fraction": valid_area_fraction,
        }
        for threshold in sca_thresholds_mm:
            label = f"sca_gt_{int(threshold) if float(threshold).is_integer() else threshold:g}mm"
            fraction, _ = weighted_fraction(swe > threshold, valid_swe, weights)
            row[label] = fraction

        # Mutually exclusive SWE-state area fractions.  These are area fractions,
        # not unweighted pixel counts.
        classes = {
            "snowfree_le_1mm": swe <= 1.0,
            "marginal_1_10mm": (swe > 1.0) & (swe <= 10.0),
            "shallow_10_25mm": (swe > 10.0) & (swe <= 25.0),
            "substantial_gt_25mm": swe > 25.0,
        }
        for name, mask in classes.items():
            fraction, _ = weighted_fraction(mask, valid_swe, weights)
            row[f"area_fraction_{name}"] = fraction
        row["area_fraction_marginal_1_25mm"] = (
            row["area_fraction_marginal_1_10mm"] + row["area_fraction_shallow_10_25mm"]
            if np.isfinite(row["area_fraction_marginal_1_10mm"])
            and np.isfinite(row["area_fraction_shallow_10_25mm"])
            else np.nan
        )

        snow_selection = valid_swe & (swe > snow_energy_threshold_mm)
        in_mean, abs_mean, albedo, energy_valid = weighted_energy_albedo(
            incoming, absorbed, weights, snow_selection, min_incoming
        )
        snow_area, _ = weighted_fraction(snow_selection, valid_swe, weights)
        row.update(
            {
                "snowcovered_energy_swe_threshold_mm": snow_energy_threshold_mm,
                "snowcovered_area_fraction": snow_area,
                "snowcovered_incoming_sw_energy_mj_m2": in_mean,
                "snowcovered_absorbed_sw_energy_mj_m2": abs_mean,
                "snowcovered_effective_albedo": albedo,
                "snowcovered_energy_valid_area_fraction": energy_valid,
            }
        )
        rows.append(row)
    return add_analysis_axis(pd.DataFrame(rows))


# -----------------------------------------------------------------------------
# Optional Noah-MP native FSNO extraction
# -----------------------------------------------------------------------------


def extract_noah_fsno(
    cfg: WorkflowConfig,
    weights: np.ndarray,
    years: tuple[int, int],
    variable_name: str,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    source = cfg.paths["noahmp_output_file"]
    metadata: dict[str, Any] = {
        "requested_variable": variable_name,
        "source": source,
        "available": False,
        "reason": None,
    }
    if not source.exists():
        metadata["reason"] = "Configured Noah-MP LDASOUT file does not exist on this machine."
        warnings.warn(str(metadata["reason"]), stacklevel=2)
        return None, metadata

    section = cfg.section("noahmp")
    state_hour = int(section["daily_state_hour"])
    with open_dataset_robust(source, decode_times=False) as ds:
        if variable_name not in ds:
            metadata["reason"] = f"{variable_name!r} is not present in the production Noah-MP LDASOUT."
            warnings.warn(str(metadata["reason"]), stacklevel=2)
            return None, metadata
        var = ds[variable_name]
        if "Time" not in var.dims:
            metadata["reason"] = f"{variable_name!r} does not contain the Time dimension; dims={var.dims}."
            warnings.warn(str(metadata["reason"]), stacklevel=2)
            return None, metadata

        times = decode_noah_times(ds, section)
        time_frame = pd.DataFrame({"index": np.arange(len(times), dtype=int), "time": times})
        time_frame["date"] = time_frame["time"].dt.normalize()
        time_frame["hour"] = time_frame["time"].dt.hour
        selected_frames = []
        for wy in years:
            start, end = analysis_bounds(wy)
            chosen = time_frame[
                time_frame["date"].between(start, end) & time_frame["hour"].eq(state_hour)
            ].copy()
            selected_frames.append(chosen)
        chosen = pd.concat(selected_frames, ignore_index=True).sort_values("time")
        if chosen.empty:
            metadata["reason"] = "No FSNO state-hour records were found in the requested Oct-Jun periods."
            warnings.warn(str(metadata["reason"]), stacklevel=2)
            return None, metadata

        # Confirm each calendar day has exactly one state-hour record.
        counts = chosen.groupby("date").size()
        if not counts.eq(1).all():
            bad = counts[counts.ne(1)].head().to_dict()
            raise ValueError(f"FSNO extraction found non-unique state-hour records on dates: {bad}")

        indices = chosen["index"].to_numpy(dtype=int)
        arrays = np.asarray(var.isel(Time=indices).values, dtype=float)
        if arrays.ndim == 2:
            arrays = arrays[None, ...]
        if arrays.shape[1:] != weights.shape:
            raise ValueError(
                f"{variable_name} grid shape {arrays.shape[1:]} does not match Noah watershed weights {weights.shape}."
            )

        rows: list[dict[str, Any]] = []
        out_of_range_count = 0
        for date, arr in zip(chosen["date"], arrays):
            finite = np.isfinite(arr)
            bad_range = finite & ((arr < -FSNO_FULL_TOLERANCE) | (arr > 1.0 + FSNO_FULL_TOLERANCE))
            out_of_range_count += int(np.count_nonzero(bad_range))
            if np.any(bad_range):
                arr = arr.copy()
                arr[bad_range] = np.nan
            # Only numerical roundoff at the boundaries is clipped; physically
            # out-of-range values above were masked and reported.
            arr = np.where(np.isfinite(arr), np.clip(arr, 0.0, 1.0), np.nan)
            valid = np.isfinite(arr)
            mean_fsno, valid_area = weighted_nanmean(arr, weights)
            classes = {
                "snowfree_0": arr <= FSNO_FULL_TOLERANCE,
                # Keep zero-SCF cells separate: the paper's strongest relative
                # ablation bias was specifically associated with very low but
                # nonzero SCF (~0.01-0.1), not snow-free ground.
                "very_low_gt0_0p1": (arr > FSNO_FULL_TOLERANCE) & (arr < 0.1),
                "partial_0p1_0p9": (arr >= 0.1) & (arr < 0.9),
                "nearly_full_0p9_lt1": (arr >= 0.9) & (arr < 1.0 - FSNO_FULL_TOLERANCE),
                "full_1": arr >= 1.0 - FSNO_FULL_TOLERANCE,
                "any_partial_gt0_lt1": (arr > FSNO_FULL_TOLERANCE) & (arr < 1.0 - FSNO_FULL_TOLERANCE),
            }
            row = {
                "date": pd.Timestamp(date),
                "water_year": int(pd.Timestamp(date).year + (pd.Timestamp(date).month >= 10)),
                "fsno_mean": mean_fsno,
                "fsno_valid_area_fraction": valid_area,
            }
            for name, mask in classes.items():
                frac, _ = weighted_fraction(mask, valid, weights)
                row[f"fsno_area_fraction_{name}"] = frac
            rows.append(row)

        metadata.update(
            {
                "available": True,
                "reason": None,
                "units": str(var.attrs.get("units", "")),
                "description": str(var.attrs.get("description", var.attrs.get("long_name", ""))),
                "state_hour": state_hour,
                "record_count": len(rows),
                "masked_out_of_range_cell_records": out_of_range_count,
            }
        )
        if out_of_range_count:
            warnings.warn(
                f"Masked {out_of_range_count} {variable_name} cell-records outside [0, 1] beyond numerical tolerance.",
                stacklevel=2,
            )
        return add_analysis_axis(pd.DataFrame(rows)), metadata


# -----------------------------------------------------------------------------
# Daily derived diagnostics and summary metrics
# -----------------------------------------------------------------------------


def enrich_daily(frame: pd.DataFrame, event_threshold_mm: float, fractional_loss_min_swe_mm: float) -> pd.DataFrame:
    out = frame.sort_values(["source", "water_year", "date"]).copy()
    groups = []
    for (source, wy), sub in out.groupby(["source", "water_year"], sort=False):
        sub = sub.sort_values("date").copy()
        swe = pd.to_numeric(sub["swe_mm"], errors="coerce")
        sub["previous_swe_mm"] = swe.shift(1)
        sub["dSWE_mm_day"] = swe.diff()
        sub["swe_gain_mm_day"] = sub["dSWE_mm_day"].clip(lower=0)
        sub["swe_loss_mm_day"] = (-sub["dSWE_mm_day"]).clip(lower=0)
        sub["is_accumulation_event"] = sub["dSWE_mm_day"] > event_threshold_mm
        sub["is_ablation_event"] = sub["dSWE_mm_day"] < -event_threshold_mm
        valid_fractional = sub["is_ablation_event"] & (sub["previous_swe_mm"] >= fractional_loss_min_swe_mm)
        sub["fractional_swe_loss"] = np.nan
        sub.loc[valid_fractional, "fractional_swe_loss"] = (
            sub.loc[valid_fractional, "swe_loss_mm_day"] / sub.loc[valid_fractional, "previous_swe_mm"]
        )
        for col in ["precip_mm", "snowfall_mm", "melt_mm", "swe_gain_mm_day", "swe_loss_mm_day"]:
            if col in sub:
                sub[f"cumulative_{col}"] = pd.to_numeric(sub[col], errors="coerce").fillna(0).cumsum()
        if "precip_mm" in sub and "snowfall_mm" in sub:
            denominator = sub["cumulative_precip_mm"]
            sub["cumulative_snowfall_fraction"] = np.where(
                denominator > 0, sub["cumulative_snowfall_mm"] / denominator, np.nan
            )
            initial_swe = swe.iloc[0] if len(swe) and np.isfinite(swe.iloc[0]) else 0.0
            sub["retention_fraction"] = np.where(
                sub["cumulative_snowfall_mm"] > 1.0,
                (swe - initial_swe) / sub["cumulative_snowfall_mm"],
                np.nan,
            )
        peak = swe.max(skipna=True)
        sub["swe_normalized_by_own_peak"] = swe / peak if np.isfinite(peak) and peak > 0 else np.nan
        sub["dSWE_7day_mean_mm_day"] = sub["dSWE_mm_day"].rolling(7, center=True, min_periods=1).mean()
        groups.append(sub)
    return pd.concat(groups, ignore_index=True).sort_values(["source", "water_year", "date"])


def build_daily_products(
    isnobal: pd.DataFrame,
    noah: pd.DataFrame,
    spatial: pd.DataFrame,
    fsno: pd.DataFrame | None,
    event_threshold_mm: float,
    fractional_loss_min_swe_mm: float,
) -> pd.DataFrame:
    required = [
        "date", "water_year", "swe_mm", "snow_depth_mm", "precip_mm", "snowfall_mm", "melt_mm",
        "incoming_sw_energy_mj_m2", "absorbed_sw_energy_mj_m2", "effective_albedo",
    ]
    require_columns(isnobal, required, "iSnobal watershed table")
    require_columns(noah, required, "Noah-MP watershed table")
    i = isnobal.copy(); i["source"] = "iSnobal"
    n = noah.copy(); n["source"] = "Noah-MP"
    daily = pd.concat([i, n], ignore_index=True, sort=False)

    spatial_merge = spatial.drop(columns=["plot_date"], errors="ignore")
    daily = daily.merge(spatial_merge, on=["date", "water_year", "source", "season_day"], how="left")
    if fsno is not None:
        fsno_merge = fsno.drop(columns=["plot_date"], errors="ignore")
        nmask = daily["source"].eq("Noah-MP")
        noah_rows = daily.loc[nmask].merge(
            fsno_merge.drop(columns=["water_year"], errors="ignore"),
            on=["date", "season_day"],
            how="left",
            suffixes=("", "_fsno"),
        )
        daily = pd.concat([daily.loc[~nmask], noah_rows], ignore_index=True, sort=False)
    daily = add_analysis_axis(daily)
    return enrich_daily(daily, event_threshold_mm, fractional_loss_min_swe_mm)


def paired_year_anomalies(daily: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "swe_mm", "snow_depth_mm", "precip_mm", "snowfall_mm", "melt_mm",
        "incoming_sw_energy_mj_m2", "absorbed_sw_energy_mj_m2", "effective_albedo",
        "sca_gt_1mm", "sca_gt_10mm", "sca_gt_25mm", "area_fraction_marginal_1_25mm",
        "snowcovered_incoming_sw_energy_mj_m2", "snowcovered_absorbed_sw_energy_mj_m2",
        "snowcovered_effective_albedo", "fsno_mean",
    ]
    variables = [v for v in variables if v in daily.columns]
    pieces = []
    for source in ["iSnobal", "Noah-MP"]:
        sub = daily[daily["source"].eq(source)].copy()
        left = sub[sub["water_year"].eq(2021)][["season_day"] + variables].copy()
        right = sub[sub["water_year"].eq(2026)][["season_day"] + variables].copy()
        merged = left.merge(right, on="season_day", how="inner", suffixes=("_2021", "_2026"))
        out = pd.DataFrame({"source": source, "season_day": merged["season_day"]})
        out["plot_date"] = out["season_day"].map(reference_date_from_season_day)
        for variable in variables:
            diff = merged[f"{variable}_2026"] - merged[f"{variable}_2021"]
            out[f"{variable}_2021"] = merged[f"{variable}_2021"]
            out[f"{variable}_2026"] = merged[f"{variable}_2026"]
            out[f"delta_{variable}_2026_minus_2021"] = diff
            out[f"delta_{variable}_7day"] = diff.rolling(7, center=True, min_periods=1).mean()
        pieces.append(out)
    anomalies = pd.concat(pieces, ignore_index=True)

    # Difference-of-differences: how much more/less anomalous Noah-MP is than iSnobal.
    i = anomalies[anomalies["source"].eq("iSnobal")].set_index("season_day")
    n = anomalies[anomalies["source"].eq("Noah-MP")].set_index("season_day")
    common = i.index.intersection(n.index)
    double_rows = pd.DataFrame({"source": "Noah-MP anomaly minus iSnobal anomaly", "season_day": common})
    double_rows["plot_date"] = [reference_date_from_season_day(v) for v in common]
    for variable in variables:
        col = f"delta_{variable}_2026_minus_2021"
        if col in i.columns and col in n.columns:
            dd = n.loc[common, col].to_numpy(float) - i.loc[common, col].to_numpy(float)
            double_rows[f"double_difference_{variable}"] = dd
            double_rows[f"double_difference_{variable}_7day"] = pd.Series(dd).rolling(7, center=True, min_periods=1).mean().to_numpy()
    return pd.concat([anomalies, double_rows], ignore_index=True, sort=False)


def value_on_month_day(sub: pd.DataFrame, month: int, day: int, column: str) -> float:
    row = sub[(sub["date"].dt.month.eq(month)) & (sub["date"].dt.day.eq(day))]
    if row.empty:
        return np.nan
    value = pd.to_numeric(row[column], errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def first_post_peak_below(sub: pd.DataFrame, column: str, threshold: float, peak_date: pd.Timestamp) -> pd.Timestamp:
    if column not in sub.columns or pd.isna(peak_date):
        return pd.NaT
    post = sub[sub["date"] >= peak_date].sort_values("date")
    values = pd.to_numeric(post[column], errors="coerce")
    hit = post[np.isfinite(values) & (values <= threshold)]
    return pd.NaT if hit.empty else pd.Timestamp(hit["date"].iloc[0])


def days_to_peak_fraction(sub: pd.DataFrame, fraction: float) -> float:
    s = sub[["date", "swe_mm"]].dropna().sort_values("date")
    if s.empty or s["swe_mm"].max() <= 0:
        return np.nan
    idx = s["swe_mm"].idxmax()
    peak_date = pd.Timestamp(s.loc[idx, "date"])
    peak = float(s.loc[idx, "swe_mm"])
    post = s[s["date"] >= peak_date]
    hit = post[post["swe_mm"] <= peak * fraction]
    if hit.empty:
        return np.nan
    return float((pd.Timestamp(hit["date"].iloc[0]) - peak_date).days)


def summarize_model_year(
    sub: pd.DataFrame,
    event_threshold_mm: float,
    fractional_loss_min_swe_mm: float,
) -> dict[str, Any]:
    source = str(sub["source"].iloc[0])
    wy = int(sub["water_year"].iloc[0])
    s = sub.sort_values("date").copy()
    swe = pd.to_numeric(s["swe_mm"], errors="coerce")
    if swe.notna().any():
        peak_idx = swe.idxmax()
        peak_swe = float(s.loc[peak_idx, "swe_mm"])
        peak_date = pd.Timestamp(s.loc[peak_idx, "date"])
    else:
        peak_swe, peak_date = np.nan, pd.NaT

    row: dict[str, Any] = {
        "source": source,
        "water_year": wy,
        "peak_swe_mm": peak_swe,
        "peak_date": peak_date,
        "days_peak_to_75pct": days_to_peak_fraction(s, 0.75),
        "days_peak_to_50pct": days_to_peak_fraction(s, 0.50),
        "days_peak_to_25pct": days_to_peak_fraction(s, 0.25),
        "swe_apr01_mm": value_on_month_day(s, 4, 1, "swe_mm"),
        "swe_may01_mm": value_on_month_day(s, 5, 1, "swe_mm"),
        "swe_jun01_mm": value_on_month_day(s, 6, 1, "swe_mm"),
        "swe_jun30_mm": value_on_month_day(s, 6, 30, "swe_mm"),
        "precip_oct_jun_mm": float(pd.to_numeric(s["precip_mm"], errors="coerce").sum(min_count=1)),
        "snowfall_oct_jun_mm": float(pd.to_numeric(s["snowfall_mm"], errors="coerce").sum(min_count=1)),
        "snowfall_oct_mar_mm": float(pd.to_numeric(s[s["date"].dt.month.isin([10, 11, 12, 1, 2, 3])]["snowfall_mm"], errors="coerce").sum(min_count=1)),
        "melt_oct_jun_mm": float(pd.to_numeric(s["melt_mm"], errors="coerce").sum(min_count=1)),
        "event_threshold_mm_day": event_threshold_mm,
        "fractional_loss_min_previous_swe_mm": fractional_loss_min_swe_mm,
    }
    if np.isfinite(row["precip_oct_jun_mm"]) and row["precip_oct_jun_mm"] > 0:
        row["snowfall_fraction_oct_jun"] = row["snowfall_oct_jun_mm"] / row["precip_oct_jun_mm"]
    else:
        row["snowfall_fraction_oct_jun"] = np.nan
    row["retention_mar31"] = value_on_month_day(s, 3, 31, "retention_fraction")
    row["retention_jun30"] = value_on_month_day(s, 6, 30, "retention_fraction")

    accum = s[s["dSWE_mm_day"] > event_threshold_mm]
    ablation = s[s["dSWE_mm_day"] < -event_threshold_mm]
    row.update(
        {
            "accumulation_event_count": int(len(accum)),
            "mean_accumulation_event_dSWE_mm_day": float(accum["dSWE_mm_day"].mean()) if len(accum) else np.nan,
            "cumulative_positive_dSWE_mm": float(pd.to_numeric(s["swe_gain_mm_day"], errors="coerce").sum(min_count=1)),
            "ablation_event_count": int(len(ablation)),
            "mean_ablation_event_loss_mm_day": float(ablation["swe_loss_mm_day"].mean()) if len(ablation) else np.nan,
            "cumulative_swe_loss_mm": float(pd.to_numeric(s["swe_loss_mm_day"], errors="coerce").sum(min_count=1)),
            "max_7day_cumulative_swe_loss_mm": float(s["swe_loss_mm_day"].rolling(7, min_periods=7).sum().max()),
            "mean_fractional_swe_loss_on_events": float(pd.to_numeric(s["fractional_swe_loss"], errors="coerce").mean()),
        }
    )

    for sca_col, pct in [("sca_gt_10mm", 0.90), ("sca_gt_10mm", 0.75), ("sca_gt_10mm", 0.50)]:
        row[f"post_peak_date_sca10_below_{int(pct * 100)}pct"] = first_post_peak_below(s, sca_col, pct, peak_date)
    if "fsno_mean" in s.columns and pd.to_numeric(s["fsno_mean"], errors="coerce").notna().any():
        for pct in [0.90, 0.75, 0.50]:
            row[f"post_peak_date_fsno_mean_below_{int(pct * 100)}pct"] = first_post_peak_below(s, "fsno_mean", pct, peak_date)

    aprjun = s[s["date"].dt.month.isin([4, 5, 6])]
    row["apr_jun_mean_effective_albedo"] = float(pd.to_numeric(aprjun["effective_albedo"], errors="coerce").mean())
    row["apr_jun_cumulative_incoming_sw_mj_m2"] = float(pd.to_numeric(aprjun["incoming_sw_energy_mj_m2"], errors="coerce").sum(min_count=1))
    row["apr_jun_cumulative_absorbed_sw_mj_m2"] = float(pd.to_numeric(aprjun["absorbed_sw_energy_mj_m2"], errors="coerce").sum(min_count=1))
    if "snowcovered_effective_albedo" in aprjun.columns:
        row["apr_jun_snowcovered_mean_effective_albedo"] = float(pd.to_numeric(aprjun["snowcovered_effective_albedo"], errors="coerce").mean())
        row["apr_jun_snowcovered_cumulative_absorbed_sw_mj_m2"] = float(
            pd.to_numeric(aprjun["snowcovered_absorbed_sw_energy_mj_m2"], errors="coerce").sum(min_count=1)
        )
    return row


def build_summary_tables(
    daily: pd.DataFrame,
    event_threshold_mm: float,
    fractional_loss_min_swe_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for source in ["iSnobal", "Noah-MP"]:
        for wy in YEARS:
            sub = daily[daily["source"].eq(source) & daily["water_year"].eq(wy)]
            if sub.empty:
                raise ValueError(f"No daily data found for {source} WY{wy} in the Oct 1-Jun 30 analysis window.")
            rows.append(summarize_model_year(sub, event_threshold_mm, fractional_loss_min_swe_mm))
    summary = pd.DataFrame(rows)

    numeric = summary.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [c for c in numeric if c != "water_year"]
    anomaly_rows = []
    for source in ["iSnobal", "Noah-MP"]:
        s = summary[summary["source"].eq(source)].set_index("water_year")
        row: dict[str, Any] = {"source": source, "comparison": "WY2026 - WY2021"}
        for col in numeric:
            if 2021 in s.index and 2026 in s.index:
                a = s.loc[2021, col]; b = s.loc[2026, col]
                row[col] = float(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan
        anomaly_rows.append(row)
    anomalies = pd.DataFrame(anomaly_rows)
    # Difference of the two model anomalies for directly comparable numeric metrics.
    i = anomalies[anomalies["source"].eq("iSnobal")]
    n = anomalies[anomalies["source"].eq("Noah-MP")]
    if not i.empty and not n.empty:
        row = {"source": "Noah-MP anomaly minus iSnobal anomaly", "comparison": "double difference"}
        for col in numeric:
            iv = i[col].iloc[0] if col in i else np.nan
            nv = n[col].iloc[0] if col in n else np.nan
            row[col] = float(nv - iv) if np.isfinite(iv) and np.isfinite(nv) else np.nan
        anomalies = pd.concat([anomalies, pd.DataFrame([row])], ignore_index=True)
    return summary, anomalies


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def format_water_year_axis(ax: plt.Axes) -> None:
    ax.set_xlim(REFERENCE_START, REFERENCE_END)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel("Month")


def clean_axes(axes: Iterable[plt.Axes] | np.ndarray) -> None:
    for ax in np.asarray(list(axes) if not isinstance(axes, np.ndarray) else axes).flat:
        ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, path: Path, cfg: WorkflowConfig, show: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg.section("plotting")["dpi"]), bbox_inches="tight")
    if not show:
        plt.close(fig)
    return path


def model_year_lines(ax: plt.Axes, daily: pd.DataFrame, column: str, ylabel: str) -> None:
    for source in ["iSnobal", "Noah-MP"]:
        for wy in YEARS:
            sub = daily[daily["source"].eq(source) & daily["water_year"].eq(wy)].sort_values("season_day")
            ax.plot(
                sub["plot_date"], sub[column],
                color=MODEL_COLORS[source], ls=YEAR_LINESTYLE[wy], lw=1.6,
            )
    ax.set_ylabel(ylabel)
    format_water_year_axis(ax)


def model_year_legend(fig: plt.Figure, y: float = 0.01) -> None:
    handles = []
    for source in ["iSnobal", "Noah-MP"]:
        for wy in YEARS:
            handles.append(
                Line2D([], [], color=MODEL_COLORS[source], ls=YEAR_LINESTYLE[wy], lw=1.6, label=f"{source} WY{wy}")
            )
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, y))


def plot_figure01(daily: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    model_year_lines(axes[0], daily, "swe_mm", "SWE (mm)")
    model_year_lines(axes[1], daily, "snow_depth_mm", "Snow depth (mm)")
    axes[0].set_xlabel("")
    model_year_legend(fig)
    clean_axes(axes)
    fig.subplots_adjust(bottom=0.14, hspace=0.12)
    return save_figure(fig, figdir / "figure01_seasonal_state.png", cfg, show)


def plot_figure02(anomalies: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    for source in ["iSnobal", "Noah-MP"]:
        sub = anomalies[anomalies["source"].eq(source)].sort_values("season_day")
        axes[0].plot(sub["plot_date"], sub["delta_swe_mm_2026_minus_2021"], color=MODEL_COLORS[source], lw=1.1, alpha=0.35)
        axes[0].plot(sub["plot_date"], sub["delta_swe_mm_7day"], color=MODEL_COLORS[source], lw=1.8, label=source)
        axes[1].plot(sub["plot_date"], sub["delta_snow_depth_mm_7day"], color=MODEL_COLORS[source], lw=1.8)
    dd = anomalies[anomalies["source"].eq("Noah-MP anomaly minus iSnobal anomaly")].sort_values("season_day")
    axes[2].plot(dd["plot_date"], dd["double_difference_swe_mm_7day"], color="black", lw=1.7)
    for ax in axes:
        ax.axhline(0, color="0.55", lw=0.8)
        format_water_year_axis(ax)
    axes[0].set_ylabel("WY2026 - WY2021\nSWE (mm)")
    axes[1].set_ylabel("WY2026 - WY2021\nSnow depth (mm)")
    axes[2].set_ylabel("Noah anomaly - iSnobal anomaly\nSWE (mm)")
    axes[0].legend()
    axes[0].set_xlabel(""); axes[1].set_xlabel("")
    clean_axes(axes)
    fig.subplots_adjust(hspace=0.15)
    return save_figure(fig, figdir / "figure02_interannual_anomaly.png", cfg, show)


def plot_figure03(daily: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    fig, axes = plt.subplots(5, 1, figsize=(15, 19), sharex=True)
    model_year_lines(axes[0], daily, "cumulative_precip_mm", "Cumulative precipitation (mm)")
    model_year_lines(axes[1], daily, "cumulative_snowfall_mm", "Cumulative snowfall (mm)")
    model_year_lines(axes[2], daily, "cumulative_snowfall_fraction", "Cumulative snowfall / precipitation")
    model_year_lines(axes[3], daily, "retention_fraction", "Season-to-date SWE retention")
    model_year_lines(axes[4], daily, "swe_mm", "SWE (mm)")
    # Retention becomes unstable before appreciable snowfall has accumulated;
    # it is intentionally left as NaN until cumulative snowfall exceeds 1 mm.
    for ax in axes[:-1]:
        ax.set_xlabel("")
    model_year_legend(fig)
    clean_axes(axes)
    fig.subplots_adjust(bottom=0.09, hspace=0.12)
    return save_figure(fig, figdir / "figure03_accumulation_input.png", cfg, show)


def plot_figure04(daily: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    model_year_lines(axes[0], daily, "dSWE_7day_mean_mm_day", "7-day mean dSWE (mm/day)")
    axes[0].axhline(0, color="0.55", lw=0.8)
    model_year_lines(axes[1], daily, "cumulative_swe_gain_mm_day", "Cumulative positive dSWE (mm)")
    model_year_lines(axes[2], daily, "cumulative_swe_loss_mm_day", "Cumulative SWE loss (mm)")
    model_year_lines(axes[3], daily, "cumulative_melt_mm", "Cumulative model melt (mm)")
    for ax in axes[:-1]:
        ax.set_xlabel("")
    model_year_legend(fig)
    clean_axes(axes)
    fig.subplots_adjust(bottom=0.10, hspace=0.12)
    return save_figure(fig, figdir / "figure04_accumulation_ablation_events.png", cfg, show)


def plot_figure05(daily: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    specs = [
        ("sca_gt_1mm", "Watershed area with SWE > 1 mm"),
        ("sca_gt_10mm", "Watershed area with SWE > 10 mm"),
        ("sca_gt_25mm", "Watershed area with SWE > 25 mm"),
        ("area_fraction_marginal_1_25mm", "Watershed area with 1 < SWE <= 25 mm"),
    ]
    for ax, (col, label) in zip(axes, specs):
        model_year_lines(ax, daily, col, f"{label}\n(fraction)")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    model_year_legend(fig)
    clean_axes(axes)
    fig.subplots_adjust(bottom=0.10, hspace=0.12)
    return save_figure(fig, figdir / "figure05_watershed_snow_coverage.png", cfg, show)


def plot_figure06_fsno(fsno: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    specs = [
        ("fsno_mean", "Watershed-mean FSNO"),
        ("fsno_area_fraction_very_low_gt0_0p1", "Area with very low FSNO\n(0 < FSNO < 0.1)"),
        ("fsno_area_fraction_partial_0p1_0p9", "Area with partial FSNO\n(0.1 <= FSNO < 0.9)"),
        ("fsno_area_fraction_nearly_full_0p9_lt1", "Area with nearly full FSNO\n(0.9 <= FSNO < 1)"),
        ("fsno_area_fraction_full_1", "Area with full FSNO\n(FSNO = 1)"),
    ]
    fig, axes = plt.subplots(len(specs), 1, figsize=(15, 18), sharex=True)
    for ax, (column, ylabel) in zip(axes, specs):
        for wy in YEARS:
            sub = fsno[fsno["water_year"].eq(wy)].sort_values("season_day")
            ax.plot(
                sub["plot_date"], sub[column], color=MODEL_COLORS["Noah-MP"],
                ls=YEAR_LINESTYLE[wy], lw=1.7, label=f"WY{wy}",
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        format_water_year_axis(ax)
    axes[0].legend()
    for ax in axes[:-1]:
        ax.set_xlabel("")
    clean_axes(axes)
    fig.subplots_adjust(hspace=0.13)
    return save_figure(fig, figdir / "figure06_noahmp_fsno.png", cfg, show)

def plot_figure07(daily: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    rows = [
        ("swe_mm", "SWE (mm)"),
        ("sca_gt_10mm", "Area with SWE > 10 mm\n(fraction)"),
        ("effective_albedo", "Effective albedo"),
        ("incoming_sw_energy_mj_m2", "Incoming SW\n(MJ/m²/day)"),
        ("absorbed_sw_energy_mj_m2", "Absorbed SW\n(MJ/m²/day)"),
        ("melt_mm", "Melt (mm/day)"),
    ]
    fig, axes = plt.subplots(len(rows), 2, figsize=(18, 23), sharex=True)
    for col_idx, source in enumerate(["iSnobal", "Noah-MP"]):
        for row_idx, (column, ylabel) in enumerate(rows):
            ax = axes[row_idx, col_idx]
            for wy in YEARS:
                sub = daily[daily["source"].eq(source) & daily["water_year"].eq(wy)].sort_values("season_day")
                values = pd.to_numeric(sub[column], errors="coerce")
                if column == "melt_mm":
                    values = values.rolling(7, center=True, min_periods=1).mean()
                ax.plot(sub["plot_date"], values, color=MODEL_COLORS[source], ls=YEAR_LINESTYLE[wy], lw=1.6, label=f"WY{wy}")
            if row_idx == 0:
                ax.set_title(source)
                ax.legend()
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            format_water_year_axis(ax)
            if row_idx < len(rows) - 1:
                ax.set_xlabel("")
    clean_axes(axes)
    fig.subplots_adjust(hspace=0.12, wspace=0.15)
    return save_figure(fig, figdir / "figure07_feedback_chain.png", cfg, show)


def plot_figure08(anomalies: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    specs = [
        ("delta_incoming_sw_energy_mj_m2_7day", "Incoming SW anomaly\n(MJ/m²/day)"),
        ("delta_effective_albedo_7day", "Effective albedo anomaly"),
        ("delta_absorbed_sw_energy_mj_m2_7day", "Absorbed SW anomaly\n(MJ/m²/day)"),
    ]
    for ax, (column, ylabel) in zip(axes[:3], specs):
        for source in ["iSnobal", "Noah-MP"]:
            sub = anomalies[anomalies["source"].eq(source)].sort_values("season_day")
            ax.plot(sub["plot_date"], sub[column], color=MODEL_COLORS[source], lw=1.7, label=source)
        ax.axhline(0, color="0.55", lw=0.8)
        ax.set_ylabel(ylabel)
        format_water_year_axis(ax)
    for source in ["iSnobal", "Noah-MP"]:
        sub = anomalies[anomalies["source"].eq(source)].sort_values("season_day").copy()
        cumulative = pd.to_numeric(sub["delta_absorbed_sw_energy_mj_m2_2026_minus_2021"], errors="coerce").fillna(0).cumsum()
        axes[3].plot(sub["plot_date"], cumulative, color=MODEL_COLORS[source], lw=1.7, label=source)
    axes[3].axhline(0, color="0.55", lw=0.8)
    axes[3].set_ylabel("Cumulative absorbed SW anomaly\n(MJ/m²)")
    format_water_year_axis(axes[3])
    axes[0].legend()
    for ax in axes[:-1]: ax.set_xlabel("")
    clean_axes(axes)
    fig.subplots_adjust(hspace=0.13)
    return save_figure(fig, figdir / "figure08_energy_anomalies.png", cfg, show)


def _spearman_text(x: pd.Series, y: pd.Series) -> str:
    valid = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return "ρ = NA"
    return f"ρ = {valid['x'].corr(valid['y'], method='spearman'):.2f}"


def plot_figure09(daily: pd.DataFrame, figdir: Path, cfg: WorkflowConfig, show: bool, event_threshold_mm: float) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    for row, source in enumerate(["iSnobal", "Noah-MP"]):
        for wy in YEARS:
            sub = daily[
                daily["source"].eq(source)
                & daily["water_year"].eq(wy)
                & (daily["dSWE_mm_day"] < -event_threshold_mm)
                & daily["sca_gt_10mm"].notna()
            ].copy()
            axes[row, 0].scatter(
                sub["sca_gt_10mm"], sub["swe_loss_mm_day"],
                s=28, alpha=0.55, marker=YEAR_MARKER[wy], color=MODEL_COLORS[source], label=f"WY{wy}",
            )
            frac = sub[sub["fractional_swe_loss"].notna()]
            axes[row, 1].scatter(
                frac["sca_gt_10mm"], frac["fractional_swe_loss"],
                s=28, alpha=0.55, marker=YEAR_MARKER[wy], color=MODEL_COLORS[source], label=f"WY{wy}",
            )
            axes[row, 0].text(
                0.03, 0.96 - (0.08 if wy == 2026 else 0.0),
                f"WY{wy}: {_spearman_text(sub['sca_gt_10mm'], sub['swe_loss_mm_day'])}",
                transform=axes[row, 0].transAxes, ha="left", va="top", fontsize=max(10, int(cfg.section('plotting')['font_size']) - 3),
            )
            axes[row, 1].text(
                0.03, 0.96 - (0.08 if wy == 2026 else 0.0),
                f"WY{wy}: {_spearman_text(frac['sca_gt_10mm'], frac['fractional_swe_loss'])}",
                transform=axes[row, 1].transAxes, ha="left", va="top", fontsize=max(10, int(cfg.section('plotting')['font_size']) - 3),
            )
        axes[row, 0].set_ylabel(f"{source}\nSWE loss (mm/day)")
        axes[row, 1].set_ylabel(f"{source}\nFractional SWE loss")
        axes[row, 0].legend()
        axes[row, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Watershed area with SWE > 10 mm (fraction)")
        ax.set_xlim(-0.02, 1.02)
    clean_axes(axes)
    fig.subplots_adjust(hspace=0.25, wspace=0.22)
    return save_figure(fig, figdir / "figure09_ablation_sensitivity.png", cfg, show)


def plot_optional_streamflow(cfg: WorkflowConfig, figdir: Path, show: bool) -> Path | None:
    path = cfg.output_dir / "tables" / "streamflow_daily_products.csv"
    if not path.exists():
        warnings.warn(f"--include-usgs was requested but {path} does not exist; skipping optional streamflow figure.", stacklevel=2)
        return None
    frame = pd.read_csv(path)
    require_columns(frame, ["date", "water_year", "iSnobal_7day_plot", "Noah-MP_7day_plot", "USGS_7day_plot"], str(path))
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["water_year"] = pd.to_numeric(frame["water_year"], errors="coerce").astype("Int64")
    frame = frame[frame["water_year"].isin(YEARS)].copy()
    frame = add_analysis_axis(frame)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    for ax, wy in zip(axes, YEARS):
        sub = frame[frame["water_year"].eq(wy)].sort_values("season_day")
        for source in ["iSnobal", "Noah-MP", "USGS"]:
            ax.plot(sub["plot_date"], sub[f"{source}_7day_plot"], color=MODEL_COLORS[source], lw=1.6, label=source)
        ax.set_ylabel(f"WY{wy}\n7-day volume (m³/day)")
        format_water_year_axis(ax)
        ax.legend(ncol=3)
    axes[0].set_xlabel("")
    clean_axes(axes)
    fig.subplots_adjust(hspace=0.12)
    return save_figure(fig, figdir / "figure10_optional_streamflow.png", cfg, show)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    configure_matplotlib(cfg)
    if args.show:
        cfg.data["plotting"]["close_after_save"] = False

    analysis_dir = cfg.output_dir / "diagnostics" / args.analysis_subdir
    table_dir = analysis_dir / "tables"
    figure_dir = analysis_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    i_table = load_table(cfg.output_dir / "tables" / "isnobal_watershed_daily.csv", YEARS)
    n_table = load_table(cfg.output_dir / "tables" / "noahmp_watershed_daily.csv", YEARS)
    weights = load_noah_weights(cfg)

    spatial_frames = []
    for source in ["isnobal", "noahmp"]:
        for wy in YEARS:
            spatial_frames.append(
                spatial_diagnostics_for_cache(
                    cfg, source, wy, weights,
                    DEFAULT_SCA_THRESHOLDS_MM,
                    float(args.snow_energy_threshold_mm),
                )
            )
    spatial = pd.concat(spatial_frames, ignore_index=True)

    fsno, fsno_metadata = extract_noah_fsno(cfg, weights, YEARS, args.fsno_variable)
    if fsno is not None:
        fsno.to_csv(table_dir / "noahmp_fsno_daily.csv", index=False)

    daily = build_daily_products(
        i_table, n_table, spatial, fsno,
        float(args.event_threshold_mm),
        float(args.fractional_loss_min_swe_mm),
    )
    anomalies = paired_year_anomalies(daily)
    summary, summary_anomalies = build_summary_tables(
        daily,
        float(args.event_threshold_mm),
        float(args.fractional_loss_min_swe_mm),
    )

    daily.to_csv(table_dir / "daily_diagnostics.csv", index=False)
    anomalies.to_csv(table_dir / "daily_anomalies_2026_minus_2021.csv", index=False)
    summary.to_csv(table_dir / "summary_metrics_by_model_year.csv", index=False)
    summary_anomalies.to_csv(table_dir / "summary_metric_anomalies.csv", index=False)

    figure_paths = [
        plot_figure01(daily, figure_dir, cfg, args.show),
        plot_figure02(anomalies, figure_dir, cfg, args.show),
        plot_figure03(daily, figure_dir, cfg, args.show),
        plot_figure04(daily, figure_dir, cfg, args.show),
        plot_figure05(daily, figure_dir, cfg, args.show),
    ]
    if fsno is not None:
        figure_paths.append(plot_figure06_fsno(fsno, figure_dir, cfg, args.show))
    figure_paths.extend(
        [
            plot_figure07(daily, figure_dir, cfg, args.show),
            plot_figure08(anomalies, figure_dir, cfg, args.show),
            plot_figure09(daily, figure_dir, cfg, args.show, float(args.event_threshold_mm)),
        ]
    )
    if args.include_usgs:
        optional = plot_optional_streamflow(cfg, figure_dir, args.show)
        if optional is not None:
            figure_paths.append(optional)

    manifest = {
        "analysis": "WY2021 versus WY2026 East River watershed diagnostic",
        "analysis_periods": {
            "WY2021": [str(analysis_bounds(2021)[0].date()), str(analysis_bounds(2021)[1].date())],
            "WY2026": [str(analysis_bounds(2026)[0].date()), str(analysis_bounds(2026)[1].date())],
        },
        "comparison_rule": "Only the common Oct 1-Jun 30 period is used for both years.",
        "divergence_rule": "No automatic divergence date is declared; daily and centered 7-day anomalies are provided.",
        "primary_sources": ["iSnobal", "Noah-MP"],
        "usgs_included": bool(args.include_usgs),
        "sca_definition": {
            "grid": "common Noah-MP grid",
            "weights": "exact fractional East River watershed weights",
            "thresholds_mm": list(DEFAULT_SCA_THRESHOLDS_MM),
            "missing_data": "renormalized over finite SWE cells; valid-area fraction is retained",
        },
        "snowcovered_energy_threshold_mm": float(args.snow_energy_threshold_mm),
        "event_threshold_mm_day": float(args.event_threshold_mm),
        "fractional_loss_min_previous_swe_mm": float(args.fractional_loss_min_swe_mm),
        "fsno": fsno_metadata,
        "tables": [
            table_dir / "daily_diagnostics.csv",
            table_dir / "daily_anomalies_2026_minus_2021.csv",
            table_dir / "summary_metrics_by_model_year.csv",
            table_dir / "summary_metric_anomalies.csv",
        ] + ([table_dir / "noahmp_fsno_daily.csv"] if fsno is not None else []),
        "figures": figure_paths,
        "final_synthesis_figure": "Intentionally not automated. Review Figures 1-9 first, then design the synthesis around the supported mechanism rather than pre-selecting a causal chain.",
    }
    save_json(manifest, analysis_dir / "analysis_manifest.json")

    print(f"Analysis complete: {analysis_dir}")
    print(f"FSNO available: {bool(fsno_metadata.get('available'))}")
    print("The final synthesis figure is intentionally deferred until the exploratory results are reviewed.")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
