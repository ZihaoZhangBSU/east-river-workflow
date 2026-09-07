#!/usr/bin/env python3
"""Compare WY2021 and WY2026 daily East River diagnostics in a 3x2 twin-axis figure.

This companion plotting script reuses outputs from:
  1) East_River_Noah_iSnobal_Separation_Diagnostic_v0.1.py
  2) East_River_LDASIN_WY2026_Forcing_Climatology_v0.1.py

Requested figure layout
-----------------------
Columns compare water years:
    left  = WY 2021
    right = WY 2026

Rows compare the same variable pair across years using twin y-axes:
    Row 1: FSNO (left axis) + delta SWE (Noah-MP - iSnobal; right axis)
    Row 2: delta effective albedo (left axis) + delta net shortwave (right axis)
    Row 3: daily mean temperature from Noah LDASIN (left axis) + delta net longwave (right axis)

Additional plotting rules from the user
---------------------------------------
* Plot only Feb 1 through Apr 30.
* Use raw daily values.
* Mark precipitation events (Noah LDASIN precip_mm_day >= 1 mm/day)
  in the albedo panels using short ticks near the bottom of the subplot.
* Only the LEFT subplot in each row carries the legend.
* Radiation panels are in W/m^2.
* Delta panels include a horizontal zero line.

Net-longwave definition used here
---------------------------------
Noah-MP:
    Daily mean net surface longwave = -mean(FIRA)
    because FIRA is "net LW radiation to atmosphere" [W/m2].

iSnobal:
    Daily mean net surface longwave = net_rad - mean(net_solar)
    where em.nc:net_rad is average net all-wave radiation [W/m2]
    and net_solar.nc:net_solar is net solar radiation [W/m2].

Delta net longwave:
    delta_net_lw_n_minus_i_w_m2 = net_lw_noah_w_m2 - net_lw_isnobal_w_m2

To stay consistent with the separation diagnostic, the iSnobal longwave field is
regridded to the common Noah-MP grid before basin averaging with the exact East
River Noah-grid fractional weights.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Allow the script to live near the repository root, examples/, or elsewhere.
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

from east_river_workflow.config import WorkflowConfig, load_config  # noqa: E402
from east_river_workflow.constants import MODEL_COLORS  # noqa: E402
from east_river_workflow.data_access import (  # noqa: E402
    decode_noah_times,
    normalized_units,
    read_isnobal_grid,
    read_noah_grid,
)
from east_river_workflow.grids import (  # noqa: E402
    fractional_polygon_weights,
    load_weights,
    regrid_area_average,
    select_watershed,
)
from east_river_workflow.plotting import configure_matplotlib  # noqa: E402
from east_river_workflow.utils import open_dataset_robust, weighted_nanmean  # noqa: E402


SCRIPT_VERSION = "0.1.0"
DEFAULT_OUTPUT_SUBDIR = "wy2021_wy2026_comparison_v0.1"
DEFAULT_SEPARATION_SUBDIR = "separation_diagnostic_v0.1"
DEFAULT_FORCING_SUBDIR = "ldasin_wy2026_forcing_v0.1"
TARGET_WATER_YEARS = (2021, 2026)
FOCUS_START = "02-01"
FOCUS_END = "04-30"
SECONDS_PER_DAY = 86400.0
MJ_TO_WM2_DAY = 1.0e6 / SECONDS_PER_DAY
DEFAULT_PRECIP_EVENT_THRESHOLD_MM_DAY = 1.0
DEFAULT_NOAH_NETLW_VARIABLE = "FIRA"
DEFAULT_ISNOBAL_NETRAD_VARIABLE = "net_rad"


@dataclass(frozen=True)
class Settings:
    output_root: Path
    separation_dir: Path
    forcing_dir: Path
    focus_start_month_day: str
    focus_end_month_day: str
    precip_event_threshold_mm_day: float
    noah_netlw_variable: str
    isnobal_netrad_variable: str
    overwrite_longwave_cache: bool

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    @property
    def cache_dir(self) -> Path:
        return self.output_root / "cache"

    def create_tree(self) -> None:
        for path in [self.output_root, self.tables_dir, self.figures_dir, self.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)


def _season_date(wy: int, month_day: str) -> pd.Timestamp:
    month, day = [int(x) for x in month_day.split("-")]
    year = wy - 1 if month >= 10 else wy
    return pd.Timestamp(year=year, month=month, day=day)


def _focus_bounds(wy: int, settings: Settings, cfg: WorkflowConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _season_date(wy, settings.focus_start_month_day)
    end = _season_date(wy, settings.focus_end_month_day)
    start = max(start, cfg.analysis_start.normalize())
    end = min(end, cfg.analysis_end.normalize())
    return start, end


def _daily_dir(root_template: str, wy: int, date: pd.Timestamp, folder_pattern: str) -> Path:
    root = Path(root_template.format(wy=wy, year=wy, start_year=wy - 1)).expanduser()
    return root / folder_pattern.format(date=date.to_pydatetime(), wy=wy)


def _save_figure(fig: plt.Figure, cfg: WorkflowConfig, settings: Settings, stem: str) -> Path:
    ext = str(cfg.section("plotting")["figure_format"])
    path = settings.figures_dir / f"{stem}.{ext}"
    fig.savefig(path, dpi=int(cfg.section("plotting")["dpi"]), bbox_inches="tight")
    if bool(cfg.section("plotting").get("close_after_save", True)):
        plt.close(fig)
    return path


def _load_noah_weights(cfg: WorkflowConfig, logger: logging.Logger) -> tuple[Any, np.ndarray]:
    noah_grid, _dem, _lat, _lon, _diag = read_noah_grid(
        cfg.data["paths"]["noahmp_geo_file"], cfg.section("noahmp")
    )
    main_weight_path = cfg.output_dir / "cache" / "east_river_fraction_noahmp.npz"
    if main_weight_path.exists():
        weights = load_weights(main_weight_path, noah_grid)
        logger.info("Loaded Noah watershed weights from %s", main_weight_path)
    else:
        logger.warning("Main Noah watershed-weight cache absent; rebuilding exact fractional weights.")
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
    return noah_grid, np.asarray(weights, dtype=float)


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _load_requested_daily_products(
    cfg: WorkflowConfig,
    settings: Settings,
    years: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sep_path = settings.separation_dir / "tables" / "separation_watershed_daily_full_period.csv"
    forcing_path = settings.forcing_dir / "tables" / "ldasin_watershed_daily_oct1_jun30.csv"
    if not sep_path.exists():
        raise FileNotFoundError(
            f"Required separation table is missing: {sep_path}\n"
            "Run East_River_Noah_iSnobal_Separation_Diagnostic_v0.1.py first."
        )
    if not forcing_path.exists():
        raise FileNotFoundError(
            f"Required forcing table is missing: {forcing_path}\n"
            "Run East_River_LDASIN_WY2026_Forcing_Climatology_v0.1.py first."
        )

    sep = _read_table(sep_path)
    forcing = _read_table(forcing_path)

    keep_sep = [
        "date", "water_year", "fsno_n_watershed_mean", "delta_swe_n_minus_i_mm",
        "delta_effective_albedo_n_minus_i", "delta_absorbed_sw_n_minus_i_mj_m2_day",
    ]
    keep_forcing = ["date", "water_year", "precip_mm_day", "temperature_mean_c"]

    missing_sep = sorted(set(keep_sep) - set(sep.columns))
    missing_forcing = sorted(set(keep_forcing) - set(forcing.columns))
    if missing_sep:
        raise KeyError(f"Separation table is missing required columns: {missing_sep}")
    if missing_forcing:
        raise KeyError(f"Forcing table is missing required columns: {missing_forcing}")

    sep = sep[keep_sep].copy()
    forcing = forcing[keep_forcing].copy()

    focus_parts_sep: list[pd.DataFrame] = []
    focus_parts_forcing: list[pd.DataFrame] = []
    for wy in years:
        start, end = _focus_bounds(wy, settings, cfg)
        focus_parts_sep.append(sep[(sep["water_year"] == wy) & (sep["date"] >= start) & (sep["date"] <= end)].copy())
        focus_parts_forcing.append(
            forcing[(forcing["water_year"] == wy) & (forcing["date"] >= start) & (forcing["date"] <= end)].copy()
        )

    sep_focus = pd.concat(focus_parts_sep, ignore_index=True)
    forcing_focus = pd.concat(focus_parts_forcing, ignore_index=True)

    sep_focus["delta_net_sw_n_minus_i_w_m2"] = sep_focus["delta_absorbed_sw_n_minus_i_mj_m2_day"] * MJ_TO_WM2_DAY
    forcing_focus["precip_event"] = forcing_focus["precip_mm_day"] >= settings.precip_event_threshold_mm_day
    return sep_focus, forcing_focus


def _weighted_mean_strict(values: np.ndarray, weights: np.ndarray, label: str) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = np.isfinite(weights) & (weights > 0)
    if not np.all(np.isfinite(values[positive])):
        bad = int(np.count_nonzero(~np.isfinite(values[positive])))
        raise ValueError(f"{label}: found {bad} non-finite positive-weight cells.")
    out, valid = weighted_nanmean(values, weights)
    if not math.isclose(valid, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: weighted coverage fraction is {valid}, expected 1.0.")
    return float(out)


def _load_noah_net_lw_daily(
    cfg: WorkflowConfig,
    settings: Settings,
    years: tuple[int, ...],
    weights: np.ndarray,
) -> pd.DataFrame:
    """Return daily Noah net surface longwave basin means [W/m^2]."""
    source = Path(cfg.data["paths"]["noahmp_output_file"]).expanduser()
    section = cfg.section("noahmp")
    rows: list[dict[str, Any]] = []
    with open_dataset_robust(source, decode_times=False) as ds:
        if settings.noah_netlw_variable not in ds:
            raise KeyError(f"{settings.noah_netlw_variable!r} not found in Noah LDASOUT: {source}")
        times = decode_noah_times(ds, section)
        frame = pd.DataFrame({"index": np.arange(len(times), dtype=int), "time": times})
        frame = frame[(frame["time"] >= cfg.analysis_start) & (frame["time"] <= cfg.analysis_end)].copy()
        frame["date"] = frame["time"].dt.normalize()
        frame["hour"] = frame["time"].dt.hour

        units = normalized_units(str(ds[settings.noah_netlw_variable].attrs.get("units", "")))
        accepted = {"w/m2", "w m-2", "w m^-2", "watt/m2", "watt m-2"}
        if units not in accepted:
            raise ValueError(
                f"Unexpected Noah net longwave units for {settings.noah_netlw_variable!r}: {units!r}"
            )

        for wy in years:
            start, end = _focus_bounds(wy, settings, cfg)
            for date in pd.date_range(start, end, freq="D"):
                day = frame[frame["date"].eq(date)].sort_values("hour")
                hours = day["hour"].astype(int).tolist()
                if hours != list(range(24)):
                    raise ValueError(f"Incomplete Noah day for {date.date()}: hours={hours}")
                indices = day["index"].to_numpy(dtype=int)
                # FIRA is net LW radiation to atmosphere.  Multiply by -1 so positive is toward surface.
                fira = np.asarray(ds[settings.noah_netlw_variable].isel(Time=indices).values, dtype=float).squeeze()
                if fira.ndim != 3:
                    raise ValueError(
                        f"Expected Noah {settings.noah_netlw_variable} to have shape (24, y, x); got {fira.shape}"
                    )
                net_lw_surface = -np.mean(fira, axis=0)
                basin_mean = _weighted_mean_strict(
                    net_lw_surface, weights, f"Noah daily net LW {date.date()}"
                )
                rows.append({
                    "date": date,
                    "water_year": int(wy),
                    "net_lw_noah_w_m2": basin_mean,
                })
    return pd.DataFrame(rows)


def _state_2d(variable, time_index: int) -> np.ndarray:
    indexer = {"time": time_index} if "time" in variable.dims else ({"Time": time_index} if "Time" in variable.dims else {})
    return np.asarray(variable.isel(indexer).values, dtype=float).squeeze()


def _read_hourly_mean_2d(path: Path, variable_name: str) -> np.ndarray:
    with open_dataset_robust(path, decode_times=True) as ds:
        if variable_name not in ds:
            raise KeyError(f"{variable_name!r} not found in {path}")
        values = np.asarray(ds[variable_name].values, dtype=float)
        units = normalized_units(str(ds[variable_name].attrs.get("units", "")))
        accepted = {"w/m2", "w m-2", "w m^-2", "watt/m2", "watt m-2"}
        if units not in accepted:
            raise ValueError(f"Unexpected units for {variable_name!r} in {path}: {units!r}")
        if values.ndim == 2:
            return values
        if values.ndim != 3:
            raise ValueError(f"Expected {variable_name!r} in {path} to be 2D or 3D; got {values.shape}")
        return np.mean(values, axis=0)


def _load_isnobal_net_lw_daily(
    cfg: WorkflowConfig,
    settings: Settings,
    years: tuple[int, ...],
    isnobal_grid,
    noah_grid,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Return daily iSnobal net surface longwave basin means [W/m^2]."""
    section = cfg.section("isnobal")
    root_template = cfg.data["paths"]["isnobal_root_template"]
    threshold = float(section["minimum_regrid_valid_coverage"])
    rows: list[dict[str, Any]] = []

    for wy in years:
        start, end = _focus_bounds(wy, settings, cfg)
        for date in pd.date_range(start, end, freq="D"):
            folder = _daily_dir(root_template, int(wy), date, section["daily_folder_pattern"])
            energy_path = folder / section["energy_filename"]
            net_solar_path = folder / section["absorbed_sw_filename"]
            if not energy_path.exists():
                raise FileNotFoundError(f"Missing iSnobal energy file for {date.date()}: {energy_path}")
            if not net_solar_path.exists():
                raise FileNotFoundError(f"Missing iSnobal net-solar file for {date.date()}: {net_solar_path}")

            with open_dataset_robust(energy_path, decode_times=True) as ds_energy:
                if settings.isnobal_netrad_variable not in ds_energy:
                    raise KeyError(
                        f"{settings.isnobal_netrad_variable!r} not found in iSnobal energy file {energy_path}"
                    )
                net_rad_var = ds_energy[settings.isnobal_netrad_variable]
                net_rad_units = normalized_units(str(net_rad_var.attrs.get("units", "")))
                accepted = {"w/m2", "w m-2", "w m^-2", "watt/m2", "watt m-2"}
                if net_rad_units not in accepted:
                    raise ValueError(
                        f"Unexpected units for {settings.isnobal_netrad_variable!r} in {energy_path}: {net_rad_units!r}"
                    )
                net_rad = _state_2d(net_rad_var, int(section["time_index"]))

            net_solar_mean = _read_hourly_mean_2d(net_solar_path, str(section["absorbed_sw_variable"]))
            net_lw_native = net_rad - net_solar_mean
            net_lw_noahgrid, _coverage = regrid_area_average(
                net_lw_native, isnobal_grid, noah_grid, minimum_valid_coverage=threshold
            )
            basin_mean = _weighted_mean_strict(
                net_lw_noahgrid, weights, f"iSnobal daily net LW {date.date()}"
            )
            rows.append({
                "date": date,
                "water_year": int(wy),
                "net_lw_isnobal_w_m2": basin_mean,
            })
    return pd.DataFrame(rows)


def _load_or_build_net_lw_table(
    cfg: WorkflowConfig,
    settings: Settings,
    years: tuple[int, ...],
    noah_grid,
    weights: np.ndarray,
    logger: logging.Logger,
) -> pd.DataFrame:
    cache_path = settings.cache_dir / "daily_net_longwave_wy2021_wy2026.csv"
    if cache_path.exists() and not settings.overwrite_longwave_cache:
        logger.info("Loading cached net-longwave table from %s", cache_path)
        frame = _read_table(cache_path)
        return frame

    isnobal_grid, _dem, _valid = read_isnobal_grid(
        cfg.data["paths"]["isnobal_topo_file"], cfg.section("isnobal")["dem_variable"]
    )
    logger.info("Computing daily Noah net longwave for WY%s and WY%s", years[0], years[1])
    noah = _load_noah_net_lw_daily(cfg, settings, years, weights)
    logger.info("Computing daily iSnobal net longwave for WY%s and WY%s", years[0], years[1])
    isnobal = _load_isnobal_net_lw_daily(cfg, settings, years, isnobal_grid, noah_grid, weights)
    frame = noah.merge(isnobal, on=["date", "water_year"], how="inner", validate="one_to_one")
    frame["delta_net_lw_n_minus_i_w_m2"] = frame["net_lw_noah_w_m2"] - frame["net_lw_isnobal_w_m2"]
    frame.to_csv(cache_path, index=False)
    logger.info("Wrote net-longwave cache to %s", cache_path)
    return frame


def _merge_all_products(
    sep: pd.DataFrame,
    forcing: pd.DataFrame,
    net_lw: pd.DataFrame,
) -> pd.DataFrame:
    merged = sep.merge(forcing, on=["date", "water_year"], how="inner", validate="one_to_one")
    merged = merged.merge(net_lw[["date", "water_year", "delta_net_lw_n_minus_i_w_m2"]], on=["date", "water_year"], how="inner", validate="one_to_one")
    merged = merged.sort_values(["water_year", "date"]).reset_index(drop=True)
    return merged


def _symmetric_limits(values: np.ndarray, pad_fraction: float = 0.08, minimum_half_range: float = 1.0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        half = minimum_half_range
    else:
        half = max(float(np.nanmax(np.abs(arr))), minimum_half_range)
    half *= (1.0 + pad_fraction)
    return -half, half


def _padded_limits(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
        span = 1.0 if math.isclose(lo, 0.0, rel_tol=0.0, abs_tol=1e-12) else abs(lo) * 0.2
        return lo - span, hi + span
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _format_date_axis(ax: plt.Axes, dates: pd.Series) -> None:
    ax.set_xlim(dates.min(), dates.max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.spines[["top"]].set_visible(False)


def _add_precip_short_ticks(ax: plt.Axes, event_dates: pd.Series) -> None:
    if len(event_dates) == 0:
        return
    ymin, ymax = ax.get_ylim()
    tick_top = ymin + 0.08 * (ymax - ymin)
    ax.vlines(event_dates, ymin=ymin, ymax=tick_top, linewidth=1.0)
    ax.set_ylim(ymin, ymax)


def make_figure(frame: pd.DataFrame, cfg: WorkflowConfig, settings: Settings) -> plt.Figure:
    configure_matplotlib(cfg)
    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    c_noah = MODEL_COLORS.get("Noah-MP", "tab:blue")
    c_isnobal = MODEL_COLORS.get("iSnobal", "tab:orange")
    # Variable colors here reflect variable identity, not model identity.
    color_primary = c_noah
    color_secondary = c_isnobal

    fig, axes = plt.subplots(3, 2, figsize=(21, 16.5), squeeze=False)
    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.07, right=0.93, hspace=0.30, wspace=0.30)

    yr_data = {wy: frame[frame["water_year"] == wy].copy() for wy in TARGET_WATER_YEARS}
    for wy, sub in yr_data.items():
        if sub.empty:
            raise ValueError(f"No merged rows available for WY{wy} in requested focus window.")

    fsno_lim = (0.0, 1.0)
    dswe_lim = _symmetric_limits(frame["delta_swe_n_minus_i_mm"].to_numpy(), minimum_half_range=10.0)
    dalb_lim = _symmetric_limits(frame["delta_effective_albedo_n_minus_i"].to_numpy(), minimum_half_range=0.02)
    dnsw_lim = _symmetric_limits(frame["delta_net_sw_n_minus_i_w_m2"].to_numpy(), minimum_half_range=5.0)
    temp_lim = _padded_limits(frame["temperature_mean_c"].to_numpy())
    dnlw_lim = _symmetric_limits(frame["delta_net_lw_n_minus_i_w_m2"].to_numpy(), minimum_half_range=5.0)

    row_specs = [
        {
            "left_col": "fsno_n_watershed_mean",
            "right_col": "delta_swe_n_minus_i_mm",
            "left_label": "FSNO",
            "right_label": "Δ SWE (mm)",
            "left_lim": fsno_lim,
            "right_lim": dswe_lim,
            "legend_labels": ("FSNO", "Δ SWE"),
            "left_is_delta": False,
            "right_is_delta": True,
            "annotate_precip": False,
        },
        {
            "left_col": "delta_effective_albedo_n_minus_i",
            "right_col": "delta_net_sw_n_minus_i_w_m2",
            "left_label": "Δ Albedo",
            "right_label": "Δ Net SW (W/m²)",
            "left_lim": dalb_lim,
            "right_lim": dnsw_lim,
            "legend_labels": ("Δ Albedo", "Δ Net SW"),
            "left_is_delta": True,
            "right_is_delta": True,
            "annotate_precip": True,
        },
        {
            "left_col": "temperature_mean_c",
            "right_col": "delta_net_lw_n_minus_i_w_m2",
            "left_label": "Temperature (°C)",
            "right_label": "Δ Net LW (W/m²)",
            "left_lim": temp_lim,
            "right_lim": dnlw_lim,
            "legend_labels": ("Temperature", "Δ Net LW"),
            "left_is_delta": False,
            "right_is_delta": True,
            "annotate_precip": False,
        },
    ]

    for j, wy in enumerate(TARGET_WATER_YEARS):
        axes[0, j].set_title(f"WY {wy}", fontweight="normal", pad=8)

    for row_idx, spec in enumerate(row_specs):
        for col_idx, wy in enumerate(TARGET_WATER_YEARS):
            sub = yr_data[wy]
            ax = axes[row_idx, col_idx]
            ax_r = ax.twinx()

            line_left, = ax.plot(
                sub["date"], sub[spec["left_col"]],
                color=color_primary, linewidth=1.5, linestyle="-",
                label=spec["legend_labels"][0],
            )
            line_right, = ax_r.plot(
                sub["date"], sub[spec["right_col"]],
                color=color_secondary, linewidth=1.5, linestyle="--",
                label=spec["legend_labels"][1],
            )

            ax.set_ylabel(spec["left_label"])
            ax_r.set_ylabel(spec["right_label"])
            ax.set_ylim(*spec["left_lim"])
            ax_r.set_ylim(*spec["right_lim"])

            _format_date_axis(ax, sub["date"])
            ax.spines["right"].set_visible(False)
            ax_r.spines["top"].set_visible(False)

            if spec["left_is_delta"]:
                ax.axhline(0.0, color=color_primary, linewidth=0.8, linestyle=":")
            if spec["right_is_delta"]:
                ax_r.axhline(0.0, color=color_secondary, linewidth=0.8, linestyle=":")

            if spec["annotate_precip"]:
                _add_precip_short_ticks(ax, sub.loc[sub["precip_event"], "date"])
                if col_idx == 0:
                    ax.text(
                        0.98, 0.06,
                        f"Short ticks: precip ≥ {settings.precip_event_threshold_mm_day:g} mm/day",
                        transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=14,
                    )

            if col_idx == 0:
                handles = [line_left, line_right]
                labels = [h.get_label() for h in handles]
                ax.legend(handles, labels, loc="upper left", frameon=True)

    fig.supxlabel("Month", y=0.095, fontsize=18)
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to east_river_config.yaml")
    parser.add_argument(
        "--output-subdir", default=DEFAULT_OUTPUT_SUBDIR,
        help=f"Output subdirectory under cfg.output_dir (default: {DEFAULT_OUTPUT_SUBDIR})",
    )
    parser.add_argument(
        "--separation-subdir", default=DEFAULT_SEPARATION_SUBDIR,
        help=f"Subdirectory under cfg.output_dir containing separation outputs (default: {DEFAULT_SEPARATION_SUBDIR})",
    )
    parser.add_argument(
        "--forcing-subdir", default=DEFAULT_FORCING_SUBDIR,
        help=f"Subdirectory under cfg.output_dir containing LDASIN forcing outputs (default: {DEFAULT_FORCING_SUBDIR})",
    )
    parser.add_argument(
        "--focus-start", default=FOCUS_START,
        help=f"Focus start MM-DD within each WY (default: {FOCUS_START})",
    )
    parser.add_argument(
        "--focus-end", default=FOCUS_END,
        help=f"Focus end MM-DD within each WY (default: {FOCUS_END})",
    )
    parser.add_argument(
        "--precip-threshold-mm-day", type=float, default=DEFAULT_PRECIP_EVENT_THRESHOLD_MM_DAY,
        help=f"Threshold for precipitation-event ticks in albedo panels (default: {DEFAULT_PRECIP_EVENT_THRESHOLD_MM_DAY})",
    )
    parser.add_argument(
        "--noah-netlw-variable", default=DEFAULT_NOAH_NETLW_VARIABLE,
        help=f"Noah-MP LDASOUT variable for net longwave (default: {DEFAULT_NOAH_NETLW_VARIABLE})",
    )
    parser.add_argument(
        "--isnobal-netrad-variable", default=DEFAULT_ISNOBAL_NETRAD_VARIABLE,
        help=f"iSnobal em.nc variable for all-wave net radiation (default: {DEFAULT_ISNOBAL_NETRAD_VARIABLE})",
    )
    parser.add_argument(
        "--overwrite-longwave-cache", action="store_true",
        help="Recompute the daily net-longwave cache even if it already exists.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    logger = logging.getLogger("wy2021_wy2026_compare")
    logger.setLevel(getattr(logging, str(cfg.section("project").get("log_level", "INFO")).upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)

    settings = Settings(
        output_root=cfg.output_dir / str(args.output_subdir),
        separation_dir=cfg.output_dir / str(args.separation_subdir),
        forcing_dir=cfg.output_dir / str(args.forcing_subdir),
        focus_start_month_day=str(args.focus_start),
        focus_end_month_day=str(args.focus_end),
        precip_event_threshold_mm_day=float(args.precip_threshold_mm_day),
        noah_netlw_variable=str(args.noah_netlw_variable),
        isnobal_netrad_variable=str(args.isnobal_netrad_variable),
        overwrite_longwave_cache=bool(args.overwrite_longwave_cache),
    )
    settings.create_tree()

    noah_grid, weights = _load_noah_weights(cfg, logger)
    sep_focus, forcing_focus = _load_requested_daily_products(cfg, settings, TARGET_WATER_YEARS)
    net_lw = _load_or_build_net_lw_table(cfg, settings, TARGET_WATER_YEARS, noah_grid, weights, logger)
    merged = _merge_all_products(sep_focus, forcing_focus, net_lw)

    merged_path = settings.tables_dir / "wy2021_wy2026_daily_merged.csv"
    merged.to_csv(merged_path, index=False)
    logger.info("Wrote merged daily comparison table to %s", merged_path)

    fig = make_figure(merged, cfg, settings)
    fig_path = _save_figure(fig, cfg, settings, "figure_wy2021_wy2026_daily_comparison")
    logger.info("Wrote figure to %s", fig_path)

    return {
        "status": "complete",
        "script_version": SCRIPT_VERSION,
        "output_root": str(settings.output_root),
        "merged_table": str(merged_path),
        "net_longwave_cache": str(settings.cache_dir / "daily_net_longwave_wy2021_wy2026.csv"),
        "figure": str(fig_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(pd.Series(result).to_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
