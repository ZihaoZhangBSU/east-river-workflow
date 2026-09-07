"""ASO extraction, QC-cleaned comparison, regridding, and elevation summaries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aso_qc import qc_cleaned_path, read_cleaned_aso_raster
from .config import WorkflowConfig
from .constants import ELEVATION_BANDS
from .data_access import decode_noah_times, read_aso_raster, read_isnobal_snow
from .grids import GridSpec, regrid_area_average, write_geotiff
from .utils import open_dataset_robust


def elevation_band_mask(elevation: np.ndarray, band: str) -> np.ndarray:
    if band not in ELEVATION_BANDS:
        raise KeyError(f"Unknown elevation band {band!r}.")
    lower, upper = ELEVATION_BANDS[band]
    valid = np.isfinite(elevation)
    if lower is not None:
        valid &= elevation >= float(lower)
    if upper is not None:
        valid &= elevation < float(upper)
    return valid


def distribution_statistics(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {
            "cell_count": 0,
            "mean_mm": np.nan,
            "median_mm": np.nan,
            "mae_mm": np.nan,
            "rmse_mm": np.nan,
            "std_mm": np.nan,
            "p05_mm": np.nan,
            "p25_mm": np.nan,
            "p75_mm": np.nan,
            "p95_mm": np.nan,
        }
    return {
        "cell_count": int(len(data)),
        "mean_mm": float(np.mean(data)),
        "median_mm": float(np.median(data)),
        "mae_mm": float(np.mean(np.abs(data))),
        "rmse_mm": float(np.sqrt(np.mean(data**2))),
        "std_mm": float(np.std(data)),
        "p05_mm": float(np.percentile(data, 5)),
        "p25_mm": float(np.percentile(data, 25)),
        "p75_mm": float(np.percentile(data, 75)),
        "p95_mm": float(np.percentile(data, 95)),
    }


def _isnobal_daily_folder(cfg: WorkflowConfig, date: pd.Timestamp) -> Path:
    wy = date.year + 1 if date.month >= 10 else date.year
    root = Path(
        cfg.data["paths"]["isnobal_root_template"].format(
            wy=wy, year=wy, start_year=wy - 1
        )
    ).expanduser()
    return root / cfg.section("isnobal")["daily_folder_pattern"].format(
        date=date.to_pydatetime(), wy=wy
    )


def _noah_state_index(times: pd.DatetimeIndex, date: pd.Timestamp, hour: int) -> int:
    target_date = pd.Timestamp(date).normalize()
    matches = np.flatnonzero((times.normalize() == target_date) & (times.hour == int(hour)))
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one Noah-MP internal state at {target_date.date()} hour {hour:02d}; "
            f"found {len(matches)}."
        )
    return int(matches[0])


def _read_noah_state(ds, index: int, variable: str, multiplier: float = 1.0) -> np.ndarray:
    if variable not in ds:
        raise KeyError(f"Required Noah-MP variable {variable!r} is absent from LDASOUT.")
    arr = np.asarray(ds[variable].isel(Time=index).values, dtype=float).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Noah-MP {variable!r} state must be 2-D after Time selection; got {arr.shape}.")
    return arr * multiplier


def _aso_density_summary(aso_swe: np.ndarray, aso_depth: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = mask & np.isfinite(aso_swe) & np.isfinite(aso_depth) & (aso_depth > 0)
    density = np.full(aso_swe.shape, np.nan, dtype=float)
    density[valid] = 1000.0 * aso_swe[valid] / aso_depth[valid]
    data = density[np.isfinite(density)]
    if not len(data):
        return {"aso_density_cell_count": 0, "aso_density_mean_kg_m3": np.nan, "aso_density_median_kg_m3": np.nan}
    return {
        "aso_density_cell_count": int(len(data)),
        "aso_density_mean_kg_m3": float(np.mean(data)),
        "aso_density_median_kg_m3": float(np.median(data)),
    }


def process_aso(
    cfg: WorkflowConfig,
    isnobal_grid: GridSpec,
    noah_grid: GridSpec,
    isnobal_dem: np.ndarray,
    noah_dem: np.ndarray,
    isnobal_weights: np.ndarray,
    noah_weights: np.ndarray,
    logger: logging.Logger,
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    """Process ASO observations and return native/common-grid products.

    The publication comparison is always on the Noah-MP grid.  Native iSnobal
    products are retained for the baseline review figures.  If ASO QC is
    enabled, only the QC-cleaned copies are used downstream.
    """
    section = cfg.section("aso")
    snow_cfg = cfg.section("isnobal")
    noah_cfg = cfg.section("noahmp")
    products: dict[tuple[str, str], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []

    # Keep the single large Noah file open for all ASO dates.
    with open_dataset_robust(cfg.data["paths"]["noahmp_output_file"], decode_times=False) as noah_ds:
        noah_times = decode_noah_times(noah_ds, noah_cfg)
        for date_text in section["dates"]:
            date = pd.Timestamp(date_text)
            state_idx = _noah_state_index(noah_times, date, int(noah_cfg["daily_state_hour"]))
            noah_swe = _read_noah_state(noah_ds, state_idx, noah_cfg["swe_variable"], 1.0)
            noah_depth = _read_noah_state(noah_ds, state_idx, noah_cfg["snow_depth_variable"], 1000.0)

            daily_folder = _isnobal_daily_folder(cfg, date)
            snow_path = daily_folder / snow_cfg["snow_filename"]
            isnobal_swe, isnobal_depth, _ = read_isnobal_snow(snow_path, snow_cfg)

            date_common_aso: dict[str, np.ndarray] = {}
            for variable, aso_pattern, isnobal_native, noah_native in [
                ("SWE", section["swe_pattern"], isnobal_swe, noah_swe),
                ("Snow depth", section["snow_depth_pattern"], isnobal_depth, noah_depth),
            ]:
                raw_aso_path = Path(cfg.data["paths"]["aso_directory"]).expanduser() / aso_pattern.format(date=date_text)
                if bool(section["quality_control"]["enabled"]):
                    cleaned_path = qc_cleaned_path(cfg, date_text, variable)
                    if not cleaned_path.exists():
                        raise FileNotFoundError(
                            f"ASO QC raster is missing: {cleaned_path}. Run ASO QC first or use run-all."
                        )
                    aso, aso_grid, aso_metadata = read_cleaned_aso_raster(cleaned_path)
                    aso_metadata["original_source"] = str(raw_aso_path)
                else:
                    aso, aso_grid, aso_metadata = read_aso_raster(raw_aso_path)

                threshold = float(section["minimum_valid_coverage"])
                aso_on_i, aso_cov_i = regrid_area_average(
                    aso, aso_grid, isnobal_grid, minimum_valid_coverage=threshold
                )
                aso_on_n, aso_cov_n = regrid_area_average(
                    aso, aso_grid, noah_grid, minimum_valid_coverage=threshold
                )
                i_on_n, i_cov_n = regrid_area_average(
                    isnobal_native, isnobal_grid, noah_grid, minimum_valid_coverage=threshold
                )

                i_native_masked = np.where(isnobal_weights > 0, isnobal_native, np.nan)
                i_on_n_masked = np.where(noah_weights > 0, i_on_n, np.nan)
                n_masked = np.where(noah_weights > 0, noah_native, np.nan)
                aso_i_masked = np.where(isnobal_weights > 0, aso_on_i, np.nan)
                aso_n_masked = np.where(noah_weights > 0, aso_on_n, np.nan)
                diff_i_native = i_native_masked - aso_i_masked
                diff_i_noah = i_on_n_masked - aso_n_masked
                diff_noah = n_masked - aso_n_masked

                arrays = {
                    "aso_on_isnobal": aso_i_masked,
                    "aso_on_noah": aso_n_masked,
                    "isnobal_native": i_native_masked,
                    "isnobal_on_noah": i_on_n_masked,
                    "noah_native": n_masked,
                    "difference_isnobal_native": diff_i_native,
                    "difference_isnobal_noah": diff_i_noah,
                    "difference_noah": diff_noah,
                    "aso_coverage_isnobal": aso_cov_i,
                    "aso_coverage_noah": aso_cov_n,
                    "isnobal_coverage_noah": i_cov_n,
                }
                products[(date_text, variable)] = {
                    "date": date,
                    "variable": variable,
                    "arrays": arrays,
                    "aso_metadata": aso_metadata,
                }
                date_common_aso[variable] = aso_n_masked

                safe = variable.lower().replace(" ", "_")
                grid_by_key = {
                    "aso_on_isnobal": isnobal_grid,
                    "aso_on_noah": noah_grid,
                    "isnobal_native": isnobal_grid,
                    "isnobal_on_noah": noah_grid,
                    "noah_native": noah_grid,
                    "difference_isnobal_native": isnobal_grid,
                    "difference_isnobal_noah": noah_grid,
                    "difference_noah": noah_grid,
                    "aso_coverage_isnobal": isnobal_grid,
                    "aso_coverage_noah": noah_grid,
                    "isnobal_coverage_noah": noah_grid,
                }
                for key, array in arrays.items():
                    write_geotiff(
                        cfg.output_dir / "regridded" / f"{date_text}_{safe}_{key}.tif",
                        array,
                        grid_by_key[key],
                    )

                # Baseline/native + common-grid summary rows.  The final Table 6
                # is filtered in workflow.py to the two common-grid products.
                for product_name, values, elevation, weights in [
                    ("iSnobal-ASO on iSnobal grid", diff_i_native, isnobal_dem, isnobal_weights),
                    ("iSnobal-ASO on Noah-MP grid", diff_i_noah, noah_dem, noah_weights),
                    ("Noah-MP-ASO on Noah-MP grid", diff_noah, noah_dem, noah_weights),
                ]:
                    for band in ELEVATION_BANDS:
                        mask = (weights > 0) & elevation_band_mask(elevation, band) & np.isfinite(values)
                        row = {
                            "date": date,
                            "variable": variable,
                            "product": product_name,
                            "elevation_band": band,
                        }
                        row.update(distribution_statistics(values[mask]))
                        summaries.append(row)

            # Add observed density summary to common-grid Table 6 rows.
            if "SWE" in date_common_aso and "Snow depth" in date_common_aso:
                for band in ELEVATION_BANDS:
                    mask = (noah_weights > 0) & elevation_band_mask(noah_dem, band)
                    density_stats = _aso_density_summary(
                        date_common_aso["SWE"], date_common_aso["Snow depth"], mask
                    )
                    for row in summaries:
                        if (
                            pd.Timestamp(row["date"]) == date
                            and row["elevation_band"] == band
                            and row["product"] in {
                                "iSnobal-ASO on Noah-MP grid",
                                "Noah-MP-ASO on Noah-MP grid",
                            }
                        ):
                            row.update(density_stats)
            logger.info("Processed ASO comparison for %s", date_text)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(cfg.output_dir / "tables" / "aso_difference_summary.csv", index=False)
    return products, summary_frame
