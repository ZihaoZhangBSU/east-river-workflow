"""Daily processing for iSnobal, hourly-in-one-file Noah-MP, SNOTEL, and USGS."""

from __future__ import annotations

from io import StringIO
import logging
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import requests

from .config import WorkflowConfig
from .constants import CFS_TO_M3_S, SECONDS_PER_DAY
from .data_access import (
    decode_noah_times,
    integrate_depth_like,
    integrate_shortwave,
    read_isnobal_energy,
    read_isnobal_precip_snowfall,
    read_isnobal_shortwave,
    read_isnobal_snow,
    time_step_seconds,
    variable_units,
)
from .grids import GridSpec, regrid_area_average
from .utils import add_water_year, depth_to_volume_m3, open_dataset_robust, save_json, weighted_nanmean, water_year_bounds


def _daily_dir(root_template: str, wy: int, date: pd.Timestamp, folder_pattern: str) -> Path:
    root = Path(root_template.format(wy=wy, year=wy, start_year=wy - 1)).expanduser()
    return root / folder_pattern.format(date=date.to_pydatetime(), wy=wy)


def spatial_cache_path(cfg: WorkflowConfig, source: str, wy: int) -> Path:
    return cfg.output_dir / "cache" / f"{source}_wy{int(wy)}_noahgrid_daily.npz"


def save_spatial_cache(path: Path, dates: list[pd.Timestamp], arrays: dict[str, list[np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {"dates": np.asarray([pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates], dtype="U10")}
    for key, values in arrays.items():
        payload[key] = np.asarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_spatial_cache(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    result["dates"] = pd.DatetimeIndex(pd.to_datetime(result["dates"]))
    return result


def _maybe_missing_files(paths: list[Path], policy: str) -> bool:
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return False
    message = "Missing required iSnobal daily files: " + ", ".join(str(p) for p in missing)
    if policy == "error":
        raise FileNotFoundError(message)
    warnings.warn(message, stacklevel=2)
    return True


def process_isnobal(
    cfg: WorkflowConfig,
    grid: GridSpec,
    noah_grid: GridSpec,
    watershed_weights: np.ndarray,
    station_lookup: list[dict[str, np.ndarray]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process iSnobal daily products and write native summaries + Noah-grid caches."""
    section = cfg.section("isnobal")
    project = cfg.section("project")
    stations = cfg.section("snotel")["stations"]
    root_template = cfg.data["paths"]["isnobal_root_template"]
    all_watershed: list[pd.DataFrame] = []
    all_station: list[pd.DataFrame] = []
    unit_diagnostics: list[dict[str, Any]] = []

    for wy in project["water_years"]:
        start, end = water_year_bounds(int(wy))
        start = max(start, cfg.analysis_start.normalize())
        end = min(end, cfg.analysis_end.normalize())
        watershed_rows: list[dict[str, Any]] = []
        station_rows: list[dict[str, Any]] = []
        dates_cache: list[pd.Timestamp] = []
        cache_arrays: dict[str, list[np.ndarray]] = {
            "swe_mm": [], "snow_depth_mm": [], "precip_mm": [], "snowfall_mm": [],
            "melt_mm": [], "swi_mm": [], "incoming_sw_energy_mj_m2": [], "absorbed_sw_energy_mj_m2": [],
            "effective_albedo": [],
        }
        logger.info("Processing iSnobal WY%s %s through %s", wy, start.date(), end.date())
        for date in pd.date_range(start, end, freq="D"):
            folder = _daily_dir(root_template, int(wy), date, section["daily_folder_pattern"])
            snow_path = folder / section["snow_filename"]
            precip_path = folder / section["precip_filename"]
            percent_path = folder / section["percent_snow_filename"]
            energy_path = folder / section["energy_filename"]
            incoming_path = folder / section["incoming_sw_filename"]
            absorbed_path = folder / section["absorbed_sw_filename"]
            required = [snow_path, precip_path, percent_path, energy_path, incoming_path, absorbed_path]
            if _maybe_missing_files(required, str(section["missing_file_policy"]).lower()):
                continue

            swe, depth, snow_time = read_isnobal_snow(snow_path, section)
            precip, snowfall, precip_diag = read_isnobal_precip_snowfall(precip_path, percent_path, section)
            swi, melt, energy_time = read_isnobal_energy(energy_path, section)
            in_mean, abs_mean, in_energy, abs_energy, albedo_native = read_isnobal_shortwave(incoming_path, absorbed_path, section)
            if not unit_diagnostics:
                unit_diagnostics.append({"source": "iSnobal", **precip_diag})

            # Native watershed means.  Effective albedo is ratio-of-integrated-energy,
            # not a simple average of per-cell albedo.
            swe_mean, swe_valid = weighted_nanmean(swe, watershed_weights)
            depth_mean, depth_valid = weighted_nanmean(depth, watershed_weights)
            precip_mean, precip_valid = weighted_nanmean(precip, watershed_weights)
            snowfall_mean, snowfall_valid = weighted_nanmean(snowfall, watershed_weights)
            melt_mean, melt_valid = weighted_nanmean(melt, watershed_weights)
            swi_mean, swi_valid = weighted_nanmean(swi, watershed_weights)
            in_e_mean, in_valid = weighted_nanmean(in_energy, watershed_weights)
            abs_e_mean, abs_valid = weighted_nanmean(abs_energy, watershed_weights)
            min_incoming = float(cfg.section("h3")["incoming_sw_minimum_mj_m2_day"])
            albedo_mean = 1.0 - abs_e_mean / in_e_mean if np.isfinite(in_e_mean) and in_e_mean > min_incoming and np.isfinite(abs_e_mean) else np.nan
            in_sw_mean, _ = weighted_nanmean(in_mean, watershed_weights)
            abs_sw_mean, _ = weighted_nanmean(abs_mean, watershed_weights)

            watershed_rows.append({
                "date": date, "water_year": int(wy),
                "swe_mm": swe_mean, "snow_depth_mm": depth_mean,
                "precip_mm": precip_mean, "snowfall_mm": snowfall_mean,
                "melt_mm": melt_mean, "swi_mm": swi_mean,
                "swi_volume_m3_day": depth_to_volume_m3(swi, watershed_weights, grid.cell_area_m2),
                "incoming_sw_w_m2": in_sw_mean, "absorbed_sw_w_m2": abs_sw_mean,
                "incoming_sw_energy_mj_m2": in_e_mean, "absorbed_sw_energy_mj_m2": abs_e_mean,
                "effective_albedo": albedo_mean,
                "swe_valid_area_fraction": swe_valid, "snow_depth_valid_area_fraction": depth_valid,
                "precip_valid_area_fraction": precip_valid, "snowfall_valid_area_fraction": snowfall_valid,
                "melt_valid_area_fraction": melt_valid, "swi_valid_area_fraction": swi_valid,
                "incoming_sw_valid_area_fraction": in_valid, "absorbed_sw_valid_area_fraction": abs_valid,
            })

            for station, lookup in zip(stations, station_lookup):
                nr, nc = int(lookup["nearest_row"]), int(lookup["nearest_col"])
                erows = np.asarray(lookup["envelope_rows"], dtype=int)
                ecols = np.asarray(lookup["envelope_cols"], dtype=int)
                def env_minmax(arr: np.ndarray) -> tuple[float, float]:
                    vals = arr[erows, ecols] if len(erows) else np.array([], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    return (float(np.min(vals)), float(np.max(vals))) if len(vals) else (np.nan, np.nan)
                swe_min, swe_max = env_minmax(swe)
                depth_min, depth_max = env_minmax(depth)
                station_rows.append({
                    "date": date, "water_year": int(wy), "station_id": int(station["id"]), "station_name": station["name"],
                    "nearest_distance_m": float(lookup["nearest_distance_m"]),
                    "swe_nearest_mm": float(swe[nr, nc]) if np.isfinite(swe[nr, nc]) else np.nan,
                    "snow_depth_nearest_mm": float(depth[nr, nc]) if np.isfinite(depth[nr, nc]) else np.nan,
                    "swe_in_noah_cell_min_mm": swe_min, "swe_in_noah_cell_max_mm": swe_max,
                    "snow_depth_in_noah_cell_min_mm": depth_min, "snow_depth_in_noah_cell_max_mm": depth_max,
                    "iSnobal_pixels_in_noah_cell": int(len(erows)),
                    "precip_nearest_mm": float(precip[nr, nc]) if np.isfinite(precip[nr, nc]) else np.nan,
                    "snowfall_nearest_mm": float(snowfall[nr, nc]) if np.isfinite(snowfall[nr, nc]) else np.nan,
                    "melt_nearest_mm": float(melt[nr, nc]) if np.isfinite(melt[nr, nc]) else np.nan,
                    "swi_nearest_mm": float(swi[nr, nc]) if np.isfinite(swi[nr, nc]) else np.nan,
                })

            # Publication/model-model cache on the Noah-MP grid.
            threshold = float(section["minimum_regrid_valid_coverage"])
            cache_daily: dict[str, np.ndarray] = {}
            for key, arr in {
                "swe_mm": swe, "snow_depth_mm": depth, "precip_mm": precip, "snowfall_mm": snowfall,
                "melt_mm": melt, "swi_mm": swi, "incoming_sw_energy_mj_m2": in_energy,
                "absorbed_sw_energy_mj_m2": abs_energy,
            }.items():
                cache_daily[key], _ = regrid_area_average(arr, grid, noah_grid, minimum_valid_coverage=threshold)
            alb = np.full(noah_grid.shape, np.nan, dtype=float)
            valid = np.isfinite(cache_daily["incoming_sw_energy_mj_m2"]) & np.isfinite(cache_daily["absorbed_sw_energy_mj_m2"]) & (cache_daily["incoming_sw_energy_mj_m2"] > min_incoming)
            alb[valid] = 1.0 - cache_daily["absorbed_sw_energy_mj_m2"][valid] / cache_daily["incoming_sw_energy_mj_m2"][valid]
            cache_daily["effective_albedo"] = alb
            dates_cache.append(date)
            for key in cache_arrays:
                cache_arrays[key].append(cache_daily[key])

        wdf = pd.DataFrame(watershed_rows)
        sdf = pd.DataFrame(station_rows)
        all_watershed.append(wdf)
        all_station.append(sdf)
        if dates_cache:
            save_spatial_cache(spatial_cache_path(cfg, "isnobal", int(wy)), dates_cache, cache_arrays)

    watershed = pd.concat(all_watershed, ignore_index=True).sort_values("date") if all_watershed else pd.DataFrame()
    station = pd.concat(all_station, ignore_index=True).sort_values(["station_id", "date"]) if all_station else pd.DataFrame()
    watershed.to_csv(cfg.output_dir / "tables" / "isnobal_watershed_daily.csv", index=False)
    station.to_csv(cfg.output_dir / "tables" / "isnobal_station_daily.csv", index=False)
    save_json(unit_diagnostics, cfg.output_dir / "diagnostics" / "isnobal_unit_report.json")
    return watershed, station


def _check_noah_time_axis(times: pd.DatetimeIndex, section: dict[str, Any]) -> dict[str, Any]:
    if times.hasnans:
        raise ValueError("Decoded Noah-MP time axis contains NaT values.")
    if times.has_duplicates:
        raise ValueError("Decoded Noah-MP time axis contains duplicate timestamps.")
    if not times.is_monotonic_increasing:
        raise ValueError("Decoded Noah-MP time axis is not strictly increasing.")
    expected_hours = float(section["expected_timestep_hours"])
    expected_seconds = expected_hours * 3600.0
    diffs = np.asarray((times[1:] - times[:-1]).total_seconds(), dtype=float)
    gaps = np.flatnonzero(~np.isclose(diffs, expected_seconds, atol=1.0))
    expected_count = section.get("expected_record_count")
    if expected_count not in {None, ""} and len(times) != int(expected_count):
        raise ValueError(f"Noah-MP record count is {len(times)}, expected {expected_count}.")
    for key, actual in [("expected_first_time", times[0]), ("expected_last_time", times[-1])]:
        configured = section.get(key)
        if configured not in {None, ""} and pd.Timestamp(configured) != actual:
            raise ValueError(f"noahmp.{key}={configured} does not match decoded Noah-MP time {actual}.")
    return {
        "record_count": int(len(times)), "first_decoded_time": str(times[0]), "last_decoded_time": str(times[-1]),
        "unexpected_gap_count": int(len(gaps)), "expected_timestep_hours": expected_hours,
    }


def _noah_slice(ds, name: str, indices: np.ndarray | list[int] | int) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Required Noah-MP variable {name!r} not found in LDASOUT.")
    variable = ds[name]
    if "Time" not in variable.dims:
        raise ValueError(f"Noah-MP variable {name!r} does not contain Time dimension: {variable.dims}")
    return np.asarray(variable.isel(Time=indices).values, dtype=float).squeeze()


def process_noahmp(
    cfg: WorkflowConfig,
    grid: GridSpec,
    watershed_weights: np.ndarray,
    station_lookup: list[dict[str, np.ndarray]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Process the single Noah-MP LDASOUT file containing hourly records."""
    section = cfg.section("noahmp")
    project = cfg.section("project")
    stations = cfg.section("snotel")["stations"]
    source = cfg.data["paths"]["noahmp_output_file"]
    all_watershed: list[pd.DataFrame] = []
    all_station: list[pd.DataFrame] = []
    daily_completeness: list[dict[str, Any]] = []
    unit_report: dict[str, str] = {}

    with open_dataset_robust(source, decode_times=False) as ds:
        times = decode_noah_times(ds, section)
        time_diag = _check_noah_time_axis(times, section)
        if cfg.analysis_start < times[0] or cfg.analysis_end > times[-1]:
            raise ValueError(
                f"Configured analysis window {cfg.analysis_start}..{cfg.analysis_end} is outside decoded Noah-MP times {times[0]}..{times[-1]}."
            )
        dt_seconds = time_step_seconds(times, float(section["expected_timestep_hours"]))
        variable_names = [
            section["swe_variable"], section["snow_depth_variable"], section["precip_variable"],
            section["snowfall_variable"], section["melt_variable"], section["snow_bottom_release_variable"],
            section["incoming_sw_variable"], section["absorbed_sw_variable"],
        ]
        for name in variable_names:
            if name not in ds:
                raise KeyError(f"Required Noah-MP variable {name!r} not found in {source}")
            unit_report[name] = variable_units(ds[name])

        time_frame = pd.DataFrame({"index": np.arange(len(times), dtype=int), "time": times})
        time_frame = time_frame[(time_frame["time"] >= cfg.analysis_start) & (time_frame["time"] <= cfg.analysis_end)].copy()
        time_frame["date"] = time_frame["time"].dt.normalize()
        time_frame["hour"] = time_frame["time"].dt.hour

        for wy in project["water_years"]:
            start, end = water_year_bounds(int(wy))
            start = max(start, cfg.analysis_start.normalize())
            end = min(end, cfg.analysis_end.normalize())
            dates = pd.date_range(start, end, freq="D")
            wrows: list[dict[str, Any]] = []
            srows: list[dict[str, Any]] = []
            dates_cache: list[pd.Timestamp] = []
            cache_arrays: dict[str, list[np.ndarray]] = {
                "swe_mm": [], "snow_depth_mm": [], "precip_mm": [], "snowfall_mm": [], "melt_mm": [],
                "qsnobot_mm": [], "incoming_sw_energy_mj_m2": [], "absorbed_sw_energy_mj_m2": [], "effective_albedo": [],
            }
            for date in dates:
                day = time_frame[time_frame["date"].eq(date)]
                hours = sorted(day["hour"].astype(int).tolist())
                complete = hours == list(range(24))
                state = day[day["hour"].eq(int(section["daily_state_hour"]))]
                has_state = len(state) == 1
                daily_completeness.append({"date": date, "water_year": int(wy), "hour_count": len(hours), "complete_00_23": complete, "has_state_hour": has_state})
                if not complete or not has_state:
                    message = f"Incomplete Noah-MP day {date.date()}: hours={hours}, state_count={len(state)}"
                    policy = str(section["daily_incomplete_policy"]).lower()
                    if policy == "error":
                        raise ValueError(message)
                    if policy == "warn":
                        warnings.warn(message, stacklevel=2)
                    continue
                indices = day.sort_values("hour")["index"].to_numpy(dtype=int)
                state_idx = int(state["index"].iloc[0])
                swe = _noah_slice(ds, section["swe_variable"], state_idx)
                depth = _noah_slice(ds, section["snow_depth_variable"], state_idx) * 1000.0
                precip = integrate_depth_like(_noah_slice(ds, section["precip_variable"], indices), unit_report[section["precip_variable"]], dt_seconds, "Noah-MP RAINRATE")
                snowfall = integrate_depth_like(_noah_slice(ds, section["snowfall_variable"], indices), unit_report[section["snowfall_variable"]], dt_seconds, "Noah-MP QSNOW")
                melt = integrate_depth_like(_noah_slice(ds, section["melt_variable"], indices), unit_report[section["melt_variable"]], dt_seconds, "Noah-MP QMELT")
                release = integrate_depth_like(_noah_slice(ds, section["snow_bottom_release_variable"], indices), unit_report[section["snow_bottom_release_variable"]], dt_seconds, "Noah-MP QSNBOT")
                in_mean, in_energy = integrate_shortwave(_noah_slice(ds, section["incoming_sw_variable"], indices), unit_report[section["incoming_sw_variable"]], dt_seconds, "Noah-MP SWFORC")
                abs_mean, abs_energy = integrate_shortwave(_noah_slice(ds, section["absorbed_sw_variable"], indices), unit_report[section["absorbed_sw_variable"]], dt_seconds, "Noah-MP FSA")
                albedo = np.full(grid.shape, np.nan, dtype=float)
                min_incoming = float(cfg.section("h3")["incoming_sw_minimum_mj_m2_day"])
                valid = np.isfinite(in_energy) & np.isfinite(abs_energy) & (in_energy > min_incoming)
                albedo[valid] = 1.0 - abs_energy[valid] / in_energy[valid]

                def wm(arr): return weighted_nanmean(arr, watershed_weights)[0]
                in_e_mean = wm(in_energy); abs_e_mean = wm(abs_energy)
                basin_albedo = 1.0 - abs_e_mean / in_e_mean if np.isfinite(in_e_mean) and in_e_mean > min_incoming and np.isfinite(abs_e_mean) else np.nan
                wrows.append({
                    "date": date, "water_year": int(wy), "swe_mm": wm(swe), "snow_depth_mm": wm(depth),
                    "precip_mm": wm(precip), "snowfall_mm": wm(snowfall), "melt_mm": wm(melt), "qsnobot_mm": wm(release),
                    "qsnobot_volume_m3_day": depth_to_volume_m3(release, watershed_weights, grid.cell_area_m2),
                    "incoming_sw_w_m2": wm(in_mean), "absorbed_sw_w_m2": wm(abs_mean),
                    "incoming_sw_energy_mj_m2": in_e_mean, "absorbed_sw_energy_mj_m2": abs_e_mean,
                    "effective_albedo": basin_albedo,
                })
                for station, lookup in zip(stations, station_lookup):
                    r, c = int(lookup["rows"][0]), int(lookup["cols"][0])
                    srows.append({
                        "date": date, "water_year": int(wy), "station_id": int(station["id"]), "station_name": station["name"],
                        "nearest_distance_m": float(lookup["distance_m"][0]),
                        "swe_mm": float(swe[r, c]) if np.isfinite(swe[r, c]) else np.nan,
                        "snow_depth_mm": float(depth[r, c]) if np.isfinite(depth[r, c]) else np.nan,
                        "precip_mm": float(precip[r, c]) if np.isfinite(precip[r, c]) else np.nan,
                        "snowfall_mm": float(snowfall[r, c]) if np.isfinite(snowfall[r, c]) else np.nan,
                        "melt_mm": float(melt[r, c]) if np.isfinite(melt[r, c]) else np.nan,
                        "qsnobot_mm": float(release[r, c]) if np.isfinite(release[r, c]) else np.nan,
                    })
                dates_cache.append(date)
                for key, arr in {
                    "swe_mm": swe, "snow_depth_mm": depth, "precip_mm": precip, "snowfall_mm": snowfall,
                    "melt_mm": melt, "qsnobot_mm": release, "incoming_sw_energy_mj_m2": in_energy,
                    "absorbed_sw_energy_mj_m2": abs_energy, "effective_albedo": albedo,
                }.items():
                    cache_arrays[key].append(arr)

            wdf = pd.DataFrame(wrows); sdf = pd.DataFrame(srows)
            all_watershed.append(wdf); all_station.append(sdf)
            if dates_cache:
                save_spatial_cache(spatial_cache_path(cfg, "noahmp", int(wy)), dates_cache, cache_arrays)

    watershed = pd.concat(all_watershed, ignore_index=True).sort_values("date") if all_watershed else pd.DataFrame()
    station = pd.concat(all_station, ignore_index=True).sort_values(["station_id", "date"]) if all_station else pd.DataFrame()
    watershed.to_csv(cfg.output_dir / "tables" / "noahmp_watershed_daily.csv", index=False)
    station.to_csv(cfg.output_dir / "tables" / "noahmp_station_daily.csv", index=False)
    pd.DataFrame(daily_completeness).to_csv(cfg.output_dir / "diagnostics" / "noahmp_daily_completeness.csv", index=False)
    save_json(unit_report, cfg.output_dir / "diagnostics" / "noahmp_unit_report.json")
    diagnostics = {**time_diag, "analysis_start": str(cfg.analysis_start), "analysis_end": str(cfg.analysis_end), "spinup_record_count": int(np.sum(times < cfg.analysis_start)), "daily_state_hour": int(section["daily_state_hour"]), "timestep_seconds": dt_seconds}
    return watershed, station, diagnostics


def process_snotel(cfg: WorkflowConfig, isnobal_station: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    section = cfg.section("snotel")
    frame = pd.read_csv(Path(cfg.data["paths"]["snotel_csv"]).expanduser())
    frame = frame.rename(columns={
        section["date_column"]: "date", section["id_column"]: "station_id",
        section["precip_column"]: "precip_observed_mm", section["swe_column"]: "swe_mm",
        section["snow_depth_column"]: "snow_depth_mm",
    })
    frame["date"] = pd.to_datetime(frame["date"])
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce").astype("Int64")
    for col in ["precip_observed_mm", "swe_mm", "snow_depth_mm"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[(frame["date"] >= cfg.analysis_start.normalize()) & (frame["date"] <= cfg.analysis_end.normalize())].copy()
    frame = add_water_year(frame)
    frame["precip_gapfilled_mm"] = frame["precip_observed_mm"]
    frame["precip_gapfill_source"] = np.where(frame["precip_observed_mm"].notna(), "observed", "missing")
    if bool(section.get("gapfill_precip_with_isnobal", True)):
        replacement = isnobal_station[["date", "station_id", "precip_nearest_mm"]].rename(columns={"precip_nearest_mm": "isnobal_gapfill_mm"})
        frame = frame.merge(replacement, on=["date", "station_id"], how="left")
        fillable = frame["precip_gapfilled_mm"].isna() & frame["isnobal_gapfill_mm"].notna()
        frame.loc[fillable, "precip_gapfilled_mm"] = frame.loc[fillable, "isnobal_gapfill_mm"]
        frame.loc[fillable, "precip_gapfill_source"] = "iSnobal nearest pixel"
    names = {int(s["id"]): s["name"] for s in section["stations"]}
    frame["station_name"] = frame["station_id"].map(names)
    gap = frame[frame["precip_observed_mm"].isna()][["date", "station_id", "station_name", "precip_gapfilled_mm", "precip_gapfill_source"]]
    gap.to_csv(cfg.output_dir / "tables" / "snotel_precip_gapfill_log.csv", index=False)
    frame.to_csv(cfg.output_dir / "tables" / "snotel_daily.csv", index=False)
    logger.info("SNOTEL precipitation gaps: %s", len(gap))
    return frame


def _fetch_usgs_daily_modern(section: dict[str, Any], cfg: WorkflowConfig) -> pd.DataFrame:
    endpoint = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
    params: dict[str, Any] | None = {
        "f": "json", "monitoring_location_id": f"USGS-{section['usgs_site']}",
        "parameter_code": section["parameter_code"], "statistic_id": section["statistic_code"],
        "datetime": f"{cfg.analysis_start.date()}/{cfg.analysis_end.date()}", "limit": int(section.get("page_size", 10000)),
    }
    key = os.environ.get(str(section.get("api_key_environment_variable", "USGS_API_KEY")))
    headers = {"X-Api-Key": key} if key else {}
    features: list[dict[str, Any]] = []
    url: str | None = endpoint
    while url:
        response = requests.get(url, params=params, headers=headers, timeout=float(section.get("request_timeout_seconds", 120)))
        response.raise_for_status(); payload = response.json(); features.extend(payload.get("features", []))
        next_links = [x.get("href") for x in payload.get("links", []) if isinstance(x, dict) and x.get("rel") == "next" and x.get("href")]
        url = str(next_links[0]) if next_links else None; params = None
    rows = []
    for feature in features:
        p = feature.get("properties", {})
        if str(p.get("parameter_code")) != str(section["parameter_code"]): continue
        rows.append({"date": pd.to_datetime(p.get("time"), errors="coerce"), "discharge_cfs": pd.to_numeric(p.get("value"), errors="coerce")})
    frame = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty: raise RuntimeError("USGS modern API returned no daily discharge.")
    frame["discharge_m3_day"] = frame["discharge_cfs"] * CFS_TO_M3_S * SECONDS_PER_DAY
    return add_water_year(frame)


def _fetch_usgs_daily_legacy(section: dict[str, Any], cfg: WorkflowConfig) -> pd.DataFrame:
    params = {"format": "rdb", "sites": section["usgs_site"], "startDT": str(cfg.analysis_start.date()), "endDT": str(cfg.analysis_end.date()), "parameterCd": section["parameter_code"], "statCd": section["statistic_code"], "siteStatus": "all"}
    response = requests.get("https://waterservices.usgs.gov/nwis/dv/", params=params, timeout=float(section.get("request_timeout_seconds", 120)))
    response.raise_for_status()
    text = "\n".join(line for line in response.text.splitlines() if not line.startswith("#"))
    frame = pd.read_csv(StringIO(text), sep="\t", dtype=str)
    frame["date"] = pd.to_datetime(frame.get("datetime"), errors="coerce")
    token = f"{section['parameter_code']}_{section['statistic_code']}"
    candidates = [c for c in frame.columns if token in c and not c.endswith("_cd")]
    if len(candidates) != 1: raise RuntimeError(f"Could not uniquely identify USGS discharge column: {candidates}")
    frame["discharge_cfs"] = pd.to_numeric(frame[candidates[0]], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["discharge_m3_day"] = frame["discharge_cfs"] * CFS_TO_M3_S * SECONDS_PER_DAY
    return add_water_year(frame[["date", "discharge_cfs", "discharge_m3_day"]])


def fetch_usgs_daily(cfg: WorkflowConfig, logger: logging.Logger) -> pd.DataFrame:
    section = cfg.section("streamflow")
    cache = cfg.output_dir / "cache" / f"usgs_{section['usgs_site']}_daily.csv"
    if cache.exists() and not bool(cfg.section("project")["overwrite"]):
        frame = pd.read_csv(cache, parse_dates=["date"])
        if not frame.empty and frame["date"].min() <= cfg.analysis_start.normalize() and frame["date"].max() >= cfg.analysis_end.normalize():
            return frame[(frame["date"] >= cfg.analysis_start.normalize()) & (frame["date"] <= cfg.analysis_end.normalize())].copy()
    mode = str(section.get("api_mode", "modern_with_legacy_fallback")).lower()
    try:
        frame = _fetch_usgs_daily_modern(section, cfg) if mode != "legacy" else _fetch_usgs_daily_legacy(section, cfg)
        source = "modern" if mode != "legacy" else "legacy"
    except Exception:
        if mode == "modern": raise
        logger.exception("Modern USGS request failed; trying legacy endpoint")
        frame = _fetch_usgs_daily_legacy(section, cfg); source = "legacy_fallback"
    frame.to_csv(cache, index=False); frame.to_csv(cfg.output_dir / "tables" / "usgs_discharge_daily.csv", index=False)
    save_json({"source_api": source, "site": section["usgs_site"], "first_date": str(frame["date"].min()), "last_date": str(frame["date"].max()), "row_count": len(frame)}, cfg.output_dir / "diagnostics" / "usgs_download_metadata.json")
    return frame
