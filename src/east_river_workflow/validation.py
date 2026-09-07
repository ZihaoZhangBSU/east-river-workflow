"""Preflight validation for the integrated East River workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import WorkflowConfig
from .data_access import decode_noah_times, read_noah_numeric_time_index, variable_units
from .grids import select_watershed
from .utils import open_dataset_robust, save_json


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} does not exist: {path}")


def validate_inputs(cfg: WorkflowConfig) -> dict[str, Any]:
    """Validate core paths, Noah time decoding, required variables and SNOTEL input.

    This intentionally validates the *single-file* Noah-MP model and never
    checks 85,439 hourly filenames or an LDASIN forcing directory.
    """
    paths = cfg.paths
    for key in ["watershed_shapefile", "isnobal_topo_file", "noahmp_output_file", "noahmp_geo_file", "snotel_csv"]:
        _require_file(paths[key], key)
    if not paths["aso_directory"].exists():
        raise FileNotFoundError(f"ASO directory does not exist: {paths['aso_directory']}")

    watershed = select_watershed(
        paths["watershed_shapefile"],
        cfg.section("watershed")["name_field"],
        cfg.section("watershed")["name_contains"],
    )
    report: dict[str, Any] = {
        "watershed": {
            "selected_feature_count": int(len(watershed)),
            "selected_name": str(watershed.iloc[0]["name"]),
            "source": str(paths["watershed_shapefile"]),
        }
    }

    noah_cfg = cfg.section("noahmp")
    required_noah = [
        noah_cfg["swe_variable"], noah_cfg["snow_depth_variable"],
        noah_cfg["precip_variable"], noah_cfg["snowfall_variable"], noah_cfg["melt_variable"],
        noah_cfg["snow_bottom_release_variable"], noah_cfg["incoming_sw_variable"], noah_cfg["absorbed_sw_variable"],
    ]
    with open_dataset_robust(paths["noahmp_output_file"], decode_times=False) as ds:
        missing = [name for name in required_noah if name not in ds]
        if missing:
            raise KeyError(f"Noah-MP LDASOUT is missing required variables: {missing}")
        times = decode_noah_times(ds, noah_cfg)
        if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("Decoded Noah-MP times must be valid, unique and increasing.")
        if len(times) > 1:
            diffs_hours = np.asarray((times[1:] - times[:-1]).total_seconds(), dtype=float) / 3600.0
            bad = np.flatnonzero(~np.isclose(diffs_hours, float(noah_cfg["expected_timestep_hours"]), atol=1 / 3600))
        else:
            bad = np.array([], dtype=int)
        if len(bad):
            raise ValueError(f"Decoded Noah-MP time axis contains {len(bad)} unexpected time gaps.")
        if cfg.analysis_start < times[0] or cfg.analysis_end > times[-1]:
            raise ValueError(
                f"Analysis window {cfg.analysis_start}..{cfg.analysis_end} is outside decoded Noah-MP times {times[0]}..{times[-1]}."
            )
        expected_count = noah_cfg.get("expected_record_count")
        if expected_count not in {None, ""} and len(times) != int(expected_count):
            raise ValueError(f"Noah record count {len(times)} != configured expected {expected_count}.")
        for key, actual in [("expected_first_time", times[0]), ("expected_last_time", times[-1])]:
            configured = noah_cfg.get(key)
            if configured not in {None, ""} and pd.Timestamp(configured) != actual:
                raise ValueError(f"noahmp.{key}={configured} does not match decoded Noah-MP time {actual}.")
        units = {name: variable_units(ds[name]) for name in required_noah}
        numeric_report = {}
        if str(noah_cfg.get("time_coordinate_mode", "")).lower() in {"numeric_index", "numeric", "index"}:
            raw_index = read_noah_numeric_time_index(ds, noah_cfg)
            numeric_report = {
                "raw_time_variable": str(noah_cfg.get("time_variable", "Time")),
                "raw_first_index": float(raw_index[0]) if len(raw_index) else None,
                "raw_last_index": float(raw_index[-1]) if len(raw_index) else None,
                "output_start_datetime_for_time_zero": str(noah_cfg.get("output_start_datetime")),
            }
        report["noahmp"] = {
            "source": str(paths["noahmp_output_file"]),
            "record_count": int(len(times)),
            "time_coordinate_mode": str(noah_cfg.get("time_coordinate_mode")),
            "first_decoded_time": str(times[0]),
            "last_decoded_time": str(times[-1]),
            "analysis_start": str(cfg.analysis_start),
            "analysis_end": str(cfg.analysis_end),
            "spinup_record_count": int(np.sum(times < cfg.analysis_start)),
            "daily_state_hour": int(noah_cfg["daily_state_hour"]),
            "units": units,
            **numeric_report,
        }

    s_cfg = cfg.section("snotel")
    s = pd.read_csv(paths["snotel_csv"])
    required_s = [s_cfg["date_column"], s_cfg["id_column"], s_cfg["swe_column"], s_cfg["snow_depth_column"], s_cfg["precip_column"]]
    missing_s = [c for c in required_s if c not in s.columns]
    if missing_s:
        raise KeyError(f"SNOTEL CSV is missing columns: {missing_s}")
    dates = pd.to_datetime(s[s_cfg["date_column"]], errors="coerce")
    report["snotel"] = {
        "rows": int(len(s)),
        "first_date": str(dates.min()),
        "last_date": str(dates.max()),
        "station_ids": sorted(pd.to_numeric(s[s_cfg["id_column"]], errors="coerce").dropna().astype(int).unique().tolist()),
    }

    # ASO raw files are checked by configured date/variable.  QC itself performs
    # detailed raster/unit checks.
    missing_aso: list[str] = []
    aso = cfg.section("aso")
    for date in aso["dates"]:
        for pattern in [aso["swe_pattern"], aso["snow_depth_pattern"]]:
            p = paths["aso_directory"] / pattern.format(date=date)
            if not p.exists():
                missing_aso.append(str(p))
    if missing_aso:
        raise FileNotFoundError("Missing configured ASO inputs: " + ", ".join(missing_aso))
    report["aso"] = {"configured_dates": list(aso["dates"]), "raw_file_count": int(2 * len(aso["dates"]))}

    save_json(report, cfg.output_dir / "diagnostics" / "validation_report.json")
    return report
