"""Configuration loading and validation for the integrated East River workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULTS: dict[str, Any] = {
    "project": {
        "analysis_start_datetime": "2017-10-01 00:00:00",
        "analysis_end_datetime": "2026-06-30 23:00:00",
        "water_years": list(range(2018, 2027)),
        "overwrite": False,
        "log_level": "INFO",
    },
    "watershed": {
        "name_field": "name",
        "name_contains": "East River",
        "fractional_chunk_size": 50000,
    },
    "isnobal": {
        "daily_folder_pattern": "run{date:%Y%m%d}",
        "snow_filename": "snow.nc",
        "precip_filename": "precip.nc",
        "percent_snow_filename": "percent_snow.nc",
        "energy_filename": "em.nc",
        "incoming_sw_filename": "hrrr_solar.nc",
        "absorbed_sw_filename": "net_solar.nc",
        "swe_variable": "specific_mass",
        "snow_depth_variable": "thickness",
        "precip_variable": "precip",
        "percent_snow_variable": "percent_snow",
        "swi_variable": "SWI",
        "snowmelt_variable": "snowmelt",
        "incoming_sw_variable": "hrrr_solar",
        "absorbed_sw_variable": "net_solar",
        "dem_variable": "dem",
        "time_index": -1,
        "missing_file_policy": "error",
        "expected_hourly_timesteps": 24,
        "minimum_regrid_valid_coverage": 0.50,
    },
    "noahmp": {
        # Production LDASOUT uses Time = 0, 1, 2, ... rather than datetimes.
        # output_start_datetime is the real datetime represented by Time=0.
        "time_coordinate_mode": "numeric_index",
        "time_variable": "Time",
        "output_start_datetime": "2016-10-01 01:00:00",
        "expected_timestep_hours": 1,
        "daily_state_hour": 23,
        "expected_record_count": None,
        "expected_first_time": None,
        "expected_last_time": None,
        "daily_incomplete_policy": "error",
        "swe_variable": "SNEQV",
        "snow_depth_variable": "SNOWH",
        "precip_variable": "RAINRATE",
        "snowfall_variable": "QSNOW",
        "melt_variable": "QMELT",
        "snow_bottom_release_variable": "QSNBOT",
        "incoming_sw_variable": "SWFORC",
        "absorbed_sw_variable": "FSA",
        "latitude_variable": "XLAT_M",
        "longitude_variable": "XLONG_M",
        "elevation_variable": "HGT_M",
    },
    "h1": {
        "start_month_day": "11-01",
        "end_month_day": "03-15",
        "event_threshold_mm_day": 1.0,
    },
    "h2": {
        "fractions": [0.75, 0.50, 0.25],
        "event_threshold_mm_day": 1.0,
        "disappearance_threshold_swe_mm": 10.0,
        "disappearance_consecutive_days": 7,
        "ablation_start_month_day": "04-01",
        "ablation_end_month_day": "06-30",
    },
    "snow_disappearance": {
        "threshold_depth_mm": 25.0,
        "consecutive_days": 7,
    },
    "h3": {
        "start_month_day": "04-01",
        "end_month_day": "06-30",
        "incoming_sw_minimum_mj_m2_day": 0.05,
        "melt_day_threshold_mm": 0.1,
    },
    "h4": {
        "season_start_month_day": "03-01",
        "season_end_month_day": "09-30",
    },
    "aso": {
        "dates": ["20180331", "20180524", "20190407", "20190610"],
        "swe_pattern": "ASO_50M_SWE_USCOGE_{date}.tif",
        "snow_depth_pattern": "ASO_50M_SD_USCOGE_{date}.tif",
        "minimum_valid_coverage": 0.50,
        "map_robust_percentile": 99.0,
        # Categorical map breaks.  null => derive clean common breaks from the
        # configured ASO comparison set.  Supply explicit lists to lock scales.
        "swe_state_breaks_mm": None,
        "snow_depth_state_breaks_mm": None,
        "swe_difference_breaks_mm": None,
        "snow_depth_difference_breaks_mm": None,
        "quality_control": {
            "enabled": True,
            "negative_clip_tolerance_mm": 1.0,
            "swe_hard_max_mm": 5000.0,
            "snow_depth_hard_max_mm": 15000.0,
            "display_percentile": 99.5,
            "top_values_per_raster": 20,
            "maximum_adjusted_cell_records": 100000,
            "density_minimum_snow_depth_mm": 100.0,
            "density_review_min_kg_m3": 50.0,
            "density_review_max_kg_m3": 700.0,
            "density_hard_max_kg_m3": 1000.0,
        },
    },
    "snotel": {
        "date_column": "Date",
        "id_column": "ID",
        "precip_column": "Precipitation Increment (mm)",
        "swe_column": "Snow Water Equivalent (mm)",
        "snow_depth_column": "Snow Depth (mm)",
        "gapfill_precip_with_isnobal": True,
        "stations": [
            {"id": 380, "name": "Butte", "latitude": 38.89433, "longitude": -106.95300, "elevation_ft": 10160},
            {"id": 737, "name": "Schofield Pass", "latitude": 39.01522, "longitude": -107.04877, "elevation_ft": 10700},
            {"id": 1141, "name": "Upper Taylor", "latitude": 38.99077, "longitude": -106.75422, "elevation_ft": 10640},
        ],
    },
    "streamflow": {
        "usgs_site": "09112500",
        "parameter_code": "00060",
        "statistic_code": "00003",
        "rolling_days": 7,
        "api_mode": "modern_with_legacy_fallback",
        "api_key_environment_variable": "USGS_API_KEY",
        "request_timeout_seconds": 120,
        "page_size": 10000,
    },
    "plotting": {
        "font_family": "Arial",
        "fallback_font": "DejaVu Sans",
        "font_size": 18,
        "dpi": 300,
        "figure_format": "png",
        "close_after_save": True,
        "line_width": 1.6,
        "watershed_swe_ymax_mm": None,
        "watershed_snow_depth_ymax_mm": None,
        "watershed_swe_y_step_mm": 100.0,
        "watershed_snow_depth_y_step_mm": 100.0,
        "station_swe_ymax_mm": None,
        "station_snow_depth_ymax_mm": None,
    },
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _normalise_legacy_keys(user: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(user)
    project = result.setdefault("project", {})
    if "analysis_start_datetime" not in project and "analysis_start_date" in project:
        project["analysis_start_datetime"] = f"{project['analysis_start_date']} 00:00:00"
    if "analysis_end_datetime" not in project and "analysis_end_date" in project:
        project["analysis_end_datetime"] = f"{project['analysis_end_date']} 23:00:00"
    return result


@dataclass(frozen=True)
class WorkflowConfig:
    data: dict[str, Any]
    source_path: Path

    @property
    def paths(self) -> dict[str, Path]:
        return {k: Path(v).expanduser() for k, v in self.data["paths"].items()}

    @property
    def output_dir(self) -> Path:
        return self.paths["output_dir"]

    @property
    def analysis_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.section("project")["analysis_start_datetime"])

    @property
    def analysis_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.section("project")["analysis_end_datetime"])

    def section(self, name: str) -> dict[str, Any]:
        return self.data[name]

    def create_output_tree(self) -> None:
        for child in ["logs", "cache", "tables", "figures", "regridded", "diagnostics"]:
            (self.output_dir / child).mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures" / "scientific").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures" / "baseline").mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> WorkflowConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as f:
        user = _normalise_legacy_keys(yaml.safe_load(f) or {})
    merged = _deep_merge(DEFAULTS, user)

    required_paths = {
        "watershed_shapefile",
        "isnobal_root_template",
        "isnobal_topo_file",
        "noahmp_output_file",
        "noahmp_geo_file",
        "snotel_csv",
        "aso_directory",
        "output_dir",
    }
    if "paths" not in merged:
        raise ValueError("The configuration must contain a 'paths' section.")
    missing = sorted(required_paths - set(merged["paths"]))
    if missing:
        raise ValueError(f"Missing required path settings: {missing}")

    start = pd.Timestamp(merged["project"]["analysis_start_datetime"])
    end = pd.Timestamp(merged["project"]["analysis_end_datetime"])
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("project analysis start/end datetimes are invalid or reversed.")

    noah = merged["noahmp"]
    mode = str(noah["time_coordinate_mode"]).lower()
    if mode not in {"numeric_index", "numeric", "index", "wrf_times", "times", "auto"}:
        raise ValueError(
            "noahmp.time_coordinate_mode must be 'numeric_index', 'wrf_times', or 'auto'."
        )
    if mode in {"numeric_index", "numeric", "index"}:
        base = noah.get("output_start_datetime")
        if base in {None, ""}:
            raise ValueError(
                "noahmp.output_start_datetime is required for numeric_index time decoding."
            )
        try:
            parsed_base = pd.Timestamp(base)
        except Exception as exc:
            raise ValueError(f"Invalid noahmp.output_start_datetime: {base!r}") from exc
        if pd.isna(parsed_base):
            raise ValueError(f"Invalid noahmp.output_start_datetime: {base!r}")
    if float(noah["expected_timestep_hours"]) <= 0:
        raise ValueError("noahmp.expected_timestep_hours must be positive.")
    if int(noah["daily_state_hour"]) not in range(24):
        raise ValueError("noahmp.daily_state_hour must be 0..23.")
    if str(noah["daily_incomplete_policy"]).lower() not in {"error", "warn", "allow"}:
        raise ValueError("noahmp.daily_incomplete_policy must be error, warn, or allow.")

    years = [int(x) for x in merged["project"]["water_years"]]
    if not years:
        raise ValueError("project.water_years must not be empty.")
    merged["project"]["water_years"] = years
    # Friendly aliases for older utility scripts.
    merged["project"]["analysis_start_date"] = str(start.date())
    merged["project"]["analysis_end_date"] = str(end.date())

    cfg = WorkflowConfig(merged, source)
    cfg.create_output_tree()
    return cfg
