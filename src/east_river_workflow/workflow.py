"""Integrated orchestration for the East River Noah-MP / iSnobal workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .aso import process_aso
from .aso_qc import run_aso_quality_control
from .config import WorkflowConfig
from .data_access import read_isnobal_grid, read_noah_grid
from .grids import (
    fractional_polygon_weights,
    load_weights,
    nearest_grid_cells,
    save_grid_diagnostics,
    save_weights,
    select_watershed,
    source_centers_within_target_cell,
)
from .logging_utils import configure_logging
from .metrics import (
    calculate_h1_h2_watershed_metrics,
    calculate_h3_metrics,
    calculate_h4_metrics,
    calculate_snotel_metrics,
    streamflow_products,
)
from .plotting import generate_baseline_figures, generate_scientific_figures
from .processing import fetch_usgs_daily, process_isnobal, process_noahmp, process_snotel
from .utils import save_json
from .validation import validate_inputs


def _load_or_build_weights(cfg, grid, watershed, stem: str, logger):
    path = cfg.output_dir / "cache" / stem
    if path.exists() and not bool(cfg.section("project")["overwrite"]):
        try:
            return load_weights(path, grid)
        except ValueError as exc:
            logger.warning("Ignoring stale watershed-weight cache %s: %s", path, exc)
    logger.info("Calculating exact East River fractional weights for %s", grid.name)
    weights = fractional_polygon_weights(
        grid,
        watershed,
        chunk_size=int(cfg.section("watershed")["fractional_chunk_size"]),
    )
    save_weights(path, weights, grid)
    return weights


def prepare_spatial(cfg: WorkflowConfig, logger):
    """Prepare East River polygon, grids, exact weights and station lookups."""
    watershed = select_watershed(
        cfg.data["paths"]["watershed_shapefile"],
        cfg.section("watershed")["name_field"],
        cfg.section("watershed")["name_contains"],
    )
    isnobal_grid, isnobal_dem, isnobal_valid = read_isnobal_grid(
        cfg.data["paths"]["isnobal_topo_file"], cfg.section("isnobal")["dem_variable"]
    )
    noah_grid, noah_dem, noah_lat, noah_lon, noah_diag = read_noah_grid(
        cfg.data["paths"]["noahmp_geo_file"], cfg.section("noahmp")
    )
    i_weights = _load_or_build_weights(cfg, isnobal_grid, watershed, "east_river_fraction_isnobal.npz", logger)
    n_weights = _load_or_build_weights(cfg, noah_grid, watershed, "east_river_fraction_noahmp.npz", logger)

    stations = cfg.section("snotel")["stations"]
    lon = [float(s["longitude"]) for s in stations]
    lat = [float(s["latitude"]) for s in stations]
    i_nearest = nearest_grid_cells(isnobal_grid, lon, lat, 1, valid_mask=isnobal_valid)
    n_nearest = nearest_grid_cells(noah_grid, lon, lat, 1)

    # For each SNOTEL station, the envelope is *all* valid iSnobal pixel centers
    # that fall inside that station's nearest Noah-MP cell footprint.
    i_station_lookup: list[dict[str, Any]] = []
    lookup_rows: list[dict[str, Any]] = []
    for station, i_lookup, n_lookup in zip(stations, i_nearest, n_nearest):
        nr, nc = int(n_lookup["rows"][0]), int(n_lookup["cols"][0])
        env = source_centers_within_target_cell(
            isnobal_grid, noah_grid, nr, nc, valid_mask=isnobal_valid
        )
        if len(env["rows"]) == 0:
            raise ValueError(
                f"No valid iSnobal pixel centers fall inside the nearest Noah-MP cell for {station['name']}."
            )
        item = {
            "nearest_row": int(i_lookup["rows"][0]),
            "nearest_col": int(i_lookup["cols"][0]),
            "nearest_distance_m": float(i_lookup["distance_m"][0]),
            "envelope_rows": env["rows"],
            "envelope_cols": env["cols"],
            "noah_row": nr,
            "noah_col": nc,
        }
        i_station_lookup.append(item)
        lookup_rows.append({
            "station_id": int(station["id"]),
            "station_name": station["name"],
            "isnobal_nearest_row": item["nearest_row"],
            "isnobal_nearest_col": item["nearest_col"],
            "isnobal_nearest_distance_m": item["nearest_distance_m"],
            "noah_nearest_row": nr,
            "noah_nearest_col": nc,
            "noah_nearest_distance_m": float(n_lookup["distance_m"][0]),
            "isnobal_pixels_within_nearest_noah_cell": int(len(env["rows"])),
        })
    pd.DataFrame(lookup_rows).to_csv(cfg.output_dir / "diagnostics" / "station_grid_lookup.csv", index=False)

    # Exact mapped watershed area from grid-cell fractions (two independent grids
    # should agree closely, and the values are recorded for auditing).
    area_i = float(np.sum(i_weights) * isnobal_grid.cell_area_m2)
    area_n = float(np.sum(n_weights) * noah_grid.cell_area_m2)
    save_grid_diagnostics(
        cfg.output_dir,
        [isnobal_grid, noah_grid],
        {
            "noah_projection": noah_diag,
            "east_river_area_m2_from_isnobal_weights": area_i,
            "east_river_area_m2_from_noah_weights": area_n,
            "relative_area_difference": abs(area_i - area_n) / np.mean([area_i, area_n]),
        },
    )
    return {
        "watershed_boundary": watershed,
        "isnobal_grid": isnobal_grid,
        "noah_grid": noah_grid,
        "isnobal_dem": isnobal_dem,
        "noah_dem": noah_dem,
        "isnobal_weights": i_weights,
        "noah_weights": n_weights,
        "isnobal_lookup": i_station_lookup,
        "noah_lookup": n_nearest,
    }


def _write_final_tables(cfg: WorkflowConfig, products: dict[str, Any]) -> None:
    products["watershed_metrics"].to_csv(cfg.output_dir / "tables" / "table3_watershed_annual_metrics.csv", index=False)
    products["snotel_metrics"].to_csv(cfg.output_dir / "tables" / "table4_snotel_annual_metrics.csv", index=False)
    products["h3_metrics"].to_csv(cfg.output_dir / "tables" / "table5_h3_annual_energy.csv", index=False)
    # Table 6 publication rows are only the two common-Noah-grid validation products.
    aso = products["aso_summary"]
    table6 = aso[aso["product"].isin(["iSnobal-ASO on Noah-MP grid", "Noah-MP-ASO on Noah-MP grid"])].copy()
    table6.to_csv(cfg.output_dir / "tables" / "table6_aso_validation.csv", index=False)
    products["h4_metrics"].to_csv(cfg.output_dir / "tables" / "table7_h4_timing.csv", index=False)


def _case_year_diagnostics(cfg: WorkflowConfig, products: dict[str, Any]) -> None:
    rows = []
    for hypothesis, cases, metric in [
        ("H1", products["h1_cases"], "Delta_SWE_Mar15_mm"),
        ("H2/H3/SNOTEL", products["h2_cases"], "Delta_t50_days"),
    ]:
        rows.extend([
            {"hypothesis": hypothesis, "case": "largest_absolute_bias", "water_year": cases["largest_year"], "signed_metric_value": cases["largest_signed_value"], "metric": metric},
            {"hypothesis": hypothesis, "case": "smallest_absolute_bias", "water_year": cases["smallest_year"], "signed_metric_value": cases["smallest_signed_value"], "metric": metric},
        ])
    pd.DataFrame(rows).to_csv(cfg.output_dir / "diagnostics" / "case_year_selection.csv", index=False)
    products["h3_metrics"][["water_year", "incoming_SW_difference_MJ_m2_QC"]].to_csv(
        cfg.output_dir / "diagnostics" / "incoming_sw_qc.csv", index=False
    )


def prepare_analysis_products(cfg: WorkflowConfig, *, logger=None, run_validation: bool = True) -> dict[str, Any]:
    """Run all preprocessing and scientific metrics once; plotting reuses outputs."""
    if logger is None:
        logger = configure_logging(cfg.output_dir, cfg.section("project")["log_level"])
    validation = validate_inputs(cfg) if run_validation else None
    spatial = prepare_spatial(cfg, logger)

    i_w, i_s = process_isnobal(
        cfg,
        spatial["isnobal_grid"],
        spatial["noah_grid"],
        spatial["isnobal_weights"],
        spatial["isnobal_lookup"],
        logger,
    )
    n_w, n_s, noah_diag = process_noahmp(
        cfg,
        spatial["noah_grid"],
        spatial["noah_weights"],
        spatial["noah_lookup"],
        logger,
    )
    save_json(noah_diag, cfg.output_dir / "diagnostics" / "noahmp_processing_diagnostics.json")
    snotel = process_snotel(cfg, i_s, logger)
    usgs = fetch_usgs_daily(cfg, logger)

    aso_qc_manifest = None
    if bool(cfg.section("aso")["quality_control"]["enabled"]):
        aso_qc_manifest = run_aso_quality_control(cfg, spatial["watershed_boundary"], logger)

    aso_products, aso_summary = process_aso(
        cfg,
        spatial["isnobal_grid"], spatial["noah_grid"],
        spatial["isnobal_dem"], spatial["noah_dem"],
        spatial["isnobal_weights"], spatial["noah_weights"], logger,
    )

    watershed_metrics, h1_cases, h2_cases = calculate_h1_h2_watershed_metrics(
        cfg, i_w, n_w, spatial["noah_weights"]
    )
    snotel_metrics = calculate_snotel_metrics(cfg, snotel, i_s, n_s)
    h3_metrics = calculate_h3_metrics(cfg, i_w, n_w)
    h4_metrics = calculate_h4_metrics(cfg, i_w, n_w, usgs)
    streamflow = streamflow_products(cfg, i_w, n_w, usgs)
    streamflow.to_csv(cfg.output_dir / "tables" / "streamflow_daily_products.csv", index=False)

    products = {
        "validation": validation,
        "logger": logger,
        **spatial,
        "isnobal_watershed": i_w,
        "isnobal_station": i_s,
        "noah_watershed": n_w,
        "noah_station": n_s,
        "noah_diagnostics": noah_diag,
        "snotel": snotel,
        "usgs": usgs,
        "aso_products": aso_products,
        "aso_summary": aso_summary,
        "aso_qc_manifest": aso_qc_manifest,
        "watershed_metrics": watershed_metrics,
        "h1_cases": h1_cases,
        "h2_cases": h2_cases,
        "snotel_metrics": snotel_metrics,
        "h3_metrics": h3_metrics,
        "h4_metrics": h4_metrics,
        "streamflow": streamflow,
    }
    _write_final_tables(cfg, products)
    _case_year_diagnostics(cfg, products)
    return products


def run_all(cfg: WorkflowConfig) -> dict[str, Any]:
    """Run the complete integrated analysis, baseline figures, and scientific figures."""
    products = prepare_analysis_products(cfg)
    baseline = generate_baseline_figures(cfg, products)
    scientific = generate_scientific_figures(cfg, products)

    qc_figures: list[str] = []
    if products.get("aso_qc_manifest"):
        manifest = products["aso_qc_manifest"]
        # run_aso_quality_control returns a dict in current code; collect any paths
        # recursively without assuming a specific manifest schema.
        def collect(obj):
            if isinstance(obj, dict):
                for v in obj.values(): collect(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj: collect(v)
            elif isinstance(obj, str) and obj.lower().endswith((".png", ".jpg", ".jpeg")):
                qc_figures.append(obj)
        collect(manifest)

    manifest = {
        "workflow_version": __version__,
        "analysis_start": str(cfg.analysis_start),
        "analysis_end": str(cfg.analysis_end),
        "baseline_figures": [str(p) for p in baseline],
        "scientific_figures": [str(p) for p in scientific],
        "aso_qc_figure_paths_found_in_manifest": qc_figures,
        "tables": [
            str(cfg.output_dir / "tables" / f) for f in [
                "table3_watershed_annual_metrics.csv",
                "table4_snotel_annual_metrics.csv",
                "table5_h3_annual_energy.csv",
                "table6_aso_validation.csv",
                "table7_h4_timing.csv",
            ]
        ],
        "daily_tables": [
            str(cfg.output_dir / "tables" / f) for f in [
                "isnobal_watershed_daily.csv", "noahmp_watershed_daily.csv",
                "isnobal_station_daily.csv", "noahmp_station_daily.csv", "snotel_daily.csv",
            ]
        ],
        "case_years": {"H1": products["h1_cases"], "H2_H3_SNOTEL": products["h2_cases"]},
    }
    save_json(manifest, cfg.output_dir / "output_manifest.json")
    products["output_manifest"] = manifest
    return products

# Backward-compatible helper for the plotting workbench.
def generate_all_figures(cfg: WorkflowConfig, products: dict[str, Any]):
    return generate_baseline_figures(cfg, products) + generate_scientific_figures(cfg, products)
