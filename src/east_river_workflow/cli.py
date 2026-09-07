"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .logging_utils import configure_logging
from .validation import validate_inputs
from .workflow import prepare_spatial, run_all
from .aso_qc import run_aso_quality_control
from .grids import select_watershed
from .plotting import configure_matplotlib


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="East River hydrologic comparison workflow")
    result.add_argument("command", choices=["validate", "prepare-spatial", "aso-qc", "run-all"])
    result.add_argument("--config", required=True, help="Path to the YAML configuration file")
    return result


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "validate":
        report = validate_inputs(cfg)
        print(json.dumps(report, indent=2, default=str))
    elif args.command == "prepare-spatial":
        logger = configure_logging(cfg.output_dir, cfg.section("project")["log_level"])
        prepare_spatial(cfg, logger)
        print(f"Spatial preparation complete: {cfg.output_dir}")
    elif args.command == "aso-qc":
        logger = configure_logging(cfg.output_dir, cfg.section("project")["log_level"])
        configure_matplotlib(cfg)
        watershed = select_watershed(
            cfg.data["paths"]["watershed_shapefile"],
            cfg.section("watershed")["name_field"],
            cfg.section("watershed")["name_contains"],
        )
        manifest = run_aso_quality_control(cfg, watershed, logger)
        print(json.dumps(manifest, indent=2, default=str))
    else:
        products = run_all(cfg)
        print(json.dumps(products["output_manifest"], indent=2, default=str))


if __name__ == "__main__":
    main()
