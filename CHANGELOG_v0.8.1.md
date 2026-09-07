# v0.8.1 hotfix

- Fixed false Noah-MP time-gap failures caused by assuming `DatetimeIndex.view("i8")` is always nanoseconds.
- All timestep calculations now use resolution-independent timedeltas via `total_seconds()`.
- Applied the same correction to Noah-MP validation/processing and iSnobal shortwave timestep handling.

# Changelog v0.8.0

- Replaced legacy 24-hour/numeric Noah-MP time decoding with hourly internal WRF `Times` from one LDASOUT file.
- Added configurable analysis window and spin-up exclusion.
- Added hour-23 Noah daily state extraction and 00–23 flux aggregation.
- Replaced LDASIN precipitation reads with LDASOUT `RAINRATE`.
- Added `QSNOW`, `QMELT`, `QSNBOT`, `SWFORC`, and `FSA` daily processing.
- Removed required `SFCRNOFF`/`UGDRNOFF` from H4.
- Made East River selection explicit and retained exact fractional polygon-cell watershed weights.
- Replaced 20-nearest-iSnobal station envelope with the requested exact set of iSnobal pixel centers inside the nearest Noah-MP cell; applies to SWE and snow depth.
- Added panel-level Pearson r and Noah-MP-minus-iSnobal mean bias to baseline watershed 3×3 figures and tight shared y-axis scaling.
- Added complete H1–H4 metric workflow, deterministic case-year selection, Tables 3–7, 15 scientific plots, and corrected labels.
- Retained all 26 baseline review plots under the current configuration.
- Retained conservative ASO QC and added All elevation category to ASO comparison.
- Updated configuration, validation, CLI, tests, README, and output manifest.

- Corrected ASO map families to the requested common-Noah-grid columns and categorical color bars.
- Restored the original source-specific `f_SWE,precip` definition (Oct 1 through each source's own peak SWE) and depth-based snow-disappearance review metric.
- Added tests for H1 algebraic closure and source-specific `f_SWE,precip`.
