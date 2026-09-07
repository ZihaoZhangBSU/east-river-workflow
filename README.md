# East River Workflow v0.8.2

Integrated East River workflow for comparing iSnobal, Noah-MP, SNOTEL, ASO, and USGS. This version implements the consolidated v0.7.0 analysis specification rather than separate Step 1 / Step 2 scripts.

## Major corrections in v0.8.2

- Noah-MP is **one configurable LDASOUT file with 85,439 hourly records**. Its production `Time` coordinate is numeric (`0, 1, 2, ..., 85438`), not a datetime. The code reconstructs datetimes from configurable `output_start_datetime` (the datetime represented by `Time=0`) and `expected_timestep_hours`. The filename is not parsed as the clock.
- The configured analysis window defaults to **2017-10-01 00:00 through 2026-06-30 23:00**; earlier Noah records are spin-up.
- Daily Noah-MP `SNEQV`/`SNOWH` are taken at **hour 23**. Hourly `RAINRATE`, `QSNOW`, `QMELT`, `QSNBOT`, `SWFORC`, and `FSA` are aggregated across complete 00–23 days.
- Noah precipitation is read directly from **LDASOUT `RAINRATE`**. `mm/timestep` is summed directly; `mm/s` is integrated by timestep duration.
- The workflow reads the full `WBDHU10.shp` layer but uses only the uniquely selected **East River** feature for scientific calculations.
- Exact fractional East River polygon/cell overlap weights are used for watershed averages.
- SNOTEL direct SWE **and snow-depth** comparison plots contain SNOTEL, nearest iSnobal pixel, nearest Noah-MP pixel, and the **min–max envelope of every valid iSnobal pixel center inside the station's nearest Noah-MP cell**.
- Baseline watershed SWE/snow-depth 3×3 figures annotate Pearson `r` and mean bias (`Noah-MP - iSnobal`) in every WY panel and use a tight shared y range.
- H1 uses **Nov 1–Mar 15** and the exact initial-state/snowfall/retention decomposition.
- H2 uses each series' own peak SWE for normalized depletion.
- H3 uses effective broadband albedo from incoming/absorbed shortwave; SCF is removed and incoming SW is not a headline attribution term.
- H4 uses **iSnobal `SWI`, Noah-MP `QSNBOT`, and USGS**. Surface/subsurface runoff is not required.
- ASO QC is run before ASO comparison; cleaned rasters are used downstream and the **All** elevation class is included.
- ASO state maps use **ASO / iSnobal / Noah-MP on the common Noah-MP grid**; difference maps use **iSnobal−Noah-MP / iSnobal−ASO / Noah-MP−ASO**, with categorical color bars. Categorical break lists can be fixed in YAML or derived consistently across all configured ASO dates.
- The default delivery contains **26 retained baseline review figures + 15 scientific figures + 8 ASO-QC figures** for the current 3-station/4-ASO-date configuration.

## Installation

```bash
conda env create -f environment.yml
conda activate east-river-workflow
pip install -e .
```

or install into an existing Python environment:

```bash
pip install -e .
```

## Configuration

Copy and edit:

```bash
cp config/east_river_config.example.yaml config/east_river_config.yaml
```

At minimum, update the paths in `paths:`. For the production Noah-MP file, keep `time_coordinate_mode: numeric_index`, `time_variable: Time`, and set `output_start_datetime` to the real datetime represented by `Time=0`. The Noah-MP filename/path, reference datetime, timestep, and analysis start/end are configuration values and can be changed without editing source code.

## Run the complete workflow

```bash
east-river validate --config config/east_river_config.yaml
east-river run-all --config config/east_river_config.yaml
```

`run-all` performs the complete integrated workflow: validation, spatial setup, iSnobal and Noah preprocessing, SNOTEL/USGS processing, ASO QC, ASO comparison, H1–H4 metrics, Tables 3–7, all baseline review figures, all scientific figures, diagnostics, and the output manifest.

## Important SNOTEL spatial comparison rule

For each station the code finds:

1. the nearest iSnobal pixel to the station;
2. the nearest Noah-MP pixel to the station;
3. every valid iSnobal pixel center that falls inside that nearest Noah-MP cell.

The direct station figures show the nearest iSnobal and Noah-MP series plus the daily iSnobal min–max range from item 3. This rule applies to both SWE and snow depth.

## Retained station diagnostics

The original station diagnostics are retained. In particular, `f_SWE,precip` is computed separately for each source as its own annual maximum SWE divided by precipitation accumulated from October 1 through that source's own maximum-SWE date. The ratio is left missing if that precipitation interval is not calendar-complete. The original snow-disappearance diagnostic remains snow-depth based (25 mm for 7 consecutive days by default), while H2 also retains its SWE-based depletion/disappearance metrics.

## Main processed tables

- `tables/isnobal_watershed_daily.csv`
- `tables/noahmp_watershed_daily.csv`
- `tables/isnobal_station_daily.csv`
- `tables/noahmp_station_daily.csv`
- `tables/snotel_daily.csv`
- `tables/table3_watershed_annual_metrics.csv`
- `tables/table4_snotel_annual_metrics.csv`
- `tables/table5_h3_annual_energy.csv`
- `tables/table6_aso_validation.csv`
- `tables/table7_h4_timing.csv`

## Figure directories

- `figures/baseline/` — retained original/review figure family (26 under current configuration)
- `figures/scientific/` — final H1–H4 figure sequence (15)
- `diagnostics/aso_qc/figures/` — ASO quality-control diagnostic figures

## Testing

```bash
pytest -q
```

The package-level test suite currently has **22 passing tests** covering numeric-index Noah time reconstruction (including the full 85,439-record range), backward-compatible WRF `Times` decoding, unit-aware hourly aggregation, event filtering/sign convention, H1 decomposition closure, source-specific `f_SWE,precip`, absolute-magnitude case selection, exact iSnobal-in-Noah-cell envelope geometry, and ASO hard-invalid QC behavior.
