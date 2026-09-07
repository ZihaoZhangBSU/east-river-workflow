# East River Workflow v0.8.2 — Implementation notes

This package is the integrated implementation of the reviewed East River analysis design. It is not split into Step 1 / Step 2 scripts.

## Core production conventions

- Noah-MP is one configurable LDASOUT file containing 85,439 hourly records.
- Production `Time` values are numeric indices (`0..85438`). The clock is reconstructed as `output_start_datetime + Time * expected_timestep_hours`; the LDASOUT filename is not parsed as the time coordinate.
- The default analysis period is 2017-10-01 00:00 through 2026-06-30 23:00; earlier records are spin-up.
- Noah-MP daily SWE (`SNEQV`) and snow depth (`SNOWH`) use the decoded 23:00 record.
- Noah-MP daily precipitation, snowfall, melt, snow-bottom release, and shortwave quantities use complete 00–23 hourly aggregation.
- LDASOUT `RAINRATE` is the Noah-MP precipitation source. `mm/timestep` is summed directly; rate units such as `mm/s` are integrated by the actual timestep.
- The full WBDHU10 layer is read, but only the uniquely selected East River feature is used for watershed calculations.
- Watershed averages use exact fractional East River polygon/grid-cell overlap weights.
- H4 uses iSnobal `SWI`, Noah-MP `QSNBOT`, and USGS discharge. Surface/subsurface runoff is not required.

## SNOTEL station representation

For each SNOTEL station the code identifies:

1. the nearest iSnobal pixel;
2. the nearest Noah-MP pixel;
3. every valid iSnobal pixel center located inside that nearest Noah-MP cell.

The direct SWE and snow-depth figures plot SNOTEL, nearest iSnobal, nearest Noah-MP, and the daily min–max iSnobal envelope from item 3.

## H1-H4

- H1: Nov 1–Mar 15 fixed accounting, exact initial-state/snowfall/retention decomposition, paired accumulation events.
- H2: each source normalized by its own peak SWE; 75/50/25% depletion timing and paired ablation events.
- H3: effective broadband albedo derived from incoming/absorbed shortwave, absorbed shortwave, melt, and SWE-loss consistency. SCF is excluded. Incoming shortwave is an input/QC quantity rather than a headline attribution panel.
- H4: SWE50 → SWI/QSNBOT release50 → USGS50 timing.

## ASO

ASO quality control runs before scientific comparison and never overwrites source rasters. The publication comparison uses the common Noah-MP grid and the East River mask. State-map columns are ASO / iSnobal / Noah-MP. Difference-map columns are iSnobal−Noah-MP / iSnobal−ASO / Noah-MP−ASO. Categorical color breaks are shared within comparison families. Elevation summaries include Lower, Middle, Upper, and All.

## Figures and tables

Under the current 3-station / 4-ASO-date configuration, the workflow targets:

- 26 retained baseline review figures;
- 15 H1-H4 scientific figures;
- 8 ASO-QC diagnostic figures;
- Tables 3–7 and supporting daily/diagnostic tables.

The baseline watershed SWE and snow-depth 3×3 figures annotate Pearson `r` and mean bias (`Noah-MP - iSnobal`) in each WY panel and use configurable tight shared y-axis ranges.

## Validation status in the delivered development environment

- Python compilation: passed.
- Automated tests: 15 passed.
- Synthetic plotting smoke checks: baseline figure family and all 15 scientific plotting functions were exercised in batches; the modified event/mechanism figures were rechecked after the final date-window/statistics corrections.
- Supplied `WBDHU10.shp`: exact East River selection was checked and returned one feature.

A complete production run was not performed in the delivery environment because the full production Noah-MP/iSnobal/ASO data paths are HPC-local. Run `east-river validate` before the production run and inspect the generated diagnostics.
