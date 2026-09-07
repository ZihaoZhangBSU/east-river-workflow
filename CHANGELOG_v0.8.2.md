# v0.8.2 numeric Noah-MP Time-index update

This release corrects the production Noah-MP time model after confirming that the single LDASOUT file stores:

- `Time = 0, 1, 2, ..., 85438`
- 85,439 total hourly records
- `Time=0` corresponds to `2016-10-01 01:00:00`
- the final index `85438` therefore corresponds to `2026-06-30 23:00:00`

## Code changes

- Added `numeric_index` Noah-MP time decoding.
- Added configurable `noahmp.output_start_datetime`; it is the datetime represented by `Time=0`.
- `expected_timestep_hours` controls the duration represented by one numeric index increment.
- The numeric `Time` axis is validated as finite, one-dimensional, numeric, and incrementing by exactly one record.
- Real datetimes are reconstructed before daily hour-23 selection or 00–23 flux aggregation.
- Validation reports both the raw numeric index range and the reconstructed datetime range.
- Backward-compatible WRF character `Times` support is retained.
- Time diagnostics no longer assume that the production file contains an internal character `Times` variable.
- The package version and output-manifest version are now `0.8.2`.

## Production config

The shipped `config/east_river_config.yaml` uses:

```yaml
noahmp:
  time_coordinate_mode: numeric_index
  time_variable: Time
  output_start_datetime: "2016-10-01 01:00:00"
  expected_timestep_hours: 1
  daily_state_hour: 23
  expected_record_count: 85439
```

The scientific analysis still starts at `2017-10-01 00:00:00`; earlier decoded Noah-MP records are spin-up.

## Tests

The test suite contains 22 passing tests, including a full-range regression check that index `85438` reconstructs to `2026-06-30 23:00:00`.
