#!/usr/bin/env python3
"""
Standalone H1 Xu first-level decomposition for the East River project.

Purpose
-------
This script does NOT modify the East River Workflow package. It reads the
existing watershed-daily CSV outputs for iSnobal and Noah-MP and computes the
H1 accumulation/ablation decomposition independently.

East River H1 window
--------------------
Initial SWE state: October 1 (end-of-day state)
Snowfall flux window: October 2 through March 15
Endpoint SWE state: March 15

The October 2 flux start is intentional. Because the October 1 SWE value is an
end-of-day state, October 1 snowfall is already reflected in that state and
must not be counted again after the initial-SWE correction.

Xu first-level framework with East River initial-SWE correction
---------------------------------------------------------------
For iSnobal (I, reference) and Noah-MP (N):

    G_I = SWE_Mar15_I - SWE_Oct01_I
    G_N = SWE_Mar15_N - SWE_Oct01_N

    S_I = cumulative snowfall_I from Oct 2 through Mar 15
    S_N = cumulative snowfall_N from Oct 2 through Mar 15

    M_I = 1 - G_I / S_I
    M_N = 1 - G_N / S_N

    epsilon_total = G_N / G_I - 1
    epsilon_A     = S_N / S_I - 1
    epsilon_M     = (M_I - M_N) / (1 - M_I)

Exact multiplicative closure:

    1 + epsilon_total = (1 + epsilon_A) * (1 + epsilon_M)

Notes
-----
- M is the Xu-style inferred ablation fraction, not an explicit model melt flux.
- M is not clipped to [0, 1]. Values outside that interval are retained and
  flagged because they reveal a mass-balance inconsistency or unusual year.
- The script requires daily columns: date, swe_mm, snowfall_mm.
- A water_year column is used when present; otherwise it is derived from date.

Example
-------
python h1_xu_first_level_standalone.py \
    --isnobal output/output1/tables/isnobal_watershed_daily.csv \
    --noah output/output1/tables/noahmp_watershed_daily.csv \
    --output h1_xu_first_level_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "swe_mm", "snowfall_mm"}


def water_year(dates: pd.Series) -> pd.Series:
    """Return water-year labels: Oct-Dec belong to the following calendar year."""
    d = pd.to_datetime(dates)
    return d.dt.year + (d.dt.month >= 10).astype(int)


def season_date(wy: int, month: int, day: int) -> pd.Timestamp:
    """Return a calendar date inside the specified water year."""
    year = wy - 1 if month >= 10 else wy
    return pd.Timestamp(year=year, month=month, day=day)


def load_daily(path: str | Path, source_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{source_name} file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise ValueError(f"{source_name} contains {bad} unparseable date value(s).")

    for col in ["swe_mm", "snowfall_mm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "water_year" not in df.columns:
        df["water_year"] = water_year(df["date"])
    else:
        df["water_year"] = pd.to_numeric(df["water_year"], errors="coerce").astype("Int64")

    if df["date"].duplicated().any():
        dup = df.loc[df["date"].duplicated(keep=False), "date"].dt.strftime("%Y-%m-%d")
        raise ValueError(
            f"{source_name} contains duplicate dates; first examples: "
            f"{dup.head(10).tolist()}"
        )

    return df.sort_values("date").reset_index(drop=True)


def value_on_date(df: pd.DataFrame, date: pd.Timestamp, column: str) -> float:
    row = df.loc[df["date"].eq(date), column]
    if len(row) != 1:
        return np.nan
    value = row.iloc[0]
    return float(value) if pd.notna(value) else np.nan


def complete_sum(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    column: str,
) -> tuple[float, bool, int, int]:
    """
    Sum a daily flux only when every calendar day is present and finite.

    Returns
    -------
    total, complete, n_expected, n_valid
    """
    expected = pd.date_range(start, end, freq="D")
    s = df.set_index("date")[column].reindex(expected)
    valid = s.notna()
    complete = bool(valid.all())
    total = float(s.sum()) if complete else np.nan
    return total, complete, len(expected), int(valid.sum())


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


def calculate_one_year(
    wy: int,
    isnobal: pd.DataFrame,
    noah: pd.DataFrame,
) -> dict:
    start = season_date(wy, 10, 1)
    flux_start = start + pd.Timedelta(days=1)
    end = season_date(wy, 3, 15)

    i = isnobal[isnobal["water_year"].eq(wy)].copy()
    n = noah[noah["water_year"].eq(wy)].copy()

    i_start = value_on_date(i, start, "swe_mm")
    n_start = value_on_date(n, start, "swe_mm")
    i_end = value_on_date(i, end, "swe_mm")
    n_end = value_on_date(n, end, "swe_mm")

    sf_i, sf_i_complete, n_expected_i, n_valid_i = complete_sum(
        i, flux_start, end, "snowfall_mm"
    )
    sf_n, sf_n_complete, n_expected_n, n_valid_n = complete_sum(
        n, flux_start, end, "snowfall_mm"
    )

    gain_i = i_end - i_start if np.all(np.isfinite([i_end, i_start])) else np.nan
    gain_n = n_end - n_start if np.all(np.isfinite([n_end, n_start])) else np.nan

    m_i = 1.0 - safe_ratio(gain_i, sf_i) if np.isfinite(sf_i) else np.nan
    m_n = 1.0 - safe_ratio(gain_n, sf_n) if np.isfinite(sf_n) else np.nan

    epsilon_total = safe_ratio(gain_n, gain_i) - 1.0 if np.isfinite(safe_ratio(gain_n, gain_i)) else np.nan
    epsilon_accum = safe_ratio(sf_n, sf_i) - 1.0 if np.isfinite(safe_ratio(sf_n, sf_i)) else np.nan

    if np.all(np.isfinite([m_i, m_n])) and (1.0 - m_i) != 0:
        epsilon_ablation = float((m_i - m_n) / (1.0 - m_i))
    else:
        epsilon_ablation = np.nan

    if np.all(np.isfinite([epsilon_accum, epsilon_ablation])):
        epsilon_reconstructed = (
            (1.0 + epsilon_accum) * (1.0 + epsilon_ablation) - 1.0
        )
    else:
        epsilon_reconstructed = np.nan

    xu_closure = (
        epsilon_total - epsilon_reconstructed
        if np.all(np.isfinite([epsilon_total, epsilon_reconstructed]))
        else np.nan
    )

    delta_start = n_start - i_start if np.all(np.isfinite([n_start, i_start])) else np.nan
    delta_end = n_end - i_end if np.all(np.isfinite([n_end, i_end])) else np.nan
    delta_gain = gain_n - gain_i if np.all(np.isfinite([gain_n, gain_i])) else np.nan

    initial_state_closure = (
        delta_end - (delta_start + delta_gain)
        if np.all(np.isfinite([delta_end, delta_start, delta_gain]))
        else np.nan
    )

    return {
        "water_year": int(wy),
        "H1_start_state_date": start.date().isoformat(),
        "H1_flux_start_date": flux_start.date().isoformat(),
        "H1_end_state_date": end.date().isoformat(),

        "SWE_Oct01_I_mm": i_start,
        "SWE_Oct01_N_mm": n_start,
        "SWE_Mar15_I_mm": i_end,
        "SWE_Mar15_N_mm": n_end,

        "SWE_gain_Oct01_Mar15_I_mm": gain_i,
        "SWE_gain_Oct01_Mar15_N_mm": gain_n,

        "Snowfall_Oct02_Mar15_I_mm": sf_i,
        "Snowfall_Oct02_Mar15_N_mm": sf_n,
        "snowfall_complete_I": sf_i_complete,
        "snowfall_complete_N": sf_n_complete,
        "snowfall_expected_days_I": n_expected_i,
        "snowfall_valid_days_I": n_valid_i,
        "snowfall_expected_days_N": n_expected_n,
        "snowfall_valid_days_N": n_valid_n,

        "M_ablation_I": m_i,
        "M_ablation_N": m_n,

        "Delta_SWE_Oct01_N_minus_I_mm": delta_start,
        "Delta_SWE_Mar15_N_minus_I_mm": delta_end,
        "Delta_corrected_SWE_gain_N_minus_I_mm": delta_gain,

        "epsilon_total_corrected": epsilon_total,
        "epsilon_accumulation": epsilon_accum,
        "epsilon_ablation": epsilon_ablation,
        "epsilon_total_reconstructed": epsilon_reconstructed,

        "initial_state_closure_error_mm": initial_state_closure,
        "xu_multiplicative_closure_error": xu_closure,

        "M_I_outside_0_1": bool(np.isfinite(m_i) and (m_i < 0 or m_i > 1)),
        "M_N_outside_0_1": bool(np.isfinite(m_n) and (m_n < 0 or m_n > 1)),
    }


def calculate_h1(
    isnobal: pd.DataFrame,
    noah: pd.DataFrame,
    years: list[int] | None = None,
) -> pd.DataFrame:
    if years is None:
        years = sorted(
            set(isnobal["water_year"].dropna().astype(int))
            & set(noah["water_year"].dropna().astype(int))
        )

    rows = [calculate_one_year(int(wy), isnobal, noah) for wy in years]
    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("No common water years were found.")

    # Warn rather than silently alter unusual Xu M values.
    flagged = out[out["M_I_outside_0_1"] | out["M_N_outside_0_1"]]
    if not flagged.empty:
        warnings.warn(
            "Xu-style M falls outside [0, 1] for water year(s): "
            + ", ".join(flagged["water_year"].astype(str))
            + ". Values are retained without clipping."
        )

    incomplete = out[~(out["snowfall_complete_I"] & out["snowfall_complete_N"])]
    if not incomplete.empty:
        warnings.warn(
            "Incomplete Oct 2-Mar 15 snowfall coverage for water year(s): "
            + ", ".join(incomplete["water_year"].astype(str))
            + ". Xu metrics for those years are left NaN."
        )

    return out


def parse_years(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone East River H1 Xu first-level decomposition."
    )
    parser.add_argument(
        "--isnobal",
        required=True,
        help="Path to isnobal_watershed_daily.csv",
    )
    parser.add_argument(
        "--noah",
        required=True,
        help="Path to noahmp_watershed_daily.csv",
    )
    parser.add_argument(
        "--output",
        default="h1_xu_first_level_metrics.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Optional comma-separated water years, e.g. 2018,2019,2020",
    )
    args = parser.parse_args()

    isnobal = load_daily(args.isnobal, "iSnobal")
    noah = load_daily(args.noah, "Noah-MP")
    years = parse_years(args.years)

    result = calculate_h1(isnobal, noah, years=years)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    display_cols = [
        "water_year",
        "Delta_SWE_Mar15_N_minus_I_mm",
        "epsilon_total_corrected",
        "epsilon_accumulation",
        "epsilon_ablation",
        "M_ablation_I",
        "M_ablation_N",
        "xu_multiplicative_closure_error",
    ]
    print(result[display_cols].to_string(index=False))
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
