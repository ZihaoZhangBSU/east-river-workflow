"""H1-H4, SNOTEL, and timing metrics for the integrated East River workflow."""

from __future__ import annotations

from typing import Iterable, Any

import numpy as np
import pandas as pd

from .processing import load_spatial_cache, spatial_cache_path
from .utils import water_year_bounds


def pearson_mean_bias(reference, model) -> dict[str, float | int]:
    x = np.asarray(reference, dtype=float).ravel()
    y = np.asarray(model, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(x, y)[0, 1])
    return {"n": int(len(x)), "pearson_r": r, "mean_bias": float(np.mean(y - x)) if len(x) else np.nan}


def paired_event_values(reference_swe, model_swe, threshold_mm_day: float, mode: str, spatial_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Return paired accumulation or ablation event values from time x row x col SWE."""
    ref = np.asarray(reference_swe, dtype=float)
    mod = np.asarray(model_swe, dtype=float)
    if ref.shape != mod.shape:
        raise ValueError(f"Paired SWE arrays must have equal shape, got {ref.shape} and {mod.shape}")
    if ref.ndim < 1 or ref.shape[0] < 2:
        return np.array([]), np.array([])
    dr = np.diff(ref, axis=0); dm = np.diff(mod, axis=0)
    if mode == "accumulation":
        x, y = dr, dm
        keep = (x > threshold_mm_day) | (y > threshold_mm_day)
    elif mode == "ablation":
        x, y = -dr, -dm
        keep = (x > threshold_mm_day) | (y > threshold_mm_day)
    else:
        raise ValueError("mode must be 'accumulation' or 'ablation'")
    valid = np.isfinite(x) & np.isfinite(y) & keep
    if spatial_mask is not None:
        mask = np.asarray(spatial_mask, dtype=bool)
        valid &= np.broadcast_to(mask, valid.shape)
    return x[valid], y[valid]


def paired_event_series(reference: pd.Series, model: pd.Series, threshold_mm_day: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
    joined = pd.concat([reference.rename("reference"), model.rename("model")], axis=1).sort_index()
    d = joined.diff()
    if mode == "accumulation":
        x = d["reference"]; y = d["model"]; keep = (x > threshold_mm_day) | (y > threshold_mm_day)
    elif mode == "ablation":
        x = -d["reference"]; y = -d["model"]; keep = (x > threshold_mm_day) | (y > threshold_mm_day)
    else:
        raise ValueError(mode)
    valid = x.notna() & y.notna() & keep
    return x[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float)



def _window_series(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Subset a daily SWE series while retaining the preceding day for dSWE."""
    s = pd.Series(series).sort_index()
    return s[(s.index >= start - pd.Timedelta(days=1)) & (s.index <= end)]


def _aligned_cache_window(ic: dict[str, Any], nc: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp):
    """Return aligned cache indices for common dates from start-1 through end."""
    idates = pd.DatetimeIndex(ic["dates"]).normalize()
    ndates = pd.DatetimeIndex(nc["dates"]).normalize()
    common = idates.intersection(ndates)
    common = common[(common >= start.normalize() - pd.Timedelta(days=1)) & (common <= end.normalize())]
    return common, idates.get_indexer(common), ndates.get_indexer(common)

def select_case_years(frame: pd.DataFrame, metric_column: str, year_column: str = "water_year") -> dict[str, Any]:
    valid = frame[[year_column, metric_column]].dropna().copy()
    if valid.empty:
        return {"largest_year": None, "smallest_year": None, "largest_signed_value": np.nan, "smallest_signed_value": np.nan}
    magnitude = valid[metric_column].abs()
    largest_idx = magnitude.idxmax(); smallest_idx = magnitude.idxmin()
    return {
        "largest_year": int(valid.loc[largest_idx, year_column]),
        "smallest_year": int(valid.loc[smallest_idx, year_column]),
        "largest_signed_value": float(valid.loc[largest_idx, metric_column]),
        "smallest_signed_value": float(valid.loc[smallest_idx, metric_column]),
    }


def _value_on_date(frame: pd.DataFrame, date: pd.Timestamp, column: str) -> float:
    subset = frame[pd.to_datetime(frame["date"]).dt.normalize().eq(date.normalize())]
    if subset.empty:
        return np.nan
    value = pd.to_numeric(subset[column], errors="coerce").dropna()
    return float(value.iloc[-1]) if len(value) else np.nan


def _season_date(wy: int, month_day: str) -> pd.Timestamp:
    month, day = [int(x) for x in month_day.split("-")]
    year = wy - 1 if month >= 10 else wy
    return pd.Timestamp(year, month, day)


def _annual_depletion(series: pd.Series, threshold_mm: float, consecutive_days: int) -> dict[str, Any]:
    s = pd.Series(series.astype(float)).sort_index().dropna()
    if s.empty:
        return {"peak_swe_mm": np.nan, "peak_date": pd.NaT, "peak_dowy": np.nan, "days_to_75": np.nan, "days_to_50": np.nan, "days_to_25": np.nan, "disappearance_date": pd.NaT, "disappearance_dowy": np.nan}
    peak = float(s.max()); peak_date = pd.Timestamp(s.idxmax())
    out = {"peak_swe_mm": peak, "peak_date": peak_date}
    wy = peak_date.year + 1 if peak_date.month >= 10 else peak_date.year
    wy_start = pd.Timestamp(wy - 1, 10, 1)
    out["peak_dowy"] = int((peak_date.normalize() - wy_start).days + 1)
    post = s.loc[s.index >= peak_date]
    norm = post / peak if peak > 0 else post * np.nan
    for frac, name in [(0.75, "75"), (0.50, "50"), (0.25, "25")]:
        candidates = norm.index[norm <= frac]
        out[f"days_to_{name}"] = int((candidates[0] - peak_date).days) if len(candidates) else np.nan
    complete = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    post_complete = complete.loc[complete.index > peak_date]
    below = post_complete.notna() & (post_complete < threshold_mm)
    run = below.rolling(consecutive_days, min_periods=consecutive_days).sum()
    endpoints = run.index[run >= consecutive_days]
    disappearance = endpoints[0] - pd.Timedelta(days=consecutive_days - 1) if len(endpoints) else pd.NaT
    out["disappearance_date"] = disappearance
    out["disappearance_dowy"] = int((disappearance.normalize() - wy_start).days + 1) if pd.notna(disappearance) else np.nan
    return out


def calculate_h1_h2_watershed_metrics(cfg, isnobal: pd.DataFrame, noah: pd.DataFrame, noah_weights: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    years = cfg.section("project")["water_years"]
    h1 = cfg.section("h1"); h2 = cfg.section("h2")
    rows = []
    for wy in years:
        i = isnobal[isnobal["water_year"].eq(wy)].copy(); n = noah[noah["water_year"].eq(wy)].copy()
        start = _season_date(wy, h1["start_month_day"]); end = _season_date(wy, h1["end_month_day"])
        i_nov = _value_on_date(i, start, "swe_mm"); n_nov = _value_on_date(n, start, "swe_mm")
        i_mar = _value_on_date(i, end, "swe_mm"); n_mar = _value_on_date(n, end, "swe_mm")
        iwin = i[(i["date"] >= start) & (i["date"] <= end)]; nwin = n[(n["date"] >= start) & (n["date"] <= end)]
        sf_i = float(iwin["snowfall_mm"].sum(min_count=1)); sf_n = float(nwin["snowfall_mm"].sum(min_count=1))
        p_i = float(iwin["precip_mm"].sum(min_count=1)); p_n = float(nwin["precip_mm"].sum(min_count=1))
        eta_i = (i_mar - i_nov) / sf_i if np.isfinite(sf_i) and sf_i != 0 and np.isfinite(i_mar) and np.isfinite(i_nov) else np.nan
        eta_n = (n_mar - n_nov) / sf_n if np.isfinite(sf_n) and sf_n != 0 and np.isfinite(n_mar) and np.isfinite(n_nov) else np.nan
        delta_mar = n_mar - i_mar if np.isfinite(n_mar) and np.isfinite(i_mar) else np.nan
        c_initial = n_nov - i_nov if np.isfinite(n_nov) and np.isfinite(i_nov) else np.nan
        eta_bar = np.nanmean([eta_i, eta_n]); sf_bar = np.nanmean([sf_i, sf_n])
        c_snowfall = eta_bar * (sf_n - sf_i) if np.isfinite(eta_bar) and np.isfinite(sf_n) and np.isfinite(sf_i) else np.nan
        c_retention = sf_bar * (eta_n - eta_i) if np.isfinite(sf_bar) and np.isfinite(eta_n) and np.isfinite(eta_i) else np.nan
        closure = delta_mar - (c_initial + c_snowfall + c_retention) if np.all(np.isfinite([delta_mar, c_initial, c_snowfall, c_retention])) else np.nan
        denom = np.nansum(np.abs([c_initial, c_snowfall, c_retention]))
        shares = [100 * abs(x) / denom if denom > 0 and np.isfinite(x) else np.nan for x in [c_initial, c_snowfall, c_retention]]

        i_series = i.set_index(pd.to_datetime(i["date"]))["swe_mm"]
        n_series = n.set_index(pd.to_datetime(n["date"]))["swe_mm"]
        idep = _annual_depletion(i_series, float(h2["disappearance_threshold_swe_mm"]), int(h2["disappearance_consecutive_days"]))
        ndep = _annual_depletion(n_series, float(h2["disappearance_threshold_swe_mm"]), int(h2["disappearance_consecutive_days"]))

        accum_stats = {"pearson_r": np.nan, "mean_bias": np.nan, "n": 0}; ablation_stats = accum_stats.copy()
        ipath = spatial_cache_path(cfg, "isnobal", wy); npath = spatial_cache_path(cfg, "noahmp", wy)
        if ipath.exists() and npath.exists():
            ic = load_spatial_cache(ipath); nc = load_spatial_cache(npath)
            mask = np.asarray(noah_weights) > 0
            common, ii, ni = _aligned_cache_window(ic, nc, start, end)
            if len(common) >= 2:
                x, y = paired_event_values(ic["swe_mm"][ii], nc["swe_mm"][ni], float(h1["event_threshold_mm_day"]), "accumulation", mask)
                accum_stats = pearson_mean_bias(x, y)
            abl_start = _season_date(wy, h2["ablation_start_month_day"]); abl_end = _season_date(wy, h2["ablation_end_month_day"])
            common, ii, ni = _aligned_cache_window(ic, nc, abl_start, abl_end)
            if len(common) >= 2:
                x, y = paired_event_values(ic["swe_mm"][ii], nc["swe_mm"][ni], float(h2["event_threshold_mm_day"]), "ablation", mask)
                ablation_stats = pearson_mean_bias(x, y)

        rows.append({
            "water_year": wy,
            "SWE_Nov1_I_mm": i_nov, "SWE_Nov1_N_mm": n_nov,
            "SWE_Mar15_I_mm": i_mar, "SWE_Mar15_N_mm": n_mar,
            "Snowfall_Nov1_Mar15_I_mm": sf_i, "Snowfall_Nov1_Mar15_N_mm": sf_n,
            "Precip_Nov1_Mar15_I_mm": p_i, "Precip_Nov1_Mar15_N_mm": p_n,
            "eta_ret_I": eta_i, "eta_ret_N": eta_n, "Delta_SWE_Mar15_mm": delta_mar,
            "C_initial_mm": c_initial, "C_snowfall_mm": c_snowfall, "C_retention_mm": c_retention,
            "share_initial_pct": shares[0], "share_snowfall_pct": shares[1], "share_retention_pct": shares[2],
            "decomposition_closure_error_mm": closure,
            "peak_SWE_I_mm": idep["peak_swe_mm"], "peak_SWE_N_mm": ndep["peak_swe_mm"],
            "peak_date_I": idep["peak_date"], "peak_date_N": ndep["peak_date"],
            "peak_DOWY_I": idep["peak_dowy"], "peak_DOWY_N": ndep["peak_dowy"],
            "days_to_75_I": idep["days_to_75"], "days_to_75_N": ndep["days_to_75"],
            "days_to_50_I": idep["days_to_50"], "days_to_50_N": ndep["days_to_50"],
            "days_to_25_I": idep["days_to_25"], "days_to_25_N": ndep["days_to_25"],
            "Delta_t50_days": ndep["days_to_50"] - idep["days_to_50"] if np.all(np.isfinite([ndep["days_to_50"], idep["days_to_50"]])) else np.nan,
            "disappearance_date_I": idep["disappearance_date"], "disappearance_date_N": ndep["disappearance_date"],
            "disappearance_DOWY_I": idep["disappearance_dowy"], "disappearance_DOWY_N": ndep["disappearance_dowy"],
            "accumulation_event_r": accum_stats["pearson_r"], "accumulation_event_mean_bias_mm_day": accum_stats["mean_bias"], "accumulation_event_n": accum_stats["n"],
            "ablation_event_r": ablation_stats["pearson_r"], "ablation_event_mean_bias_mm_day": ablation_stats["mean_bias"], "ablation_event_n": ablation_stats["n"],
        })
    table = pd.DataFrame(rows)
    return table, select_case_years(table, "Delta_SWE_Mar15_mm"), select_case_years(table, "Delta_t50_days")




def _complete_precip_to_date(frame: pd.DataFrame, wy: int, end_date: pd.Timestamp, column: str) -> tuple[float, bool]:
    """Sum daily precipitation from Oct 1 through end_date only if calendar-complete."""
    if pd.isna(end_date) or frame.empty or column not in frame.columns:
        return np.nan, False
    start = pd.Timestamp(wy - 1, 10, 1)
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        return np.nan, False
    series = frame.set_index(pd.to_datetime(frame["date"]).dt.normalize())[column]
    series = pd.to_numeric(series, errors="coerce")
    target = pd.date_range(start, end, freq="D")
    aligned = series.reindex(target)
    complete = bool(len(aligned) == len(target) and aligned.notna().all())
    return (float(aligned.sum()) if complete else np.nan), complete

def calculate_snotel_metrics(cfg, snotel: pd.DataFrame, isnobal: pd.DataFrame, noah: pd.DataFrame) -> pd.DataFrame:
    h1 = cfg.section("h1"); h2 = cfg.section("h2"); snow_cfg = cfg.section("snow_disappearance"); rows = []
    for station in cfg.section("snotel")["stations"]:
        sid = int(station["id"])
        for wy in cfg.section("project")["water_years"]:
            s = snotel[(snotel["station_id"] == sid) & (snotel["water_year"] == wy)].sort_values("date")
            i = isnobal[(isnobal["station_id"] == sid) & (isnobal["water_year"] == wy)].sort_values("date")
            n = noah[(noah["station_id"] == sid) & (noah["water_year"] == wy)].sort_values("date")
            swe_series = {
                "SNOTEL": s.set_index(pd.to_datetime(s["date"])) ["swe_mm"] if len(s) else pd.Series(dtype=float),
                "iSnobal": i.set_index(pd.to_datetime(i["date"])) ["swe_nearest_mm"] if len(i) else pd.Series(dtype=float),
                "Noah": n.set_index(pd.to_datetime(n["date"])) ["swe_mm"] if len(n) else pd.Series(dtype=float),
            }
            depth_series = {
                "SNOTEL": s.set_index(pd.to_datetime(s["date"])) ["snow_depth_mm"] if len(s) else pd.Series(dtype=float),
                "iSnobal": i.set_index(pd.to_datetime(i["date"])) ["snow_depth_nearest_mm"] if len(i) else pd.Series(dtype=float),
                "Noah": n.set_index(pd.to_datetime(n["date"])) ["snow_depth_mm"] if len(n) else pd.Series(dtype=float),
            }
            dep = {name: _annual_depletion(ser, float(h2["disappearance_threshold_swe_mm"]), int(h2["disappearance_consecutive_days"])) for name, ser in swe_series.items()}
            depth_dep = {name: _annual_depletion(ser, float(snow_cfg["threshold_depth_mm"]), int(snow_cfg["consecutive_days"])) for name, ser in depth_series.items()}
            start = _season_date(wy, h1["start_month_day"]); end = _season_date(wy, h1["end_month_day"])
            sprecip = s[(s["date"] >= start) & (s["date"] <= end)]["precip_gapfilled_mm"].sum(min_count=1) if len(s) else np.nan
            iprecip = i[(i["date"] >= start) & (i["date"] <= end)]["precip_nearest_mm"].sum(min_count=1) if len(i) else np.nan
            nprecip = n[(n["date"] >= start) & (n["date"] <= end)]["precip_mm"].sum(min_count=1) if len(n) else np.nan
            row = {
                "station_id": sid, "station_name": station["name"], "water_year": wy,
                "SNOTEL_precip_Nov1_Mar15_mm": sprecip,
                "iSnobal_precip_Nov1_Mar15_mm": iprecip,
                "Noah_precip_Nov1_Mar15_mm": nprecip,
            }
            for name, prefix in [("SNOTEL", "SNOTEL"), ("iSnobal", "I"), ("Noah", "N")]:
                d = dep[name]; dd = depth_dep[name]
                row.update({
                    f"peak_SWE_{prefix}_mm": d["peak_swe_mm"],
                    f"peak_date_{prefix}": d["peak_date"],
                    f"days_to_75_{prefix}": d["days_to_75"],
                    f"days_to_50_{prefix}": d["days_to_50"],
                    f"days_to_25_{prefix}": d["days_to_25"],
                    f"disappearance_date_{prefix}": d["disappearance_date"],
                    f"snow_depth_disappearance_date_{prefix}": dd["disappearance_date"],
                })
            # Original/review f_SWE,precip metric uses each source's own precipitation
            # accumulated from Oct 1 through that source's annual maximum-SWE date.
            for source, prefix, frame, pcol in [
                ("SNOTEL", "SNOTEL", s, "precip_gapfilled_mm"),
                ("iSnobal", "I", i, "precip_nearest_mm"),
                ("Noah", "N", n, "precip_mm"),
            ]:
                total, complete = _complete_precip_to_date(frame, wy, dep[source]["peak_date"], pcol)
                row[f"precip_to_peak_{prefix}_mm"] = total
                row[f"precip_complete_to_peak_{prefix}"] = complete
                peak = dep[source]["peak_swe_mm"]
                row[f"f_SWE_precip_{prefix}"] = peak / total if complete and np.isfinite(total) and total > 0 and np.isfinite(peak) else np.nan

            sseries = swe_series["SNOTEL"]
            for model_name, prefix in [("iSnobal", "I_vs_SNOTEL"), ("Noah", "N_vs_SNOTEL")]:
                mseries = swe_series[model_name]
                joined = pd.concat([sseries.rename("s"), mseries.rename("m")], axis=1).dropna()
                row[f"mean_SWE_bias_{prefix}_mm"] = float((joined["m"] - joined["s"]).mean()) if len(joined) else np.nan
                for mode in ["accumulation", "ablation"]:
                    threshold = float(h1["event_threshold_mm_day"] if mode == "accumulation" else h2["event_threshold_mm_day"])
                    if mode == "accumulation":
                        event_start, event_end = start, end
                    else:
                        event_start = _season_date(wy, h2["ablation_start_month_day"])
                        event_end = _season_date(wy, h2["ablation_end_month_day"])
                    x, y = paired_event_series(_window_series(sseries, event_start, event_end), _window_series(mseries, event_start, event_end), threshold, mode)
                    stats = pearson_mean_bias(x, y)
                    row[f"{mode}_r_{prefix}"] = stats["pearson_r"]
                    row[f"{mode}_mean_bias_{prefix}_mm_day"] = stats["mean_bias"]
                    row[f"{mode}_n_{prefix}"] = stats["n"]
            rows.append(row)
    return pd.DataFrame(rows)

def calculate_h3_metrics(cfg, isnobal: pd.DataFrame, noah: pd.DataFrame) -> pd.DataFrame:
    h3 = cfg.section("h3"); rows = []
    for wy in cfg.section("project")["water_years"]:
        start = _season_date(wy, h3["start_month_day"]); end = _season_date(wy, h3["end_month_day"])
        i = isnobal[(isnobal["date"] >= start) & (isnobal["date"] <= end)].copy()
        n = noah[(noah["date"] >= start) & (noah["date"] <= end)].copy()
        m = i.merge(n, on="date", suffixes=("_I", "_N"))
        EinI = m["incoming_sw_energy_mj_m2_I"].sum(min_count=1); EabsI = m["absorbed_sw_energy_mj_m2_I"].sum(min_count=1)
        EinN = m["incoming_sw_energy_mj_m2_N"].sum(min_count=1); EabsN = m["absorbed_sw_energy_mj_m2_N"].sum(min_count=1)
        alphaI = 1 - EabsI / EinI if np.isfinite(EinI) and EinI > 0 else np.nan
        alphaN = 1 - EabsN / EinN if np.isfinite(EinN) and EinN > 0 else np.nan
        daily_valid = m[["incoming_sw_energy_mj_m2_I", "incoming_sw_energy_mj_m2_N", "effective_albedo_I", "effective_albedo_N"]].notna().all(axis=1)
        albedo_term = np.nan
        if daily_valid.any():
            v = m[daily_valid]
            mean_incoming = 0.5 * (v["incoming_sw_energy_mj_m2_I"] + v["incoming_sw_energy_mj_m2_N"])
            delta_alpha = v["effective_albedo_N"] - v["effective_albedo_I"]
            albedo_term = float((-mean_incoming * delta_alpha).sum())
        swe_i = m["swe_mm_I"]; swe_n = m["swe_mm_N"]
        loss_i = -swe_i.diff(); loss_n = -swe_n.diff()
        melt_day = (m["melt_mm_I"] > float(h3["melt_day_threshold_mm"])) | (m["melt_mm_N"] > float(h3["melt_day_threshold_mm"]))
        extra_loss = (loss_n - loss_i)[melt_day & loss_i.notna() & loss_n.notna()]
        rows.append({
            "water_year": wy, "effective_albedo_I": alphaI, "effective_albedo_N": alphaN,
            "cumulative_absorbed_SW_I_MJ_m2": EabsI, "cumulative_absorbed_SW_N_MJ_m2": EabsN,
            "Delta_absorbed_SW_MJ_m2": EabsN - EabsI if np.all(np.isfinite([EabsI, EabsN])) else np.nan,
            "albedo_driven_consistency_MJ_m2": albedo_term,
            "cumulative_snowmelt_I_mm": m["melt_mm_I"].sum(min_count=1), "cumulative_QMELT_N_mm": m["melt_mm_N"].sum(min_count=1),
            "mean_extra_Noah_SWE_loss_mm_day": float(extra_loss.mean()) if len(extra_loss) else np.nan,
            "incoming_SW_difference_MJ_m2_QC": EinN - EinI if np.all(np.isfinite([EinI, EinN])) else np.nan,
        })
    return pd.DataFrame(rows)


def cumulative_half_date(dates: pd.Series | pd.DatetimeIndex, values: pd.Series | np.ndarray) -> pd.Timestamp:
    d = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))); v = np.asarray(values, dtype=float)
    valid = np.isfinite(v) & (v >= 0)
    d, v = d[valid], v[valid]
    if not len(v) or np.sum(v) <= 0: return pd.NaT
    c = np.cumsum(v); idx = int(np.searchsorted(c, 0.5 * c[-1], side="left"))
    return pd.Timestamp(d[min(idx, len(d) - 1)])


def calculate_h4_metrics(cfg, isnobal: pd.DataFrame, noah: pd.DataFrame, usgs: pd.DataFrame) -> pd.DataFrame:
    h4 = cfg.section("h4"); h2 = cfg.section("h2"); rows = []
    for wy in cfg.section("project")["water_years"]:
        start = _season_date(wy, h4["season_start_month_day"]); end = _season_date(wy, h4["season_end_month_day"])
        end = min(end, cfg.analysis_end.normalize())
        i = isnobal[isnobal["water_year"].eq(wy)].sort_values("date")
        n = noah[noah["water_year"].eq(wy)].sort_values("date")
        u = usgs[usgs["water_year"].eq(wy)].sort_values("date")
        idep = _annual_depletion(i.set_index(pd.to_datetime(i["date"]))["swe_mm"], float(h2["disappearance_threshold_swe_mm"]), int(h2["disappearance_consecutive_days"]))
        ndep = _annual_depletion(n.set_index(pd.to_datetime(n["date"]))["swe_mm"], float(h2["disappearance_threshold_swe_mm"]), int(h2["disappearance_consecutive_days"]))
        swe50_i = idep["peak_date"] + pd.to_timedelta(idep["days_to_50"], unit="D") if pd.notna(idep["peak_date"]) and np.isfinite(idep["days_to_50"]) else pd.NaT
        swe50_n = ndep["peak_date"] + pd.to_timedelta(ndep["days_to_50"], unit="D") if pd.notna(ndep["peak_date"]) and np.isfinite(ndep["days_to_50"]) else pd.NaT
        ii = i[(i["date"] >= start) & (i["date"] <= end)]; nn = n[(n["date"] >= start) & (n["date"] <= end)]; uu = u[(u["date"] >= start) & (u["date"] <= end)]
        swi50 = cumulative_half_date(ii["date"], ii["swi_mm"])
        q50 = cumulative_half_date(nn["date"], nn["qsnobot_mm"])
        usgs50 = cumulative_half_date(uu["date"], uu["discharge_m3_day"])
        def dd(a,b): return int((a-b).days) if pd.notna(a) and pd.notna(b) else np.nan
        rows.append({
            "water_year": wy, "SWE50_I": swe50_i, "SWE50_N": swe50_n, "SWI50": swi50, "QSNBOT50": q50,
            "release_lag_I_days": dd(swi50, swe50_i), "release_lag_N_days": dd(q50, swe50_n),
            "Delta_SWE50_days": dd(swe50_n, swe50_i), "Delta_Release50_days": dd(q50, swi50),
            "USGS50": usgs50, "SWI50_minus_USGS50_days": dd(swi50, usgs50), "QSNBOT50_minus_USGS50_days": dd(q50, usgs50),
        })
    return pd.DataFrame(rows)


def streamflow_products(cfg, isnobal_watershed: pd.DataFrame, noah_watershed: pd.DataFrame, usgs: pd.DataFrame) -> pd.DataFrame:
    """Retained v0.5-style 7-day review product, using QSNBOT rather than runoff."""
    merged = isnobal_watershed[["date", "water_year", "swi_volume_m3_day"]].merge(
        noah_watershed[["date", "qsnobot_volume_m3_day"]], on="date", how="outer"
    ).merge(usgs[["date", "discharge_m3_day"]], on="date", how="outer").sort_values("date")
    merged["date"] = pd.to_datetime(merged["date"])
    roll = int(cfg.section("streamflow")["rolling_days"])
    for source, col in [("iSnobal", "swi_volume_m3_day"), ("Noah-MP", "qsnobot_volume_m3_day"), ("USGS", "discharge_m3_day")]:
        merged[f"{source}_7day_plot"] = merged[col].rolling(roll, center=True, min_periods=1).mean()
        merged[f"{source}_normalized"] = np.nan
    for wy in cfg.section("project")["water_years"]:
        mask = merged["water_year"].eq(wy)
        for source in ["iSnobal", "Noah-MP", "USGS"]:
            col = f"{source}_7day_plot"; mx = merged.loc[mask, col].max(skipna=True)
            if np.isfinite(mx) and mx > 0: merged.loc[mask, f"{source}_normalized"] = merged.loc[mask, col] / mx
    return merged
