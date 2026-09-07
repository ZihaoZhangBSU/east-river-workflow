"""Baseline review figures and H1-H4 scientific figures for East River."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from .aso import elevation_band_mask
from .config import WorkflowConfig
from .constants import ELEVATION_BANDS, MODEL_COLORS
from .grids import GridSpec
from .metrics import paired_event_series, paired_event_values, pearson_mean_bias
from .processing import load_spatial_cache, spatial_cache_path
from .utils import water_year_bounds


def configure_matplotlib(cfg: WorkflowConfig) -> str:
    section = cfg.section("plotting")
    requested = section["font_family"]
    available = {font.name for font in fm.fontManager.ttflist}
    family = requested if requested in available else section["fallback_font"]
    mpl.rcParams.update({
        "font.family": family,
        "font.size": section["font_size"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": section["dpi"],
        "figure.dpi": 120,
    })
    return family


def _save(fig: plt.Figure, cfg: WorkflowConfig, stem: str, *, scientific: bool = False) -> Path:
    extension = cfg.section("plotting")["figure_format"]
    folder = cfg.output_dir / "figures" / ("scientific" if scientific else "baseline")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.{extension}"
    fig.savefig(path, bbox_inches="tight", dpi=cfg.section("plotting")["dpi"])
    if bool(cfg.section("plotting").get("close_after_save", True)):
        plt.close(fig)
    return path


def _format_water_year_axis(ax: plt.Axes, wy: int) -> None:
    start, end = water_year_bounds(int(wy))
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.spines[["top", "right"]].set_visible(False)


def _tight_upper(values: Iterable[float], *, step: float, manual: float | None) -> float:
    if manual is not None:
        return float(manual)
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float(step)
    top = float(np.max(arr)) * 1.05
    return float(np.ceil(top / step) * step)


def _paired_daily_stats(i: pd.DataFrame, n: pd.DataFrame, column: str) -> dict[str, float | int]:
    pair = i[["date", column]].merge(n[["date", column]], on="date", suffixes=("_I", "_N")).dropna()
    return pearson_mean_bias(pair[f"{column}_I"].to_numpy(), pair[f"{column}_N"].to_numpy())


def plot_watershed_comparison(cfg: WorkflowConfig, isnobal: pd.DataFrame, noah: pd.DataFrame, variable: str) -> Path:
    """Required 3x3 baseline hydrograph with r and mean bias in every WY panel."""
    specs = {
        "SWE": ("swe_mm", "Watershed-average SWE (mm)", "watershed_mean_swe", "watershed_swe_ymax_mm", "watershed_swe_y_step_mm"),
        "Snow depth": ("snow_depth_mm", "Watershed-average snow depth (mm)", "watershed_mean_snow_depth", "watershed_snow_depth_ymax_mm", "watershed_snow_depth_y_step_mm"),
    }
    column, ylabel, stem, max_key, step_key = specs[variable]
    years = cfg.section("project")["water_years"]
    plotting = cfg.section("plotting")
    ymax = _tight_upper(
        [isnobal[column].max(skipna=True), noah[column].max(skipna=True)],
        step=float(plotting[step_key]),
        manual=plotting.get(max_key),
    )
    fig, axes = plt.subplots(3, 3, figsize=(21, 16.5), sharey=True)
    for ax, wy in zip(axes.flat, years):
        i = isnobal[isnobal["water_year"].eq(wy)].sort_values("date")
        n = noah[noah["water_year"].eq(wy)].sort_values("date")
        ax.plot(i["date"], i[column], color=MODEL_COLORS["iSnobal"], lw=1.6)
        ax.plot(n["date"], n[column], color=MODEL_COLORS["Noah-MP"], lw=1.6)
        stats = _paired_daily_stats(i, n, column)
        rtxt = "NA" if not np.isfinite(stats["pearson_r"]) else f"{stats['pearson_r']:.2f}"
        btxt = "NA" if not np.isfinite(stats["mean_bias"]) else f"{stats['mean_bias']:+.1f} mm"
        ax.text(0.03, 0.95, f"WY {wy}", transform=ax.transAxes, ha="left", va="top")
        ax.text(0.97, 0.95, f"r = {rtxt}\nMean bias = {btxt}", transform=ax.transAxes, ha="right", va="top")
        ax.set_ylim(0, ymax)
        _format_water_year_axis(ax, wy)
    for ax in axes[:, 0]: ax.set_ylabel(ylabel)
    for ax in axes[-1, :]: ax.set_xlabel("Month")
    fig.legend(
        handles=[
            Line2D([], [], color=MODEL_COLORS["iSnobal"], lw=1.6, label="iSnobal"),
            Line2D([], [], color=MODEL_COLORS["Noah-MP"], lw=1.6, label="Noah-MP"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015),
    )
    fig.subplots_adjust(top=0.98, bottom=0.09, hspace=0.30, wspace=0.16)
    return _save(fig, cfg, stem)


def plot_station_comparison(cfg: WorkflowConfig, station_id: int, station_name: str, snotel: pd.DataFrame, isnobal: pd.DataFrame, noah: pd.DataFrame, variable: str) -> Path:
    """SNOTEL direct comparison including exact iSnobal-within-Noah-cell envelope."""
    if variable == "SWE":
        obs_col, i_col, i_min, i_max, n_col, ylabel, slug, max_key, step = (
            "swe_mm", "swe_nearest_mm", "swe_in_noah_cell_min_mm", "swe_in_noah_cell_max_mm", "swe_mm", "SWE (mm)", "swe", "station_swe_ymax_mm", 50.0,
        )
    else:
        obs_col, i_col, i_min, i_max, n_col, ylabel, slug, max_key, step = (
            "snow_depth_mm", "snow_depth_nearest_mm", "snow_depth_in_noah_cell_min_mm", "snow_depth_in_noah_cell_max_mm", "snow_depth_mm", "Snow depth (mm)", "snow_depth", "station_snow_depth_ymax_mm", 100.0,
        )
    s = snotel[snotel["station_id"].eq(station_id)].copy()
    i = isnobal[isnobal["station_id"].eq(station_id)].copy()
    n = noah[noah["station_id"].eq(station_id)].copy()
    ymax = _tight_upper([s[obs_col].max(skipna=True), i[i_max].max(skipna=True), n[n_col].max(skipna=True)], step=step, manual=cfg.section("plotting").get(max_key))
    fig, axes = plt.subplots(3, 3, figsize=(21, 16.5), sharey=True)
    for ax, wy in zip(axes.flat, cfg.section("project")["water_years"]):
        ss = s[s["water_year"].eq(wy)].sort_values("date")
        ii = i[i["water_year"].eq(wy)].sort_values("date")
        nn = n[n["water_year"].eq(wy)].sort_values("date")
        ax.fill_between(ii["date"], ii[i_min], ii[i_max], color=MODEL_COLORS["iSnobal"], alpha=0.18, linewidth=0)
        ax.plot(ii["date"], ii[i_col], color=MODEL_COLORS["iSnobal"], lw=1.5)
        ax.plot(nn["date"], nn[n_col], color=MODEL_COLORS["Noah-MP"], lw=1.5)
        ax.plot(ss["date"], ss[obs_col], color=MODEL_COLORS["SNOTEL"], lw=1.4)
        ax.text(0.03, 0.95, f"WY {wy}", transform=ax.transAxes, ha="left", va="top")
        ax.set_ylim(0, ymax); _format_water_year_axis(ax, wy)
    for ax in axes[:, 0]: ax.set_ylabel(ylabel)
    for ax in axes[-1, :]: ax.set_xlabel("Month")
    fig.text(0.5, 0.985, f"{station_name}", ha="center", va="top")
    fig.legend(handles=[
        Line2D([], [], color=MODEL_COLORS["SNOTEL"], label="SNOTEL"),
        Line2D([], [], color=MODEL_COLORS["iSnobal"], label="Nearest iSnobal pixel"),
        Line2D([], [], color=MODEL_COLORS["Noah-MP"], label="Nearest Noah-MP pixel"),
        Patch(facecolor=MODEL_COLORS["iSnobal"], alpha=0.18, label="iSnobal min-max within nearest Noah-MP pixel"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(top=0.95, bottom=0.11, hspace=0.30, wspace=0.16)
    return _save(fig, cfg, f"station_{station_id}_{slug}_comparison")


def _station_error_table(snotel_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in snotel_metrics.iterrows():
        for model, prefix in [("iSnobal", "I"), ("Noah-MP", "N")]:
            a = pd.Timestamp(r.get(f"snow_depth_disappearance_date_{prefix}")) if pd.notna(r.get(f"snow_depth_disappearance_date_{prefix}")) else pd.NaT
            b = pd.Timestamp(r.get("snow_depth_disappearance_date_SNOTEL")) if pd.notna(r.get("snow_depth_disappearance_date_SNOTEL")) else pd.NaT
            rows.append({"station_id": int(r["station_id"]), "water_year": int(r["water_year"]), "source": model, "disappearance_error_days": (a-b).days if pd.notna(a) and pd.notna(b) else np.nan})
    return pd.DataFrame(rows)


def plot_snow_disappearance_errors(cfg: WorkflowConfig, metrics: pd.DataFrame) -> Path:
    errors = _station_error_table(metrics)
    years = cfg.section("project")["water_years"]; stations = cfg.section("snotel")["stations"]
    models = ["iSnobal", "Noah-MP"]; fig, ax = plt.subplots(figsize=(18, 8)); x=np.arange(len(years), dtype=float); width=0.12
    hatches=["", "//", "xx"]; offsets=np.linspace(-2.5*width,2.5*width,len(stations)*len(models)); k=0
    for station,hatch in zip(stations,hatches):
        for model in models:
            sub=errors[(errors.station_id.eq(int(station["id"]))) & errors.source.eq(model)].set_index("water_year")
            vals=[sub["disappearance_error_days"].get(y,np.nan) for y in years]
            ax.bar(x+offsets[k], vals, width=width, color=MODEL_COLORS[model], hatch=hatch, edgecolor="black", linewidth=.4); k+=1
    ax.axhline(0,color="black",lw=.8); ax.set_xticks(x,[str(y) for y in years]); ax.set_xlabel("Water year"); ax.set_ylabel("Snow-disappearance error (days)\nmodel minus SNOTEL"); ax.spines[["top","right"]].set_visible(False)
    first=ax.legend(handles=[Patch(facecolor=MODEL_COLORS[m],label=m) for m in models],title="Model",loc="upper left"); ax.add_artist(first)
    ax.legend(handles=[Patch(facecolor="white",edgecolor="black",hatch=h,label=s["name"]) for s,h in zip(stations,hatches)],title="Station",loc="upper right")
    return _save(fig,cfg,"snow_disappearance_errors")


def plot_accumulated_precip_and_swe(cfg: WorkflowConfig, snotel: pd.DataFrame, isnobal: pd.DataFrame, noah: pd.DataFrame) -> Path:
    fig,axes=plt.subplots(3,1,figsize=(20,15),sharex=True)
    for ax,station in zip(axes,cfg.section("snotel")["stations"]):
        sid=int(station["id"])
        sources=[("SNOTEL",snotel[snotel.station_id.eq(sid)],"swe_mm","precip_gapfilled_mm"),("iSnobal",isnobal[isnobal.station_id.eq(sid)],"swe_nearest_mm","precip_nearest_mm"),("Noah-MP",noah[noah.station_id.eq(sid)],"swe_mm","precip_mm")]
        for source,frame,swe_col,pcol in sources:
            frame=frame.sort_values("date").copy(); frame["cum_precip"]=frame.groupby("water_year")[pcol].cumsum()
            ax.plot(frame.date,frame[swe_col],color=MODEL_COLORS[source],lw=1.4); ax.plot(frame.date,frame.cum_precip,color=MODEL_COLORS[source],lw=1.4,ls="--")
        ax.text(.01,.95,station["name"],transform=ax.transAxes,ha="left",va="top"); ax.set_ylabel("SWE / accumulated\nprecipitation (mm)"); ax.spines[["top","right"]].set_visible(False)
    axes[-1].set_xlabel("Date")
    handles=[]
    for source in ["SNOTEL","iSnobal","Noah-MP"]:
        handles += [Line2D([],[],color=MODEL_COLORS[source],ls="-",label=f"{source} SWE"),Line2D([],[],color=MODEL_COLORS[source],ls="--",label=f"{source} precipitation")]
    fig.legend(handles=handles,loc="lower center",ncol=3,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.10,hspace=.20)
    return _save(fig,cfg,"station_accumulated_precipitation_and_swe")


def plot_f_swe_precip(cfg: WorkflowConfig, metrics: pd.DataFrame) -> Path:
    fig,axes=plt.subplots(1,3,figsize=(21,7),sharey=True)
    for ax,station in zip(axes,cfg.section("snotel")["stations"]):
        sub=metrics[metrics.station_id.eq(int(station["id"]))].sort_values("water_year")
        x=np.arange(len(sub));
        for src, ratio_col in [
            ("SNOTEL", "f_SWE_precip_SNOTEL"),
            ("iSnobal", "f_SWE_precip_I"),
            ("Noah-MP", "f_SWE_precip_N"),
        ]:
            ax.plot(sub.water_year, sub[ratio_col], marker="o", color=MODEL_COLORS[src], label=src)
        ax.text(.03,.95,station["name"],transform=ax.transAxes,ha="left",va="top"); ax.set_xlabel("Water year"); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Peak SWE / precipitation through peak")
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[s],marker="o",label=s) for s in ["SNOTEL","iSnobal","Noah-MP"]],loc="lower center",ncol=3,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.18,wspace=.12)
    return _save(fig,cfg,"station_f_swe_precip")


def _dowy(date: Any, wy: int) -> float:
    if pd.isna(date): return np.nan
    return float((pd.Timestamp(date)-pd.Timestamp(wy-1,10,1)).days+1)


def plot_annual_max_swe_timing(cfg: WorkflowConfig, metrics: pd.DataFrame) -> Path:
    fig,axes=plt.subplots(1,3,figsize=(21,7),sharey=True)
    for ax,station in zip(axes,cfg.section("snotel")["stations"]):
        sub=metrics[metrics.station_id.eq(int(station["id"]))].sort_values("water_year")
        for src,prefix in [("SNOTEL","SNOTEL"),("iSnobal","I"),("Noah-MP","N")]:
            vals=[_dowy(d,int(wy)) for d,wy in zip(sub[f"peak_date_{prefix}"],sub.water_year)]
            ax.plot(sub.water_year,vals,marker="o",color=MODEL_COLORS[src])
        ax.text(.03,.95,station["name"],transform=ax.transAxes,ha="left",va="top"); ax.set_xlabel("Water year"); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Day of water year of maximum SWE")
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[s],marker="o",label=s) for s in ["SNOTEL","iSnobal","Noah-MP"]],loc="lower center",ncol=3,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.18,wspace=.12)
    return _save(fig,cfg,"station_annual_max_swe_timing")


def _plot_grid(ax: plt.Axes,data:np.ndarray,grid:GridSpec,*,vmin=None,vmax=None,cmap="viridis",norm=None):
    origin="upper" if grid.transform.e<0 else "lower"
    return ax.imshow(data,extent=grid.extent,origin=origin,cmap=cmap,vmin=None if norm else vmin,vmax=None if norm else vmax,norm=norm,interpolation="nearest")


def _overlay_boundary(ax:plt.Axes,boundary,grid:GridSpec):
    boundary.to_crs(grid.crs).boundary.plot(ax=ax,color="black",linewidth=.7); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]


def _auto_categorical_breaks(maximum: float, *, difference: bool) -> np.ndarray:
    """Return clean categorical map boundaries using the observed robust range.

    The exact break numbers can be supplied in YAML.  When they are omitted,
    this helper derives a compact, reproducible set from the full comparison
    set so the ASO maps remain categorical without baking study-specific
    maxima into source code.
    """
    maximum = float(max(maximum, 1.0))
    magnitude = 10.0 ** np.floor(np.log10(maximum))
    rounded = np.ceil(maximum / magnitude * 2.0) / 2.0 * magnitude
    if difference:
        positive = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.00]) * rounded
        return np.concatenate((-positive[:0:-1], positive))
    return np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.00]) * rounded


def _aso_breaks(cfg: WorkflowConfig, variable: str, products: dict, *, difference: bool) -> np.ndarray:
    aso = cfg.section("aso")
    key = "swe_difference_breaks_mm" if difference and variable == "SWE" else (
        "snow_depth_difference_breaks_mm" if difference else (
            "swe_state_breaks_mm" if variable == "SWE" else "snow_depth_state_breaks_mm"
        )
    )
    supplied = aso.get(key)
    if supplied is not None:
        arr = np.asarray(supplied, dtype=float)
        if arr.ndim != 1 or len(arr) < 3 or not np.all(np.diff(arr) > 0):
            raise ValueError(f"aso.{key} must be a strictly increasing list with at least 3 values.")
        if difference and not (arr[0] < 0 < arr[-1]):
            raise ValueError(f"aso.{key} must span negative and positive differences.")
        return arr
    vals = []
    for date in aso["dates"]:
        a = products[(date, variable)]["arrays"]
        if difference:
            fields = [a["isnobal_on_noah"] - a["noah_native"], a["difference_isnobal_noah"], a["difference_noah"]]
            vals.extend([np.abs(f[np.isfinite(f)]) for f in fields if np.isfinite(f).any()])
        else:
            fields = [a["aso_on_noah"], a["isnobal_on_noah"], a["noah_native"]]
            vals.extend([f[np.isfinite(f)] for f in fields if np.isfinite(f).any()])
    if not vals:
        return _auto_categorical_breaks(1.0, difference=difference)
    combined = np.concatenate(vals)
    robust = float(np.percentile(combined, float(aso["map_robust_percentile"])))
    return _auto_categorical_breaks(robust, difference=difference)


def plot_aso_field_matrix(cfg,products,variable,isnobal_grid,noah_grid,boundary)->Path:
    """ASO, iSnobal, Noah-MP side-by-side on the common Noah-MP grid.

    This follows the user's corrected map design: no native-grid first column,
    no grid-description text, and one shared categorical color scale.
    """
    dates=cfg.section("aso")["dates"]
    fig,axes=plt.subplots(len(dates),3,figsize=(18,5*len(dates)),squeeze=False)
    for c,t in enumerate(["ASO","iSnobal","Noah-MP"]): axes[0,c].set_title(t)
    bounds = _aso_breaks(cfg, variable, products, difference=False)
    cmap = plt.get_cmap("viridis", len(bounds)-1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    im=None
    for r,date in enumerate(dates):
        a=products[(date,variable)]["arrays"]
        fields=[a["aso_on_noah"],a["isnobal_on_noah"],a["noah_native"]]
        for c,f in enumerate(fields):
            im=_plot_grid(axes[r,c],f,noah_grid,cmap=cmap,norm=norm)
            _overlay_boundary(axes[r,c],boundary,noah_grid)
        axes[r,0].text(-.05,.5,date,transform=axes[r,0].transAxes,ha="right",va="center",rotation=90)
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),boundaries=bounds,ticks=bounds,fraction=.018,pad=.01,spacing="proportional")
    cb.set_label(f"{variable} (mm)")
    fig.subplots_adjust(left=.07,right=.91,top=.95,bottom=.03,hspace=.12,wspace=.05)
    return _save(fig,cfg,f"aso_{variable.lower().replace(' ','_')}_model_fields")


def plot_aso_difference_matrix(cfg,products,variable,isnobal_grid,noah_grid,boundary)->Path:
    """Common-grid categorical differences: I-N, I-ASO, N-ASO."""
    dates=cfg.section("aso")["dates"]
    fig,axes=plt.subplots(len(dates),3,figsize=(18,5*len(dates)),squeeze=False)
    for c,t in enumerate(["iSnobal - Noah-MP","iSnobal - ASO","Noah-MP - ASO"]): axes[0,c].set_title(t)
    bounds = _aso_breaks(cfg, variable, products, difference=True)
    cmap = plt.get_cmap("RdBu_r", len(bounds)-1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    im=None
    for r,date in enumerate(dates):
        a=products[(date,variable)]["arrays"]
        fields=[a["isnobal_on_noah"]-a["noah_native"],a["difference_isnobal_noah"],a["difference_noah"]]
        for c,f in enumerate(fields):
            im=_plot_grid(axes[r,c],f,noah_grid,cmap=cmap,norm=norm)
            _overlay_boundary(axes[r,c],boundary,noah_grid)
        axes[r,0].text(-.05,.5,date,transform=axes[r,0].transAxes,ha="right",va="center",rotation=90)
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),boundaries=bounds,ticks=bounds,fraction=.018,pad=.01,spacing="proportional")
    cb.set_label(f"Difference in {variable} (mm)")
    fig.subplots_adjust(left=.07,right=.91,top=.95,bottom=.03,hspace=.12,wspace=.05)
    return _save(fig,cfg,f"aso_{variable.lower().replace(' ','_')}_differences")


def plot_elevation_distributions(cfg,date,variable,products,isnobal_dem,noah_dem,isnobal_weights,noah_weights)->Path:
    a=products[(date,variable)]["arrays"]
    specs=[("iSnobal-ASO\niSnobal grid",a["difference_isnobal_native"],isnobal_dem,isnobal_weights),("iSnobal-ASO\nNoah-MP grid",a["difference_isnobal_noah"],noah_dem,noah_weights),("Noah-MP-ASO\nNoah-MP grid",a["difference_noah"],noah_dem,noah_weights)]
    bands=list(ELEVATION_BANDS); fig,axes=plt.subplots(1,3,figsize=(21,7),sharey=True)
    positions=np.arange(len(bands)); width=.65
    for ax,(title,vals,elev,w) in zip(axes,specs):
        data=[vals[(w>0)&elevation_band_mask(elev,b)&np.isfinite(vals)] for b in bands]
        ax.boxplot(data,positions=positions,widths=width,showfliers=False); ax.axhline(0,color="black",lw=.8); ax.set_xticks(positions,bands); ax.set_xlabel("Elevation category"); ax.set_title(title); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel(f"Model - ASO {variable} bias (mm)"); fig.subplots_adjust(wspace=.12); return _save(fig,cfg,f"aso_{date}_{variable.lower().replace(' ','_')}_elevation_distributions")


def plot_streamflow(cfg:WorkflowConfig,frame:pd.DataFrame,normalized:bool)->Path:
    years=cfg.section("project")["water_years"]; fig,axes=plt.subplots(3,3,figsize=(21,16.5),sharey=normalized)
    if normalized:
        columns={s:f"{s}_normalized" for s in ["iSnobal","Noah-MP","USGS"]}; ylabel="Normalized 7-day moving average"; stem="streamflow_timing_normalized_7day"
    else:
        columns={s:f"{s}_7day_plot" for s in ["iSnobal","Noah-MP","USGS"]}; ylabel="7-day moving average (m³/day)"; stem="streamflow_physical_volume_7day"
    for ax,wy in zip(axes.flat,years):
        sub=frame[frame.water_year.eq(wy)]
        for source,col in columns.items(): ax.plot(sub.date,sub[col],color=MODEL_COLORS[source],lw=1.5)
        ax.text(.03,.95,f"WY {wy}",transform=ax.transAxes,ha="left",va="top"); _format_water_year_axis(ax,wy)
    for ax in axes[:,0]: ax.set_ylabel(ylabel)
    for ax in axes[-1,:]: ax.set_xlabel("Month")
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[s],label=s) for s in ["iSnobal","Noah-MP","USGS"]],loc="lower center",ncol=3,bbox_to_anchor=(.5,.015)); fig.subplots_adjust(bottom=.09,hspace=.30,wspace=.16); return _save(fig,cfg,stem)


# -----------------------------------------------------------------------------
# Scientific figures
# -----------------------------------------------------------------------------

def scientific_figure_01_decision_logic(cfg:WorkflowConfig)->Path:
    fig,ax=plt.subplots(figsize=(16,8)); ax.axis("off")
    nodes=[(.05,.62,.20,.18,"Observed pattern\nNoah-MP SWE lower"),(.30,.62,.20,.18,"H1 accumulation\nBias present by Mar 15?"),(.55,.62,.20,.18,"H2 ablation\nFaster after peak control?"),(.32,.20,.25,.20,"H3 mechanism\nAlbedo → absorbed SW → melt"),(.67,.20,.25,.20,"H4 consequence\nSWE timing → SWI/QSNBOT → USGS")]
    for x,y,w,h,text in nodes: ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.2)); ax.text(x+w/2,y+h/2,text,ha="center",va="center")
    arrows=[((.25,.71),(.30,.71)),((.50,.71),(.55,.71)),((.66,.62),(.52,.40)),((.57,.30),(.67,.30))]
    for a,b in arrows: ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.2))
    ax.text(.55,.48,"if faster ablation is confirmed",ha="center",va="center",fontsize=max(11,cfg.section("plotting")["font_size"]-2))
    return _save(fig,cfg,"figure01_decision_logic",scientific=True)


def scientific_figure_02_h1_case_hydrographs(cfg,i,n,h1_cases)->Path:
    years=[int(h1_cases["largest_year"]),int(h1_cases["smallest_year"])]; fig,axes=plt.subplots(2,2,figsize=(18,12),sharex="col")
    for c,(wy,label) in enumerate(zip(years,["Largest accumulation-bias year","Smallest accumulation-bias year"])):
        ii=i[i.water_year.eq(wy)]; nn=n[n.water_year.eq(wy)]
        for r,(col,ylabel) in enumerate([("swe_mm","SWE (mm)"),("snow_depth_mm","Snow depth (mm)")]):
            ax=axes[r,c]; ax.plot(ii.date,ii[col],color=MODEL_COLORS["iSnobal"],lw=1.6); ax.plot(nn.date,nn[col],color=MODEL_COLORS["Noah-MP"],lw=1.6); ax.text(.03,.95,f"{label} (WY {wy})",transform=ax.transAxes,ha="left",va="top"); ax.set_ylabel(ylabel); _format_water_year_axis(ax,wy)
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[s],label=s) for s in ["iSnobal","Noah-MP"]],loc="lower center",ncol=2,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.11,hspace=.15,wspace=.15); return _save(fig,cfg,"figure02_h1_case_hydrographs",scientific=True)


def scientific_figure_03_h1_decomposition(cfg,table)->Path:
    t=table.sort_values("water_year"); x=np.arange(len(t)); fig,axes=plt.subplots(2,1,figsize=(18,12),sharex=True)
    cols=[("C_initial_mm","Initial state"),("C_snowfall_mm","Snowfall amount"),("C_retention_mm","Retention efficiency")]
    bottoms_pos=np.zeros(len(t)); bottoms_neg=np.zeros(len(t))
    for col,label in cols:
        v=t[col].to_numpy(float); pos=np.where(v>0,v,0); neg=np.where(v<0,v,0); axes[0].bar(x,pos,bottom=bottoms_pos,label=label); axes[0].bar(x,neg,bottom=bottoms_neg); bottoms_pos+=pos; bottoms_neg+=neg
    axes[0].plot(x,t["Delta_SWE_Mar15_mm"],marker="o",color="black",lw=1.2,label="Mar 15 SWE bias"); axes[0].axhline(0,color="black",lw=.8); axes[0].set_ylabel("Contribution to Noah-MP - iSnobal SWE (mm)")
    share_cols=[("share_initial_pct","Initial state"),("share_snowfall_pct","Snowfall amount"),("share_retention_pct","Retention efficiency")]; bottom=np.zeros(len(t))
    for col,label in share_cols: v=t[col].to_numpy(float); axes[1].bar(x,v,bottom=bottom,label=label); bottom+=np.nan_to_num(v)
    axes[1].set_ylabel("Absolute contribution share (%)"); axes[1].set_ylim(0,100); axes[1].set_xticks(x,[str(y) for y in t.water_year],rotation=45); axes[1].set_xlabel("Water year"); axes[0].legend(loc="lower center",bbox_to_anchor=(.5,-.20),ncol=4); axes[0].spines[["top","right"]].set_visible(False); axes[1].spines[["top","right"]].set_visible(False); fig.subplots_adjust(hspace=.32); return _save(fig,cfg,"figure03_h1_contribution_decomposition",scientific=True)


def _density_panel(ax,x,y,xlabel,ylabel,label):
    """Same-unit paired density panel with Pearson r and mean(y-x) bias."""
    valid=np.isfinite(x)&np.isfinite(y); x=np.asarray(x)[valid]; y=np.asarray(y)[valid]
    if len(x):
        ax.hist2d(x,y,bins=65,norm=LogNorm(),cmap="viridis"); lo=min(np.min(x),np.min(y),0); hi=max(np.max(x),np.max(y),1); ax.plot([lo,hi],[lo,hi],ls="--",color="black",lw=.9); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        st=pearson_mean_bias(x,y); ax.text(.04,.06,f"r = {st['pearson_r']:.2f}\nMean bias = {st['mean_bias']:+.2f}",transform=ax.transAxes,ha="left",va="bottom")
    ax.text(.03,.95,label,transform=ax.transAxes,ha="left",va="top"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.spines[["top","right"]].set_visible(False)


def _mechanism_density_panel(ax,x,y,xlabel,ylabel,label,mean_label):
    """Cross-variable H3 density panel.

    A model-model *difference* is already plotted on y, so subtracting x from y
    would be dimensionally invalid (e.g., albedo versus energy).  Report the
    Pearson association and the mean y-response instead.
    """
    valid=np.isfinite(x)&np.isfinite(y); x=np.asarray(x,dtype=float)[valid]; y=np.asarray(y,dtype=float)[valid]
    if len(x):
        ax.hist2d(x,y,bins=65,norm=LogNorm(),cmap="viridis")
        ax.axhline(0,color="0.45",lw=.8); ax.axvline(0,color="0.45",lw=.8)
        if len(x) >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            r=float(np.corrcoef(x,y)[0,1])
        else:
            r=np.nan
        rtxt="NA" if not np.isfinite(r) else f"{r:.2f}"
        mean_y=float(np.nanmean(y))
        ax.text(.04,.06,f"r = {rtxt}\n{mean_label} = {mean_y:+.2f}",transform=ax.transAxes,ha="left",va="bottom")
    ax.text(.03,.95,label,transform=ax.transAxes,ha="left",va="top"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.spines[["top","right"]].set_visible(False)


def _season_plot_date(wy: int, month_day: str) -> pd.Timestamp:
    month, day = [int(v) for v in month_day.split("-")]
    return pd.Timestamp(wy - 1 if month >= 10 else wy, month, day)


def _aligned_spatial_window(cfg: WorkflowConfig, wy: int, start: pd.Timestamp, end: pd.Timestamp):
    ic = load_spatial_cache(spatial_cache_path(cfg, "isnobal", wy))
    nc = load_spatial_cache(spatial_cache_path(cfg, "noahmp", wy))
    idates = pd.DatetimeIndex(ic["dates"]).normalize()
    ndates = pd.DatetimeIndex(nc["dates"]).normalize()
    common = idates.intersection(ndates)
    common = common[(common >= start.normalize() - pd.Timedelta(days=1)) & (common <= end.normalize())]
    ii = idates.get_indexer(common); ni = ndates.get_indexer(common)
    return ic, nc, common, ii, ni


def scientific_figure_04_h1_accumulation_density(cfg,h1_cases,noah_weights)->Path:
    fig,axes=plt.subplots(1,2,figsize=(18,8)); threshold=float(cfg.section("h1")["event_threshold_mm_day"])
    for ax,key,label in zip(axes,["largest_year","smallest_year"],["Largest accumulation-bias year","Smallest accumulation-bias year"]):
        wy=int(h1_cases[key]); start=_season_plot_date(wy,cfg.section("h1")["start_month_day"]); end=_season_plot_date(wy,cfg.section("h1")["end_month_day"]); ic,nc,common,ii,ni=_aligned_spatial_window(cfg,wy,start,end); x,y=paired_event_values(ic["swe_mm"][ii],nc["swe_mm"][ni],threshold,"accumulation",spatial_mask=noah_weights>0); _density_panel(ax,x,y,"iSnobal daily +dSWE (mm/day)","Noah-MP daily +dSWE (mm/day)",f"{label} (WY {wy})")
    fig.subplots_adjust(wspace=.22); return _save(fig,cfg,"figure04_h1_accumulation_cell_day_density",scientific=True)


def _normalized_curve(frame:pd.DataFrame,col="swe_mm") -> tuple[np.ndarray,np.ndarray]:
    f=frame[["date",col]].dropna().sort_values("date");
    if f.empty or f[col].max()<=0: return np.array([]),np.array([])
    idx=f[col].idxmax(); peak_date=pd.Timestamp(f.loc[idx,"date"]); post=f[f.date>=peak_date]; return (post.date-peak_date).dt.days.to_numpy(),(post[col]/f.loc[idx,col]).to_numpy(float)


def scientific_figure_05_h2_normalized_depletion(cfg,i,n,h2_cases)->Path:
    fig,axes=plt.subplots(1,2,figsize=(18,8),sharey=True)
    for ax,key,label in zip(axes,["largest_year","smallest_year"],["Largest ablation-bias year","Smallest ablation-bias year"]):
        wy=int(h2_cases[key]); ii=i[i.water_year.eq(wy)]; nn=n[n.water_year.eq(wy)]
        for source,frame in [("iSnobal",ii),("Noah-MP",nn)]: x,y=_normalized_curve(frame); ax.plot(x,y,color=MODEL_COLORS[source],lw=1.6)
        for level in [.75,.5,.25]: ax.axhline(level,color="0.75",lw=.7,ls="--")
        ax.text(.03,.95,f"{label} (WY {wy})",transform=ax.transAxes,ha="left",va="top"); ax.set_xlabel("Days since each model peak SWE"); ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("SWE / peak SWE"); fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[s],label=s) for s in ["iSnobal","Noah-MP"]],loc="lower center",ncol=2,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.16,wspace=.12); return _save(fig,cfg,"figure05_h2_normalized_depletion",scientific=True)


def scientific_figure_06_h2_ablation_density(cfg,h2_cases,noah_weights)->Path:
    fig,axes=plt.subplots(1,2,figsize=(18,8)); threshold=float(cfg.section("h2")["event_threshold_mm_day"])
    for ax,key,label in zip(axes,["largest_year","smallest_year"],["Largest ablation-bias year","Smallest ablation-bias year"]):
        wy=int(h2_cases[key]); start=_season_plot_date(wy,cfg.section("h2")["ablation_start_month_day"]); end=_season_plot_date(wy,cfg.section("h2")["ablation_end_month_day"]); ic,nc,common,ii,ni=_aligned_spatial_window(cfg,wy,start,end); x,y=paired_event_values(ic["swe_mm"][ii],nc["swe_mm"][ni],threshold,"ablation",spatial_mask=noah_weights>0); _density_panel(ax,x,y,"iSnobal daily SWE loss (mm/day)","Noah-MP daily SWE loss (mm/day)",f"{label} (WY {wy})")
    fig.subplots_adjust(wspace=.22); return _save(fig,cfg,"figure06_h2_ablation_cell_day_density",scientific=True)


def scientific_figure_07_snotel_normalized(cfg,snotel,i_station,n_station,h2_cases)->Path:
    years=[int(h2_cases["largest_year"]),int(h2_cases["smallest_year"])]; fig,axes=plt.subplots(2,3,figsize=(21,12),sharey=True)
    for r,(wy,label) in enumerate(zip(years,["Largest ablation-bias year","Smallest ablation-bias year"])):
        for c,station in enumerate(cfg.section("snotel")["stations"]):
            sid=int(station["id"]); ax=axes[r,c]
            specs=[("SNOTEL",snotel[(snotel.station_id.eq(sid))&snotel.water_year.eq(wy)],"swe_mm"),("iSnobal",i_station[(i_station.station_id.eq(sid))&i_station.water_year.eq(wy)],"swe_nearest_mm"),("Noah-MP",n_station[(n_station.station_id.eq(sid))&n_station.water_year.eq(wy)],"swe_mm")]
            for source,frame,col in specs: x,y=_normalized_curve(frame,col); ax.plot(x,y,color=MODEL_COLORS[source],lw=1.5)
            for level in [.75,.5,.25]: ax.axhline(level,color="0.8",lw=.6,ls="--")
            ax.text(.03,.95,f"{station['name']}\n{label} (WY {wy})",transform=ax.transAxes,ha="left",va="top"); ax.set_xlabel("Days since each series peak"); ax.spines[["top","right"]].set_visible(False)
    for ax in axes[:,0]: ax.set_ylabel("SWE / own peak SWE")
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[s],label=s) for s in ["SNOTEL","iSnobal","Noah-MP"]],loc="lower center",ncol=3,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.12,hspace=.20,wspace=.13); return _save(fig,cfg,"figure07_snotel_normalized_depletion",scientific=True)


def _snotel_event_figure(cfg,snotel,i_station,n_station,wy:int,stem:str,label:str)->Path:
    fig,axes=plt.subplots(3,2,figsize=(18,18)); th=float(cfg.section("h1")["event_threshold_mm_day"])
    for r,station in enumerate(cfg.section("snotel")["stations"]):
        sid=int(station["id"]); s=snotel[(snotel.station_id.eq(sid))&snotel.water_year.eq(wy)].set_index("date")["swe_mm"]; ii=i_station[(i_station.station_id.eq(sid))&i_station.water_year.eq(wy)].set_index("date")["swe_nearest_mm"]; nn=n_station[(n_station.station_id.eq(sid))&n_station.water_year.eq(wy)].set_index("date")["swe_mm"]
        for c,mode in enumerate(["accumulation","ablation"]):
            ax=axes[r,c]; allvals=[]
            if mode=="accumulation": start=_season_plot_date(wy,cfg.section("h1")["start_month_day"]); end=_season_plot_date(wy,cfg.section("h1")["end_month_day"])
            else: start=_season_plot_date(wy,cfg.section("h2")["ablation_start_month_day"]); end=_season_plot_date(wy,cfg.section("h2")["ablation_end_month_day"])
            def window(z): return z[(z.index>=start-pd.Timedelta(days=1))&(z.index<=end)]
            for source,m in [("iSnobal",ii),("Noah-MP",nn)]:
                x,y=paired_event_series(window(s),window(m),th,mode); ax.scatter(x,y,s=15,alpha=.65,color=MODEL_COLORS[source]); st=pearson_mean_bias(x,y); allvals.extend(x.tolist()+y.tolist()); ax.text(.97,.06 if source=="iSnobal" else .16,f"{source}: r={st['pearson_r']:.2f}, MB={st['mean_bias']:+.2f}",transform=ax.transAxes,ha="right",va="bottom",fontsize=max(10,cfg.section("plotting")["font_size"]-4))
            hi=max(allvals) if allvals else 1; ax.plot([0,hi],[0,hi],ls="--",color="black",lw=.8); ax.set_xlim(0,hi*1.03); ax.set_ylim(0,hi*1.03); ax.text(.03,.95,station["name"],transform=ax.transAxes,ha="left",va="top"); ax.set_xlabel("SNOTEL daily +dSWE (mm/day)" if mode=="accumulation" else "SNOTEL daily SWE loss (mm/day)"); ax.set_ylabel("Model daily +dSWE (mm/day)" if mode=="accumulation" else "Model daily SWE loss (mm/day)"); ax.spines[["top","right"]].set_visible(False)
    axes[0,0].set_title(f"Accumulation — {label} (WY {wy})"); axes[0,1].set_title(f"Ablation — {label} (WY {wy})"); fig.legend(handles=[Line2D([],[],marker="o",ls="",color=MODEL_COLORS[s],label=s) for s in ["iSnobal","Noah-MP"]],loc="lower center",ncol=2,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.08,hspace=.30,wspace=.22); return _save(fig,cfg,stem,scientific=True)


def scientific_figure_08_snotel_events(cfg,s,i,n,h2_cases)->Path: return _snotel_event_figure(cfg,s,i,n,int(h2_cases["largest_year"]),"figure08_snotel_events_largest_h2","Largest ablation-bias year")
def scientific_figure_09_snotel_events(cfg,s,i,n,h2_cases)->Path: return _snotel_event_figure(cfg,s,i,n,int(h2_cases["smallest_year"]),"figure09_snotel_events_smallest_h2","Smallest ablation-bias year")


def scientific_figure_10_h3_chain(cfg,i,n,h2_cases)->Path:
    years=[int(h2_cases["largest_year"]),int(h2_cases["smallest_year"])]; fig,axes=plt.subplots(4,2,figsize=(18,18),sharex="col")
    for c,(wy,label) in enumerate(zip(years,["Largest ablation-bias year","Smallest ablation-bias year"])):
        ii=i[i.water_year.eq(wy)].copy(); nn=n[n.water_year.eq(wy)].copy(); peak_i=ii.loc[ii.swe_mm.idxmax(),"date"] if len(ii) else pd.NaT; peak_n=nn.loc[nn.swe_mm.idxmax(),"date"] if len(nn) else pd.NaT
        for source,frame,peak in [("iSnobal",ii,peak_i),("Noah-MP",nn,peak_n)]:
            post=frame[frame.date>=peak].copy(); x=(post.date-pd.Timestamp(peak)).dt.days
            axes[0,c].plot(x,post.swe_mm/post.swe_mm.max(),color=MODEL_COLORS[source],lw=1.5); axes[1,c].plot(x,post.effective_albedo,color=MODEL_COLORS[source],lw=1.5); axes[2,c].plot(x,post.absorbed_sw_w_m2,color=MODEL_COLORS[source],lw=1.5); axes[3,c].plot(x,post.melt_mm,color=MODEL_COLORS[source],lw=1.5)
        axes[0,c].text(.03,.95,f"{label} (WY {wy})",transform=axes[0,c].transAxes,ha="left",va="top")
    for r,ylabel in enumerate(["SWE / peak SWE","Effective albedo","Absorbed SW (W m⁻²)","Daily melt (mm/day)"]): axes[r,0].set_ylabel(ylabel)
    for ax in axes[-1,:]: ax.set_xlabel("Days after each model peak SWE")
    for ax in axes.flat: ax.spines[["top","right"]].set_visible(False)
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[s],label=s) for s in ["iSnobal","Noah-MP"]],loc="lower center",ncol=2,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.07,hspace=.18,wspace=.15); return _save(fig,cfg,"figure10_h3_case_chain",scientific=True)


def scientific_figure_11_h3_density(cfg,h2_cases,noah_weights)->Path:
    years=[int(h2_cases["largest_year"]),int(h2_cases["smallest_year"])]; fig,axes=plt.subplots(2,2,figsize=(18,14))
    for r,(wy,label) in enumerate(zip(years,["Largest ablation-bias year","Smallest ablation-bias year"])):
        start=_season_plot_date(wy,cfg.section("h3")["start_month_day"]); end=_season_plot_date(wy,cfg.section("h3")["end_month_day"]); ic,nc,dates,ii,ni=_aligned_spatial_window(cfg,wy,start,end); common=len(dates); mask=np.broadcast_to((noah_weights>0),(common,)+noah_weights.shape)
        da=nc["effective_albedo"][ni]-ic["effective_albedo"][ii]; dsw=nc["absorbed_sw_energy_mj_m2"][ni]-ic["absorbed_sw_energy_mj_m2"][ii]; loss_i=-np.diff(ic["swe_mm"][ii],axis=0); loss_n=-np.diff(nc["swe_mm"][ni],axis=0); extra=loss_n-loss_i
        v=mask & np.isfinite(da)&np.isfinite(dsw); _mechanism_density_panel(axes[r,0],da[v],dsw[v],"Delta effective albedo (Noah-MP - iSnobal)","Delta absorbed SW (MJ m⁻² day⁻¹)",f"{label} (WY {wy})","Mean Delta absorbed SW")
        v2=mask[1:] & np.isfinite(dsw[1:]) & np.isfinite(extra); _mechanism_density_panel(axes[r,1],dsw[1:][v2],extra[v2],"Delta absorbed SW (MJ m⁻² day⁻¹)","Extra Noah-MP SWE loss (mm/day)",f"{label} (WY {wy})","Mean extra Noah SWE loss")
    fig.subplots_adjust(hspace=.25,wspace=.22); return _save(fig,cfg,"figure11_h3_mechanistic_density",scientific=True)


def scientific_figure_12_h3_annual(cfg,h3)->Path:
    t=h3.sort_values("water_year"); fig,axes=plt.subplots(1,3,figsize=(21,7));
    axes[0].plot(t.water_year,t.effective_albedo_I,marker="o",color=MODEL_COLORS["iSnobal"],label="iSnobal"); axes[0].plot(t.water_year,t.effective_albedo_N,marker="o",color=MODEL_COLORS["Noah-MP"],label="Noah-MP"); axes[0].set_ylabel("Effective albedo")
    x=np.arange(len(t)); axes[1].bar(x-.18,t.Delta_absorbed_SW_MJ_m2,width=.36,label="Delta absorbed SW"); axes[1].bar(x+.18,t.albedo_driven_consistency_MJ_m2,width=.36,label="Albedo-driven consistency term"); axes[1].set_xticks(x,[str(y) for y in t.water_year],rotation=45); axes[1].set_ylabel("Cumulative energy (MJ m⁻²)")
    axes[2].plot(t.water_year,t.cumulative_snowmelt_I_mm,marker="o",color=MODEL_COLORS["iSnobal"],label="iSnobal melt"); axes[2].plot(t.water_year,t.cumulative_QMELT_N_mm,marker="o",color=MODEL_COLORS["Noah-MP"],label="Noah-MP melt"); ax2=axes[2].twinx(); ax2.plot(t.water_year,t.mean_extra_Noah_SWE_loss_mm_day,marker="s",ls="--",color="black",label="Extra Noah SWE loss"); axes[2].set_ylabel("Cumulative melt (mm)"); ax2.set_ylabel("Extra Noah SWE loss (mm/day)")
    for ax in axes: ax.set_xlabel("Water year"); ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(loc="best"); axes[1].legend(loc="best"); axes[2].legend(loc="upper left"); ax2.legend(loc="upper right"); fig.subplots_adjust(wspace=.28); return _save(fig,cfg,"figure12_h3_annual_consistency",scientific=True)


def scientific_figure_13_aso_maps(cfg,products,noah_grid,boundary)->Path:
    """Common-Noah-grid ASO state and difference map matrix.

    Columns follow the reviewed design exactly:
    ASO, iSnobal, Noah-MP, iSnobal-Noah-MP, iSnobal-ASO, Noah-MP-ASO.
    All maps use categorical breaks, with one common state scale and one common
    difference scale for each variable across all configured ASO dates.
    """
    dates=cfg.section("aso")["dates"]
    rows=[(d,v) for d in dates for v in ["SWE","Snow depth"]]
    fig,axes=plt.subplots(len(rows),6,figsize=(31,4.2*len(rows)),squeeze=False)
    labels=["ASO","iSnobal","Noah-MP","iSnobal - Noah-MP","iSnobal - ASO","Noah-MP - ASO"]
    for c,t in enumerate(labels): axes[0,c].set_title(t)

    state_settings={}
    diff_settings={}
    for var in ["SWE","Snow depth"]:
        sb=_aso_breaks(cfg,var,products,difference=False)
        db=_aso_breaks(cfg,var,products,difference=True)
        sc=plt.get_cmap("viridis",len(sb)-1); dc=plt.get_cmap("RdBu_r",len(db)-1)
        state_settings[var]=(sb,sc,BoundaryNorm(sb,sc.N,clip=True))
        diff_settings[var]=(db,dc,BoundaryNorm(db,dc.N,clip=True))

    last_state={}; last_diff={}
    for r,(date,var) in enumerate(rows):
        a=products[(date,var)]["arrays"]
        states=[a["aso_on_noah"],a["isnobal_on_noah"],a["noah_native"]]
        diffs=[a["isnobal_on_noah"]-a["noah_native"],a["difference_isnobal_noah"],a["difference_noah"]]
        sb,sc,sn=state_settings[var]; db,dc,dn=diff_settings[var]
        for c,f in enumerate(states):
            last_state[var]=_plot_grid(axes[r,c],f,noah_grid,cmap=sc,norm=sn)
            _overlay_boundary(axes[r,c],boundary,noah_grid)
        for c,f in enumerate(diffs,start=3):
            last_diff[var]=_plot_grid(axes[r,c],f,noah_grid,cmap=dc,norm=dn)
            _overlay_boundary(axes[r,c],boundary,noah_grid)
        axes[r,0].text(-.04,.5,f"{date}\n{var}",transform=axes[r,0].transAxes,ha="right",va="center",rotation=90)

    # Place compact categorical color bars along the bottom; separate SWE/depth
    # scales are necessary because the physical ranges differ substantially.
    for j,var in enumerate(["SWE","Snow depth"]):
        sb=state_settings[var][0]; db=diff_settings[var][0]
        cax1=fig.add_axes([0.12+0.45*j,0.018,0.18,0.010])
        cax2=fig.add_axes([0.31+0.45*j,0.018,0.18,0.010])
        cb1=fig.colorbar(last_state[var],cax=cax1,orientation="horizontal",boundaries=sb,ticks=sb,spacing="proportional")
        cb2=fig.colorbar(last_diff[var],cax=cax2,orientation="horizontal",boundaries=db,ticks=db,spacing="proportional")
        cb1.set_label(f"{var} (mm)"); cb2.set_label(f"{var} difference (mm)")
    fig.subplots_adjust(left=.07,right=.995,top=.97,bottom=.075,hspace=.08,wspace=.04)
    return _save(fig,cfg,"figure13_aso_state_difference_maps",scientific=True)


def scientific_figure_14_aso_elevation(cfg,products,noah_dem,noah_weights)->Path:
    dates=cfg.section("aso")["dates"]; bands=list(ELEVATION_BANDS); fig,axes=plt.subplots(len(dates),2,figsize=(18,5*len(dates)),squeeze=False)
    for r,date in enumerate(dates):
        for c,var in enumerate(["SWE","Snow depth"]):
            a=products[(date,var)]["arrays"]; xpos=np.arange(len(bands)); data_i=[]; data_n=[]
            for b in bands:
                mask=(noah_weights>0)&elevation_band_mask(noah_dem,b); data_i.append(a["difference_isnobal_noah"][mask & np.isfinite(a["difference_isnobal_noah"])]); data_n.append(a["difference_noah"][mask & np.isfinite(a["difference_noah"])])
            # paired box positions for the two models in each category
            pos_i=xpos-.18; pos_n=xpos+.18; bi=axes[r,c].boxplot(data_i,positions=pos_i,widths=.30,patch_artist=True,showfliers=False); bn=axes[r,c].boxplot(data_n,positions=pos_n,widths=.30,patch_artist=True,showfliers=False)
            for p in bi["boxes"]: p.set_facecolor("none"); p.set_edgecolor(MODEL_COLORS["iSnobal"])
            for p in bn["boxes"]: p.set_facecolor("none"); p.set_edgecolor(MODEL_COLORS["Noah-MP"])
            axes[r,c].axhline(0,color="black",lw=.7); axes[r,c].set_xticks(xpos,bands); axes[r,c].set_ylabel(f"Model - ASO {var} bias (mm)"); axes[r,c].text(.03,.95,date,transform=axes[r,c].transAxes,ha="left",va="top"); axes[r,c].spines[["top","right"]].set_visible(False)
    fig.legend(handles=[Patch(facecolor="none",edgecolor=MODEL_COLORS["iSnobal"],label="iSnobal - ASO"),Patch(facecolor="none",edgecolor=MODEL_COLORS["Noah-MP"],label="Noah-MP - ASO")],loc="lower center",ncol=2,bbox_to_anchor=(.5,.01)); fig.subplots_adjust(bottom=.06,hspace=.26,wspace=.20); return _save(fig,cfg,"figure14_aso_elevation_distributions",scientific=True)


def scientific_figure_15_h4(cfg,i,n,usgs,h4)->Path:
    fig,axes=plt.subplots(2,2,figsize=(18,14)); years=cfg.section("project")["water_years"]; example=int(h4.iloc[np.nanargmax(np.abs(h4.Delta_SWE50_days.to_numpy(float)))] ["water_year"]) if len(h4) and np.isfinite(h4.Delta_SWE50_days).any() else int(years[0])
    ii=i[i.water_year.eq(example)].sort_values("date"); nn=n[n.water_year.eq(example)].sort_values("date"); uu=usgs[usgs.water_year.eq(example)].sort_values("date")
    for source,frame in [("iSnobal",ii),("Noah-MP",nn)]: x,y=_normalized_curve(frame); axes[0,0].plot(x,y,color=MODEL_COLORS[source],lw=1.5)
    axes[0,0].axhline(.5,color="0.7",ls="--",lw=.8); axes[0,0].set_xlabel("Days after each model peak SWE"); axes[0,0].set_ylabel("SWE / peak SWE"); axes[0,0].text(.03,.95,f"Example WY {example}",transform=axes[0,0].transAxes,ha="left",va="top")
    # cumulative release/flow curves on common seasonal elapsed-day clock
    start=pd.Timestamp(year=example, month=3, day=1) if example==example else pd.NaT
    series=[("iSnobal SWI",ii,"swi_mm",MODEL_COLORS["iSnobal"]),("Noah-MP QSNBOT",nn,"qsnobot_mm",MODEL_COLORS["Noah-MP"]),("USGS",uu,"discharge_m3_day",MODEL_COLORS["USGS"])]
    for lab,f,col,color in series:
        f=f[f.date>=start].copy(); vals=pd.to_numeric(f[col],errors="coerce").fillna(0).clip(lower=0).to_numpy(); total=np.sum(vals); frac=np.cumsum(vals)/total if total>0 else np.full(len(vals),np.nan); x=(pd.to_datetime(f.date)-start).dt.days; axes[1,0].plot(x,frac,color=color,lw=1.5,label=lab)
    axes[1,0].axhline(.5,color="0.7",ls="--",lw=.8); axes[1,0].set_xlabel("Days after Mar 1"); axes[1,0].set_ylabel("Fraction of seasonal total"); axes[1,0].legend(loc="best")
    valid=h4[["Delta_SWE50_days","Delta_Release50_days"]].dropna(); axes[0,1].scatter(valid.Delta_SWE50_days,valid.Delta_Release50_days,s=45); st=pearson_mean_bias(valid.Delta_SWE50_days,valid.Delta_Release50_days); axes[0,1].axhline(0,color="0.6",lw=.7); axes[0,1].axvline(0,color="0.6",lw=.7); axes[0,1].set_xlabel("Delta SWE50 (days)"); axes[0,1].set_ylabel("Delta release50 (days)"); axes[0,1].text(.97,.06,f"r = {st['pearson_r']:.2f}\nMean bias = {st['mean_bias']:+.2f} days",transform=axes[0,1].transAxes,ha="right",va="bottom")
    axes[1,1].plot(h4.water_year,h4.SWI50_minus_USGS50_days,marker="o",color=MODEL_COLORS["iSnobal"],label="SWI50 - USGS50"); axes[1,1].plot(h4.water_year,h4.QSNBOT50_minus_USGS50_days,marker="o",color=MODEL_COLORS["Noah-MP"],label="QSNBOT50 - USGS50"); axes[1,1].axhline(0,color="0.6",lw=.7); axes[1,1].set_xlabel("Water year"); axes[1,1].set_ylabel("50% timing error (days)"); axes[1,1].legend(loc="best")
    for ax in axes.flat: ax.spines[["top","right"]].set_visible(False)
    fig.subplots_adjust(hspace=.26,wspace=.22); return _save(fig,cfg,"figure15_h4_timing_linkage",scientific=True)


def generate_baseline_figures(cfg:WorkflowConfig,products:dict[str,Any])->list[Path]:
    configure_matplotlib(cfg); figs=[]
    for variable in ["SWE","Snow depth"]:
        figs.append(plot_watershed_comparison(cfg,products["isnobal_watershed"],products["noah_watershed"],variable))
        for station in cfg.section("snotel")["stations"]:
            figs.append(plot_station_comparison(cfg,int(station["id"]),station["name"],products["snotel"],products["isnobal_station"],products["noah_station"],variable))
        figs.append(plot_aso_field_matrix(cfg,products["aso_products"],variable,products["isnobal_grid"],products["noah_grid"],products["watershed_boundary"]))
        figs.append(plot_aso_difference_matrix(cfg,products["aso_products"],variable,products["isnobal_grid"],products["noah_grid"],products["watershed_boundary"]))
        for date in cfg.section("aso")["dates"]: figs.append(plot_elevation_distributions(cfg,date,variable,products["aso_products"],products["isnobal_dem"],products["noah_dem"],products["isnobal_weights"],products["noah_weights"]))
    figs += [plot_snow_disappearance_errors(cfg,products["snotel_metrics"]),plot_accumulated_precip_and_swe(cfg,products["snotel"],products["isnobal_station"],products["noah_station"]),plot_f_swe_precip(cfg,products["snotel_metrics"]),plot_annual_max_swe_timing(cfg,products["snotel_metrics"]),plot_streamflow(cfg,products["streamflow"],False),plot_streamflow(cfg,products["streamflow"],True)]
    return figs


def generate_scientific_figures(cfg:WorkflowConfig,products:dict[str,Any])->list[Path]:
    configure_matplotlib(cfg); return [
        scientific_figure_01_decision_logic(cfg),
        scientific_figure_02_h1_case_hydrographs(cfg,products["isnobal_watershed"],products["noah_watershed"],products["h1_cases"]),
        scientific_figure_03_h1_decomposition(cfg,products["watershed_metrics"]),
        scientific_figure_04_h1_accumulation_density(cfg,products["h1_cases"],products["noah_weights"]),
        scientific_figure_05_h2_normalized_depletion(cfg,products["isnobal_watershed"],products["noah_watershed"],products["h2_cases"]),
        scientific_figure_06_h2_ablation_density(cfg,products["h2_cases"],products["noah_weights"]),
        scientific_figure_07_snotel_normalized(cfg,products["snotel"],products["isnobal_station"],products["noah_station"],products["h2_cases"]),
        scientific_figure_08_snotel_events(cfg,products["snotel"],products["isnobal_station"],products["noah_station"],products["h2_cases"]),
        scientific_figure_09_snotel_events(cfg,products["snotel"],products["isnobal_station"],products["noah_station"],products["h2_cases"]),
        scientific_figure_10_h3_chain(cfg,products["isnobal_watershed"],products["noah_watershed"],products["h2_cases"]),
        scientific_figure_11_h3_density(cfg,products["h2_cases"],products["noah_weights"]),
        scientific_figure_12_h3_annual(cfg,products["h3_metrics"]),
        scientific_figure_13_aso_maps(cfg,products["aso_products"],products["noah_grid"],products["watershed_boundary"]),
        scientific_figure_14_aso_elevation(cfg,products["aso_products"],products["noah_dem"],products["noah_weights"]),
        scientific_figure_15_h4(cfg,products["isnobal_watershed"],products["noah_watershed"],products["usgs"],products["h4_metrics"]),
    ]

# Backward-compatible name used by older notebooks.
def generate_all_figures(cfg:WorkflowConfig,products:dict[str,Any])->list[Path]:
    return generate_baseline_figures(cfg,products)+generate_scientific_figures(cfg,products)
