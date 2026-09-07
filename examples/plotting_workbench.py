# %% [markdown]
# # East River plotting workbench
# This is the plain-Python/JupyterLab companion to plotting_workbench.ipynb.
# Run cells after an editable package installation.

# %%
from pathlib import Path
import matplotlib.pyplot as plt

from east_river_workflow.config import load_config
from east_river_workflow.plotting import (
    configure_matplotlib,
    plot_accumulated_precip_and_swe,
    plot_annual_max_swe_timing,
    plot_aso_difference_matrix,
    plot_aso_field_matrix,
    plot_elevation_distributions,
    plot_f_swe_precip,
    plot_snow_disappearance_errors,
    plot_station_comparison,
    plot_streamflow,
    plot_watershed_comparison,
)
from east_river_workflow.workflow import prepare_analysis_products, generate_all_figures

CONFIG_CANDIDATES = [Path("../config/east_river_config.yaml"), Path("config/east_river_config.yaml")]
CONFIG_PATH = next(path.resolve() for path in CONFIG_CANDIDATES if path.exists())
cfg = load_config(CONFIG_PATH)
cfg.data["plotting"]["close_after_save"] = False
configure_matplotlib(cfg)

# %% [markdown]
# ## Prepare products once

# %%
products = prepare_analysis_products(cfg, run_validation=True)

# %% [markdown]
# ## Generate every standard figure

# %%
figure_paths = generate_all_figures(cfg, products)
plt.show()

# %% [markdown]
# ## Example: rerun one watershed figure after editing plotting.py

# %%
path = plot_watershed_comparison(
    cfg, products["isnobal_watershed"], products["noah_watershed"], "SWE"
)
plt.show()
path

# %% [markdown]
# ## Example: rerun one station figure

# %%
station = cfg.section("snotel")["stations"][0]
path = plot_station_comparison(
    cfg, int(station["id"]), station["name"], products["snotel"],
    products["isnobal_station"], products["noah_station"], "SWE"
)
plt.show()
path

# %%
plt.close("all")
