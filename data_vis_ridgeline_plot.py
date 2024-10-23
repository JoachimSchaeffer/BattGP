import os

import matplotlib.pyplot as plt
import numpy as np

import src.config as cfg
from joypy_modified import joypy
from src.batt_data import data_utils
from src.path_setup import setup_paths

plt.style.use("seaborn-v0_8-white")

setup_paths()
data_utils.build_data_cache()


columns_eval = [
    "T_Cell_1_2",
    "T_Cell_3_4",
    "T_Cell_5_6",
    "T_Cell_7_8",
    "I_Batt",
    "U_Batt",
    "SOC_Batt",
]
cell_volt_container = []
ibat_container = []
soc_container = []
temp_container = []
bat_age_container = []
sampling_time_container = []
labels = []

for batt_id in range(1, 29):
    labels.append(f"Battery {batt_id}")
    volt_cols = list(["U_Cell_" + str(cellnr) for cellnr in range(1, 9)])
    df = data_utils.read_battery_data(batt_id, keep_columns=columns_eval + volt_cols)
    # get all voltage data
    cell_volt_container.append(df[volt_cols].to_numpy().flatten())
    # get all ibat data
    ibat_container.append(df["I_Batt"].to_numpy().flatten())
    # get all soc data
    soc_container.append(df["SOC_Batt"].to_numpy().flatten())
    # get all temperature data
    temp_container.append(
        df[["T_Cell_1_2", "T_Cell_3_4", "T_Cell_5_6", "T_Cell_7_8"]]
        .to_numpy()
        .flatten()
    )
    # get all battery age data
    bat_age_container.append((df.index[-1] - df.index[0]).days)
    # get all sampling time data
    sampling_times_series = df.index.to_series().diff().dt.total_seconds()
    sampling_time_container.append(sampling_times_series.to_numpy().flatten())


xlabel_size = 55
plots = ["ibat", "cell_volt", "soc", "temp", "sampling_time"]


def add_labels(
    axes,
    labels,
    xlabel,
    title,
    filename,
    fontsize=60,
    title_fontsize=70,
    base="results/data_vis",
):
    for ax, label in zip(axes, labels):
        ax.set_ylabel(
            label.split(" ")[-1],
            fontsize=fontsize - 20,
            rotation="horizontal",
            va="top",
            ha="right",
        )
        ax.set_yticklabels([])
        ax.set_yticks([])
    axes[-1].set_xlabel(xlabel, fontsize=fontsize)
    fig.suptitle(title, fontsize=title_fontsize, y=1.01)
    os.makedirs(base, exist_ok=True)
    fig.savefig(os.path.join(base, filename), bbox_inches="tight")
    plt.close(fig)


# replace NaNs with 9999 in sampling time
sampling_time_container = [
    np.where(np.isnan(x), 9999, x) for x in sampling_time_container
]
# create a copy of the sampling time container
sampling_time_container_copy = sampling_time_container.copy()
# Replace all values larger than 100 with 0
weights = [np.where(x > 100, 0, x) for x in sampling_time_container_copy]


# Copy ibat container and filter fo values >200 and smaller than -800
# Slighly hacky but very few outlier data points are filtered, not changing the overall distribution
if "ibat" in plots:
    print("Plotting ibat")
    ibat_container_copy = ibat_container.copy()
    ibat_container_copy = [x[(x > -400) & (x < 200)] for x in ibat_container_copy]

    fig, axes = joypy.joyplot(
        ibat_container_copy,
        ylim=(1e-7, 10),
        figsize=(15, 15),
        xlabelsize=xlabel_size,
        ylabels=True,
        labels=labels,
        color=cfg.COLORS[0],
        overlap=0.7,
        bw=0.5,
        logscale=True,
    )
    add_labels(
        axes, labels, "Current [A]", "Log Density of Current", "joyplot_ibat.pdf"
    )

if "cell_volt" in plots:
    print("Plotting cell voltage")
    fig, axes = joypy.joyplot(
        cell_volt_container,
        ylim=(1e-6, 100),
        figsize=(15, 15),
        xlabelsize=xlabel_size,
        ylabels=True,
        labels=labels,
        color=cfg.COLORS[1],
        overlap=0.7,
        bw=0.005,
        logscale=True,
    )
    add_labels(
        axes,
        labels,
        "Cell Voltage [V]",
        "Log Density of Cell Voltage",
        "joyplot_cell_volt.pdf",
    )

if "soc" in plots:
    print("Plotting SOC")
    fig, axes = joypy.joyplot(
        soc_container,
        ylim=(1e-4, 100),
        figsize=(15, 15),
        xlabelsize=xlabel_size,
        ylabels=True,
        labels=labels,
        color=cfg.COLORS[2],
        overlap=0.5,
        bw=0.015,
        logscale=True,
    )
    add_labels(axes, labels, "SOC [%]", "Log Density of SOC", "joyplot_soc.pdf")

if "temp" in plots:
    print("Plotting temperature")
    # Filter out all temperatures smaller than -20, these are values set by the BMS as invalid
    temp_container_filtered = [x[x > -20] for x in temp_container]
    # Create a copy of the filtered temperature container and filter out all temperatures larger than 60
    temp_container_filtered_ = temp_container_filtered.copy()
    temp_container_filtered_ = [x[x < 60] for x in temp_container_filtered_]
    fig, axes = joypy.joyplot(
        temp_container_filtered_,
        ylim=(0, 0.2),
        figsize=(15, 15),
        xlabelsize=xlabel_size,
        ylabels=True,
        labels=labels,
        color=cfg.COLORS[3],
        overlap=0.6,
        bw=0.6,
        logscale=False,
    )
    add_labels(
        axes, labels, "Temperature [°C]", "Density of Temperature", "joyplot_temp.pdf"
    )

    fig, axes = joypy.joyplot(
        temp_container_filtered,
        ylim=(1e-5, 0.2),
        figsize=(15, 15),
        xlabelsize=xlabel_size,
        ylabels=True,
        labels=labels,
        color=cfg.COLORS[3],
        overlap=0.5,
        bw=0.6,
        logscale=True,
    )
    add_labels(
        axes,
        labels,
        "Temperature [°C]",
        "Log Density of Temperature",
        "joyplot_logtemp.pdf",
    )
