import datetime
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .. import config as cfg
from ..batt_data.batt_data import BattData
from ..gp.plotting import plot_r0_prediction
from ..plot_utils import add_second_datetime_xaxis_below
from .batt_cell_gp_protocol import IBatteryCellGP
from .battgp import BattGPResult


def plot_fault_probabilities(
    r0_fault_df: pd.DataFrame,
    gp_res: BattGPResult,
    x_padding_percent: float = 0.75,
    save: bool = True,
    single_row: bool = True,
    legend: bool = False,
    **kwargs,
) -> None:
    """Plot fault probabilities"""
    if len(r0_fault_df.columns) != 5:
        Warning("Not all calculated fault probabilities are plotted.")

    if single_row:
        nrows = 1
        # scale = 1 / 1.8
        figsize = (6.2, 1.5)
    else:
        nrows = 5
        figsize = (12, 14)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=figsize,
        sharex=True,
        # gridspec_kw={"height_ratios": [5, 5, 5]},
    )
    if single_row:
        axs = [axs]

    axs: List[plt.Axes]

    # t = np.linspace(0, self.batt_data.age, 300)
    for i, cell in enumerate(gp_res.cellmodels):
        color = cfg.COLORS[np.mod(i, len(cfg.COLORS))]
        linestyle = cfg.LINESTYLES[np.mod(i, len(cfg.LINESTYLES))]
        if single_row:
            cols = [f"R{cell.cellnr} band_i fault prob"]
        else:
            cols = [
                f"R{cell.cellnr} band_i fault prob",
                f"R_upper{cell.cellnr} band_i fault prob",
                f"R_lower{cell.cellnr} band_i fault prob",
                f"R{cell.cellnr} thres fault prob",
            ]
        for i, col in enumerate(cols):
            axs[i].plot(
                r0_fault_df["t"],
                r0_fault_df[col],
                label=f"Cell {cell.cellnr}",
                color=color,
                linestyle=linestyle,
            )

    if cfg.PLOT_WEAKEST_LINK:
        axs[0].plot(
            r0_fault_df["t"],
            r0_fault_df["Weakest_link_stat"],
            label="Weakest Link",
            color="#0000f4",
            linestyle=(0, (1, 2)),
            linewidth=0.75,
        )

    if single_row:
        ax2 = axs[0].twinx()
        ax2.plot(
            r0_fault_df["t"],
            r0_fault_df["R0 mean_gp cells var"],
            color="black",
            linestyle="solid",
            label="Variance of $R_{t}$ Mean",
        )
        ax2.set_ylabel("Variance")
        ax2.set_ylim(0, 1e-6)
    else:
        axs[4].plot(
            r0_fault_df["t"],
            r0_fault_df["R0 mean_gp cells var"],
            color="black",
            linestyle="solid",
        )

    legend_dict = {
        "markerscale": 0.2,
        "loc": "upper left",
        "ncols": int((i + 1) / 2),
    }
    if single_row:
        # Merge the two legends
        handles, labels = axs[0].get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
        axs[0].legend(handles, labels, ncols=2, markerscale=0.2, loc="upper left")
        # if "band" in kwargs:
        #     band = kwargs["band"]
        # else:
        #     band = cfg.BAND
        # axs[0].set_title(f"Battery #{batt_data.id}, Band: {1000*band:.2f} mOhm")
        axs[0].set_ylabel("Fault Probability")
        if not legend:
            axs[0].legend().remove()
    else:
        for i, ax in enumerate(axs):
            ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)
            if i != len(axs) - 1:
                ax.legend(**legend_dict)
                ax.set_xlabel("")
                ax.set_ylabel("Fault Probability")
            else:
                ax.set_xlabel("Age [days]")
                ax.set_ylabel("R0 Variance")
            if not legend:
                ax.legend().remove()

        axs[0].set_title("R-Band")
        axs[1].set_title("R-Band Upper")
        axs[2].set_title("R-Band Lower")
        axs[3].set_title("R Upper Threshold")
        axs[4].set_title("Variance of Cell R Mean")
        # fig.suptitle(
        #     f"Battery #{batt_data.id}, Band: {cfg.BAND} mOhm",
        #     y=0.95,
        # )

    x_padding_abs = gp_res.batt_data.age * x_padding_percent / 100
    axs[-1].set_xlim(-x_padding_abs, r0_fault_df["t"].values[-1] + x_padding_abs)
    axs[-1].set_xlabel("Age [days]")
    # start_datetime = self.batt_data.df.index[0].to_pydatetime()
    # add_second_datetime_xaxis_below(fig, axs, start_datetime)
    if save:
        if "save_path" in kwargs:
            save_path = kwargs["save_path"]
        else:
            save_path = cfg.PATH_GP_MODELING
        # make directory if not exist
        save_path_batt = os.path.join(save_path, gp_res.batt_data.id)
        os.makedirs(save_path_batt, exist_ok=True)
        if "causal" in kwargs and kwargs["causal"]:
            trailing_name = "_causal"
        else:
            trailing_name = "_acausal"
        if "r0_band" in kwargs:
            trailing_name += f"_band{1000*kwargs['r0_band']:.2f}"
        figname = os.path.join(
            save_path_batt,
            f"Batt{gp_res.batt_data.id}_fault{trailing_name}_prob.pdf",
        )
        fig.savefig(figname, bbox_inches="tight")
    # plt.show()


# The following functions are only for plotting
def plot_cell_r0_predictions(
    gp_res: BattGPResult,
    single_plot: bool = False,
    save: bool = False,
    trailing_name: str = "",
    legend: bool = False,
    add_datetime_xaxis: bool = False,
    y_lim: Tuple[Optional[float], Optional[float]] = [None, None],
    force_ylim: bool = False,
    **plotargs,
) -> None:
    """Plot mean and variance of GP model for each cell."""

    axs: List[plt.Axes]

    if single_plot:
        figsize = (10 / 1.15, 5.5 / 1.3)
        fig, axs = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [13, 3, 2]},
        )
    else:
        figsize = (25, 12.5)
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=figsize, sharex=True)
        plt.subplots_adjust(hspace=0.5)

    ref_op = f"Ref. OP: {gp_res.ref_op.disp_str()}"
    fig.suptitle(f"System: #{gp_res.batt_data.id} {ref_op}", fontsize=17, y=0.938)

    if single_plot:
        for ax, cellmodel in enumerate(gp_res.cellmodels):
            color = cfg.COLORS[np.mod(ax, len(cfg.COLORS))]
            linestyle = cfg.LINESTYLES[np.mod(ax, len(cfg.LINESTYLES))]
            plot_r0_prediction_cellmodel(
                gp_res.get_cell_data(cellmodel.cellnr, signals=["t", "r0", "r0var"]),
                cellmodel,
                gp_res.batt_data,
                ax=axs[0],
                label=f"Cell {cellmodel.cellnr}",
                **plotargs,
                title="",
                color=color,
                linestyle=linestyle,
            )
        # Remove xlabel from top plot
        axs[0].set_xlabel("")
        (y_lim_low, y_lim_high) = axs[0].get_ylim()
        if force_ylim and y_lim is not None:
            y_lim_low = y_lim[0]
            y_lim_high = y_lim[1]
        elif not force_ylim:
            if y_lim[0] is not None:
                y_lim_low = np.min([y_lim[0], y_lim_low])
            if y_lim[1] is not None:
                y_lim_high = np.max([y_lim[1], y_lim_high])
        axs[0].set_ylim(y_lim_low, y_lim_high)

        # Plot the data as histograms
        (Xt, yt) = gp_res.cellmodels[-1].get_training_data()
        data_points = Xt[:, 0]
        bins = int(np.floor(np.max(data_points) + 1) / 2)
        axs[2].hist(data_points, bins=bins, color="red")
        # set y axis to log scale
        axs[2].set_yscale("log")
        if "y_lim_data" in plotargs:
            axs[2].set_ylim(plotargs["y_lim_data"])
        else:
            axs[2].set_ylim([0.8, 1001])
        axs[2].yaxis.set_major_locator(mticker.LogLocator(base=100))
        axs[2].set_ylabel("Data")
        # activate horizontal grid
        axs[2].grid(axis="y", linestyle="--", alpha=0.5)

        mean_op = gp_res.batt_data.mean_op

        mean_op_str = f"Mean selected data: {mean_op.disp_str()}"
        axs[2].set_title(
            f"# data points: {len(Xt[:, 0])}, {mean_op_str}", y=0.89, fontsize=11.5
        )
        # "\n" f"{ref_op_str}"

        # get the temperature of each cell and put it in a pandas data frame
        temp = []
        for cellmodel in gp_res.cellmodels:
            (Xt, yt) = gp_res.cellmodels[-1].get_training_data()
            temp.append(Xt[:, 3])
        time = Xt[:, 0]
        # round the time up to the next integer
        time = np.ceil(time)
        temp = np.array(temp).T
        temp_df = pd.DataFrame(
            temp, columns=[f"Cell {i}" for i in range(1, temp.shape[1] + 1)]
        )
        temp_df["Time"] = time
        # now group by time and get the average temperature for each day
        temp_df = temp_df.groupby("Time").mean()
        # get the time and mean temperature for each day
        time = temp_df.index
        temp = temp_df.values.mean(axis=1)

        axs[1].scatter(time, temp, color="#156B90", label="Mean Temp.", s=np.sqrt(0.95))
        axs[1].set_ylim(9, 55)
        axs[1].set_ylabel("T [°C]")

        if legend:
            # Get the legend handles and labels
            handles, labels = axs[0].get_legend_handles_labels()
            # Remove all legend handles with "±σ"
            list_tuples = [
                (handle, label)
                for i, (handle, label) in enumerate(zip(handles, labels))
                if "±σ" not in label or i in [0, 1]
            ]
            # Unzip the list of tuples into two lists
            handles, labels = list(zip(*list_tuples))
            # Find the index of the first legend handle with "±σ"
            idx = next(i for i, label in enumerate(labels) if "±σ" in label)
            labels_ = labels[idx].split(": ")[1]
            # Move the legend handle with "±σ" to the end of the list
            handles = handles[:idx] + handles[idx + 1 :] + (handles[idx],)  # noqa: E203
            labels = labels[:idx] + labels[idx + 1 :] + (labels_,)  # noqa: E203
            axs[0].legend(handles, labels)
        else:
            axs[0].get_legend().remove()
        axs[-1].set_xlabel("Age [days]")
    else:
        for cellmodel, ax in zip(gp_res.cellmodels, axs.ravel()):
            plot_r0_prediction_cellmodel(
                gp_res.get_cell_data(cellmodel.cellnr, signals=["t", "r0", "r0var"]),
                cellmodel,
                gp_res.batt_data,
                ax=ax,
                **plotargs,
            )

    # Reset the figure size , theres a bug somewhere causing an unwanted resize
    fig.set_size_inches(figsize[0], figsize[1])
    start_datetime = gp_res.batt_data.df.index[0].to_pydatetime()
    if single_plot and add_datetime_xaxis:
        add_second_datetime_xaxis_below(fig, axs, start_datetime)

    # iterate through all ticks and labels and increase the fotnsize
    for ax in axs.reshape((-1,)):
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(14)
        for item in [ax.xaxis.get_label(), ax.yaxis.get_label()]:
            item.set_fontsize(16)

    # fig.tight_layout()
    if save:
        # make directory if not exist
        if "save_path" not in plotargs:
            save_path = cfg.PATH_GP_MODELING
        else:
            save_path = plotargs["save_path"]
        save_path_batt = os.path.join(save_path, gp_res.batt_data.id)
        os.makedirs(save_path_batt, exist_ok=True)
        if single_plot:
            figname = os.path.join(
                save_path_batt,
                f"Batt{gp_res.batt_data.id}_R_single_axis{trailing_name}.pdf",
            )
        else:
            figname = os.path.join(
                save_path_batt,
                f"Batt{gp_res.batt_data.id}_R_multiaxis{trailing_name}.pdf",
            )
        fig.savefig(figname, bbox_inches="tight")
        if "save_config" in plotargs and plotargs["save_config"]:
            # Save a copy of src/config.py as a txt file in the same folder
            config_file = os.path.join(save_path, "config.txt")
            with open(config_file, "w") as f:
                f.write(f"config.py as of {datetime.datetime.now()}\n\n")
                with open("src/config.py", "r") as f2:
                    f.write(f2.read())
    # plt.show()


def plot_r0_prediction_cellmodel(
    df: pd.DataFrame,
    cellmodel: IBatteryCellGP,
    batt_data: BattData,
    ax: Optional[plt.Axes] = None,
    plot_to_age: Optional[float] = None,
    extrapolation_horizon: Optional[float] = None,
    title: Optional[str] = None,
    plot_data: bool = False,
    color: Optional[tuple[float, float, float] | str] = None,
    label: Optional[str] = None,
    linestyle: Optional[str] = None,
    add_datetime_xaxis: bool = False,
    **kwargs,
) -> None:
    """Plot mean function of GP model for cell."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    if cellmodel.cellnr == -1:
        datalabel = "Pack"
    else:
        datalabel = f"Cell {cellmodel.cellnr}"

    n = df.shape[0]

    if n >= 1000:
        idx = np.round(np.linspace(0, n - 1, 500)).astype(int)
        # Make sure to include the last index
        idx[-1] = n - 1
    else:
        idx = np.arange(n)

    plot_r0_prediction(
        ax=ax,
        plot_to_age=plot_to_age,
        extrapolation_horizon=extrapolation_horizon,
        color=color,
        label=label,
        linestyle=linestyle,
        age=batt_data.age,
        time=df["t"].values[idx],
        mean=df["r0"].values[idx],
        var=df["r0var"].values[idx],
        datalabel=datalabel,
        **kwargs,
    )

    if plot_data:
        (xt, yt) = cellmodel.get_training_data()
        xt = xt[:, 0]
        ax.plot(
            xt,
            yt * 1000,
            "r.",
            markersize=0.75,
            label=label,
        )
        # plotting.add_measurements(ax, xt, yt * 1000, "data")

    if add_datetime_xaxis and fig is not None:
        start_dt = batt_data.df.index[0]
        add_second_datetime_xaxis_below(fig, [ax], start_dt)

    if title is None:
        ax.set_title(f"GP-Model, {datalabel} (Battery {batt_data.id})")
    else:
        ax.set_title(title)
