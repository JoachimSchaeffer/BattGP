from typing import Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from src import config as cfg
from src.batt_data import data_utils
from src.batt_data.batt_data import BattData, SegmentCriteria
from src.batt_models.battcellgp_full import BatteryCellGP_Full
from src.gp.plotting import add_gp_data
from src.gp.spatiotemporal_gp import ApproxSpatioTemporalGP
from src.gp.wiener_kernel_temporal import WienerTemporalKernel
from src.path_setup import setup_paths
from src.plot_utils import get_basis_vectors
from src.plotting_setup import setup_plots

setup_paths()
data_utils.build_data_cache()


def get_normalized_distance(
    xt: np.ndarray, op: np.ndarray, lengthscale_rbf: np.ndarray
):
    xt = xt - op
    xt = xt / lengthscale_rbf.reshape((1, -1))

    return np.linalg.norm(xt, axis=1)


def get_basis_marks(basis_vectors, op_ref, lengthscale_rbf, idx, noise_variance):
    # remove op_ref
    basis_vectors = basis_vectors[np.any(basis_vectors != op_ref, axis=1), :]

    x = basis_vectors[:, idx]

    x = np.unique(x)

    data = []

    for xx in x:
        op_refx = op_ref.copy()
        op_refx[idx] = xx
        bvs = basis_vectors[basis_vectors[:, idx] == xx]
        dists = get_normalized_distance(bvs, op_refx, lengthscale_rbf)

        value = np.sqrt(noise_variance * (1 - np.exp(-2 * dists)))

        # data.append((xx, sum(dists)))
        data.append((xx, min(value)))

    data.sort(key=lambda v: v[1], reverse=True)

    return data


def map_mark_color(v) -> Tuple[float, float, float]:
    MAX_V = 1e-3
    MAX_C = 0.9

    if v > MAX_V:
        c = MAX_C
    else:
        c = v * MAX_C / MAX_V

    return (c, c, c)


# Settings

# MODE = "single" # : Single cell
MODE = "all_cells"  # : All cells of one system
batt_id = "8"
# cell_nb will be ignored if MODE=="all_Cells"
cell_nb = 1
# Use seaborn style defaults and set the default figure size
setup_plots()

labelfontsize = 16
titlefontsize = 18


# max_age = None -> battdata.age
# MAX_AGE = 100
MAX_AGE = None

# BASIS_VECTOR_STRATEGY = "kmeans"
BASIS_VECTOR_STRATEGY = "uniform"

PLOT_BASISMARKS = True
PLOT_LENGTHSCALES = False

ocv_path = "data/ocv_linear_approx.csv"

segment_criteria: SegmentCriteria = SegmentCriteria(
    soc_upper_limit=cfg.SOC_UPPER_LIMIT,
    soc_lower_limit=cfg.SOC_LOWER_LIMIT,
    ibat_upper_limit=cfg.Ibat_UPPER_LIMIT,
    ibat_lower_limit=cfg.Ibat_LOWER_LIMIT,
    t_upper_limit=cfg.T_UPPER_LIMIT,
    t_lower_limit=cfg.T_LOWER_LIMIT,
)

cell_characterstics = data_utils.read_cell_characteristics(path=ocv_path)

# Lets go!
battdata = BattData(
    batt_id,
    cell_characterstics,
    segment_selection=True,
    segment_criteria=segment_criteria,
    gap_removal=cfg.GAP_REMOVAL,
)

# Manual way of creating the model.
battdata.update_op()

if MAX_AGE is None:
    MAX_AGE = battdata.age
colors = cfg.COLORS

r_range = (8, 24)

if MAX_AGE is None:
    MAX_AGE = battdata.age
colors = cfg.COLORS

if MODE == "all_cells":
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    cells = [i for i in range(1, 9)]
elif MODE == "single_cell":
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharey=True)
    cells = [cell_nb]
else:
    raise NotImplementedError(
        f"Mode must be either 'single_cell' or 'all_cells', not {MODE}"
    )


for cell_nb in cells:
    (xt, yt) = battdata.generateTrainingData(cell_nb)
    xt = xt.astype(float)
    yt = yt.astype(float)

    yt = yt[xt[:, 0] <= MAX_AGE]
    xt = xt[xt[:, 0] <= MAX_AGE, :]

    xte = xt

    tt = xt[:, 0]
    xt = xt[:, 1:]
    Ts = 1 / 24

    # op = np.array([[battdata.mean_op.I, battdata.mean_op.SOC, battdata.mean_op.T]])
    op = np.array([[-15, 90, 25]])

    params = BatteryCellGP_Full.get_default_parameters()

    basis_vectors = get_basis_vectors(
        battdata,
        ref_op=op.reshape((-1,)),
        nbasis=(5, 5, 5),
    )

    kernel_s = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))
    kernel_s.outputscale = params["outputscale_rbf"]
    kernel_s.base_kernel.lengthscale = params["lengthscale_rbf"]

    stgp = ApproxSpatioTemporalGP(
        basis_vectors,
        kernel_s,
        WienerTemporalKernel(params["outputscale_wiener"]),
        params["noise_variance"],
        smoothing_steps=None,
    )

    t_end = MAX_AGE

    t_stgp = np.arange(0, t_end, Ts)
    r0_stgp = np.zeros_like(t_stgp)
    r0var_stgp = np.zeros_like(t_stgp)
    ds_cnt = np.zeros_like(t_stgp)

    (r0_stgp[[0]], r0var_stgp[[0]]) = stgp.predict(op)

    min_sampling_time = np.min(np.diff(tt))
    max_samples = int(np.ceil(Ts / min_sampling_time))

    idx1 = 0

    for i in tqdm.trange(len(t_stgp) - 1):
        idx0 = np.argmax(tt[idx1 : idx1 + max_samples] >= t_stgp[i]) + idx1
        idx1 = np.argmax(tt[idx0 : idx0 + max_samples] > t_stgp[i + 1]) + idx0

        n = idx1 - idx0

        if n > 0:
            stgp.update(xt[idx0:idx1, :], yt[idx0:idx1])
            ds_cnt[[i + 1]] = n

        (r0_stgp[[i + 1]], r0var_stgp[[i + 1]]) = stgp.predict(op)

        stgp.time_step(Ts)

    op_ref = op.reshape((-1,))

    t = MAX_AGE

    x_I = np.linspace(
        segment_criteria.ibat_lower_limit, segment_criteria.ibat_upper_limit, 1000
    )
    x_I = np.linspace(-30, -5, 1000)

    x_SOC = np.linspace(
        segment_criteria.soc_lower_limit, segment_criteria.soc_upper_limit, 1000
    )

    x_T = np.linspace(
        segment_criteria.t_lower_limit, segment_criteria.t_upper_limit, 1000
    )

    X = np.zeros((len(x_I), 4))

    X[:, 0] = t
    X[:, 1] = x_I
    X[:, 2] = op_ref[1]
    X[:, 3] = op_ref[2]

    (r0_Ir, r0var_Ir) = stgp.predict(X[:, 1:])
    r0_Ir = r0_Ir.reshape((-1,))

    X[:, 1] = op_ref[0]
    X[:, 2] = x_SOC
    X[:, 3] = op_ref[2]

    (r0_SOCr, r0var_SOCr) = stgp.predict(X[:, 1:])
    r0_SOCr = r0_SOCr.reshape((-1,))

    X[:, 0] = t
    X[:, 1] = op_ref[0]
    X[:, 2] = op_ref[1]
    X[:, 3] = x_T

    (r0_Tr, r0var_Tr) = stgp.predict(X[:, 1:])
    r0_Tr = r0_Tr.reshape((-1,))

    basis_marks_I = get_basis_marks(
        basis_vectors,
        op_ref,
        np.array(params["lengthscale_rbf"]),
        0,
        params["noise_variance"],
    )
    basis_marks_SOC = get_basis_marks(
        basis_vectors,
        op_ref,
        np.array(params["lengthscale_rbf"]),
        1,
        params["noise_variance"],
    )
    basis_marks_T = get_basis_marks(
        basis_vectors,
        op_ref,
        np.array(params["lengthscale_rbf"]),
        2,
        params["noise_variance"],
    )

    if MODE == "all_cells":
        color = "k"
        color_two = "k"
        color_three = colors[cell_nb - 1]
        linestyle = cfg.LINESTYLES[cell_nb - 1]
    elif MODE == "single_cell":
        color = "orange"
        color_two = "yellow"
        color_three = "red"
        linestyle = "-"
    add_gp_data(
        axs[0],
        f"R(I, {op_ref[1]:.0f}% SOC, {op_ref[2]:.0f}°C)",
        color_three,
        x_I,
        m_post=r0_Ir * 1000,
        var_post=r0var_Ir * 1e6,
        hatch=None,
        linestyle=linestyle,
    )
    axs[0].set_ylim(r_range)
    axs[0].axvline(op_ref[0], label=f"I Ref.: {op_ref[0]:.0f} A", color=color)

    if PLOT_BASISMARKS:
        for x, d in basis_marks_I:
            axs[0].axvline(
                x, color=map_mark_color(d), linestyle="--", label="Basis Points"
            )

    if PLOT_LENGTHSCALES:
        axs[0].axvspan(
            op_ref[0] - params["lengthscale_rbf"][0],
            op_ref[0] + params["lengthscale_rbf"][0],
            color=color_two,
            alpha=0.2,
            label="I_ref ± lengthscale",
        )
    axs[0].set_xlabel("I (A)", fontsize=labelfontsize)
    axs[0].set_ylabel("R (mΩ)", fontsize=labelfontsize)
    axs[0].set_title("Current", fontsize=titlefontsize - 1)

    if MODE == "all_cells":
        axs[0].tick_params(axis="x", labelsize=labelfontsize - 2)
        axs[0].tick_params(axis="y", labelsize=labelfontsize - 2)

    # Remove duplicate legend entries
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys())

    add_gp_data(
        axs[1],
        f"R({op_ref[0]:.0f} A, SOC, {op_ref[2]:.0f}°C)",
        color_three,
        x_SOC,
        m_post=r0_SOCr * 1000,
        var_post=r0var_SOCr * 1e6,
        hatch=None,
        linestyle=linestyle,
    )
    axs[1].set_ylim(r_range)
    axs[1].axvline(op_ref[1], label=f"SOC Ref.: {op_ref[1]:.0f}%", color=color)

    if PLOT_BASISMARKS:
        for x, d in basis_marks_SOC:
            axs[1].axvline(
                x, color=map_mark_color(d), linestyle="--", label="Basis Points"
            )

    if PLOT_LENGTHSCALES:
        axs[1].axvspan(
            op_ref[1] - params["lengthscale_rbf"][1],
            op_ref[1] + params["lengthscale_rbf"][1],
            color="k",
            alpha=0.2,
            label="SOC_ref ± lengthscale",
        )
    axs[1].set_title("SOC", fontsize=titlefontsize - 1)

    if MODE == "all_cells":
        axs[1].tick_params(axis="x", labelsize=labelfontsize - 2)
        axs[1].tick_params(axis="y", labelsize=labelfontsize - 2)

    # Remove duplicate legend entries
    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys())

    if MODE == "single_cell":
        axs[1].set_title(
            "Operating Characteristics Temperature", fontsize=titlefontsize - 1
        )
    else:
        axs[1].set_title("SOC", fontsize=titlefontsize - 1)
        axs[1].tick_params(axis="x", labelsize=labelfontsize - 2)
        axs[1].tick_params(axis="y", labelsize=labelfontsize - 2)

    handles, labels = axs[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[1].legend(by_label.values(), by_label.keys())

    # ax: plt.Axes = plt.subplot(1, 3, 3)
    add_gp_data(
        axs[2],
        f"R({op_ref[0]:.0f} A, {op_ref[1]:.0f}% SOC, T)",
        color_three,
        x_T,
        m_post=r0_Tr * 1000,
        var_post=r0var_Tr * 1e6,
        hatch=None,
        linestyle=linestyle,
    )

    axs[2].set_ylim(r_range)
    axs[2].axvline(op_ref[2], label=f"T Ref.: {op_ref[2]:.0f}°C", color=color)

    if PLOT_BASISMARKS:
        for x, d in basis_marks_T:
            axs[2].axvline(
                x, color=map_mark_color(d), linestyle="--", label="Basis Points"
            )

    if PLOT_LENGTHSCALES:
        axs[2].axvspan(
            op_ref[2] - params["lengthscale_rbf"][2],
            op_ref[2] + params["lengthscale_rbf"][2],
            color="k",
            alpha=0.2,
            label="T_ref ± lengthscale",
        )

    axs[2].set_xlabel("T (°C)", fontsize=labelfontsize)

    if MODE == "single_cell":
        axs[2].set_title(
            "Operating Characteristics Temperature", fontsize=titlefontsize - 1
        )
    else:
        axs[2].set_title("Temperature", fontsize=titlefontsize - 1)
        axs[2].tick_params(axis="x", labelsize=labelfontsize - 2)
        axs[2].tick_params(axis="y", labelsize=labelfontsize - 2)

    # Remove duplicate legend entries
    handles, labels = axs[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[2].legend(by_label.values(), by_label.keys())

    if MODE == "single_cell":
        axs[1].set_ylabel("R (mΩ)", fontsize=labelfontsize)
        axs[2].set_ylabel("R (mΩ)", fontsize=labelfontsize)
        fig.suptitle(
            f"Battery #{batt_id}, Cell {cell_nb}, t = {t_end:.0f} days (data points: {round(sum(ds_cnt))}) \n Recursive Spatio-Temporal GP Model",
            fontsize=titlefontsize,
        )
        plt.tight_layout()
        if MAX_AGE is not None:
            plt.savefig(
                f"results/{batt_id}_cell{cell_nb}_operating_chracteristics_{MAX_AGE:.0f}days.pdf",
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"results/{batt_id}_cell{cell_nb}_operating_chracteristics.pdf",
                bbox_inches="tight",
            )
    else:
        fig.suptitle(
            f"Operating Characteristics for System #{batt_id}, t = {t_end:.0f} days (data points: {round(sum(ds_cnt))})",
            fontsize=titlefontsize,
        )
        plt.tight_layout()
        if MAX_AGE is not None:
            plt.savefig(
                f"results/{batt_id}_all_cell_operating_chracteristics_{MAX_AGE:.0f}days.pdf",
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                f"results/{batt_id}_all_cell_operating_chracteristics.pdf",
                bbox_inches="tight",
            )
