from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def add_truth(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str = "truth"):
    ax.plot(x.reshape((-1,)), y, "k:", label=label)


def add_measurements(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str = "meas."):
    ax.plot(x.reshape((-1,)), y, "bo", label=label)


def add_gp_data(
    ax: plt.Axes,
    label: str,
    color: str,
    x: np.ndarray,
    m_prior=None,
    var_prior=None,
    m_post=None,
    var_post=None,
    x_base=None,
    hatch="/",
    linestyle="-",
):
    x = x.reshape((-1,))

    if m_prior is not None:
        ax.plot(x, m_prior, "--", color=color, label=f"{label}: μ prior")

    if var_prior is not None:
        ax.fill_between(
            x,
            m_prior - np.sqrt(var_prior),
            m_prior + np.sqrt(var_prior),
            color=color,
            alpha=0.1,
            hatch=hatch,
            linewidth=0,
            label=f"{label}: ±σ prior",
        )

    if m_post is not None:
        ax.plot(x, m_post, color=color, linestyle=linestyle, label=f"{label}: μ post")

    if var_post is not None:
        ax.fill_between(
            x,
            m_post - np.sqrt(var_post),
            m_post + np.sqrt(var_post),
            color="gray",
            alpha=0.2,
            hatch=hatch,
            linewidth=0,
            label=f"{label}: ±σ post",
        )

    if x_base is not None:
        for i, xb in enumerate(x_base):
            ax.axvline(
                xb,
                color=color,
                linewidth=0.5,
                label=f"{label}: x_base" if i == 0 else "_",
            )


def plot_r0_prediction(
    ax: Optional[plt.Axes] = None,
    plot_to_age: Optional[float] = None,
    extrapolation_horizon: Optional[float] = None,
    color: Optional[tuple[float, float, float] | str] = None,
    label: Optional[str] = None,
    linestyle: Optional[str] = None,
    x_padding_percent: float = 0.75,
    age: Optional[float] = None,
    time: np.ndarray = None,
    mean: np.ndarray = None,
    var: np.ndarray = None,
    datalabel: float = "Pack",
    **kwargs,
) -> None:
    """Plot mean function of GP model for cell."""

    assert not ((plot_to_age is not None) and (extrapolation_horizon is not None))

    if extrapolation_horizon is not None:
        plot_to_age = age + extrapolation_horizon
        # Make figure wider
        fig = plt.gcf()
        # Get the current size
        fig_size = fig.get_size_inches()
        stretch = 1 + extrapolation_horizon / age
        fig.set_size_inches(stretch * fig_size[0], fig_size[1])
    elif plot_to_age is None:
        plot_to_age = age

    plotargs = {}

    if var is not None:
        plotargs["var_post"] = var * 1000**2

    plotargs["linestyle"] = linestyle
    xlabel = "Age [days]"

    if color is None:
        color = "blue"

    add_gp_data(ax, datalabel, color, time, m_post=mean * 1000, **plotargs)
    ax.set_xlabel(xlabel)

    if label is None:
        label = R"R($I_{op}$, SOC$_{op}$, T$_{op}$) + R(t)"

    if extrapolation_horizon is not None:
        ax.axvline(age, label="_", color="gray")

    ax.set_ylabel(R"R [m$\Omega$]")
    ax.legend(frameon=True, fancybox=True, framealpha=1, loc="upper left")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.5)

    x_padding_abs = plot_to_age * x_padding_percent / 100
    ax.set_xlim(-x_padding_abs, plot_to_age + x_padding_abs)
