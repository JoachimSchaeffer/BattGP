# %% Demo/comparison of recursive GP (Huber) and spatio-temporal GP (Sarkka)

from typing import Tuple

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import get_ipython
from matplotlib import patches
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D, art3d
from plotly.offline import init_notebook_mode

import src.gp.spatiotemporal_gp as stgpm
from src.gp.wiener_kernel import WienerKernel
from src.gp.wiener_kernel_temporal import WienerTemporalKernel
from src.plotting_setup import setup_plots

init_notebook_mode()

try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        get_ipython().run_line_magic("load_ext", "autoreload")
        get_ipython().run_line_magic("autoreload", "2")
except NameError:
    pass
# Use seaborn style defaults and set the default figure size
setup_plots()


# %% Parameterize example


# "true" function
def f(t, s, noise_var=0) -> np.ndarray:
    y = -np.cos(2 * np.pi * t / 40) + np.arctan(2 * np.pi * 0.5 * s / 4) + 2
    return y + np.random.normal(0, np.sqrt(noise_var), y.shape)


# sampling time
TS = 0.1

# end of considered time interval (starting at zero)
T_END = 10

# time intervals in which measurements are recorded
# (all measurements must be aligned with the sampling time, therefore we don't allow
# here to specify the measurement times directly)
# T_MEAS_RANGES = [(0, 2), (3, 4), (5, 9)]
T_MEAS_RANGES = [(0, 10)]

# standard deviation (actual) measurement
STD_Y_MEAS = 0.125 / 2

# spatial range
S_RANGE = (-4, 4)

# reference point on spatial axis
S_REF = 0

# number of spatial basis vectors (inducing points)
N_SPATIAL_BASIS_VECTORS = 5

# GP parameter
STD_Y_GP = 0.25 / 4

OUTPUTSCALE_RBF = 5
LENGTHSCALE_RBF = 2

OUTPUTSCALE_WIENER = 2
OUTPUTSCALE_WIENER2 = 0.2

RESTRICT_MEAS_POINTS_TO_BASIS_VECTORS = False

nb = N_SPATIAL_BASIS_VECTORS

sb = np.linspace(S_RANGE[0], S_RANGE[1], nb)


# %% Create base grid and plot true function

t = np.arange(0, T_END, TS)
s = np.arange(-4, 4, 0.1)

y_true = f(t, S_REF)

(t_grid, s_grid) = np.meshgrid(t, s)
Y = f(t_grid, s_grid)

figax: Tuple[Figure, Axes3D] = plt.subplots(subplot_kw={"projection": "3d"})
(fig, ax) = figax

ax.plot_surface(t_grid, s_grid, Y, linewidth=0, alpha=0.5)
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("y")

p = patches.Rectangle((0, -0.5), 10, 4, linewidth=0, color="black", alpha=0.2)
ax.add_patch(p)
art3d.patch_2d_to_3d(p, S_REF, "y")
ax.plot3D(t, np.ones_like(t) * S_REF, y_true, color="black")

xq_surf = np.hstack((t_grid.reshape((-1, 1)), s_grid.reshape((-1, 1))))


# %% Generate (and plot) training data (uniformly sampled over time)

vb = np.zeros(t.shape, dtype=bool)

for lb, ub in T_MEAS_RANGES:
    vb = vb | ((t >= lb) & (t < ub))

tt = t[vb]

n_meas = len(tt)

if RESTRICT_MEAS_POINTS_TO_BASIS_VECTORS:
    st = np.random.choice(sb, len(tt))
else:
    st = np.random.uniform(S_RANGE[0], S_RANGE[1], len(tt))

yt = f(tt, st, STD_Y_MEAS**2)


figax: Tuple[Figure, Axes3D] = plt.subplots(subplot_kw={"projection": "3d"})
(fig, ax) = figax

ax.plot_surface(t_grid, s_grid, Y, linewidth=0, alpha=0.5)
ax.plot(tt, st, yt, "*")
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("y")

p = patches.Rectangle((0, -0.5), 10, 4, linewidth=0, color="black", alpha=0.2)
ax.add_patch(p)
art3d.patch_2d_to_3d(p, S_REF, "y")
ax.plot3D(t, np.ones_like(t) * S_REF, y_true, color="black")


# %% Create ExactGP-class for comparison


class ExactGP_(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        kernel: gpytorch.kernels.kernel,
        noise_cov: float,
    ):
        super().__init__(
            train_x,
            train_y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )

        self.likelihood.noise = torch.tensor(noise_cov)

        self.mean_module = gpytorch.means.ZeroMean()

        self.covar_module = kernel
        self.eval()
        self.likelihood.eval()

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = self(x)

        y_mean = out.mean.detach().cpu().numpy()
        y_var = out.variance.detach().cpu().numpy()

        return (y_mean, y_var)


# %% Construct and evaluate exact GP and recursive GP (Huber)

tb = tt.copy()

x_base = np.vstack([[tb[i], sb[j]] for i, j in np.ndindex((len(tb), nb))])

xt = np.hstack((tt.reshape((-1, 1)), st.reshape((-1, 1))))

mean_module = gpytorch.means.ZeroMean()

kernel_wiener = WienerKernel(active_dims=[0])
kernel_rbf = gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[1])
kernel = gpytorch.kernels.ScaleKernel(kernel_wiener) + gpytorch.kernels.ScaleKernel(
    kernel_rbf
)

kernel.kernels[0].outputscale = torch.tensor(OUTPUTSCALE_WIENER)
kernel.kernels[1].outputscale = torch.tensor(OUTPUTSCALE_RBF)
kernel.kernels[1].base_kernel.lengthscale = torch.tensor(LENGTHSCALE_RBF)

y_egp = np.nan * np.zeros_like(tt)
var_y_egp = np.nan * np.zeros_like(tt)

y_egp_smooth = np.nan * np.zeros_like(t)
var_y_egp_smooth = np.nan * np.zeros_like(t)

k_xs = np.array([10, 50, 99])
y_egp_xs = np.zeros((len(s), len(k_xs)))
ixs = 0

for i in range(len(yt)):
    egp = ExactGP_(
        torch.tensor(xt[: i + 1]), torch.tensor(yt[: i + 1]), kernel, STD_Y_GP**2
    )
    (y_egp[[i]], var_y_egp[[i]]) = egp.predict(torch.tensor([[tt[i], S_REF]]))

    if ixs < len(k_xs) and i == k_xs[ixs]:
        (y_egp_xs[:, ixs], _) = egp.predict(
            torch.tensor(
                np.hstack(
                    (tt[i] * np.ones_like(s.reshape((-1, 1))), s.reshape((-1, 1)))
                )
            )
        )
        ixs += 1


(y_egp_smooth, var_y_egp_smooth) = egp.predict(
    torch.tensor(
        np.hstack((t.reshape((-1, 1)), S_REF * np.ones_like(t.reshape((-1, 1)))))
    )
)

kernel_s = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1))
kernel_s[0].outputscale = OUTPUTSCALE_RBF
kernel_s[1].base_kernel.lengthscale = torch.tensor(LENGTHSCALE_RBF)

stgp = stgpm.ApproxSpatioTemporalGP(
    sb.reshape((-1, 1)),
    kernel_s,
    WienerTemporalKernel(OUTPUTSCALE_WIENER),
    STD_Y_GP**2,
    smoothing_steps=-1,
)

stgp2 = stgpm.ApproxSpatioTemporalGP(
    sb.reshape((-1, 1)),
    kernel_s,
    WienerTemporalKernel(OUTPUTSCALE_WIENER2),
    STD_Y_GP**2,
    smoothing_steps=-1,
)

nt = xt.shape[0]

y_stgp = np.nan * np.zeros_like(t)
var_y_stgp = np.nan * np.zeros_like(t)

y_stgp2 = np.nan * np.zeros_like(t)
var_y_stgp2 = np.nan * np.zeros_like(t)


i = 0
for k in range(len(t)):
    if np.min(np.abs(tt - t[k])) < 0.5 * TS:
        stgp.update(np.array([[xt[i, 1]]]), np.array([yt[i]]))
        stgp2.update(np.array([[xt[i, 1]]]), np.array([yt[i]]))
        i += 1

    (y_stgp[[k]], var_y_stgp[[k]]) = stgp.predict(np.array([[S_REF]]))
    (y_stgp2[[k]], var_y_stgp2[[k]]) = stgp2.predict(np.array([[S_REF]]))

    if k < len(t) - 1:
        stgp.time_step(t[k + 1] - t[k])
        stgp2.time_step(t[k + 1] - t[k])

# Pt = stgp.P[:2, :2]
(t_stgp_smooth, y_stgp_smooth, var_y_stgp_smooth) = stgp.smooth(np.array([[S_REF]]))
(t_stgp2_smooth, y_stgp2_smooth, var_y_stgp2_smooth) = stgp2.smooth(np.array([[S_REF]]))

y_stgp_smooth = y_stgp_smooth.reshape((-1,))
var_y_stgp_smooth = var_y_stgp_smooth.reshape((-1,))

y_stgp2_smooth = y_stgp2_smooth.reshape((-1,))
var_y_stgp2_smooth = var_y_stgp2_smooth.reshape((-1,))


# # Jump to the last time step to compare the results
# stgp.time_step(t[-1] - stgp.t)

sq = np.arange(sb[0], sb[-1], 0.05).reshape((-1, 1))
(yq, vyq) = stgp.predict(sq)

# %%
figax: Tuple[Figure, Axes3D] = plt.subplots(subplot_kw={"projection": "3d"})
(fig, ax) = figax

ax.plot_surface(t_grid, s_grid, Y, linewidth=0, alpha=0.5)

ax.plot(tt, st, yt, "*k")
ax.set_xlabel("t")
ax.set_ylabel("x_s")
ax.set_zlabel("y")

p = patches.Rectangle((0, -0.5), 10, 4, linewidth=0, color="black", alpha=0.2)
ax.add_patch(p)
art3d.patch_2d_to_3d(p, S_REF, "y")
ax.plot3D(t, np.ones_like(t) * S_REF, y_true, color="black")
ax.plot3D(tt, np.ones_like(tt) * S_REF, y_egp, color="blue")
ax.plot3D(t, np.ones_like(t) * S_REF, y_egp_smooth, color="orange")

for i, idx in enumerate(k_xs):
    ax.plot3D(np.ones_like(s) * tt[idx], s, y_egp_xs[:, i], color="cyan")

ax.set_zlim(-0.5, 3.5)


# %%
ax = plt.subplot(3, 1, 1)
plt.plot(t, y_true, "gray", label="ground truth")
plt.plot(tt, y_egp, "blue", label="exact gp (online)")
ax.fill_between(
    tt,
    y_egp - np.sqrt(var_y_egp),
    y_egp + np.sqrt(var_y_egp),
    color="blue",
    alpha=0.1,
    hatch="/",
    linewidth=0,
    label="exact gp (online): ±σ prior",
)
plt.plot(t, y_egp_smooth, "orange", label="exact gp (offline)")
ax.fill_between(
    t,
    y_egp_smooth - np.sqrt(var_y_egp_smooth),
    y_egp_smooth + np.sqrt(var_y_egp_smooth),
    color="orange",
    alpha=0.1,
    hatch="+",
    linewidth=0,
    label="exact gp (offline): ±σ prior",
)
plt.legend()
plt.ylabel("mean(y)")
plt.title(f"mean value of y at x_s_ref = {S_REF}")
ax.set_ylim([0, 2.5])

ax = plt.subplot(3, 1, 2)
plt.plot(t, y_true, "gray", label="ground truth")
plt.plot(t, y_stgp, "blue", label="stGP (online)")
ax.fill_between(
    t,
    y_stgp - np.sqrt(var_y_stgp),
    y_stgp + np.sqrt(var_y_stgp),
    color="blue",
    alpha=0.1,
    hatch="/",
    linewidth=0,
    label="stGP (online): ±σ prior",
)
plt.plot(t_stgp_smooth, y_stgp_smooth, "orange", label="stGP (smoothing)")
ax.fill_between(
    t,
    y_stgp_smooth - np.sqrt(var_y_stgp_smooth),
    y_stgp_smooth + np.sqrt(var_y_stgp_smooth),
    color="orange",
    alpha=0.1,
    hatch="+",
    linewidth=0,
    label="stGP gp (offline): ±σ prior",
)
plt.legend()
plt.ylabel("mean(y)")
ax.set_ylim([0, 2.5])

ax = plt.subplot(3, 1, 3)
plt.plot(t, y_true, "gray", label="ground truth")
plt.plot(t, y_stgp2, "blue", label="stGP (online)")
ax.fill_between(
    t,
    y_stgp2 - np.sqrt(var_y_stgp2),
    y_stgp2 + np.sqrt(var_y_stgp2),
    color="blue",
    alpha=0.1,
    hatch="/",
    linewidth=0,
    label="stGP (online): ±σ prior",
)
plt.plot(t_stgp2_smooth, y_stgp2_smooth, "orange", label="stGP (smoothing)")
ax.fill_between(
    t,
    y_stgp2_smooth - np.sqrt(var_y_stgp2_smooth),
    y_stgp2_smooth + np.sqrt(var_y_stgp2_smooth),
    color="orange",
    alpha=0.1,
    hatch="+",
    linewidth=0,
    label="stGP gp (offline): ±σ prior",
)
plt.legend()
plt.ylabel("mean(y)")
ax.set_ylim([0, 2.5])

plt.xlabel("t")

# %%
