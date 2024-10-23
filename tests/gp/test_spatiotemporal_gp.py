import unittest
from typing import Optional

import gpytorch
import numpy as np
import torch

from src.gp.recursive_gp import RecursiveGP
from src.gp.spatiotemporal_gp import ApproxSpatioTemporalGP
from src.gp.wiener_kernel import WienerKernel
from src.gp.wiener_kernel_temporal import WienerTemporalKernel


def _foo(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.cos(0.5 * np.pi * x).sum(axis=1) - np.cos(2 * np.pi * t / 40)


def _setup_wiener_rbf_kernel(
    outputscale_wiener,
    dims_rbf: int,
    outputscale_rbf,
    lengthscale_rbf,
    device: Optional[torch.device] = None,
) -> gpytorch.kernels.Kernel:
    kernel_wiener = WienerKernel(active_dims=[0]).to(device)
    kernel_rbf = gpytorch.kernels.RBFKernel(
        ard_num_dims=dims_rbf, active_dims=[i for i in range(1, dims_rbf + 1)]
    ).to(device)
    kernel = gpytorch.kernels.ScaleKernel(
        kernel_wiener, device=device
    ) + gpytorch.kernels.ScaleKernel(kernel_rbf, device=device)

    kernel.kernels[0].outputscale = torch.tensor(outputscale_wiener, device=device)
    kernel.kernels[1].outputscale = torch.tensor(outputscale_rbf, device=device)
    kernel.kernels[1].base_kernel.lengthscale = torch.tensor(
        lengthscale_rbf, device=device
    )

    return kernel


def _setup_rgp_wiener_rbf(
    outputscale_wiener,
    outputscale_rbf,
    lengthscale_rbf,
    noise_var,
    x_base,
    device: Optional[torch.device] = None,
) -> RecursiveGP:
    ns = x_base.shape[1] - 1

    x_base = torch.tensor(x_base)

    kernel = _setup_wiener_rbf_kernel(
        outputscale_wiener, ns, outputscale_rbf, lengthscale_rbf, device
    )

    gp = RecursiveGP(x_base, Y=None, kernel=kernel, noise_var=noise_var, device=device)

    return gp


def _setup_rbf_kernel(
    dims: int,
    outputscale: float,
    lengthscale: float,
    device: Optional[torch.device] = None,
) -> gpytorch.kernels.Kernel:
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=dims)
    kernel = gpytorch.kernels.ScaleKernel(kernel)

    kernel.outputscale = torch.tensor(outputscale)
    kernel.base_kernel.lengthscale = torch.tensor(lengthscale)

    if device is not None:
        kernel.outputscale = kernel.outputscale.to(device=device)
        kernel.base_kernel.lengthscale = kernel.base_kernel.lengthscale.to(
            device=device
        )

    return kernel


class _ExactGP(gpytorch.models.ExactGP):
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

    def predict(self, x, full_cov=False):
        assert full_cov is False

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = self(x)

        y_mean = out.mean.detach().cpu().numpy()
        y_var = out.variance.detach().cpu().numpy()

        return (y_mean, y_var)


class TestSpatioTemporalGP(unittest.TestCase):

    def test_compare_stgp_rgp(self):
        # If a rGP possesses as basis vectors the "product" of the training time points
        # and the basis points of a stGP, both GPs should give the same results.

        # TODO: This doesn't seem to hold, if the first sampling time point is not 0.0
        #       This should be revisited!

        tt = np.unique(np.random.uniform(0.0, 10.0, 10))
        tt[0] = 0.0

        SPATIAL_DIMS = 3
        OUTPUTSCALE_WIENER = 10.0
        OUTPUTSCALE_RBF = 3.0
        LENGTHSCALE_RBF = 2.0
        NOISE_VAR = 0.1

        s_base = np.random.uniform(-5.0, 5.0, (20, SPATIAL_DIMS))

        nt = len(tt)
        ns = s_base.shape[0]
        x_base = np.zeros((nt * ns, SPATIAL_DIMS + 1))
        idx = 0

        for idx_t, idx_s in np.ndindex(nt, ns):
            x_base[idx, 0] = tt[idx_t]
            x_base[idx, 1:] = s_base[idx_s, :]
            idx += 1

        st = np.random.uniform(-5.0, 5.0, (nt, SPATIAL_DIMS))

        yt = _foo(tt, st)

        sq = np.random.uniform(-5.0, 5.0, (50, SPATIAL_DIMS))
        xq = np.hstack((np.zeros((sq.shape[0], 1)), sq))

        rgp = _setup_rgp_wiener_rbf(
            outputscale_wiener=OUTPUTSCALE_WIENER,
            outputscale_rbf=OUTPUTSCALE_RBF,
            lengthscale_rbf=LENGTHSCALE_RBF,
            noise_var=NOISE_VAR,
            x_base=x_base,
        )

        stgp = ApproxSpatioTemporalGP(
            torch.tensor(s_base),
            _setup_rbf_kernel(
                dims=SPATIAL_DIMS,
                outputscale=OUTPUTSCALE_RBF,
                lengthscale=LENGTHSCALE_RBF,
            ),
            WienerTemporalKernel(outputscale=OUTPUTSCALE_WIENER),
            NOISE_VAR,
        )

        for i in range(len(tt)):
            dt = tt[i] - stgp.t

            stgp.time_step(dt)

            if i == 0:
                # check values after first time step, before updates
                xq[:, 0] = tt[i]
                [yqr, yqvarr] = rgp.predict(xq, full_cov=True)
                [yqst, yqvarst] = stgp.predict(sq, full_cov=True)

                # (actually, the mean should be still exactly zero)
                self.assertLess(np.linalg.norm(yqr - yqst), 1e-14)
                self.assertLess(
                    np.linalg.norm(yqvarr - yqvarst), 1e-12 * np.linalg.norm(yqvarr)
                )

            xts = np.hstack((tt[[i]].reshape((1, 1)), st[[i], :])).reshape((1, -1))
            rgp.update(xts, yt[[i]])
            stgp.update(st[[i], :], yt[[i]])

            xq[:, 0] = tt[i]
            [yqr, yqvarr] = rgp.predict(xq, full_cov=True)
            [yqst, yqvarst] = stgp.predict(sq, full_cov=True)

            [yqst2, yqvarst2] = stgp.predict(sq, full_cov=False)

            self.assertLess(np.linalg.norm(yqr - yqst), 1e-6 * np.linalg.norm(yqr))
            self.assertLess(
                np.linalg.norm(yqvarr - yqvarst), 1e-6 * np.linalg.norm(yqvarr)
            )

            self.assertEqual(np.linalg.norm(yqst - yqst2), 0.0)
            self.assertLess(
                np.linalg.norm(np.diag(yqvarst) - yqvarst2),
                1e-12 * np.linalg.norm(yqvarst2),
            )

    def test_compare_stgp_egp(self):
        # If all training data is given at spatial points that are basis vectors of the
        # stGP, than it should be exact and its output should equal (even at other
        # query points) the output of an eGP.

        tt = np.unique(np.random.uniform(0.0, 10.0, 10))

        SPATIAL_DIMS = 3
        OUTPUTSCALE_WIENER = 10.0
        OUTPUTSCALE_RBF = 3.0
        LENGTHSCALE_RBF = 2.0
        NOISE_VAR = 0.1

        s_base = np.random.uniform(-5.0, 5.0, (20, SPATIAL_DIMS))

        nt = len(tt)
        ns = s_base.shape[0]

        idxt = np.random.choice(ns, nt)
        st = s_base[idxt, :]

        yt = _foo(tt, st)
        xt = np.hstack((tt.reshape((-1, 1)), st))

        sq = np.random.uniform(-5.0, 5.0, (50, SPATIAL_DIMS))
        xq = np.hstack((np.zeros((sq.shape[0], 1)), sq))

        ekernel = _setup_wiener_rbf_kernel(
            outputscale_wiener=OUTPUTSCALE_WIENER,
            dims_rbf=SPATIAL_DIMS,
            outputscale_rbf=OUTPUTSCALE_RBF,
            lengthscale_rbf=LENGTHSCALE_RBF,
        )

        stgp = ApproxSpatioTemporalGP(
            torch.tensor(s_base),
            _setup_rbf_kernel(
                dims=SPATIAL_DIMS,
                outputscale=OUTPUTSCALE_RBF,
                lengthscale=LENGTHSCALE_RBF,
            ),
            WienerTemporalKernel(outputscale=OUTPUTSCALE_WIENER),
            NOISE_VAR,
        )

        for i in range(len(tt)):
            dt = tt[i] - stgp.t

            stgp.time_step(dt)
            stgp.update(st[[i], :], yt[[i]])
            [yqst, yqvarst] = stgp.predict(sq, full_cov=False)

            egp = _ExactGP(
                torch.tensor(xt[: i + 1, :]),
                torch.tensor(yt[: i + 1]),
                ekernel,
                NOISE_VAR,
            )
            xq[:, 0] = tt[i]
            [yqe, yqvare] = egp.predict(torch.tensor(xq), full_cov=False)

            self.assertLess(np.linalg.norm(yqe - yqst), 1e-6 * np.linalg.norm(yqe))
            self.assertLess(
                np.linalg.norm(yqvare - yqvarst), 1e-6 * np.linalg.norm(yqvare)
            )

    def test_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU available")

        tt = np.unique(np.random.uniform(0.0, 10.0, 10))

        SPATIAL_DIMS = 3
        OUTPUTSCALE_WIENER = 10.0
        OUTPUTSCALE_RBF = 3.0
        LENGTHSCALE_RBF = 2.0
        NOISE_VAR = 0.1

        s_base = np.random.uniform(-5.0, 5.0, (20, SPATIAL_DIMS))

        nt = len(tt)

        st = np.random.uniform(-5.0, 5.0, (nt, SPATIAL_DIMS))

        yt = _foo(tt, st)

        sq = np.random.uniform(-5.0, 5.0, (50, SPATIAL_DIMS))

        # get reference soultion using the cpu
        stgp_cpu = ApproxSpatioTemporalGP(
            torch.tensor(s_base),
            _setup_rbf_kernel(
                dims=SPATIAL_DIMS,
                outputscale=OUTPUTSCALE_RBF,
                lengthscale=LENGTHSCALE_RBF,
            ),
            WienerTemporalKernel(outputscale=OUTPUTSCALE_WIENER),
            NOISE_VAR,
        )

        for i in range(len(tt)):
            dt = tt[i] - stgp_cpu.t

            stgp_cpu.time_step(dt)
            stgp_cpu.update(st[[i], :], yt[[i]])

        (yq_cpu, yqvar_cpu) = stgp_cpu.predict(sq, full_cov=True)

        # Lets use the GPU
        device = torch.device("cuda")
        torch.set_default_device(device)

        stgp_cuda = ApproxSpatioTemporalGP(
            torch.tensor(s_base),
            _setup_rbf_kernel(
                dims=SPATIAL_DIMS,
                outputscale=OUTPUTSCALE_RBF,
                lengthscale=LENGTHSCALE_RBF,
            ),
            WienerTemporalKernel(outputscale=OUTPUTSCALE_WIENER),
            NOISE_VAR,
            device=device,
        )

        for i in range(len(tt)):
            dt = tt[i] - stgp_cuda.t

            stgp_cuda.time_step(dt)
            stgp_cuda.update(st[[i], :], yt[[i]])

        (yq_cuda, yqvar_cuda) = stgp_cuda.predict(sq, full_cov=True)

        torch.set_default_device("cpu")

        self.assertLess(np.linalg.norm(yq_cpu - yq_cuda), 1e-6 * np.linalg.norm(yq_cpu))

        self.assertLess(
            np.linalg.norm(yqvar_cpu - yqvar_cuda),
            1e-6 * np.linalg.norm(yqvar_cpu),
        )


if __name__ == "__main__":
    unittest.main()
