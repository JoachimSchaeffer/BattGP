import unittest
from typing import Optional

import gpytorch
import numpy as np
import torch

from src.gp.recursive_gp import RecursiveGP
from src.gp.standard_models import ScaledRBFModel


def _foo(x: np.ndarray) -> np.ndarray:
    return np.cos(0.5 * np.pi * x).sum(axis=1)


def _setup_rbf_kernel(
    outputscale, lengthscale, device: Optional[torch.device] = None
) -> gpytorch.kernels.Kernel:
    kernel = gpytorch.kernels.RBFKernel()
    kernel = gpytorch.kernels.ScaleKernel(kernel)

    kernel.outputscale = torch.tensor(outputscale)
    kernel.base_kernel.lengthscale = torch.tensor(lengthscale)

    if device is not None:
        kernel.outputscale = kernel.outputscale.to(device=device)
        kernel.base_kernel.lengthscale = kernel.base_kernel.lengthscale.to(
            device=device
        )

    return kernel


def _setup_rgp_rbf(
    outputscale, lengthscale, noise_var, x_base, device: Optional[torch.device] = None
):

    x_base = torch.tensor(x_base)

    kernel = _setup_rbf_kernel(outputscale, lengthscale)
    gp = RecursiveGP(x_base, Y=None, kernel=kernel, noise_var=noise_var, device=device)

    return gp


class TestRecursiveGP(unittest.TestCase):

    def test_rbf_1d_1basepoint(self):

        x = np.array([[1.0]])
        gp = _setup_rgp_rbf(outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x)
        gp.update(x, np.array([10.0]))
        (yq, var_yq) = gp.predict(x)

        # If we have only one training point that corresponds to a base vector, and
        # the measurement variance equals the prior's variance, we expect to predict
        # for the same point a value that is the mean of the prior (0) and the training
        # data, with the variance also the half of the prior's/measurement variance
        self.assertAlmostEqual(yq[0], 5.0)
        self.assertAlmostEqual(var_yq[0], 1.5)

        # If we add another training point at the same location, we expect a weighted
        # mean value, weighted by the reciprocals of the variances.
        # The variance of the result is just 3.0 / 3, as we have three input
        # informations, each with a variance of 3.0
        gp.update(x, np.array([[10.0]]))
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], (5.0 / 1.5 + 10.0 / 3.0) / (1 / 1.5 + 1 / 3.0))
        self.assertAlmostEqual(var_yq[0], 3.0 / 3)

        # We get the same result if we update the GP with both measurements at the same
        # time.
        gp = _setup_rgp_rbf(outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x)
        gp.update(np.array([[1.0], [1.0]]), np.array([10.0, 10.0]))
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], (5.0 / 1.5 + 10.0 / 3.0) / (1 / 1.5 + 1 / 3.0))
        self.assertAlmostEqual(var_yq[0], 3.0 / 3)

        # If the measurement variance is zero, we get the measured value back, with a
        # variance of zero.
        # We should see a gaussian curve centered at 1 and with a variance of 2.0**2
        # (because of the lengthscale of 2.0), so we can check this also.
        gp = _setup_rgp_rbf(outputscale=3.0, lengthscale=2.0, noise_var=0.0, x_base=x)
        gp.update(x, np.array([10.0]))
        (yq, var_yq) = gp.predict(np.array([[1.0], [0.0], [3.0]]))
        self.assertAlmostEqual(yq[0], 10.0)
        self.assertAlmostEqual(var_yq[0], 0.0)

        self.assertAlmostEqual(yq[1], 10.0 * np.exp(-((0 - 1) ** 2) / (2 * 4.0)), 5)
        self.assertAlmostEqual(yq[2], 10.0 * np.exp(-((3 - 1) ** 2) / (2 * 4.0)), 5)

        # The variance is like an "inverted" gaussian function, were the exponential
        # is squared (thus the argument of the exponential is multiplied by two)
        self.assertAlmostEqual(
            var_yq[1], 3.0 - 3.0 * np.exp(-((0 - 1) ** 2) / (1 * 4.0)), 5
        )

        self.assertAlmostEqual(
            var_yq[2], 3.0 - 3.0 * np.exp(-((3 - 1) ** 2) / (1 * 4.0)), 5
        )

    def test_rbf_1d_3basepoints(self):
        # The expected results of the test above don't change if we simply add more
        # base points
        x = np.array([[1.0]])
        x_base = np.array([[1.0], [1.2], [-3.0]])
        gp = _setup_rgp_rbf(
            outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x_base
        )
        gp.update(x, np.array([10.0]))
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], 5.0)
        self.assertAlmostEqual(var_yq[0], 1.5)

        gp.update(x, np.array([[10.0]]))
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], (5.0 / 1.5 + 10.0 / 3.0) / (1 / 1.5 + 1 / 3.0), 5)
        self.assertAlmostEqual(var_yq[0], 3.0 / 3, 5)

        gp = _setup_rgp_rbf(
            outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x_base
        )
        gp.update(np.array([[1.0], [1.0]]), np.array([10.0, 10.0]))

        self.assertAlmostEqual(yq[0], (5.0 / 1.5 + 10.0 / 3.0) / (1 / 1.5 + 1 / 3.0), 5)
        self.assertAlmostEqual(var_yq[0], 3.0 / 3, 5)

        gp = _setup_rgp_rbf(outputscale=3.0, lengthscale=2.0, noise_var=0.0, x_base=x)
        gp.update(x, np.array([10.0]))
        (yq, var_yq) = gp.predict(np.array([[1.0], [0.0], [3.0]]))
        self.assertAlmostEqual(yq[0], 10.0)
        self.assertAlmostEqual(var_yq[0], 0.0)

        self.assertAlmostEqual(yq[1], 10.0 * np.exp(-((0 - 1) ** 2) / (2 * 4.0)), 5)
        self.assertAlmostEqual(yq[2], 10.0 * np.exp(-((3 - 1) ** 2) / (2 * 4.0)), 5)

        self.assertAlmostEqual(
            var_yq[1], 3.0 - 3.0 * np.exp(-((0 - 1) ** 2) / (1 * 4.0)), 5
        )

        self.assertAlmostEqual(
            var_yq[2], 3.0 - 3.0 * np.exp(-((3 - 1) ** 2) / (1 * 4.0)), 5
        )

    def test_rbf_3d_1basepoint(self):
        # Also, if we use a higher dimensional space, the expected results don't change.
        x = np.array([[1.0, 2.0, -1.0]])

        gp = _setup_rgp_rbf(outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x)
        gp.update(x, np.array([10.0]))
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], 5.0)
        self.assertAlmostEqual(var_yq[0], 1.5)

        gp.update(x, np.array([[10.0]]))
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], (5.0 / 1.5 + 10.0 / 3.0) / (1 / 1.5 + 1 / 3.0))
        self.assertAlmostEqual(var_yq[0], 3.0 / 3)

        gp = _setup_rgp_rbf(outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x)
        gp.update(
            np.array([[1.0, 2.0, -1.0], [1.0, 2.0, -1.0]]),
            np.array([10.0, 10.0]),
        )
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], (5.0 / 1.5 + 10.0 / 3.0) / (1 / 1.5 + 1 / 3.0))
        self.assertAlmostEqual(var_yq[0], 3.0 / 3)

        gp = _setup_rgp_rbf(outputscale=3.0, lengthscale=2.0, noise_var=0.0, x_base=x)
        gp.update(x, np.array([10.0]))
        (yq, var_yq) = gp.predict(
            np.array([[1.0, 2.0, -1.0], [1.0, 1.0, -1.0], [1.0, 4.0, -1.0]])
        )
        self.assertAlmostEqual(yq[0], 10.0)
        self.assertAlmostEqual(var_yq[0], 0.0)

        self.assertAlmostEqual(yq[1], 10.0 * np.exp(-((1 - 2) ** 2) / (2 * 4.0)), 5)
        self.assertAlmostEqual(yq[2], 10.0 * np.exp(-((4 - 2) ** 2) / (2 * 4.0)), 5)

        self.assertAlmostEqual(
            var_yq[1], 3.0 - 3.0 * np.exp(-((0 - 1) ** 2) / (1 * 4.0)), 5
        )

        self.assertAlmostEqual(
            var_yq[2], 3.0 - 3.0 * np.exp(-((3 - 1) ** 2) / (1 * 4.0)), 5
        )

    def test_trainingspoints_equal_basepoints(self):
        # If the training points are a subset of the base points, the recursive GP
        # should be exact.

        x_base = np.random.uniform(-5.0, 5.0, (20, 3))

        nt = 50
        # leave some base points out by purpose
        idx_t = np.random.random_integers(0, x_base.shape[0] - 5, (nt,))
        xt = x_base[idx_t, :]

        yt = _foo(xt)

        xq = np.random.uniform(-5.0, 5.0, (50, 3))

        # Generate reference solution on CPU
        gp = _setup_rgp_rbf(
            outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x_base
        )
        gp.update(xt, yt)

        (yq, var_yq) = gp.predict(xq, True)

        # reference solution
        gp_ref = ScaledRBFModel(
            torch.tensor(xt),
            torch.tensor(yt),
            noise_variance=3.0,
            outputscale=3.0,
            lengthscale=2.0,
        )

        (yq_ref, var_yq_ref) = gp_ref.predict(xq, True)

        self.assertLess(np.linalg.norm(yq_ref - yq), 1e-5 * np.linalg.norm(yq_ref))
        self.assertLess(
            np.linalg.norm(var_yq_ref - var_yq), 1e-5 * np.linalg.norm(var_yq_ref)
        )

    def test_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("no GPU available")

        x_base = np.random.uniform(-5.0, 5.0, (5, 3))

        nt = 50
        xt = np.random.uniform(-5.0, 5.0, (nt, 3))
        yt = _foo(xt)

        xq = np.random.uniform(-5.0, 5.0, (50, 3))

        # Generate reference solution on CPU
        gp_cpu = _setup_rgp_rbf(
            outputscale=3.0, lengthscale=2.0, noise_var=3.0, x_base=x_base
        )
        gp_cpu.update(xt, yt)

        (yq_cpu, var_yq_cpu) = gp_cpu.predict(xq, True)

        # Lets use the GPU
        device = torch.device("cuda")
        torch.set_default_device(device)

        gp_cuda = _setup_rgp_rbf(
            outputscale=3.0,
            lengthscale=2.0,
            noise_var=3.0,
            x_base=x_base,
            device=device,
        )

        gp_cuda.update(xt, yt)

        (yq_cuda, var_yq_cuda) = gp_cuda.predict(xq, True)

        torch.set_default_device("cpu")

        self.assertLess(np.linalg.norm(yq_cpu - yq_cuda), 1e-6 * np.linalg.norm(yq_cpu))

        self.assertLess(
            np.linalg.norm(var_yq_cpu - var_yq_cuda),
            1e-6 * np.linalg.norm(var_yq_cpu),
        )


if __name__ == "__main__":
    unittest.main()
