import gpytorch
import numpy as np
import torch

from . import training


class ScaledRBFModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        noise_variance,
        outputscale,
        lengthscale,
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ScaledRBFModel, self).__init__(
            train_x.type(torch.float32),
            train_y.type(torch.float32),
            likelihood,
        )
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.likelihood.noise = noise_variance
        self.covar_module.outputscale = outputscale
        self.covar_module.base_kernel.lengthscale = lengthscale

        self.eval()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(
        self, xq: np.ndarray, full_cov: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = self(torch.tensor(xq, dtype=torch.float32))

        yq = out.mean.detach().cpu().numpy()

        if full_cov:
            var_yq = out._covar.detach().cpu().numpy()
        else:
            var_yq = np.diag(out._covar.detach().cpu().numpy())

        return (yq, var_yq)

    def optimize(self, **kwargs):
        return training.train_exact_gp(
            self, self.train_inputs[0], self.train_targets, **kwargs
        )


class SparseScaledRBFModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        inducing_points: torch.Tensor,
        noise_variance,
        outputscale,
        lengthscale,
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(SparseScaledRBFModel, self).__init__(
            train_x.type(torch.float32),
            train_y.type(torch.float32),
            likelihood,
        )
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

        self.likelihood.noise = noise_variance
        self.base_covar_module.outputscale = outputscale
        self.base_covar_module.base_kernel.lengthscale = lengthscale

        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module,
            inducing_points=inducing_points.type(torch.float32),
            likelihood=likelihood,
        )
        self.eval()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            out = self(torch.tensor(xq, dtype=torch.float32))

        yq = out.mean.detach().cpu().numpy()
        var_yq = np.diag(out._covar.detach().cpu().numpy())

        return (yq, var_yq)

    def optimize(self, **kwargs):
        return training.train_exact_gp(
            self, self.train_inputs[0], self.train_targets, **kwargs
        )
