from typing import Iterable

import gpytorch
import gpytorch.constraints
import torch

from src import gpytorch_utils

from ..gp.wiener_kernel import WienerKernel


class BatteryCellGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        *,
        n_devices: int = 1,
        output_device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        self.device_ = train_x.device

        super().__init__(
            train_x,
            train_y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )

        self.mean_module = gpytorch.means.ZeroMean()

        kernel_wiener = WienerKernel(active_dims=[0])
        kernel_rbf = gpytorch.kernels.RBFKernel(ard_num_dims=3, active_dims=[1, 2, 3])
        kernel = gpytorch.kernels.ScaleKernel(
            kernel_wiener
        ) + gpytorch.kernels.ScaleKernel(kernel_rbf)
        self.n_devices = n_devices
        if self.n_devices > 1:
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                kernel,
                device_ids=range(self.n_devices),
                output_device=output_device,
            )
        elif self.n_devices == 1:
            self.covar_module = kernel
        else:
            raise ValueError("n_devices must be an integer and >= 1")

        self.to(train_x.device)

    def to(self, device: torch.device):
        self.device_ = device
        self.likelihood.to(device)
        self.mean_module.to(device)
        self.covar_module.to(device)
        super().to(device)

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def noise_variance(self):
        return self.likelihood.noise

    @noise_variance.setter
    def noise_variance(self, value: torch.Tensor | float):
        self.likelihood.noise = gpytorch_utils.get_tensor(value).to(self.device_)

    @property
    def noise_variance_constraint(self):
        return self.likelihood.noise_covar.raw_noise_constraint

    @noise_variance_constraint.setter
    def noise_variance_constraint(self, value: gpytorch_utils.ScalarConstraintType):
        constraint = gpytorch_utils.get_scalar_gpytorch_constraint(value).to(
            self.device_
        )
        self.likelihood.noise_covar.raw_noise_constraint = constraint

    @property
    def outputscale_wiener(self):
        if self.n_devices > 1:
            return self.covar_module.base_kernel.kernels[0].outputscale
        else:
            return self.covar_module.kernels[0].outputscale

    @outputscale_wiener.setter
    def outputscale_wiener(self, value: torch.Tensor | float):
        if self.n_devices > 1:
            self.covar_module.base_kernel.kernels[0].outputscale = (
                gpytorch_utils.get_tensor(value).to(self.device_)
            )
        else:
            self.covar_module.kernels[0].outputscale = gpytorch_utils.get_tensor(
                value
            ).to(self.device_)

    @property
    def outputscale_wiener_constraint(self):
        if self.n_devices > 1:
            return self.covar_module.base_kernel.kernels[0].raw_outputscale_constraint
        else:
            return self.covar_module.kernels[0].raw_outputscale_constraint

    @outputscale_wiener_constraint.setter
    def outputscale_wiener_constraint(self, value: gpytorch_utils.ScalarConstraintType):
        constraint = gpytorch_utils.get_scalar_gpytorch_constraint(value).to(
            self.device_
        )
        if self.n_devices > 1:
            self.covar_module.base_kernel.kernels[0].raw_outputscale_constraint = (
                constraint
            )
        else:
            self.covar_module.kernels[0].raw_outputscale_constraint = constraint

    @property
    def outputscale_rbf(self):
        if self.n_devices > 1:
            return self.covar_module.base_kernel.kernels[1].outputscale
        else:
            return self.covar_module.kernels[1].outputscale

    @outputscale_rbf.setter
    def outputscale_rbf(self, value: torch.Tensor | float):
        if self.n_devices > 1:
            self.covar_module.base_kernel.kernels[1].outputscale = (
                gpytorch_utils.get_tensor(value).to(self.device_)
            )
        else:
            self.covar_module.kernels[1].outputscale = gpytorch_utils.get_tensor(
                value
            ).to(self.device_)

    @property
    def outputscale_rbf_constraint(self):
        if self.n_devices > 1:
            return self.covar_module.base_kernel.kernels[1].raw_outputscale_constraint
        else:
            return self.covar_module.kernels[1].raw_outputscale_constraint

    @outputscale_rbf_constraint.setter
    def outputscale_rbf_constraint(self, value: gpytorch_utils.ScalarConstraintType):
        constraint = gpytorch_utils.get_scalar_gpytorch_constraint(value).to(
            self.device_
        )
        if self.n_devices > 1:
            self.covar_module.base_kernel.kernels[1].raw_outputscale_constraint = (
                constraint
            )
        else:
            self.covar_module.kernels[1].raw_outputscale_constraint = constraint

    @property
    def lengthscale_rbf(self):
        if self.n_devices > 1:
            return self.covar_module.base_kernel.kernels[1].base_kernel.lengthscale
        else:
            return self.covar_module.kernels[1].base_kernel.lengthscale

    @lengthscale_rbf.setter
    def lengthscale_rbf(self, value: torch.Tensor | Iterable[float] | float):
        if self.n_devices > 1:
            self.covar_module.base_kernel.kernels[1].base_kernel.lengthscale = (
                gpytorch_utils.get_tensor(value).to(self.device_)
            )
        else:
            self.covar_module.kernels[1].base_kernel.lengthscale = (
                gpytorch_utils.get_tensor(value).to(self.device_)
            )

    @property
    def lengthscale_rbf_constraint(self):
        if self.n_devices > 1:
            return self.covar_module.base_kernel.kernels[
                1
            ].base_kernel.raw_lengthscale_constraint
        else:
            return self.covar_module.kernels[1].base_kernel.raw_lengthscale_constraint

    @lengthscale_rbf_constraint.setter
    def lengthscale_rbf_constraint(self, value: gpytorch_utils.VectorConstraintType):
        constraint = gpytorch_utils.get_vector_gpytorch_constraint(value).to(
            self.device_
        )
        if self.n_devices > 1:
            self.covar_module.base_kernel.kernels[1].raw_lengthscale_constraint = (
                constraint
            )
        else:
            self.covar_module.kernels[1].raw_lengthscale_constraint = constraint
