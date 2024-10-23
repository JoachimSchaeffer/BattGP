from copy import deepcopy
from typing import Any, Optional

import gpytorch
import numpy as np
import pandas as pd
import torch

from .. import config as cfg
from ..batt_data.batt_data import BattData
from ..gp.spatiotemporal_gp import ApproxSpatioTemporalGP
from ..gp.wiener_kernel_temporal import WienerTemporalKernel
from ..operating_point import Op
from .batt_cell_gp_protocol import IBatteryCellGP
from .cell_stgp import (
    CleanupNegativeVariancesStrategyStr,
    apply_stgp,
    cleanup_negative_variances,
)
from .cellnr import get_cell_tag


class BatteryCellGP_SpatioTemporal(IBatteryCellGP):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        basis_vectors: np.ndarray,
        sampling_time_sec: int,
        max_batch_size: int,
        cellnr: Optional[int] = None,
        **kwargs,
    ):
        self.params: dict[str, Any] = self.get_default_parameters()

        self.max_batch_size = max_batch_size
        self.sampling_time_sec = sampling_time_sec
        self._cellnr = cellnr

        UNLOGGED_PARAMS = {"device"}

        for k, v in kwargs.items():
            if k in UNLOGGED_PARAMS:
                continue

            if k not in self.params:
                raise ValueError(f"unknown keyword parameter '{k}'")
            self.params[k] = v

        self.dtype_ = self.params["dtype"]
        self.device_ = kwargs.get("device", torch.device("cpu"))

        self.xt = x
        self.yt = y

        self.basis_vectors = torch.tensor(
            basis_vectors, dtype=self.dtype_, device=self.device_
        )
        noise_var = self.params["noise_variance"]

        dims = self.xt.shape[1] - 1
        kernel_s = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dims)
        )
        kernel_s[0].outputscale = self.params["outputscale_rbf"]
        kernel_s[1].base_kernel.lengthscale = self.params["lengthscale_rbf"]

        self.model = ApproxSpatioTemporalGP(
            self.basis_vectors,
            kernel_s,
            WienerTemporalKernel(self.params["outputscale_wiener"]),
            noise_var,
            smoothing_steps=-1,
            device=self.device_,
        )

    @property
    def cellnr(self) -> int:
        return self._cellnr

    @staticmethod
    def get_default_parameters() -> dict[str, Any]:
        params = {
            "noise_variance": cfg.NOISE_VARIANCE,
            "outputscale_wiener": cfg.OUTPUTSCALE_WIENER,
            "outputscale_rbf": cfg.OUTPUTSCALE_RBF,
            "lengthscale_rbf": cfg.LENGTHSCALE_RBF,
            "dtype": cfg.DTYPE,
            "output_device": None,
        }
        return params

    def get_parameters(self) -> dict[str, Any]:
        return deepcopy(self.params)

    def train_hyperparameters(self, messages: bool = True) -> np.ndarray:
        m1 = "Hyperparameter optimization not implemented for spatiotemporal GP models."
        m2 = "Consider unsing a standard GP model with a subset of the data or implement this yourself and start a PR."
        raise NotImplementedError(f"{m1}\n{m2}")

    def predict_r0_op(
        self,
        op: Op,
        smooth: bool,
        cleanup_neg_var_strategy: Optional[CleanupNegativeVariancesStrategyStr] = None,
    ) -> pd.DataFrame:
        cell_tag = get_cell_tag(self.cellnr)

        df = apply_stgp(
            self.model,
            self.xt,
            self.yt,
            op,
            self.sampling_time_sec,
            smooth,
            self.max_batch_size,
            cell_tag,
        )

        if cleanup_neg_var_strategy is not None:
            cleanup_negative_variances(df, cleanup_neg_var_strategy, self.model)

        return df

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.xt, torch.Tensor):
            x = self.xt.detach().cpu().numpy()
        else:
            x = self.xt

        if isinstance(self.yt, torch.Tensor):
            y = self.yt.detach().cpu().numpy()
        else:
            y = self.yt

        return (x, y)


def build_cellmodel_spatio_temporal(
    cellnr: int,
    batt_data: BattData,
    basis_vectors: np.ndarray,
    sampling_time_sec: int,
    max_age: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> BatteryCellGP_SpatioTemporal:

    (x, y) = batt_data.generateTrainingData(
        cellnr, max_training_data=-1, max_age=max_age
    )

    return BatteryCellGP_SpatioTemporal(
        x,
        y,
        basis_vectors,
        sampling_time_sec,
        max_batch_size,
        cellnr,
        device=device,
        **kwargs,
    )
