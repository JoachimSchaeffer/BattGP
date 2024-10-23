import os
from copy import deepcopy
from typing import Any, Optional

import gpytorch
import numpy as np
import pandas as pd
import torch

from .. import config as cfg
from ..batt_data.batt_data import BattData
from ..gp import training
from ..operating_point import Op
from .batt_cell_gp_protocol import (
    IBatteryCellGP,
)
from .cell_gp import BatteryCellGP
from .cellnr import get_causal_tag, get_cell_tag


def _create_hyperparams_df(
    params: dict,
) -> pd.DataFrame:
    df = pd.DataFrame(
        index=[
            "Noise Variance",
            "Wiener Outputscale",
            "RBF Outputscale",
            "RBF Lengthscale 1",
            "RBF Lengthscale 2",
            "RBF Lengthscale 3",
        ],
        columns=["params"],
        data=[
            [params["noise_variance"]],
            [params["outputscale_wiener"]],
            [params["outputscale_rbf"]],
            [params["lengthscale_rbf"][0]],
            [params["lengthscale_rbf"][1]],
            [params["lengthscale_rbf"][2]],
        ],
    )
    return df


class BatteryCellGP_Full(IBatteryCellGP):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, cellnr: Optional[int] = None, **kwargs
    ):
        self.params: dict[str, Any] = self.get_default_parameters()
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

        x = torch.tensor(x, dtype=self.dtype_, device=self.device_)
        y = torch.tensor(y, dtype=self.dtype_, device=self.device_)

        self.model: BatteryCellGP = BatteryCellGP(x, y, **kwargs)

        self.model.noise_variance_constraint = self.params["noise_variance_range"]
        self.model.noise_variance = self.params["noise_variance"]

        self.model.outputscale_wiener_constraint = self.params[
            "outputscale_wiener_range"
        ]

        self.model.outputscale_wiener = self.params["outputscale_wiener"]

        self.model.outputscale_rbf_constraint = self.params["outputscale_rbf_range"]
        self.model.outputscale_rbf = self.params["outputscale_rbf"]

        self.model.lengthscale_rbf_constraint = self.params["lengthscale_rbf_range"]
        self.model.lengthscale_rbf = self.params["lengthscale_rbf"]

        self.model.eval()
        self.model.likelihood.eval()

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
            "noise_variance_range": cfg.NOISE_VARIANCE_RANGE,
            "outputscale_wiener_range": cfg.OUTPUTSCALE_WIENER_RANGE,
            "outputscale_rbf_range": cfg.OUTPUTSCALE_RBF_RANGE,
            "lengthscale_rbf_range": cfg.LENGTHSCALE_RBF_RANGE,
            "max_iter": cfg.OPTIM_MAX_ITER,
            "rel_tol": cfg.OPTIM_REL_TOL,
            "lr": cfg.OPTIM_LR,
            "dtype": cfg.DTYPE,
            "n_devices": 1,
            "output_device": None,
        }
        return params

    def get_parameters(self) -> dict[str, Any]:
        return deepcopy(self.params)

    def save_hyperparameters(
        self,
        path: str,
    ) -> None:

        hyperparams_df = _create_hyperparams_df(self.params)
        hyperparams_df.loc["Marginal Likelihood"] = self.marginallikelihood
        # hyperparams_df.loc["Training Time"] = self.stats["training_time"]
        path = os.path.join(path, f"{self._cellnr}hyperparams.csv")
        hyperparams_df.to_csv(path)

    def train_hyperparameters(self, messages: bool = True) -> np.ndarray:
        x_train = self.model.train_inputs[0]
        y_train = self.model.train_targets

        # Here you can choose how you want to train the model
        # training.train_exact_gp_botorch(
        if cfg.HYPER_OPT_PARAMS["opt_algorithm"] == "torch_lbfgs":
            trainer = training.train_exact_gp_lbfgs
        elif cfg.HYPER_OPT_PARAMS["opt_algorithm"] == "botorch_lbfgs_B":
            trainer = training.train_exact_gp_botorch
        elif cfg.HYPER_OPT_PARAMS["opt_algorithm"] == "torch_adam":
            trainer = training.train_exact_gp_adam
        else:
            algo = cfg.HYPER_OPT_PARAMS["opt_algorithm"]
            raise ValueError(f"{algo} is not implemented as optimization algorithm.")

        losses = trainer(
            self.model,
            x_train,
            y_train,
            loss_scale=len(y_train),
            max_iter=self.params["max_iter"],
            rel_ftol=self.params["rel_tol"],
            lr=self.params["lr"],
            messages=messages,
        )

        self.params["noise_variance"] = float(self.model.noise_variance)
        self.params["outputscale_wiener"] = float(self.model.outputscale_wiener)
        self.params["outputscale_rbf"] = float(self.model.outputscale_rbf)
        self.params["lengthscale_rbf"] = tuple(
            self.model.lengthscale_rbf.detach().cpu().numpy()[0]
        )

        if isinstance(losses, np.ndarray):
            self.marginallikelihood = losses[-1]
        else:
            self.marginallikelihood = losses

        return losses

    def predict(
        self, x: np.ndarray, full_cov: bool = False, no_cov: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_ = torch.tensor(x, device=self.device_, dtype=self.dtype_).contiguous()
            out = self.model(x_)

        y = out.mean.detach().cpu().numpy()

        if no_cov:
            return y

        y_var = out.variance.detach().cpu().numpy().reshape(-1)

        if full_cov:
            n = x.shape[0]
            covmatrix = np.full(
                (n, n),
                np.nan,
                dtype=np.float32 if self.dtype_ == torch.float32 else np.float64,
            )

            for i in range(n):
                covmatrix[i, i] = y_var[i]

            y_var = covmatrix

        return (y, y_var)

    def predict_r0_op(self, op: Op, t: np.ndarray) -> pd.DataFrame:

        X = np.column_stack(
            (
                t,
                np.ones(len(t)) * op.I,
                np.ones(len(t)) * op.SOC,
                np.ones(len(t)) * op.T,
            )
        )

        (r0, r0var) = self.predict(X, full_cov=False)

        cell_tag = get_cell_tag(self._cellnr)
        causal_tag = get_causal_tag(False)

        df = pd.DataFrame(
            {
                "t": t,
                f"r0_{causal_tag}_{cell_tag}": r0,
                f"r0var_{causal_tag}_{cell_tag}": r0var,
            }
        )
        return df

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.model.train_inputs[0].detach().cpu().numpy(),
            self.model.train_targets.detach().cpu().numpy().reshape((-1,)),
        )


def build_cellmodel_full(
    cellnr: int,
    batt_data: BattData,
    max_training_data: int,
    max_age: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> BatteryCellGP_Full:

    (x, y) = batt_data.generateTrainingData(cellnr, max_training_data, max_age)

    return BatteryCellGP_Full(x, y, cellnr, device=device, **kwargs)
