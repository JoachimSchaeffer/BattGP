from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch

from .. import config as cfg
from ..batt_data.batt_data import BattData
from .basis_vector_selection import BasisVectorStrategyStr, get_basis_vectors
from .battcellgp_spatiotemporal import build_cellmodel_spatio_temporal
from .battgp import BattGP, BattGPResult
from .cell_stgp import CleanupNegativeVariancesStrategyStr
from .ref_strategy import RefStrategy


class BattGP_SpatioTemporal(BattGP):
    def __init__(
        self,
        batt_data: BattData,
        *,
        sampling_time_sec: int,
        max_batch_size: Optional[int] = None,
        max_age: Optional[int] = None,
        ref_strategy: RefStrategy = RefStrategy("mean"),
        device: torch.device = torch.device("cpu"),
        save_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            batt_data,
            max_age=max_age,
            max_training_data=None,
            ref_strategy=ref_strategy,
            save_path=save_path,
        )

        self.max_batch_size = max_batch_size
        self.sampling_time_sec = sampling_time_sec

        if "basis_vectors" in kwargs:
            self.basis_vectors = kwargs["basis_vectors"]
            kwargs.pop("basis_vectors")
        elif "basis_vector_strategy" in kwargs:
            self.basis_vectors = self._get_basis_vectors(**kwargs)
            kwargs.pop("nbasis")
            kwargs.pop("basis_vector_strategy")
        else:
            self.basis_vectors = self._get_basis_vectors()

        self.packmodel = build_cellmodel_spatio_temporal(
            -1,
            self.batt_data,
            self.basis_vectors,
            self.sampling_time_sec,
            max_age=self.max_age,
            max_batch_size=max_batch_size,
            device=device,
            **kwargs,
        )

        for cellnr in self.batt_data.cell_nrs:
            self.cellmodels.append(
                build_cellmodel_spatio_temporal(
                    cellnr,
                    self.batt_data,
                    self.basis_vectors,
                    self.sampling_time_sec,
                    max_age=self.max_age,
                    max_batch_size=max_batch_size,
                    device=device,
                    **kwargs,
                )
            )

    def _get_basis_vectors(
        self,
        basis_vector_strategy: BasisVectorStrategyStr = "kmeans",
        nbasis: Tuple[int, int, int] = (4, 4, 4),
        **kwargs,
    ) -> torch.Tensor:
        """Set basis vectors for the spatiotemporal GP model"""

        return get_basis_vectors(
            basis_vector_strategy,
            nbasis,
            self.batt_data,
            self.max_training_data,
            self.max_age,
            self.ref_op.into_array(),
            cfg.LENGTHSCALE_RBF,
            **kwargs,
        )

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "sampling_time_sec": self.sampling_time_sec,
            "max_batch_size": self.max_batch_size,
            "ref_point": self.get_operating_point().disp_str(),
            "segment_criteria": "Saving segment criteria not implemented yet, see config.py",
            "gap_removal": "Saving gap removal not implemented yet, see config.py",
            "ocv_path": "Saving ocv path not implemented yet, see config.py",
        }

    def predict_cell_r0_op(
        self,
        smooth: bool = True,
        max_age: Optional[int] = None,
        cleanup_neg_var_strategy: CleanupNegativeVariancesStrategyStr = "prior",
        save: bool = True,
    ) -> BattGPResult:
        if max_age is None:
            max_age = self.max_age

        # (x, y) = self.batt_data.generateTrainingData(-1, max_age=max_age)
        print(f"    processing pack (0/{len(self.cellmodels)})")

        df: pd.DataFrame = self.packmodel.predict_r0_op(
            op=self.ref_op,
            smooth=smooth,
            cleanup_neg_var_strategy=cleanup_neg_var_strategy,
        )

        for cellmodel in self.cellmodels:
            cellnr = cellmodel.cellnr
            print(f"    processing cell {cellnr} ({cellnr}/{len(self.cellmodels)})")

            cell_df = cellmodel.predict_r0_op(
                op=self.ref_op,
                smooth=smooth,
                cleanup_neg_var_strategy=cleanup_neg_var_strategy,
            )
            df = df.merge(cell_df)

        if save and (self.save_path is not None):
            self.save_df(df)

        return BattGPResult(self.batt_data, self.cellmodels, self.ref_op, df)
