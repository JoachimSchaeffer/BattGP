import gc
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..batt_data.batt_data import BattData
from .batt_cell_gp_protocol import IBatteryCellGP
from .battcellgp_full import build_cellmodel_full
from .battgp import BattGP, BattGPResult
from .ref_strategy import RefStrategy


class BattGP_Full(BattGP):
    def __init__(
        self,
        batt_data: BattData,
        *,
        max_training_data: Optional[int] = None,
        max_age: Optional[int] = None,
        ref_strategy: RefStrategy = RefStrategy("mean"),
        device: torch.device = torch.device("cpu"),
        save_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if max_training_data is None:
            max_training_data: int = 2000
            print(
                f"Max training data set to {max_training_data}, because no values was passed for max_training_data"
            )

        super().__init__(
            batt_data,
            max_training_data=max_training_data,
            max_age=max_age,
            ref_strategy=ref_strategy,
            save_path=save_path,
        )

        self.packmodel: IBatteryCellGP = build_cellmodel_full(
            -1,
            self.batt_data,
            max_training_data=self.max_training_data,
            max_age=self.max_age,
            device=device,
            **kwargs,
        )

        for cellnr in self.batt_data.cell_nrs:
            self.cellmodels.append(
                build_cellmodel_full(
                    cellnr,
                    self.batt_data,
                    max_training_data=self.max_training_data,
                    max_age=self.max_age,
                    device=device,
                    **kwargs,
                )
            )

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "ref_point": self.get_operating_point().disp_str(),
            "segment_criteria": "Saving segment criteria not implemented yet, see config.py",
            "gap_removal": "Saving gap removal not implemented yet, see config.py",
            "ocv_path": "Saving ocv path not implemented yet, see config.py",
        }

    def predict_cell_r0_op(
        self,
        destroy_after_run: bool = True,
        add_time_steps: bool = False,
        save: bool = True,
    ) -> BattGPResult:
        """Computes all R0 values for all cells in the battery.

        Computes all R0 values for all cells in the battery and sets operating point and
        stores them in a dataframe.
        """
        # compute R0 values for all cells
        # t = np.linspace(0, self.batt_data.age, 300)
        # get t from the packmodel
        t = self.cellmodels[0].model.train_inputs[0][:, 0]
        # add additional time steps to t, such that the resulting vectore has maximal a time step of 1 day
        if add_time_steps:
            t = np.concatenate(
                (
                    t,
                    np.linspace(
                        t[0], self.batt_data.age, int(self.batt_data.age - t[0])
                    ),
                )
            )
            # sort t
            self.t = np.sort(t)
        else:
            self.t = np.linspace(t[0].detach().cpu(), self.batt_data.age, 300)

        df: pd.DataFrame = self.packmodel.predict_r0_op(op=self.ref_op, t=self.t)

        if destroy_after_run:
            del self.packmodel.model
            gc.collect()
            torch.cuda.empty_cache()

        for i, cellmodel in enumerate(self.cellmodels):
            # start = time.time()
            cell_df = cellmodel.predict_r0_op(op=self.ref_op, t=self.t)
            # end = time.time()
            # print(end-start)
            df = df.merge(cell_df)

            if destroy_after_run:
                # Certainly not the most elegant way of clearing memory on the GPU.
                # It works however.
                if i < len(self.cellmodels) - 1:
                    del cellmodel.model
                    gc.collect()
                    torch.cuda.empty_cache()

        if save and self.save_path is not None:
            self.save_df(df)

        return BattGPResult(self.batt_data, self.cellmodels, self.ref_op, df)
