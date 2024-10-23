import concurrent.futures
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Union

import pandas as pd

from ..batt_data.batt_data import BattData
from ..operating_point import Op
from .batt_cell_gp_protocol import IBatteryCellGP
from .cellnr import get_causal_tag, get_cell_tag
from .ref_strategy import RefStrategy


@dataclass
class BattGPResult:
    batt_data: BattData
    cellmodels: List[IBatteryCellGP]
    ref_op: Op
    df: pd.DataFrame

    def get_cell_data(
        self,
        cellnrs: Union[Iterable[int], int],
        signals: Optional[Iterable[str]] = None,
        causal: bool = False,
        missing_behaviour: Literal["error", "ignore"] = "error",
    ) -> pd.DataFrame:
        """Get signals from the result data

        signals is a iterable of string values that specifies the data that is to be
        read from the result. The signal names correspond to the column names
        without the causal/acausal "tag" and the cell identifier.

        If cellnrs is a scalar int value, the columns of the returned dataframe have
        the same name as the specified signals.
        If cellnrs is an iterable over int values, the columns of the returned
        dataframe contain the cell identifier, but still not the causal/acausal tag.
        (The motivation for this behaviour is that this enables to write functions
        that use the result and can be applied to both data.

        The columns in the returned dataframe are sorted like the signals in signal.
        If the data of multiple cells are retrieved, the data is grouped by the signals.
        For example:
            gpresult.get_cell_data([1, 2, 3, 4], ["t", "r0", "r0var"])
        results in a dataframe with the columns
            t, r0_c1, r0_c2, r0_c3, r0_c4, r0var_c1, r0var_c2, r0var_c3, r0var_c4
        """

        CELL_COLS = ["r0", "r0var", "dr0", "dr0var"]

        if signals is None:
            signals = ["t", "ds_count", "r0", "r0var", "dr0", "dr0var"]

        if isinstance(cellnrs, int):
            cellnrs = [cellnrs]
            use_cell_tag = False
        else:
            use_cell_tag = True

        causal_tag = get_causal_tag(causal)

        source_names: List[str] = []
        target_names_map: Dict[str, str] = {}

        for s in signals:
            if s not in CELL_COLS:
                source_names.append(s)
                target_names_map[s] = s
            else:
                for cellnr in cellnrs:
                    cell_tag = get_cell_tag(cellnr)

                    source_name = f"{s}_{causal_tag}_{cell_tag}"
                    source_names.append(source_name)

                    if use_cell_tag:
                        target_names_map[source_name] = f"{s}_{cell_tag}"
                    else:
                        target_names_map[source_name] = s

        columns = [s for s in source_names if s in self.df.columns]

        if len(columns) < len(source_names):
            if missing_behaviour == "error":
                delta = set(source_names) - set(columns)
                raise ValueError(
                    f"signal(s) {', '.join(delta)} not available in the result"
                )

        return self.df[columns].rename(columns=target_names_map)


class BattGP:
    def __init__(
        self,
        batt_data: BattData,
        *,
        max_training_data: Optional[int] = None,
        max_age: Optional[int] = None,
        save_path: Optional[str] = None,
        ref_strategy: RefStrategy = RefStrategy("mean"),
    ) -> None:
        self.packmodel: IBatteryCellGP
        self.cellmodels: List[IBatteryCellGP] = []

        self.batt_data: BattData = batt_data
        self.max_training_data = max_training_data

        if max_age is None:
            self.max_age: int = batt_data.age
        else:
            self.max_age: int = max_age

        self.ref_strategy = ref_strategy
        if ref_strategy.is_median():
            self.ref_op = self.batt_data.median_op
        elif ref_strategy.is_mean():
            self.ref_op = self.batt_data.mean_op
        else:
            self.ref_op = ref_strategy.get_manual_value()
        print(f"Reference operating point: {self.ref_op}")

        if save_path is None:
            self.save_path = None
        else:
            self.save_path = os.path.join(save_path, self.batt_data.id)
            os.makedirs(self.save_path, exist_ok=True)

    def set_operating_point_to_mean(
        self,
    ) -> None:
        """Set operating point

        Set operating point for each cell to evaluate mean function of GP model for
        each cell."""

        op = self.batt_data.mean_op

        print(f"Battery operating point set to mean: {op.disp_str()}")
        self.set_operating_point(op, verbose=False)

    def set_operating_point_to_median(
        self,
    ) -> None:
        """Set operating point

        Set operating point for each cell to evaluate mean function of GP model for
        each cell."""

        op = self.batt_data.median_op

        print(f"Battery operating point set to median: {op.disp_str()}")
        self.set_operating_point(op, verbose=False)

    def set_operating_point(self, op: Op, verbose: bool = True) -> None:
        """Set operating point

        Set operating point for each cell to evaluate mean function of GP model for
        each cell."""
        if verbose:
            print(f"Battery operating point set to: {op.disp_str}")

        self.ref_op = op

    def get_operating_point(self) -> Op:
        """Get operating point"""
        return self.ref_op

    def get_cell_model(self, cellnr: int) -> IBatteryCellGP:
        if cellnr == -1:
            return self.packmodel

        for cell in self.cellmodels:
            if cell.cellnr == cellnr:
                return cell

        raise ValueError(f"cell {cellnr} does not exist")

    def train_hyperparameters(
        self, parallelize: bool = False, messages: bool = True
    ) -> None:
        """Optimizing hyperparamters."""
        if not parallelize:
            self.packmodel.train_hyperparameters(messages=messages)

            for cellmodel in self.cellmodels:
                cellmodel.train_hyperparameters(messages=messages)

        else:
            futures = []
            # start the thread pool
            with concurrent.futures.ThreadPoolExecutor(
                1 + len(self.cellmodels)
            ) as executor:
                futures.append(
                    executor.submit(
                        lambda: _execute_with_message(
                            self.packmodel.train_hyperparameters(messages=False),
                            "training of packmodel finished",
                        )
                    )
                )

                for i, cellmodel in enumerate(self.cellmodels):
                    futures.append(
                        executor.submit(
                            lambda: _execute_with_message(
                                cellmodel.train_hyperparameters(messages=False),
                                f"training of cellmodel {i + 1} finished",
                            )
                        )
                    )

            futures.wait(futures)

    def save_hyperparameters(self, path: str) -> None:
        """Save hyperparameters"""
        save_path = os.path.join(path, self.batt_id)
        os.makedirs(save_path, exist_ok=True)
        self.packmodel.save_hyperparameters(save_path)

        for cellmodel in self.cellmodels:
            cellmodel.save_hyperparameters(save_path)

    def _get_parameters(self) -> Dict[str, Any]:
        return {}

    def get_parameters(self) -> Dict[str, Any]:
        return self._get_parameters()

    def save_df(
        self,
        df: Optional[pd.DataFrame],
    ) -> None:
        params = self.get_parameters()

        filename_data = os.path.join(self.save_path, "battgpf_df.feather")
        filename_info = os.path.join(self.save_path, "battgpf_info.json")
        # Fix the saving here and in the other file.
        if df is None:
            with open(filename_info, "w") as fil:
                json.dump(params, fil)
            return

        # delete existing files, to prevent that the data and info file doesn't belong
        # together if the writing of the feather files succedes but not the writing of
        # the json file
        try:
            os.remove(filename_data)
        except FileNotFoundError:
            pass

        try:
            os.remove(filename_info)
        except FileNotFoundError:
            pass

        df.to_feather(filename_data)
        with open(filename_info, "w") as fil:
            json.dump(params, fil)


def _execute_with_message(func: Callable, message: str):
    func()
    print(message)
