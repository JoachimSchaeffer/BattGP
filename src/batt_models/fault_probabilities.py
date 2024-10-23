from typing import List, Tuple

import numpy as np
import pandas as pd

from .battgp import BattGPResult
from .fault_evaluation import (
    calc_outside_band_probabilities,
    calc_over_threshold_probability,
    calc_r0_cells_var,
)


def calc_fault_probabilities(
    gp_res: BattGPResult,
    causal: bool,
    r0_band: float,
    r0_upper_threshold: float,
) -> pd.DataFrame:
    cellnumbers = gp_res.batt_data.cell_nrs

    r0_cells_mean_gp = gp_res.get_cell_data(cellnumbers, ["t", "r0"], causal)
    r0_cells_var_gp = gp_res.get_cell_data(cellnumbers, ["t", "r0var"], causal)

    (df_faults, df_mean_mean_gp) = _calc_fault_probabilities(
        r0_cells_mean_gp, r0_cells_var_gp, cellnumbers, r0_band, r0_upper_threshold
    )

    df_faults = pd.merge(df_faults, df_mean_mean_gp, left_index=True, right_index=True)

    return df_faults


def _calc_fault_probabilities(
    r0_cells_mean_gp: pd.DataFrame,
    r0_cells_var_gp: pd.DataFrame,
    cellnumbers: List[int],
    r0_band: float,
    r0_upper_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # For development purposes, you can experiment with whether you want to
    # cal the mean by using all cells or all other cells. Default: all other cells.

    cols_mean = [col for col in r0_cells_mean_gp.columns if col != "t"]
    cols_var = [col for col in r0_cells_var_gp.columns if col != "t"]
    r0_cells = r0_cells_mean_gp[cols_mean].values
    r0var_cells = r0_cells_var_gp[cols_var].values

    # variant with band_mean_without_eval_cell = True must be evaluated
    # independently of the fact if "~mean_i" is part of the "modes" or not, as
    # this also returns the mean-values that have to be provided in every case
    (prob_outside_band, prob_above_band, prob_below_band, r0_mean_mean) = (
        calc_outside_band_probabilities(
            r0_cells,
            r0var_cells,
            r0_band,
            band_mean_without_eval_cell=True,
            return_intermediate_arrays=True,
        )
    )

    r0_mean_mean_gp = {
        f"~{col}": r0_mean_mean[:, i] for (i, col) in enumerate(cols_mean)
    }
    r0_mean_mean_gp = pd.DataFrame(r0_mean_mean_gp)
    r0_mean_mean_gp["R0 mean mean_gp"] = r0_mean_mean_gp.mean(axis=1)

    r0_fault = {}

    # Look up the probability for the gaussian distribution based on mean and variance
    # and the threshold value
    for i, x in enumerate(cellnumbers):
        r0_fault[f"R{x} band_i fault prob"] = prob_outside_band[:, i]
        r0_fault[f"R_upper{x} band_i fault prob"] = prob_above_band[:, i]
        r0_fault[f"R_lower{x} band_i fault prob"] = prob_below_band[:, i]

    prob_threshold = calc_over_threshold_probability(
        r0_cells, r0var_cells, r0_upper_threshold
    )

    for i, x in enumerate(cellnumbers):
        r0_fault[f"R{x} thres fault prob"] = prob_threshold[:, i]

    r0_fault["R0 mean_gp cells var"] = calc_r0_cells_var(r0_cells)

    u_band_cols = [f"R_upper{i} band_i fault prob" for i in cellnumbers]
    l_band_cols = [f"R_lower{i} band_i fault prob" for i in cellnumbers]
    r0_fault_band = np.array(
        [
            r0_fault[col1] + r0_fault[col2]
            for col1, col2 in zip(u_band_cols, l_band_cols)
        ]
    )
    r0_fault["Weakest_link_stat"] = 1 - np.prod(1 - r0_fault_band, axis=0)

    r0_fault_df = pd.DataFrame(r0_fault)

    r0_fault_df.insert(0, "t", r0_cells_mean_gp["t"])

    return r0_fault_df, r0_mean_mean_gp
