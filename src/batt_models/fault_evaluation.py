from typing import Dict, Tuple

import numpy as np
import scipy.special


def normal_cdf(x: np.ndarray, mean: np.array, std: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * scipy.special.erf((x - mean) / (np.sqrt(2) * std))


def hodges_lehmann_estimator(x: np.ndarray):
    """
    Simple robust estimator of location using the Hodges-Lehmann estimator.
    https://doi.org/10.1214/aoms/1177704172
    """
    n = len(x)
    return np.median([(x[i] + x[j]) / 2 for i in range(n) for j in range(i, n)])


def get_fault_evaluation(
    r0_cells: np.ndarray,
    r0var_cells: np.ndarray,
    r0_band_delta: float,
    r0_upper_threshold: float,
) -> Dict[str, np.ndarray]:
    band_props = calc_outside_band_probabilities(
        r0_cells, r0var_cells, r0_band_delta, return_intermediate_arrays=True
    )

    fault_evaluation = {
        "P_outside_band": band_props[0],
        "P_above_band": band_props[1],
        "P_below_band": band_props[2],
    }

    fault_evaluation["P_over_threshold"] = calc_over_threshold_probability(
        r0_cells, r0var_cells, r0_upper_threshold
    )

    fault_evaluation["cells_var"] = calc_r0_cells_var(r0_cells)

    return fault_evaluation


def calc_outside_band_probabilities(
    r0_cells: np.ndarray,
    r0var_cells: np.ndarray,
    r0_band_delta: float,
    band_mean_without_eval_cell: bool = True,
    return_intermediate_arrays: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Resistance fault probability (cell deviating from the mean)
    # 1. Calculate the mean resistance of all cells ecxept the cell in question
    # 2. Calculate the probability that the resistance of the cell in question is
    # outside a band around the mean resistance of the other cells

    n_cells = r0_cells.shape[1]

    if band_mean_without_eval_cell:
        r0_mean = np.zeros_like(r0_cells)

        for ic in range(n_cells):
            # Mean resistance of all cells except the cell in question
            colum_mask = np.ones((n_cells,), dtype=bool)
            colum_mask[ic] = False
            r0_mean[:, ic] = np.array(
                [
                    hodges_lehmann_estimator(r0_cells[i, colum_mask])
                    for i in range(r0_cells.shape[0])
                ]
            ).reshape((-1))
    else:
        r0_mean = r0_cells.mean(axis=1).reshape((-1, 1))

    # Look up the probability for the gaussian distribution based on mean and variance and
    # the threshold value
    r0std_cells = np.sqrt(r0var_cells)

    # Resistance band upper probability
    prob_above = 1 - normal_cdf(r0_mean + r0_band_delta, r0_cells, r0std_cells)

    # Resistance band lower probability
    prob_below = normal_cdf(r0_mean - r0_band_delta, r0_cells, r0std_cells)

    # Resistance band probability
    prob_outside_band = prob_above + prob_below

    if return_intermediate_arrays:
        return (prob_outside_band, prob_above, prob_below, r0_mean)
    else:
        return prob_outside_band


def calc_over_threshold_probability(
    r0_cells: np.ndarray,
    r0var_cells: np.ndarray,
    r0_upper_threshold: float,
) -> np.ndarray:

    return 1 - normal_cdf(r0_upper_threshold, r0_cells, np.sqrt(r0var_cells))


def calc_r0_cells_var(r0_cells: np.ndarray) -> np.ndarray:
    return r0_cells.var(axis=1)
