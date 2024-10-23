from typing import Literal, Optional

import numpy as np
import pandas as pd
import tqdm

from ..gp.spatiotemporal_gp import ApproxSpatioTemporalGP
from ..operating_point import Op
from .cellnr import get_causal_tag

CleanupNegativeVariancesStrategyStr = Literal["prior", "eps"]


def apply_stgp(
    model: ApproxSpatioTemporalGP,
    xt: np.ndarray,
    yt: np.ndarray,
    ref_op: Op,
    sampling_time_sec: int,
    smooth: bool,
    max_batch_size: Optional[int],
    cell_tag: Optional[str] = None,
) -> pd.DataFrame:
    """Apply spatiotemporal GP"""

    tt = xt[:, 0]
    st = xt[:, 1:]
    min_sampling_time = np.min(np.diff(tt))

    Ts = sampling_time_sec / (60 * 60 * 24)
    t_end = tt[-1]

    t_r = np.arange(0, t_end, Ts)
    ds_cnt = np.zeros_like(t_r, dtype=int)
    r0_r = np.zeros_like(t_r)
    r0var_r = np.zeros_like(t_r)
    dr0_r = np.zeros_like(t_r)
    dr0var_r = np.zeros_like(t_r)

    (r0_r[[0]], r0var_r[[0]]) = model.predict(ref_op.into_row_vector())
    (dr0_r[[0]], dr0var_r[[0]]) = model.get_time_derivative()

    idx_t1 = 0

    for i in tqdm.tqdm(range(len(t_r) - 1), leave=False):
        # Determine maximal number of samples within the current sampling interval to
        # avoid checking to many rows in the lines after.
        max_samples = int(np.ceil((t_r[i + 1] - t_r[i]) / min_sampling_time))

        # Get data indices for first and one behind last sample of current sampling
        # interval.
        idx_t0 = np.argmax(tt[idx_t1 : idx_t1 + max_samples] >= t_r[i]) + idx_t1
        idx_t1 = np.argmax(tt[idx_t0 : idx_t0 + max_samples] > t_r[i + 1]) + idx_t0

        cur_st = st[idx_t0:idx_t1, :]
        cur_yt = yt[idx_t0:idx_t1]

        # Update model with data from current sampling interval.
        # If match_batch_size is specified and there are more samples than this value
        # within the current sampling interval, update model by chunks
        n = len(cur_yt)
        batch_size = n if max_batch_size is None else max_batch_size
        idx0 = 0

        while idx0 < n:
            idx1 = idx0 + batch_size

            if idx1 > n:
                idx1 = n

            model.update(cur_st[idx0:idx1, :], cur_yt[idx0:idx1])

            idx0 = idx1

        ds_cnt[[i + 1]] = n

        (r0_r[[i + 1]], r0var_r[[i + 1]]) = model.predict(ref_op.into_row_vector())
        (dr0_r[[i + 1]], dr0var_r[[i + 1]]) = model.get_time_derivative()

        model.time_step(Ts)

    if cell_tag is None or cell_tag == "":
        cell_tag = ""
    else:
        cell_tag = f"_{cell_tag}"

    if smooth:
        (_, r0_rs, r0var_rs) = model.smooth(
            ref_op.into_row_vector(), show_progress=True
        )

        causal_tag = get_causal_tag(True)
        acausal_tag = get_causal_tag(False)

        df = pd.DataFrame(
            {
                "t": t_r,
                "ds_count": ds_cnt,
                f"r0_{causal_tag}{cell_tag}": r0_r,
                f"r0var_{causal_tag}{cell_tag}": r0var_r,
                f"dr0_{causal_tag}{cell_tag}": dr0_r,
                f"dr0var_{causal_tag}{cell_tag}": dr0var_r,
                f"r0_{acausal_tag}{cell_tag}": r0_rs.reshape((-1,)),
                f"r0var_{acausal_tag}{cell_tag}": r0var_rs.reshape((-1,)),
            }
        )
    else:
        causal_tag = get_causal_tag(True)

        df = pd.DataFrame(
            {
                "t": t_r,
                "ds_count": ds_cnt,
                f"r0_{causal_tag}{cell_tag}": r0_r,
                f"r0var_{causal_tag}{cell_tag}": r0var_r,
                f"dr0_{causal_tag}{cell_tag}": dr0_r,
                f"dr0var_{causal_tag}{cell_tag}": dr0var_r,
            }
        )
    return df


def cleanup_negative_variances(
    df: pd.DataFrame,
    strategy: CleanupNegativeVariancesStrategyStr = "prior",
    model: Optional[ApproxSpatioTemporalGP] = None,
):
    """
    sets all negative values for the variance to the prior variance of the GP

    Alters the argument in place.
    """
    if strategy == "prior":
        prior_variance = model.spatial_kernel[0].outputscale.detach().cpu().numpy()
    elif strategy == "eps":
        prior_variance = 1e-8

    var_cols = [c for c in df.columns if c.startswith("r0var")]
    for c in var_cols:
        vb = df[c] < 0
        df.loc[vb, c] = prior_variance
