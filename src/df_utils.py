from datetime import timedelta
from typing import Literal, Optional

import numpy as np
import pandas as pd


def remove_gaps(df: pd.DataFrame, gap_days: int = 90) -> pd.DataFrame:
    gaps = df.index.to_series().diff()
    gap_pos = np.where(gaps > timedelta(days=gap_days))
    # Get the length of the sections between the gaps.
    section_lengths = []
    for i in range(len(gap_pos[0])):
        section_lengths.append(df.index[-1] - df.index[gap_pos[0][-i - 1]])
        if section_lengths[-1] > timedelta(days=20):
            break
    if len(gap_pos[0]) == 1:
        # pick the longer section
        df_1 = df.iloc[: gap_pos[0][0]]  # noqa: E203
        df_2 = df.iloc[gap_pos[0][0] :]  # noqa: E203
        # get the time difference between the first and last timestamp
        time_diff_1 = df_1.index[-1] - df_1.index[0]
        time_diff_2 = df_2.index[-1] - df_2.index[0]
        if time_diff_1 > time_diff_2:
            df = df_1
        else:
            df = df_2
    elif len(gap_pos[0]) > 0:
        # Cut the dataframe to the last section correspodning to
        # gap_pos[0][-i-1]
        df = df.iloc[gap_pos[0][-i - 1] :]  # noqa: E203

    return df


def sample(
    df: pd.DataFrame,
    n: int,
    strategy: Literal[
        "random", "even_time_spacing", "even_idx_spacing"
    ] = "even_idx_spacing",
    include_first_and_last: bool = True,
    random_state: Optional[int] = None,
) -> pd.DataFrame:

    if strategy == "random":
        # ensure that the first and last dataset are always part of the sampled
        # data
        df_start, df_end = df.iloc[0], df.iloc[-1]
        df = df.sample(n, random_state=random_state)

        if include_first_and_last:
            df.iloc[0], df.iloc[-1] = df_start, df_end

    elif strategy == "even_idx_spacing":
        # evenly step through data
        idx_select = np.linspace(0, len(df) - 1, n, dtype=int)
        df = df.iloc[idx_select]

    elif strategy == "even_time_spacing":
        # Alternatively sample form data frame with roughly equal spacing in time
        weights = np.diff(df.index)
        # add first weight to be the same as the current first weight
        weights = np.insert(weights, 0, weights[0])
        # Convert weights to float in seconds
        weights = weights.astype("timedelta64[s]").astype(int)
        # use a moving average to smooth the weights
        filter_width = int(len(weights) / 100)
        weights = np.convolve(
            weights, np.ones(filter_width) / filter_width, mode="same"
        )
        # turn weights into a probability distribution
        weights = weights / np.sum(weights)
        # turn weights into a dataframe with the same index as the original dataframe
        weights = pd.Series(weights, index=df.index)

        df_start, df_end = df.iloc[0], df.iloc[-1]
        df = df.sample(n, random_state=42, weights=weights)

        if include_first_and_last:
            df.iloc[0], df.iloc[-1] = df_start, df_end

    return df
