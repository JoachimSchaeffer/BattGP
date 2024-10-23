"""Functions to access the data cache.

This module serves as an abstraction layer such that no other module has to know the
format of the cache files as well as the naming convention.

The "outer world" communicates with the cache only by the battery ids (str) and the
Pandas dataframes holding the data.

On the other hand, this module doesn't care about the data stored within the dataframes,
provided that the data types are serializable by the used file format.

Thus, if the data format should change from feather to something else, only the
function within this module should be changed.

The cache is active if the constant PATH_DATA_CACHE in the global config module is not
None but gives the path that is used for caching the data.
"""

import os
from glob import glob
from typing import List

import pandas as pd

from .. import config as cfg


def get_cache_filename(batt_id: str) -> str:
    """Return file name for the cached data of the specified battery id.

    If no PATH_DATA_CACHE is specified in the global config, this function returns an
    emtpy str.
    If the PATH_DATA_CACHE is specified, it returns always a file name, even if this
    file does not exist.
    """
    if cfg.PATH_DATA_CACHE is None:
        return ""

    return os.path.join(cfg.PATH_DATA_CACHE, f"{batt_id}.feather")


def is_cached(batt_id: str) -> bool:
    """Return True iff the specified battery id is available in the cache."""
    return (cfg.PATH_DATA_CACHE is not None) and os.path.isfile(
        get_cache_filename(batt_id)
    )


def save_dataframe(batt_id: str, df: pd.DataFrame):
    """Save the dataframe df for the battery id batt_id in the cache.

    Existing data for this battery in the cache gets overwritten.

    Raises a RuntimeError if no PATH_DATA_CACHE is specified.
    """
    if cfg.PATH_DATA_CACHE is None:
        raise RuntimeError("No config.PATH_DATA_CACHE specified")

    df.reset_index().to_feather(get_cache_filename(batt_id))


def load_dataframe(batt_id: str) -> pd.DataFrame:
    """Load the data for the specified battery from the cache.

    Raises a ValueError if this battery doesn't have cached data.
    """
    if not is_cached(batt_id):
        raise ValueError(f"No cached data for battery id '{batt_id}'")

    df = pd.read_feather(get_cache_filename(batt_id))
    df.set_index(df.columns[0], inplace=True)

    return df


def get_cached_battery_ids() -> List[str]:
    """Get list of all battery ids that have cached data."""
    if cfg.PATH_DATA_CACHE is None:
        return []

    ids = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob(os.path.join(cfg.PATH_DATA_CACHE, "*.feather"))
    ]

    if all(id.isnumeric() for id in ids):
        return sorted(ids, key=float)
    else:
        return sorted(ids)
