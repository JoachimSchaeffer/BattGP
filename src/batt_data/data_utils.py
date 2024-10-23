"""Functions to load the raw battery data.

This module provides function to read the battery data from disc, either directly from
the csv files or by using the local data cache.
The preferred way is to load the data by crating an BattData object, in which case one
don't use this module directly, with the execption of `read_cell_characteristics`.

Also, to avoid larger breaks due to caching during work, the function
`build_data_cache` can invoked once to cache the data of all available batteries in one
go.

The data sources and cache are controlled by the following constants in the global
config module:
    config.PATH_FIELDDATA_DATA
                Either path to the directory where the csv files are stored
                or path to the zip file that containts the csv files.
                In each case the expected format is that the data of a battery system
                is stored in the root of the specified folder or archive in a csv with
                the name
                    data_sys_ID.csv
                where ID might be any string used to name the specific battery.

                This constant can be also None. In this case the PATH_DATA_CACHE must
                be available and it must contain the battery which data is to be loaded.

    config.PATH_DATA_CACHE
                Path to the directory where the battery data is cached in a compressed
                and faster loadable data format.
                If this constant is None, the cache is deactivated.

    config.PATH_FIELDDATA_CELL_CHARACTERISTIC
                Path to the csv file with the SOC-OCV curve.
                This file must have to columns named SOC and OCV containing the SOC in
                percent and OCV in volt. The SOC values must be strictly increasing and
                cover at least the interval from 0.1 % to 99.9 %.

    With the default arguments the cache data is preferred, and if the field data (csv)
    are used, it is written to the cache after it is read.
"""

import os
import warnings
from glob import glob
from typing import List, Literal, Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from tqdm import tqdm

from .. import config as cfg
from ..exceptions import InvalidConfigError
from . import cache
from .cell_characteristics import CellCharacteristics
from .data_columns import DATA_COLUMNS_MAPPING

_FIELD_DATA_ZIP_FOLDER = "field_data/"
_FIELD_DATA_CSV_PREFIX = "data_sys_"
_FIELD_DATA_CSV_EXT = ".csv"


def read_cell_characteristics(
    path: Optional[str] = None,
    df_ocv: Optional[pd.DataFrame] = None,
    interpolator: Literal["akima", "pchip"] = "pchip",
) -> CellCharacteristics:
    """Read SOC-OCV curve.

    Reads the SOC-OCV curve that is provided normally in a csv file, which format is
    described in the module level documentation.

    By default the data is read from config.PATH_FIELDDATA_CELL_CHARACTERISTIC.
    Alternatively a csv file can be specified directly by the `path` argument, or
    one can provide the SOC-OCV curve by a dataframe `df_ocv`, which needs to provide
    the same columns with the same restrictions as the csv file.

    The provided datapoints are later interpolated, and with `interpolator` the method
    can be chosen between "pchip" (default) and "akima".
    """
    if path is not None and df_ocv is not None:
        raise InvalidConfigError(
            "Either 'path' or 'df_ocv' may be given, but not both."
        )
    elif path is not None:
        df_ocv = pd.read_csv(path, header=[0])
    elif df_ocv is None:
        df_ocv = pd.read_csv(cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC, header=[0])

    # interpolate simple OCV curve
    if interpolator == "akima":
        simple_ocv = Akima1DInterpolator(df_ocv.SOC, df_ocv.OCV)
    elif interpolator == "pchip":
        simple_ocv = PchipInterpolator(df_ocv.SOC, df_ocv.OCV, extrapolate=True)
    else:
        raise ValueError(
            f"Unknown interpolator string, only akima and pchip supported: {interpolator}"
        )
    return CellCharacteristics(ocv_at_soc=simple_ocv)


def build_data_cache(
    rebuild_cache: bool = False,
):
    """Serialize data to cache folder.

    Reads the data of all available batteries in config.PATH_FIELDDATA_DATA and writes
    it in the cache.
    If the data of a battery is allready available in the cache, this battery is skipped
    (without checking of the source data has changed.)
    To force overwritting existing data, the function can be called with
    `rebuild_cache=True`.

    The basic clean up of read_battery_fielddata is applied.
    """
    batt_ids_folders = get_fieldata_battery_ids()

    # check if batt_id_folders exists and is not empty
    if not batt_ids_folders:
        print(
            "No batteries found in data folder! "
            f"Put time series data extracted in Path: {cfg.PATH_FIELDDATA_DATA} "
            " in subfolders named by ID."
        )

    print(f"{len(batt_ids_folders)} batteries found in field data folder or archive")

    if rebuild_cache:
        batts_to_cache = batt_ids_folders
    else:
        batt_ids_cached = cache.get_cached_battery_ids()
        batts_to_cache = list(set(batt_ids_folders) - set(batt_ids_cached))

        allready_cached_count = len(batt_ids_folders) - len(batts_to_cache)

        if allready_cached_count > 0:
            print(
                f"{allready_cached_count} of these are allready cached and therefore skipped.\n"
                "Recaching can be forced by setting 'rebuild_cache=True' when calling this function."
            )

    for batt_id in (pbar := tqdm(_sort_batt_ids(batts_to_cache))):
        pbar.set_description(f"read batt_id '{batt_id}'")
        read_battery_data(batt_id, write_to_cache=True, print_msgs=False)


def read_battery_data(
    batt_id: str,
    resample_T: Optional[pd.DateOffset | pd.Timedelta | str] = None,
    keep_columns: Optional[List[str]] = None,
    use_cache: bool = True,
    write_to_cache: bool = True,
    print_msgs: bool = False,
) -> pd.DataFrame:
    """
    Reads the battery data.

    This function manages automatically if the data is read from the raw csv or the
    cache and should be used normally to load battery data.

    Parameters:
    ----------
    subsample_T: frequency to resample Data: e.g '5min', omit if all data shall be kept
                 This operation is applied after writing the data to the cache.

    keep_columns: list fo columns that shall be kept. If set to None (default) all
                  columns that are given an import name in the module data_columns are
                  kept.
                  This operation is applied BEFORE writing the data to the cache.

    use_cache: If `True` (default) the cached data is used (if available).
               If `False` the cached data is not used (even if no raw csv data is
               available)

    write_to_cache: If `True` the data is written into the cache iff it was read from
                    the raw csv. (I.e. the cache is never overwritten by data read from
                    the cache itself.)

    print_msgs: If `True` (default) some status messages are emitted.
    """

    if cfg.PATH_DATA_CACHE is None:
        if use_cache:
            warnings.warn(
                "No cache folder specified with config.PATH_DATA_CACHE, ignoring 'use_cache=True'"
            )
            use_cache = False
        if write_to_cache:
            warnings.warn(
                "No cache folder specified with config.PATH_DATA_CACHE, ignoring 'write_to_cache=True'"
            )
            write_to_cache = False

    if not use_cache or not cache.is_cached(batt_id):
        df = read_battery_fielddata(batt_id, keep_columns)

        # convert to float32/int32 to save memory
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(
            np.float32
        )
        df[df.select_dtypes(np.int64).columns] = df.select_dtypes(np.int64).astype(
            np.int32
        )

        # locally save compressed version of dataframe
        if write_to_cache:
            cache.save_dataframe(batt_id, df)
    else:
        if print_msgs:
            print("loading frame from serialized file on hard drive...")

        df = cache.load_dataframe(batt_id)

    if keep_columns is not None:
        drop_cols = set(df.columns).difference(set(keep_columns))

        df.drop(columns=drop_cols, inplace=True)

    if resample_T:
        df = df.resample(resample_T).mean()

    if print_msgs:
        print(f"loaded dataframe for: {batt_id} with subsampling: {resample_T}")

    return df


def get_fieldata_battery_ids() -> List[str]:
    """Get list of battery ids available in config.PATH_FIELDDATA_DATA

    Returns [] and emits a warning if config.PATH_FIELDDATA_DATA is None.
    """

    if cfg.PATH_FIELDDATA_DATA is None:
        warnings.warn(
            "No folder or archive for field data specified with config.PATH_FIELDDATA_DATA"
        )
        return []
    elif not os.path.exists(cfg.PATH_FIELDDATA_DATA):
        raise InvalidConfigError(
            f"Path config.PATH_FIELDDATA_DATA = '{cfg.PATH_FIELDDATA_DATA}' does not exist."
        )
    elif os.path.isdir(cfg.PATH_FIELDDATA_DATA):
        csv_files = [
            os.path.basename(f)
            for f in glob(
                os.path.join(
                    cfg.PATH_FIELDDATA_DATA,
                    f"{_FIELD_DATA_CSV_PREFIX}*{_FIELD_DATA_CSV_EXT}",
                )
            )
        ]

        return _get_batt_ids(csv_files)
    else:
        with ZipFile(cfg.PATH_FIELDDATA_DATA, "r") as archive:
            files = archive.namelist()

        files = [
            f[len(_FIELD_DATA_ZIP_FOLDER) :]
            for f in files
            if f.startswith(_FIELD_DATA_ZIP_FOLDER)
        ]
        return _get_batt_ids(files)


def read_battery_fielddata(
    batt_id: str,
    unknown_columns_action: Literal["ignore", "keep", "error"] = "error",
    unused_columns_action: Literal["check_and_ignore", "ignore", "keep"] = "ignore",
) -> pd.DataFrame:
    """Read raw csv files for specified battery

    If keep_columns provides a list with column names, only those are returned.
    By default (keep_columns=None) only the columns that are given an import name in
    the module data_columns are read.

    Some basic clean up of the data is applied (dropping nan and inf values).
    """

    all_batt_ids = get_fieldata_battery_ids()

    if batt_id not in all_batt_ids:
        raise ValueError(f"Battery ID '{batt_id}' not available in provided field data")

    if os.path.isdir(cfg.PATH_FIELDDATA_DATA):
        file = os.path.join(
            cfg.PATH_FIELDDATA_DATA,
            f"{_FIELD_DATA_CSV_PREFIX}{batt_id}{_FIELD_DATA_CSV_EXT}",
        )

        df = _check_and_clean(
            pd.read_csv(file, index_col=0, parse_dates=True),
            unknown_columns_action=unknown_columns_action,
            unused_columns_action=unused_columns_action,
        )
    else:
        with ZipFile(cfg.PATH_FIELDDATA_DATA, "r") as archive:
            file = f"{_FIELD_DATA_ZIP_FOLDER}{_FIELD_DATA_CSV_PREFIX}{batt_id}{_FIELD_DATA_CSV_EXT}"

            with archive.open(file, "r") as fil:
                df = _check_and_clean(
                    pd.read_csv(fil, index_col=0, parse_dates=True),
                    unknown_columns_action=unknown_columns_action,
                    unused_columns_action=unused_columns_action,
                )

    return df


def _sort_batt_ids(ids: List[str]) -> List[str]:
    if all(id.isnumeric() for id in ids):
        return sorted(ids, key=float)
    else:
        return sorted(ids)


def _get_batt_id(file: str) -> Optional[str]:

    (basename, ext) = os.path.splitext(file)

    if ext != _FIELD_DATA_CSV_EXT:
        return None

    if ("\\" in basename) or ("/" in basename):
        return None

    if not basename.startswith(_FIELD_DATA_CSV_PREFIX):
        return None

    return basename[len(_FIELD_DATA_CSV_PREFIX) :]


def _get_batt_ids(files: List[str]) -> List[str]:
    batt_ids = [id for file in files if (id := _get_batt_id(file)) is not None]

    return _sort_batt_ids(batt_ids)


def _check_and_clean(
    df: pd.DataFrame,
    unknown_columns_action: Literal["ignore", "keep", "error"],
    unused_columns_action: Literal["check_and_ignore", "ignore", "keep"],
) -> pd.DataFrame:
    """Preprocess dataframe

    step 1: remove NaN and inf rows
    step 2: drop irrelevant columns
    """

    cols = set(df.columns)
    known_cols = set(k for k in DATA_COLUMNS_MAPPING.keys())
    necessary_cols = set(k for (k, v) in DATA_COLUMNS_MAPPING.items() if v is not None)

    if unused_columns_action in ("check_and_ignore", "keep"):
        expected_cols = known_cols
    else:
        expected_cols = necessary_cols

    missing_necessary_cols = expected_cols - cols

    if len(missing_necessary_cols) > 0:
        raise ValueError(
            f"the following data columns are missing: {', '.join(missing_necessary_cols)}"
        )

    unknown_cols = cols - known_cols

    if unknown_columns_action == "error" and len(unknown_cols) > 0:
        raise ValueError(
            f"the read data contains the following unknown columns: {', '.join(unknown_cols)}"
        )

    if unused_columns_action == "keep":
        keep_columns = expected_cols
    else:
        keep_columns = necessary_cols

    if unknown_columns_action == "keep":
        keep_columns += unknown_cols

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    drop_cols = set(df.columns).difference(set(keep_columns))

    if len(drop_cols) > 0:
        df.drop(columns=drop_cols, inplace=True)

    df.rename(
        columns={k: v for (k, v) in DATA_COLUMNS_MAPPING.items() if v is not None},
        inplace=True,
    )

    return df
