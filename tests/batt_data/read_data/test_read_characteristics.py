import pathlib
import unittest

import pandas as pd

import src.config as cfg
from src.batt_data.data_utils import (
    read_cell_characteristics,
)

# path to the archive with the test fielddata, relative to this file
_REL_OCV = "../../data/ocv_linear_approx.csv"


class TestReadCharacteristics(unittest.TestCase):

    def test_read_characteristics(self):
        ocv_path = pathlib.Path(__file__).parent.resolve().joinpath(_REL_OCV)

        cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC = ocv_path
        _ = read_cell_characteristics()

    def test_read_characteristics_explicit_path(self):
        ocv_path = pathlib.Path(__file__).parent.resolve().joinpath(_REL_OCV)

        cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC = None
        _ = read_cell_characteristics(ocv_path)

    def test_read_characteristics_from_dataframe(self):
        ocv = pd.DataFrame.from_dict(
            {"SOC": [0.0, 20.0, 90.0, 100.0], "OCV": [3.36, 3.4, 3.41, 3.45]}
        )

        cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC = None
        _ = read_cell_characteristics(df_ocv=ocv)

    def test_read_characteristics_use_akima(self):
        ocv = pd.DataFrame.from_dict(
            {"SOC": [0.0, 20.0, 90.0, 100.0], "OCV": [3.36, 3.4, 3.41, 3.45]}
        )

        cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC = None
        _ = read_cell_characteristics(df_ocv=ocv, interpolator="akima")


if __name__ == "__main__":
    unittest.main()
