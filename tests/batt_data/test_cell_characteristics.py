import unittest

import numpy as np
import pandas as pd

from src.batt_data.data_utils import (
    read_cell_characteristics,
)


class TestCellCharacteristics(unittest.TestCase):

    def test_ocv_lookup_scalars(self):
        ocv = pd.DataFrame.from_dict({"SOC": [0.0, 100.0], "OCV": [3.36, 3.45]})

        cc = read_cell_characteristics(df_ocv=ocv, interpolator="pchip")

        self.assertAlmostEqual(cc.ocv_lookup(50.0), (3.36 + 3.45) / 2)
        self.assertAlmostEqual(cc.ocv_lookup(50.0, extrapolate=True), (3.36 + 3.45) / 2)

    def test_ocv_lookup_arrays(self):
        ocv = pd.DataFrame.from_dict({"SOC": [0.0, 100.0], "OCV": [3.36, 3.45]})

        cc = read_cell_characteristics(df_ocv=ocv, interpolator="pchip")

        soc = np.array([25.0, 50.0, 75.0])
        ocv_expected = 3.36 + (3.45 - 3.36) * np.array([0.25, 0.5, 0.75])
        self.assertTrue(np.allclose(cc.ocv_lookup(soc), ocv_expected))
        self.assertTrue(np.allclose(cc.ocv_lookup(soc, extrapolate=True), ocv_expected))

    def test_ocv_lookup_extrapolate_option(self):
        ocv = pd.DataFrame.from_dict({"SOC": [0.0, 100.0], "OCV": [3.36, 3.45]})

        cc = read_cell_characteristics(df_ocv=ocv, interpolator="pchip")

        self.assertAlmostEqual(cc.ocv_lookup(0.0, extrapolate=True), 3.36)
        self.assertAlmostEqual(cc.ocv_lookup(100.0, extrapolate=True), 3.45)

        self.assertAlmostEqual(cc.ocv_lookup(0.0), 3.36 + 0.001 * (3.45 - 3.36))
        self.assertAlmostEqual(cc.ocv_lookup(100.0), 3.36 + 0.999 * (3.45 - 3.36))

    def test_ocv_lookup_extrapolate_arrays(self):
        ocv = pd.DataFrame.from_dict({"SOC": [0.0, 100.0], "OCV": [3.36, 3.45]})

        cc = read_cell_characteristics(df_ocv=ocv, interpolator="pchip")

        soc = np.array([-50.0, 0.0, 0.1, 50.0, 99.9, 100.0, 120])
        ocv_expected_extrap = 3.36 + (3.45 - 3.36) * soc / 100
        ocv_expected_noextrap = 3.36 + (3.45 - 3.36) * soc.clip(0.1, 99.9) / 100

        self.assertTrue(
            np.allclose(cc.ocv_lookup(soc, extrapolate=True), ocv_expected_extrap)
        )
        self.assertTrue(np.allclose(cc.ocv_lookup(soc), ocv_expected_noextrap))


if __name__ == "__main__":
    unittest.main()
