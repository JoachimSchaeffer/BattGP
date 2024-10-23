import pathlib
import unittest

import src.config as cfg
from src.batt_data.data_utils import (
    read_battery_data,
)

# path to the archive with the test fielddata, relative to this file
_REL_CACHE = "../../data/cache"


class TestReadData(unittest.TestCase):

    def test_read_data(self):

        cache_dir = pathlib.Path(__file__).parent.resolve().joinpath(_REL_CACHE)

        cfg.PATH_DATA_CACHE = cache_dir
        # PATH_FIELDDATA_DATA should be never used in these tests, but to be sure
        # that we will not inadvertently use the "production" directory, we set it
        # to None.
        cfg.PATH_FIELDDATA_DATA = None

        df = read_battery_data("3")

        self.assertGreater(len(df), 0)

    def test_read_data_select_columns(self):

        cache_dir = pathlib.Path(__file__).parent.resolve().joinpath(_REL_CACHE)

        cfg.PATH_DATA_CACHE = cache_dir
        # PATH_FIELDDATA_DATA should be never used in these tests, but to be sure
        # that we will not inadvertently use the "production" directory, we set it
        # to None.
        cfg.PATH_FIELDDATA_DATA = None

        keep_columns = ["U_Batt", "SOC_Batt"]
        df = read_battery_data("3", keep_columns=keep_columns)

        self.assertGreater(len(df), 0)
        self.assertSetEqual(set(df.columns), set(keep_columns))


if __name__ == "__main__":
    unittest.main()
