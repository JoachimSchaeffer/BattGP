import pathlib
import tempfile
import unittest

import src.config as cfg
from src.batt_data import cache
from src.batt_data.batt_data import BattData
from src.batt_data.data_utils import read_cell_characteristics

# path to the archive with the test fielddata, relative to this file
_REL_ARCHIVE = "../data/field_data_test.zip"
_REL_CACHE = "../data/cache"
_REL_OCV = "../data/ocv_linear_approx.csv"


class TestBattData_ReadFieldData(unittest.TestCase):

    def test_batt_data_init_from_field_data(self):
        archive = pathlib.Path(__file__).parent.resolve().joinpath(_REL_ARCHIVE)
        # cache_path = pathlib.Path(__file__).parent.resolve().joinpath(_REL_CACHE)

        ocv_path = pathlib.Path(__file__).parent.resolve().joinpath(_REL_OCV)

        cfg.PATH_FIELDDATA_DATA = archive

        cell_ocv = read_cell_characteristics(ocv_path)

        with tempfile.TemporaryDirectory() as cache_dir:
            cfg.PATH_DATA_CACHE = cache_dir

            battdata = BattData("3", cell_ocv)

            # as PATH_DATA_CACHE is not None, the read data should be cached now
            self.assertTrue(cache.is_cached("3"))

        self.assertEqual(battdata.id, "3")

    def test_batt_data_init_from_cache(self):
        cache_path = pathlib.Path(__file__).parent.resolve().joinpath(_REL_CACHE)

        ocv_path = pathlib.Path(__file__).parent.resolve().joinpath(_REL_OCV)

        cfg.PATH_DATA_CACHE = cache_path
        cfg.PATH_FIELDDATA_DATA = "<something invalid>"

        cell_ocv = read_cell_characteristics(ocv_path)

        battdata = BattData("3", cell_ocv)

        self.assertEqual(battdata.id, "3")


if __name__ == "__main__":
    unittest.main()
