import os
import pathlib
import tempfile
import unittest
from glob import glob
from zipfile import ZipFile

import src.config as cfg
from src.batt_data import cache
from src.batt_data.data_utils import (
    build_data_cache,
    read_battery_data,
    read_battery_fielddata,
)

# path to the archive with the test fielddata, relative to this file
_REL_ARCHIVE = "../../data/field_data_test.zip"


class TestReadFieldData(unittest.TestCase):

    def test_read_field_data_unpacked(self):
        archive = pathlib.Path(__file__).parent.resolve().joinpath(_REL_ARCHIVE)

        with tempfile.TemporaryDirectory() as raw_data_folder:
            with ZipFile(archive) as zf:
                zf.extractall(raw_data_folder)

            cfg.PATH_FIELDDATA_DATA = f"{raw_data_folder}/field_data"

            df = read_battery_fielddata("3")

        self.assertGreater(len(df), 0)

    def test_read_field_data_zip(self):
        archive = pathlib.Path(__file__).parent.resolve().joinpath(_REL_ARCHIVE)

        cfg.PATH_FIELDDATA_DATA = archive

        df = read_battery_fielddata("3")

        self.assertGreater(len(df), 0)

    def test_build_data_cache_unpacked(self):
        archive = pathlib.Path(__file__).parent.resolve().joinpath(_REL_ARCHIVE)

        with (
            tempfile.TemporaryDirectory() as raw_data_folder,
            tempfile.TemporaryDirectory() as cache_folder,
        ):
            with ZipFile(archive) as zf:
                zf.extractall(raw_data_folder)

            cfg.PATH_FIELDDATA_DATA = f"{raw_data_folder}/field_data"
            cfg.PATH_DATA_CACHE = cache_folder

            build_data_cache()

            # check manually if 3 and 7 are cached
            cached_files = glob(f"{cache_folder}/*")
            cached_files = {os.path.basename(p) for p in cached_files}

            self.assertSetEqual(cached_files, {"3.feather", "14.feather"})

            # while we're at it, lets test the cache helper function
            self.assertTrue(cache.is_cached("3"))
            self.assertTrue(cache.is_cached("14"))
            self.assertFalse(cache.is_cached(""))
            self.assertFalse(cache.is_cached("33"))
            self.assertFalse(cache.is_cached("6"))

            self.assertSetEqual(set(cache.get_cached_battery_ids()), {"3", "14"})

    def test_build_data_cache_zip(self):
        archive = pathlib.Path(__file__).parent.resolve().joinpath(_REL_ARCHIVE)

        with (tempfile.TemporaryDirectory() as cache_folder,):
            cfg.PATH_FIELDDATA_DATA = archive
            cfg.PATH_DATA_CACHE = cache_folder

            build_data_cache()

            # check manually if 3 and 7 are cached
            cached_files = glob(f"{cache_folder}/*")
            cached_files = {os.path.basename(p) for p in cached_files}

            self.assertSetEqual(cached_files, {"3.feather", "14.feather"})

            # while we're at it, lets test the cache helper function
            self.assertTrue(cache.is_cached("3"))
            self.assertTrue(cache.is_cached("14"))
            self.assertFalse(cache.is_cached(""))
            self.assertFalse(cache.is_cached("33"))
            self.assertFalse(cache.is_cached("6"))

            self.assertSetEqual(set(cache.get_cached_battery_ids()), {"3", "14"})

    def test_load_data(self):
        archive = pathlib.Path(__file__).parent.resolve().joinpath(_REL_ARCHIVE)

        with tempfile.TemporaryDirectory() as cache_folder:
            cfg.PATH_FIELDDATA_DATA = archive
            cfg.PATH_DATA_CACHE = cache_folder

            df = read_battery_data("3", write_to_cache=False)

            self.assertGreater(len(df), 0)
            self.assertEqual(len(cache.get_cached_battery_ids()), 0)

            df2 = read_battery_data("3")

            self.assertGreater(len(df2), 0)
            self.assertEqual(len(cache.get_cached_battery_ids()), 1)
            self.assertTrue(cache.is_cached("3"))

            # Making the cfg.PATH_FIELDDATA_DATA invalid
            # there should be no error afterwards, as the data of batters 7 should
            # be read from the cache
            cfg.PATH_FIELDDATA_DATA = None

            df3 = read_battery_data("3")
            self.assertGreater(len(df3), 0)


if __name__ == "__main__":
    unittest.main()
