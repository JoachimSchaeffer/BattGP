import os

from . import config as cfg


def _create_path(path: str) -> None:
    if not os.path.isdir(path):
        os.mkdir(path)


def setup_paths():
    if cfg.PATH_DATA_CACHE is not None:
        _create_path(cfg.PATH_DATA_CACHE)

    _create_path(cfg.PATH_FIGURES_DATA_VIS)
