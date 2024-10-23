from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .. import config as cfg
from .. import df_utils
from ..exceptions import InsufficientDataError
from ..operating_point import Op
from .cell_characteristics import CellCharacteristics
from .data_utils import read_battery_data


@dataclass
class SegmentCriteria:
    soc_upper_limit: float
    soc_lower_limit: float
    ibat_upper_limit: float
    ibat_lower_limit: float
    t_upper_limit: float
    t_lower_limit: float


class BattData:
    """Battery data class.
    Contains information about the battery, such as the OCV curve, the
    data aframe with the data, the segmment selection, SOC calcualtion etc.
    """

    def __init__(
        self,
        batt_id: str,
        cell_characteristics: CellCharacteristics,
        segment_criteria: Optional[SegmentCriteria] = None,
        segment_selection: bool = True,
        gap_removal: bool = True,
        min_data_threshold: int = 1000,
    ):
        self.id: str = batt_id
        self.cell_characteristics: CellCharacteristics = cell_characteristics
        self.min_data_threshold = min_data_threshold

        # Information about segment selection.
        if segment_criteria is not None:
            self.segment_criteria: SegmentCriteria = segment_criteria
        else:
            self.segment_criteria: SegmentCriteria = SegmentCriteria(
                soc_upper_limit=cfg.SOC_UPPER_LIMIT,
                soc_lower_limit=cfg.SOC_LOWER_LIMIT,
                ibat_upper_limit=cfg.Ibat_UPPER_LIMIT,
                ibat_lower_limit=cfg.Ibat_LOWER_LIMIT,
                t_upper_limit=cfg.T_UPPER_LIMIT,
                t_lower_limit=cfg.T_LOWER_LIMIT,
            )

        self.df: pd.DataFrame = read_battery_data(self.id)

        (self.cells_series, self.cells_parallel) = self._determine_battery_type()

        self.df["OCV"] = self.cell_characteristics.ocv_lookup(
            self.df["SOC_Batt"].values
        ).astype(np.float64)

        self.df["T_max"] = self.df[self.T_cols].max(axis=1)
        self.df.join(_get_voltage_statistics(self.df[self.cell_voltage_cols]))

        if segment_selection:
            # Run the segment selection.
            self._select_segments()
        elif gap_removal:
            # Run the gap removal, which is otherwise done in the segment selection.
            self.df = df_utils.remove_gaps(self.df, cfg.GAP_REMOVAL_THRESHOLD)
            self._reset_time()
        else:
            self._reset_time()

        X, _ = self.generateTrainingData(
            -1, max_training_data=cfg.NB_DATAPOINTS, max_age=None
        )

        self.mean_op = Op(np.nan, np.nan, np.nan)
        self.median_op = Op(np.nan, np.nan, np.nan)
        self.update_op(X)

    @property
    def age(self) -> int:
        return self.df["time"].iloc[-1]

    def _reset_time(self) -> None:
        """Reset the time column to start at 0."""
        self.df["time"] = (self.df.index - self.df.index[0]).total_seconds() / (
            60 * 60 * 24
        )

    def _determine_battery_type(self) -> Tuple[int, int]:
        """
        Check the battery type and set the cell count and cell columns.
        This function is implemented as a method to allow for easy extension to
        other battery types.
        """
        return (8, 1)

    @property
    def cell_nrs(self) -> List[int]:
        return [i + 1 for i in range(self.cells_series * self.cells_parallel)]

    @property
    def batt_type_str(self) -> str:
        return f"{self.cells_series}s{self.cells_parallel}p"

    @property
    def cell_voltage_cols(self) -> List[str]:
        return [f"U_Cell_{i + 1}" for i in range(self.cells_series)]

    @property
    def T_cols(self) -> List[str]:
        return sorted(set(cfg.TEMP_MAP.values()))

    def _select_segments(self, additional_gap_removal: bool = True) -> pd.DataFrame:
        """
        Function selects datasetions for the estimation of R0.
        The selection is based on the following criteria set in the config file.
        """
        self.df = df_utils.remove_gaps(self.df, cfg.GAP_REMOVAL_THRESHOLD)

        Ibat_max = self.cells_parallel * self.segment_criteria.ibat_upper_limit
        Ibat_min = self.cells_parallel * self.segment_criteria.ibat_lower_limit
        eps_cnv = cfg.CNV_LIMIT
        self.df = self.df.query(
            f"SOC_Batt < {self.segment_criteria.soc_upper_limit} "
            f"& SOC_Batt > {self.segment_criteria.soc_lower_limit} "
            f"& I_Batt <= {Ibat_max} "
            f"& I_Batt >= {Ibat_min} "
            f"& I_CNV_Cell_1 < {eps_cnv} "
            f"& I_CNV_Cell_2 < {eps_cnv} "
            f"& I_CNV_Cell_3 < {eps_cnv} "
            f"& I_CNV_Cell_4 < {eps_cnv} "
            f"& I_CNV_Cell_5 < {eps_cnv} "
            f"& I_CNV_Cell_6 < {eps_cnv} "
            f"& I_CNV_Cell_7 < {eps_cnv} "
            f"& I_CNV_Cell_8 < {eps_cnv} "
            f"& I_CNV_Cell_1 > {-eps_cnv} "
            f"& I_CNV_Cell_2 > {-eps_cnv} "
            f"& I_CNV_Cell_3 > {-eps_cnv} "
            f"& I_CNV_Cell_4 > {-eps_cnv} "
            f"& I_CNV_Cell_5 > {-eps_cnv} "
            f"& I_CNV_Cell_6 > {-eps_cnv} "
            f"& I_CNV_Cell_7 > {-eps_cnv} "
            f"& I_CNV_Cell_8 > {-eps_cnv} "
            f"& T_Cell_1_2 > {self.segment_criteria.t_lower_limit} "
            f"& T_Cell_3_4 > {self.segment_criteria.t_lower_limit} "
            f"& T_Cell_5_6 > {self.segment_criteria.t_lower_limit} "
            f"& T_Cell_7_8 > {self.segment_criteria.t_lower_limit} "
            f"& T_Cell_1_2 < {self.segment_criteria.t_upper_limit} "
            f"& T_Cell_3_4 < {self.segment_criteria.t_upper_limit} "
            f"& T_Cell_5_6 < {self.segment_criteria.t_upper_limit} "
            f"& T_Cell_7_8 < {self.segment_criteria.t_upper_limit}"
        )
        # Remove data points with identical time stamps (duplicates), keep the first
        self.df = self.df[~self.df.index.duplicated(keep="first")]
        self._assert_nb_data_points()

        if additional_gap_removal:
            self.df = df_utils.remove_gaps(self.df, cfg.GAP_REMOVAL_THRESHOLD)

        self._reset_time()

        self._assert_nb_data_points()

    def _assert_nb_data_points(self) -> None:
        train_data_count = len(self.df)

        if train_data_count < self.min_data_threshold:
            raise InsufficientDataError(
                f"For battery '{self.id}' only {train_data_count} datapoints qualified with current data filter settings, "
                f"which is less than the minimum of {self.min_data_threshold}."
            )

    def generateTrainingData(
        self,
        cell_nr: int,
        max_training_data: Optional[int] = None,
        max_age: Optional[int] = None,
        strategy: Literal[
            "random", "even_time_spacing", "even_idx_spacing"
        ] = "even_idx_spacing",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate train input matrix X and target matrix Y."""

        if not (
            (cell_nr == -1)
            or ((cell_nr >= 1) and (cell_nr <= len(self.cell_voltage_cols)))
        ):
            raise ValueError(
                "cell_nr must be -1 (pack mode) or "
                f"between 1 and the total number of cells ({len(self.cell_voltage_cols)}), "
                f"but it is {cell_nr}"
            )

        # This dataframe will eventually contain the columns needed to build the
        # training data.
        # (It needs the columns time, I, SOC, T, U and R, where U is used as an
        # intermediate signal used to calculate R.)
        df_data = self.df[["time"]].copy()
        df_data["SOC"] = self.df["SOC_Batt"]

        # flag to indicate pack measurement and data generation for entire pack
        # not cell level
        if cell_nr == -1:
            # pack ocv is sum of all cell ocv
            df_data["OCV"] = self.cells_series * self.df["OCV"]
            # get mean temperature out of all readings as proxy for pack temperature
            df_data["T"] = self.df[self.T_cols].mean(axis=1)
            df_data["I"] = self.df["I_Batt"] / self.cells_parallel
            df_data["U"] = self.df["U_Batt"]

        else:  # cell mode
            df_data["OCV"] = self.df["OCV"]
            df_data["I"] = (self.df["I_Batt"] / self.cells_parallel) + self.df[
                f"I_CNV_Cell_{cell_nr}"
            ]
            df_data["T"] = self.df[cfg.TEMP_MAP[cell_nr]]
            df_data["U"] = self.df[f"U_Cell_{cell_nr}"]

        if max_age is not None:
            df_data = df_data[df_data["time"] <= max_age]

        self._assert_nb_data_points()

        df_data["R"] = (df_data["U"] - df_data["OCV"]) / df_data["I"]

        if cfg.SMOOTHING_ENABLED:
            df_data = df_data.resample(cfg.SMOOTH_MIN_WINDOW).mean()

        df_data = df_data[df_data["R"].notna()]
        assert len(df_data["R"]) > 0, "no datapoints qualified"

        if (
            (max_training_data is not None)
            and (max_training_data != -1)
            and (len(df_data) > max_training_data)
        ):
            df_data = df_utils.sample(
                df_data, max_training_data, strategy, include_first_and_last=True
            )

        X = df_data[["time", "I", "SOC", "T"]].to_numpy().astype(np.float64)
        Y = (
            df_data["R"]
            .to_numpy()
            .reshape((-1,))
            # .round(decimals=cfg.DECIMALS_R0)
            .astype(np.float64)
        )
        return X, Y

    def update_op(self, X: Optional[np.ndarray] = None, verbose: bool = False) -> None:
        """
        Compute the mean and median operating point (I, SOC, Temperature) for pack.
        """

        if X is None:
            self.mean_op.I = self.df["I_Batt"].mean()
            self.mean_op.SOC = self.df["SOC_Batt"].mean()
            self.mean_op.T = np.median(self.df[self.T_cols].mean())

            self.median_op.I = self.df["I_Batt"].median()
            self.median_op.SOC = self.df["SOC_Batt"].median()
            self.median_op.T = np.median(self.df[self.T_cols])

        else:
            self.mean_op.I = X[:, 1].mean()
            self.mean_op.SOC = X[:, 2].mean()
            self.mean_op.T = X[:, 3].mean()

            self.median_op.I = np.median(X[:, 1])
            self.median_op.SOC = np.median(X[:, 2])
            self.median_op.T = np.median(X[:, 3])

        if verbose:
            print(f"Battery mean operating point computed: {self.mean_op.disp_str()}")


def _get_voltage_statistics(voltage_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate voltage statistics, mean subtracted voltage, voltage stdv."""
    df_stat = pd.DataFrame(index=voltage_data.index)

    voltage_data = voltage_data.values

    n_cells = voltage_data.shape[1]

    voltage_sum = np.sum(voltage_data, axis=1)

    for c in range(n_cells):
        df_stat[f"U_Cell_WO{c + 1}_mean"] = (voltage_sum - voltage_data[:, c]) / (
            n_cells - 1
        )

    df_stat["CellVoltageVar"] = np.var(voltage_data, axis=1, ddof=1)
    df_stat["CellVoltageStdv"] = np.sqrt(df_stat["CellVoltageVar"])

    return df_stat
