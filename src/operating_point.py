from dataclasses import dataclass

import numpy as np


@dataclass
class Op:
    I: float  # noqa: E741
    SOC: float
    T: float

    def into_array(self) -> np.ndarray:
        return np.array([self.I, self.SOC, self.T])

    def into_row_vector(self) -> np.ndarray:
        return np.array([self.I, self.SOC, self.T]).reshape(1, -1)

    def disp_str(self) -> str:
        return f"I = {self.I:.2f} A, SOC = {self.SOC:.2f} %, T = {self.T:.2f} °C"
