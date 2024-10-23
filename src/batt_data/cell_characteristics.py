from dataclasses import dataclass

import numpy as np
from scipy import interpolate


@dataclass
class CellCharacteristics:
    ocv_at_soc: interpolate.interp1d

    def ocv_lookup(
        self, soc: float | np.ndarray, extrapolate: bool = False
    ) -> np.ndarray:
        if not extrapolate:
            soc = np.clip(soc, 0.1, 99.9)

        return self.ocv_at_soc(soc)
