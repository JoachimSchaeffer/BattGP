from typing import Optional, Tuple

import numpy as np


class TemporalKernel:
    """
    Abstract base class for temporal kernels that defines the interface and also handles
    the caching of previous calculated kalman matrices.
    """

    def __init__(self):
        self._cache: Optional[Tuple[float, float, float]] = None

    def get_kalman_matrices(
        self,
        time_step_size: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache is not None and self._cache[0] == time_step_size:
            return (self._cache[1], self._cache[2])

        if time_step_size == 0.0:
            A = np.eye(self.order)
            Q = np.zeros(self.order)
        else:
            (A, Q) = self._get_kalman_matrices(time_step_size)

        self._cache = [time_step_size, A, Q]

        return (A, Q)

    @property
    def order(self) -> int:
        raise NotImplementedError("TemporalKernel is abstract")

    def get_initial_covariance(self) -> np.ndarray:
        raise NotImplementedError("TemporalKernel is abstract")

    def _get_kalman_matrices(self, ts: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("TemporalKernel is abstract")

    def get_time_derivative(self, z: np.ndarray, P: np.ndarray) -> Tuple[float, float]:
        raise NotImplementedError("TemporalKernel is abstract")


class ZeroTemporalKernel(TemporalKernel):
    def __init__(self):
        super().__init__()

    @property
    def order(self) -> int:
        return 0

    def get_initial_covariance(self) -> np.ndarray:
        return np.array([]).reshape((0, 0))

    def _get_kalman_matrices(self, Ts: float):
        if Ts == -1.0:
            raise NotImplementedError("continuous time case not implemented")

        A = np.array([[]]).reshape((0, 0))
        Q = np.array([[]]).reshape((0, 0))

        return (A, Q)

    def get_time_derivative(
        self, _z: np.ndarray, _P: np.ndarray
    ) -> Tuple[float, float]:
        return (0.0, 0.0)
