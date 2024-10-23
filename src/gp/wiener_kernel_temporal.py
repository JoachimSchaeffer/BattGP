import numpy as np

from .temporal_kernel import TemporalKernel


class WienerTemporalKernel(TemporalKernel):
    def __init__(self, outputscale: float = 1.0):
        super().__init__()
        self._outputscale = outputscale

    @property
    def order(self) -> int:
        return 2

    @property
    def outputscale(self) -> float:
        return self._outputscale

    @outputscale.setter
    def outputscale(self, outputscale: float):
        if outputscale != self._outputscale:
            self._outputscale = outputscale
            self._cache = None

    def get_initial_covariance(self) -> np.ndarray:
        return np.array([[0.0, 0.0], [0.0, 0.0]])

    def _get_kalman_matrices(self, Ts: float):
        if Ts == -1.0:
            raise NotImplementedError("continuous time case not implemented")

        A = np.array([[1, Ts], [0, 1]], dtype=float)
        Q = self._outputscale * np.array([[Ts**3 / 3, Ts**2 / 2], [Ts**2 / 2, Ts]])

        return (A, Q)

    def get_time_derivative(self, z: np.ndarray, P: np.ndarray) -> float:
        return (z[1], P[1, 1])
