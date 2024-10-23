from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import gpytorch
import numpy as np
import torch
import tqdm

from .temporal_kernel import TemporalKernel


@dataclass
class _SmootherData:
    zcorr: np.ndarray
    Pcorr: np.ndarray
    ts: float


def _update_kalman_pred_z_P(z: np.ndarray, P: np.ndarray, A: np.ndarray, Q: np.ndarray):
    nt = A.shape[0]

    # z+ = A z-
    z[:nt, :] = A @ z[:nt, :]

    # P+ = A P- A^* + Q
    P[:nt, :nt] = A @ P[:nt, :nt] @ A.T + Q
    P[nt:, :nt] = P[nt:, :nt] @ A.T
    P[:nt, nt:] = A @ P[:nt, nt:]


class ApproxSpatioTemporalGP:
    """
    Approximate spatial-temporal GP [Sarkka et al], [Huber]

    This GP is exact regarding the temporal dimension and uses an approximation based on
    basis vectors (inducing points) [Huber] regarding the spatial dimensions.

    The temporal dimension is implicit when updating and querying the GP. I.e. the
    input data never contains the temporal value.
    The current temporal value follows from the sum of the sizes of all previously
    executed time steps.

    This class also is able to perfom smoothing ("backward steps"). To opt in, the
    number of maximal smoothing steps must be specified when constructing an object of
    this class.
    """

    def __init__(
        self,
        spatial_X_base: torch.Tensor,
        spatial_kernel: gpytorch.kernels.Kernel,
        temporal_kernel: TemporalKernel,
        noise_var: float,
        smoothing_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        smoothing_steps: Optional[int] (default: None)
            Number of past steps for which the Rauch-Tung-Striebel smoother can be
            applied.
            None : No past means and covariances are stored, thus no smoothing is
                   possible.
            0    : Same as None.
            > 0  : Number of steps for which the past means and covarianves are stored.
                   The data is stored in a deque, thus newer data overwrites older data.
            -1   : All past means and covarianves are stored.
        """
        # Assert that the dimensions of the spatial_X_base agree with the spatial kernel
        if len(spatial_X_base[0]) != spatial_kernel.base_kernel.ard_num_dims:
            raise ValueError(
                "The number of spatial dimensions in the spatial_X_base does not "
                "agree with the number of dimensions in the spatial kernel."
            )

        if device is None:
            device = spatial_kernel.device

        self._device = device
        self._dtype_torch = torch.float64
        self._dtype_np = np.float64

        if spatial_kernel is None:
            spatial_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # Check whether spatial_X_base is a torch tensor
        if not isinstance(spatial_X_base, torch.Tensor):
            spatial_X_base = torch.tensor(
                spatial_X_base, dtype=self._dtype_torch, device=device
            )
        self.X_b = spatial_X_base
        self.spatial_kernel = spatial_kernel
        if device is not None:
            self.spatial_kernel = self.spatial_kernel.to(device)

        self.noise_var = noise_var

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Kbb = self.spatial_kernel.forward(self.X_b, self.X_b)

        self.Kbb = Kbb.detach().cpu().numpy()
        self.invKbb = np.linalg.pinv(self.Kbb, rcond=1e-8)

        self.temporal_kernel = temporal_kernel
        nt = self.temporal_kernel.order

        nb = self.X_b.shape[0]

        Pt_init = self.temporal_kernel.get_initial_covariance()
        Ps_init = self.Kbb

        self.P: np.ndarray = np.block(
            [
                [Pt_init, np.zeros((nt, nb))],
                [np.zeros((nb, nt)), Ps_init],
            ]
        )

        self.z = np.zeros((nt + nb, 1))

        self.t = 0.0

        self._smoothing: Optional[deque[_SmootherData]] = None

        if (smoothing_steps is not None) and (smoothing_steps != 0):
            self._smoothing = deque(
                maxlen=None if smoothing_steps == -1 else smoothing_steps
            )

    @property
    def t(self) -> float:
        return self._t

    @t.setter
    def t(self, t: float):
        self._t = t

    def time_step(self, time_step_size: float):
        if self._smoothing is not None:
            self._smoothing.append(
                _SmootherData(self.z.copy(), self.P.copy(), time_step_size)
            )

        (At, Qt) = self.temporal_kernel.get_kalman_matrices(time_step_size)

        _update_kalman_pred_z_P(self.z, self.P, At, Qt)

        self._t += time_step_size

    def _get_output_matrix(self, X_t: np.ndarray):
        X_t = torch.tensor(X_t, dtype=self._dtype_torch, device=self._device)

        nm = X_t.shape[0]
        nb = self.Kbb.shape[0]
        nt = self.temporal_kernel.order

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Ktb = self.spatial_kernel.forward(X_t, self.X_b)
            Ktt = self.spatial_kernel.forward(X_t, X_t)

        Ktb = Ktb.detach().cpu().numpy()
        Ktt = Ktt.detach().cpu().numpy()

        H = np.zeros((nm, nt + nb))

        if nt > 0:
            H[:, 0] = 1

        H2 = Ktb @ self.invKbb
        H[:, nt:] = H2

        return (H, Ktt)

    def _get_covariance(
        self, H: np.ndarray, Ktt: np.ndarray, P: np.ndarray
    ) -> np.ndarray:
        nt = self.temporal_kernel.order
        return Ktt + H @ P @ H.T - H[:, nt:] @ self.Kbb @ H[:, nt:].T

    def update(self, Xs_t: np.ndarray, Y_t: np.ndarray) -> None:
        nm = Xs_t.shape[0]

        (H, Ktt) = self._get_output_matrix(Xs_t)
        C = self._get_covariance(H, Ktt, self.P)

        v = Y_t.reshape((-1, 1)) - H @ self.z

        K = (
            self.P
            @ H.T
            @ torch.linalg.inv(
                torch.tensor(
                    C + np.eye(nm) * self.noise_var, dtype=torch.float64, device="cpu"
                )
            )
            .detach()
            .numpy()
        )

        self.z = self.z + K @ v
        self.P = self.P - K @ H @ self.P

    def predict(
        self,
        X_q: np.ndarray,
        full_cov: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        (H, Kqq) = self._get_output_matrix(X_q)
        C = self._get_covariance(H, Kqq, self.P)

        Y_q = (H @ self.z).reshape((-1,))

        if not full_cov:
            C = np.diag(C)

        return (Y_q, C)

    def predict_in(
        self,
        time_step: float,
        X_q: np.ndarray,
        full_cov: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if time_step == 0:
            return self.predict(X_q, full_cov)

        (At, Qt) = self.temporal_kernel.get_kalman_matrices(time_step)

        z = self.z.copy()
        P = self.P.copy()

        _update_kalman_pred_z_P(z, P, At, Qt)

        (H, Kqq) = self._get_output_matrix(X_q)
        C = self._get_covariance(H, Kqq, P)

        Y_q = (H @ z).reshape((-1,))

        if not full_cov:
            C = np.diag(C)

        return (Y_q, C)

    def get_time_derivative(self) -> Tuple[float, float]:
        return self.temporal_kernel.get_time_derivative(self.z, self.P)

    def smooth(
        self,
        X_q: np.ndarray,
        n_steps: Optional[int] = None,
        show_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._smoothing is None:
            raise ValueError(
                "Method smooth can only be used if object was created with "
                "non-None argument 'smoothing'"
            )

        if n_steps is None:
            n_steps = len(self._smoothing) + 1
        else:
            n_steps = min(n_steps, len(self._smoothing) + 1)

        (H, Kqq) = self._get_output_matrix(X_q)

        t = np.zeros((n_steps,))
        Y_q = np.zeros((n_steps, X_q.shape[0]))
        C = np.zeros_like(Y_q)

        tcur = self._t
        zcur = self.z
        Pcur = self.P

        for i in tqdm.tqdm(
            reversed(range(n_steps)),
            leave=False,
            disable=not show_progress,
            total=n_steps,
        ):
            if i < n_steps - 1:
                zc = self._smoothing[i].zcorr
                Pc = self._smoothing[i].Pcorr
                tsc = self._smoothing[i].ts
                tcur -= tsc

                (At, Qt) = self.temporal_kernel.get_kalman_matrices(tsc)

                zp = zc.copy()
                Pp = Pc.copy()

                _update_kalman_pred_z_P(zp, Pp, At, Qt)

                nt = At.shape[0]
                ns = Pp.shape[0] - nt
                A = np.block(
                    [[At, np.zeros((nt, ns))], [np.zeros((ns, nt)), np.eye(ns)]]
                )

                G = (
                    Pc
                    @ A.T
                    @ torch.linalg.inv(
                        torch.tensor(Pp, dtype=torch.float64, device="cpu")
                    )
                    .detach()
                    .numpy()
                )
                zcur = zc + G @ (zcur - zp)
                Pcur = Pc + G @ (Pcur - Pp) @ G.T

            t[i] = tcur
            Y_q[[i], :] = H @ zcur
            C[[i], :] = np.diag(self._get_covariance(H, Kqq, Pcur))

        return (t, Y_q, C)
