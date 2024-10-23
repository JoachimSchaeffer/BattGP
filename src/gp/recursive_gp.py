from typing import Optional

import gpytorch
import numpy as np
import torch


class RecursiveGP:
    """
    Recursive GP regression based on F. Huber et al.
    Recursive update of the covariance matrix based on batches.
    This implementation is a mix of numpy and pytorch and is not optimized for speed.
    """

    def __init__(
        self,
        X_base: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        kernel: Optional[gpytorch.kernels.Kernel] = None,
        noise_var: float = 1.0,
        mean_function: Optional[gpytorch.means.Mean] = None,
        max_batch_size: int = 100,
        device: Optional[torch.device] = None,
    ):
        self._dtype_torch = X_base.dtype
        self._dtype_np = np.float64

        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.X_b = X_base
        self.kernel = kernel

        if device is not None:
            self.device = device
        else:
            self.device = self.X_b.device

        self.X_b = self.X_b.to(self.device)
        self.kernel = self.kernel.to(self.device)

        self.noise_var = noise_var

        if mean_function is None:
            self.mean_function = gpytorch.means.ZeroMean()
        else:
            self.mean_function = mean_function

        self.max_batch_size = max_batch_size

        # Constants (for constant base vectors), are used every inference step
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean_b = self.mean_function.forward(X_base)
            K_b = self.kernel.forward(X_base, X_base)

        self.mean_b = mean_b.detach().cpu().numpy()
        self.K_b = K_b.detach().cpu().numpy()
        self.invK_b = np.linalg.pinv(self.K_b, rcond=1e-8)

        # Recursion start
        self.mu_g_prev = self.mean_b
        self.C_g_prev = self.K_b

        if Y is not None:
            self.update(X_base, Y)

    @staticmethod
    def from_exactgp(
        base_model: gpytorch.models.ExactGP,
        use_y: bool = True,
        max_batch_size: int = 100,
    ):
        if use_y:
            Y = base_model.train_targets.detach().cpu().numpy()
        else:
            Y = None

        model = RecursiveGP(
            base_model.train_inputs.detach().cpu().numpy(),
            Y=Y,
            kernel=base_model.covar_module,
            noise_var=float(base_model.likelihood.noise_covar),
            mean_function=base_model.mean_module,
            max_batch_size=max_batch_size,
        )

        return model

    def update(self, X_t: np.ndarray, Y_t: np.ndarray) -> None:
        """
        update the model with new data and batch process if necessary
        """

        if X_t.dtype != self._dtype_np:
            X_t = X_t.astype(self._dtype_np)

        if len(X_t) > self.max_batch_size:
            X_t_batches = np.array_split(X_t, len(X_t) // self.max_batch_size)
            Y_t_batches = np.array_split(Y_t, len(Y_t) // self.max_batch_size)

            assert len(X_t_batches) == len(
                Y_t_batches
            ), "X_t and Y_t have different batch sizes"

            for X_t, Y_t in zip(X_t_batches, Y_t_batches):
                self._inference_and_update_step(X_t, Y_t)
        else:
            self._inference_and_update_step(X_t, Y_t)

    def predict(
        self,
        X_t: np.ndarray,
        full_cov: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        predict state given test input X_t, returns means and variances
        """
        m_t, C_t = self._inference_step(X_t, full_cov=full_cov)[:2]

        # add no noise_var to be consistent with gpytorch

        return m_t.reshape((-1,)), C_t

    def _inference_step(
        self, X_t: np.ndarray, full_cov: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_t = torch.tensor(X_t, dtype=self._dtype_torch, device=self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            Ktb = self.kernel.forward(X_t, self.X_b)
            mu_t = self.mean_function.forward(X_t)

            if full_cov:
                Ktt = self.kernel.forward(X_t, X_t)
            else:
                Ktt = self.kernel.forward(X_t, X_t, diag=True)

        Ktb = Ktb.detach().cpu().numpy()
        Ktt = Ktt.detach().cpu().numpy()
        mu_t = mu_t.detach().cpu().numpy()

        J_t = Ktb @ self.invK_b
        mu_p_t = mu_t + J_t @ (self.mu_g_prev - self.mean_b)

        if full_cov:
            C_p_t = Ktt + J_t @ (self.C_g_prev - self.K_b) @ J_t.T
        else:  # only diagonal - faster
            C_p_t = Ktt + np.diag(J_t @ (self.C_g_prev - self.K_b) @ J_t.T)

        return mu_p_t, C_p_t, J_t

    def _inference_and_update_step(self, X_t: np.ndarray, Y_t: np.ndarray) -> None:
        # batch data if necessary
        # 1. Inference at step t
        mu_p_t, C_p_t, J_t = self._inference_step(X_t)
        # 2. Update step t-1
        # compute Gain matrix
        try:
            G_t = (
                self.C_g_prev
                @ J_t.T
                @ np.linalg.pinv(
                    C_p_t + self.noise_var * np.eye(C_p_t.shape[0]), rcond=1e-8
                )
            )
        except np.linalg.LinAlgError:
            print("SVD convergence failed -- skipping update")
            return

        # update mean function
        mu_g_t = self.mu_g_prev + G_t @ (Y_t - mu_p_t)
        # update covariance matrix
        C_g_t = self.C_g_prev - G_t @ J_t @ self.C_g_prev
        # set current to prev
        self.mu_g_prev = mu_g_t
        self.C_g_prev = C_g_t

        return
