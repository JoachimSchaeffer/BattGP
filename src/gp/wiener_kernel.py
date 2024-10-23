import gpytorch
import torch

# import numpy as np


class WienerKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def forward(self, x1, x2, **params):
        distance = self.covar_dist(x1, x2, **params)

        only_diag = params.get("diag", False)

        if only_diag:
            minval = torch.minimum(x1, x2)
            # raise Exception
        else:
            minval = distance.clone()

            for c in range(x2.shape[0]):
                minval[:, c] = torch.minimum(x1, x2[c]).reshape((-1,))

            # if minval.shape[0] == minval.shape[1]:
            #     tmp = torch.pow(minval, 3) / 3 + distance * torch.pow(minval, 2) / 2
            #     tmp = tmp.detach().cpu().numpy()
            #     print("a", minval.shape)
            #     print(tmp)
            #     np.linalg.cholesky(tmp)
            #     print("b")

        return torch.pow(minval, 3) / 3 + distance * torch.pow(minval, 2) / 2
