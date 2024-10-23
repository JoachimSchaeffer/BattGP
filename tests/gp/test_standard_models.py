import unittest

import gpytorch
import numpy as np
import torch

from src.gp.standard_models import ScaledRBFModel


class TestStandardModels(unittest.TestCase):

    def test_rbf_1d(self):

        x = np.array([[1.0]])

        gp = ScaledRBFModel(
            torch.Tensor(x),
            torch.Tensor([10.0]),
            noise_variance=3.0,
            outputscale=3.0,
            lengthscale=2.0,
        )

        # deactivate "debug" to prevent warning that gp is called with training data
        with gpytorch.settings.debug(False):
            (yq, var_yq) = gp.predict(x)

        # If we have only one training point that corresponds to a base vector, and
        # the measurement variance equals the prior's variance, we expect to predict
        # for the same point a value that is the mean of the prior (0) and the training
        # data, with the variance also the half of the prior's/measurement variance
        self.assertAlmostEqual(yq[0], 5.0)
        self.assertAlmostEqual(var_yq[0], 1.5, places=5)

        # If we add another training point at the same location, we expect a weighted
        # mean value, weighted by the reciprocals of the variances.
        gp = ScaledRBFModel(
            torch.Tensor([[1.0], [1.0]]),
            torch.Tensor([10.0, 10.0]),
            noise_variance=3.0,
            outputscale=3.0,
            lengthscale=2.0,
        )
        (yq, var_yq) = gp.predict(x)

        self.assertAlmostEqual(yq[0], (5.0 / 1.5 + 10.0 / 3.0) / (1 / 1.5 + 1 / 3.0), 5)
        self.assertAlmostEqual(var_yq[0], 3.0 / 3, 5)


if __name__ == "__main__":
    unittest.main()
