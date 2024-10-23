import unittest

import numpy as np
import scipy.linalg

from src.gp.wiener_kernel_temporal import WienerTemporalKernel


class TestWienerTemporalKernel(unittest.TestCase):

    def test_kernel(self):

        # create kernel with default outputscale (= 1)
        outputscale = 1
        kernel = WienerTemporalKernel()
        self.assertEqual(kernel.outputscale, outputscale)

        self.assertEqual(kernel.order, 2)
        self.assertEqual(np.linalg.norm(kernel.get_initial_covariance()), 0)

        Ts = 0.2
        (A, Q) = kernel.get_kalman_matrices(Ts)

        Ar = scipy.linalg.expm(Ts * np.array([[0, 1.0], [0, 0]]))
        Qr = outputscale * np.array([[Ts**3 / 3, Ts**2 / 2], [Ts**2 / 2, Ts]])

        self.assertLess(np.linalg.norm(A - Ar), 1e-12 * np.linalg.norm(Ar))
        self.assertLess(np.linalg.norm(Q - Qr), 1e-12 * np.linalg.norm(Qr))

        # kernel should be static
        (A, Q) = kernel.get_kalman_matrices(Ts)

        self.assertLess(np.linalg.norm(A - Ar), 1e-12 * np.linalg.norm(Ar))
        self.assertLess(np.linalg.norm(Q - Qr), 1e-12 * np.linalg.norm(Qr))

        # other time step size
        Ts = 0.3
        (A, Q) = kernel.get_kalman_matrices(Ts)

        Ar = scipy.linalg.expm(Ts * np.array([[0, 1.0], [0, 0]]))
        Qr = outputscale * np.array([[Ts**3 / 3, Ts**2 / 2], [Ts**2 / 2, Ts]])

        self.assertLess(np.linalg.norm(A - Ar), 1e-12 * np.linalg.norm(Ar))
        self.assertLess(np.linalg.norm(Q - Qr), 1e-12 * np.linalg.norm(Qr))

        # change outputscale
        outputscale = 5
        kernel.outputscale = outputscale

        self.assertEqual(kernel.outputscale, outputscale)

        (A, Q) = kernel.get_kalman_matrices(Ts)

        Ar = scipy.linalg.expm(Ts * np.array([[0, 1.0], [0, 0]]))
        Qr = outputscale * np.array([[Ts**3 / 3, Ts**2 / 2], [Ts**2 / 2, Ts]])

        self.assertLess(np.linalg.norm(A - Ar), 1e-12 * np.linalg.norm(Ar))
        self.assertLess(np.linalg.norm(Q - Qr), 1e-12 * np.linalg.norm(Qr))

        # create kernel with specified outputscale
        outputscale = 2
        kernel = WienerTemporalKernel(outputscale)
        self.assertEqual(kernel.outputscale, outputscale)

        (A, Q) = kernel.get_kalman_matrices(Ts)

        Ar = scipy.linalg.expm(Ts * np.array([[0, 1.0], [0, 0]]))
        Qr = outputscale * np.array([[Ts**3 / 3, Ts**2 / 2], [Ts**2 / 2, Ts]])

        self.assertLess(np.linalg.norm(A - Ar), 1e-12 * np.linalg.norm(Ar))
        self.assertLess(np.linalg.norm(Q - Qr), 1e-12 * np.linalg.norm(Qr))


if __name__ == "__main__":
    unittest.main()
