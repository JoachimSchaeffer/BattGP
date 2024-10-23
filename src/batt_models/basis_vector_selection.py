from typing import Literal, Tuple

import numpy as np
from sklearn.cluster import KMeans

from ..batt_data.batt_data import BattData

BasisVectorStrategyStr = Literal["kmeans", "uniform", "manual"]


def get_basis_vectors(
    strategy: BasisVectorStrategyStr,
    nbasis: Tuple[int, int, int],
    batt_data: BattData,
    max_training_data: int,
    age: int,
    ref_op: np.ndarray,
    lengthscale_rbf: Tuple[float, float, float],
    **kwargs,
):
    # Range divided by the lengthscale. Forget about points too far away.
    # nbasis points must be uneven, to put the center point at the mean where we
    # are evaluating the GP.
    #  nbasis = np.array([7, 7, 7])
    if strategy == "kmeans":
        # Create a grid of uniformly spaced basis vectors
        (x, y) = batt_data.generateTrainingData(-1, max_training_data, age)
        n_clusters = np.prod(nbasis)
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(
            x[:, 1:]
        )
        basis_vectors = kmeans.cluster_centers_
    elif strategy == "uniform":
        # "Linearly spaced basis vectors"
        # Issues: Assymetric, the op/mean is not part of the basis vectors!

        # spacing_multiplier = np.array([0.6, 0.6, 0.5])
        spacing_multiplier = np.array([0.5, 0.5, 0.5])
        # Center the basis vectors around the mean

        first_dim_points = np.linspace(
            ref_op[0] - spacing_multiplier[0] * lengthscale_rbf[0],
            np.min(
                [
                    ref_op[0] + spacing_multiplier[0] * lengthscale_rbf[0],
                    batt_data.segment_criteria.ibat_upper_limit,
                ]
            ),
            nbasis[0],
        )
        second_dim_points = np.linspace(
            np.max(
                [
                    ref_op[1] - spacing_multiplier[1] * lengthscale_rbf[1],
                    batt_data.segment_criteria.soc_lower_limit,
                ]
            ),
            np.min(
                [
                    ref_op[1] + spacing_multiplier[1] * lengthscale_rbf[1],
                    batt_data.segment_criteria.soc_upper_limit,
                ]
            ),
            nbasis[1],
        )

        third_dim_points = np.linspace(
            np.max(
                [
                    ref_op[2] - spacing_multiplier[2] * lengthscale_rbf[2],
                    batt_data.segment_criteria.t_lower_limit,
                ]
            ),
            np.min(
                [
                    ref_op[2] + spacing_multiplier[2] * lengthscale_rbf[2],
                    batt_data.segment_criteria.t_upper_limit,
                ]
            ),
            nbasis[2],
        )
        basis_vectors = np.zeros((nbasis[0] * nbasis[1] * nbasis[2], 3))
        for i, p1 in enumerate(first_dim_points):
            for j, p2 in enumerate(second_dim_points):
                for k, p3 in enumerate(third_dim_points):
                    basis_vectors[i * nbasis[1] * nbasis[2] + j * nbasis[2] + k] = [
                        p1,
                        p2,
                        p3,
                    ]
        # print(basis_vectors)

    elif strategy == "manual":
        if "basis_points" not in kwargs:
            ValueError(
                "basis_points must be passed to the model, if strategy is manual"
            )

        first_dim_points = [-120, -115, -110, -125, -130]
        second_dim_points = [82, 88, 94, 74, 68]
        third_dim_points = [25, 30, 35, 20, 15]
        # Online setting, we'd want a common refercen point!
        nbasis = np.array(
            [len(first_dim_points), len(second_dim_points), len(third_dim_points)]
        )
        basis_vectors = np.zeros((nbasis[0] * nbasis[1] * nbasis[2], 3))
        for i in range(nbasis[0]):
            for j in range(nbasis[1]):
                for k in range(nbasis[2]):
                    basis_vectors[i * nbasis[1] * nbasis[2] + j * nbasis[2] + k] = [
                        first_dim_points[i],
                        second_dim_points[j],
                        third_dim_points[k],
                    ]
    else:
        ValueError(f"Strategy: {strategy} is not implemented")

    # Check whether ref_op is part of the basis vectors
    if not np.any(np.all(basis_vectors == ref_op, axis=1)):
        basis_vectors = np.vstack(
            [
                basis_vectors,
                ref_op,
            ]
        )

    return basis_vectors
