from typing import Any, Callable, Optional, Protocol

import numpy as np
import pandas as pd

from ..operating_point import Op


class IBatteryCellGP(Protocol):
    """Interface for battery cell models.

    As the purpose of this repository is to research and evaluate different concepts,
    this interface is designed to be agnostic to the underlying model.
    Therefore only Python base datatypes and numpy arrays are used within this
    interface. This will cost some performance due to unnecessary conversions, but this
    should not be an issue for the purposes of this project.

    On the other side, the data structure is fixed:
    - There are four input signals, that are given in the following order:
        1. time in days
        2. battery current in A
        3. SOC in percent
        4. temperature in °C

    - There is one target variable
        1. internal resistance R in Ohm

    The input data is always given as a [N x 4] numpy array (even if there is only one
    dataset).
    The target data is always given as a [N] numpy vector.

    Also, every class that implements this interface should store the data used to build
    the model.
    """

    def __init__(self, x_train, y_train, **kwargs):
        """Initialize model with data."""
        ...

    @property
    def cellnr(self) -> int:
        """Cellnr property"""
        ...

    @staticmethod
    def get_default_parameters() -> dict[str, Any]:
        """Returns a dictionary with the default parameters."""
        ...

    def get_parameters(self) -> dict[str, Any]:
        """Returns a dictionary with the set parameters."""
        ...

    def train_hyperparameters(self, messages: bool = True) -> Optional[Any]:
        """Train the hyperparameters of the underlying model.

        The return value of this method, if there is any, should give some information
        about the training. It should only be used for the development of the concrete
        model. General code should not rely on the return value.

        Parameters of the training process should be parameters of the overall model
        and therefore provided with the constructor.
        """
        ...

    def predict_r0_op(self, op: Op, *args) -> pd.DataFrame:
        """Returns prediction of target for given input values

        Returns a tuple with the predicted values as well as the variances of these
        values (full_cov = False) or the full covariance matrix of the predictions
        (full_cov = True).
        If the underlying model doesn't provide any sensible measure for the
        uncertainty, a vector or matrix with nan-values is returned.

        If no_cov = True, than only the predicted values are returned directly (i.e. not
        packed into a tuple) as ndarray.
        """
        ...

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the training data as a tuple of two numpy arrays."""
        ...

    def save_hyperparameters(self, path: str):
        """Save hyperparameters to file"""
        ...


BatteryCellGPFactory = Callable[[np.ndarray, np.ndarray], IBatteryCellGP]
