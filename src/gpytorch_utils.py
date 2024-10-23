"""

Some convenience functions to create pytorch tensors or gpytorch constraints from
(iterables of) floats.
"""

import math
from typing import Iterable

import gpytorch.constraints
import torch

ScalarConstraintType = tuple[float, float]
VectorConstraintType = Iterable[tuple[float, float]]


def get_scalar_gpytorch_constraint(
    value: ScalarConstraintType,
) -> gpytorch.constraints.Interval:
    infinit_lower_bound = value[0] < 0 and math.isinf(value[0])
    infinit_upper_bound = value[1] > 0 and math.isinf(value[1])

    if infinit_lower_bound and infinit_upper_bound:
        return gpytorch.constraints.Interval(-math.inf, math.inf)
    elif infinit_upper_bound:
        if value[0] == 0.0:
            return gpytorch.constraints.Positive()
        else:
            return gpytorch.constraints.GreaterThan(value[0])
    elif infinit_lower_bound:
        return gpytorch.constraints.LessThan(value[1])
    else:
        return gpytorch.constraints.Interval(value[0], value[1])


def get_tensor_from_iterable(values: Iterable[float]) -> torch.Tensor:
    n = len(values)

    t = torch.zeros((n,))

    for i, v in enumerate(values):
        t[i] = float(v)

    return t


def get_tensor(values: torch.Tensor | Iterable[float] | float) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    elif isinstance(values, Iterable):
        return get_tensor_from_iterable(values)
    else:
        return get_tensor_from_iterable([values])


def get_vector_gpytorch_constraint(
    values: VectorConstraintType,
) -> gpytorch.constraints.Interval:
    lower_bounds = [v[0] for v in values]
    upper_bounds = [v[1] for v in values]

    infinit_lower_bounds = all(lb < 0 and math.isinf(lb) for lb in lower_bounds)
    infinit_upper_bounds = all(ub > 0 and math.isinf(ub) for ub in upper_bounds)

    if infinit_lower_bounds and infinit_upper_bounds:
        return gpytorch.constraints.Interval(-math.inf, math.inf)
    elif infinit_upper_bounds:
        if all(lb == 0.0 for lb in lower_bounds):
            return gpytorch.constraints.Positive()
        else:
            return gpytorch.constraints.GreaterThan(
                get_tensor_from_iterable(lower_bounds)
            )
    elif infinit_lower_bounds:
        return gpytorch.constraints.LessThan(get_tensor_from_iterable(upper_bounds))
    else:
        return gpytorch.constraints.Interval(
            get_tensor_from_iterable(lower_bounds),
            get_tensor_from_iterable(upper_bounds),
        )
