from enum import Enum
from typing import Literal, Optional, Union

from ..operating_point import Op


class _StrategyType(Enum):
    Mean = 0
    Median = 1
    Manual = 2


class RefStrategy:
    _type: _StrategyType
    _value: Optional[Op]

    def __init__(self, strategy: Union[Literal["mean", "median"] | Op]):
        if isinstance(strategy, str):
            if strategy == "mean":
                self._type = _StrategyType.Mean
                self._value = None
            elif strategy == "median":
                self._type = _StrategyType.Median
                self._value = None
            else:
                raise ValueError("Invalid value for 'strategy'")

        elif isinstance(strategy, Op):
            self._type = _StrategyType.Manual
            self._value = strategy

        else:
            raise ValueError("Invalid value for 'strategy'")

    def is_mean(self) -> bool:
        return self._type == _StrategyType.Mean

    def is_median(self) -> bool:
        return self._type == _StrategyType.Median

    def is_manual(self) -> bool:
        return self._type == _StrategyType.Median

    def get_manual_value(self) -> Op:
        if self._type != _StrategyType.Manual:
            raise ValueError("manual value can only be given if strategy is 'manual'")

        return self._value
