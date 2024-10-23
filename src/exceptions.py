class BattGPBaseException(Exception):
    """Base exception"""


class InsufficientDataError(BattGPBaseException):
    """Not enough data"""


class InvalidConfigError(BattGPBaseException):
    """Invalid config"""
