from enum import Enum


class SchedulerType(str, Enum):
    GREEDY = "greedy"
    POLYNOMIAL = "polynomial"
    PLATEAU = "plateau"
    FIXED = "fixed"


class OptimizerType(str, Enum):
    ADAMW = "adam"
    SGD = "sgd"


class DatasetType(str, Enum):
    HONMA = "honma"
    HANSEN = "hansen"


class LossReductionType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
