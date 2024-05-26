from enum import Enum


class SchedulerType(str, Enum):
    GREEDY = "greedy"
    POLYNOMIAL = "polynomial"
    PLATEAU = "plateau"
    ONE_CYCLE = "one-cycle"
    FIXED = "fixed"


class OptimizerType(str, Enum):
    ADAMW = "adam"
    SGD = "sgd"


class DatasetType(str, Enum):
    HONMA = "honma"
    HANSEN = "hansen"
    COMBINED = "combined"


class LossReductionType(str, Enum):
    SUM = "sum"
    MEAN = "mean"


class NormType(str, Enum):
    NONE = "none"
    LAYER = "layer"
    RMS = "rms"
    CRMS = "crms"
    MAX = "max"


class AttentionType(str, Enum):
    MHA = "mha"
    FISH = "fish"
    LINEAR = "linear"


class ResidualType(str, Enum):
    PRENORM = "prenorm"
    REZERO = "rezero"
