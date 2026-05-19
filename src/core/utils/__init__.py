

from enum import Enum

__all__ = ["UnknownModuleError", "TaskType", "LearningType", "Device"]

class UnknownModuleError(Exception):
    """Exception raised for unknown module."""
    pass
class TaskType(Enum):
    """Task type for the pipeline."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    RECONSTRUCTION = "reconstruction"
    PRETRAIN = "pretrain"
class LearningType(str, Enum):
    """Learning type defining how the model learns from the dataset samples."""

    ONE_CLASS = "one_class"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    FULLY_SUPERVISED = "fully_supervised"
class DictCFG(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)
class Device(str, Enum):
    """Device on which to compute metrics."""

    CPU = "cpu"
    GPU = "cuda"