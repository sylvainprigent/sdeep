"""Interfaces for SDeep modules"""

from .dataset import SDataset
from .eval import SEval
from .model import SModel
from .transform import STransform
from .workflow import SWorkflow


__all__ = [
    "SDataset",
    "SEval",
    "SModel",
    "STransform",
    "SWorkflow"
]
