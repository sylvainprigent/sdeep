"""Module to implement methods for evaluating models during validation steps or after training"""

from .classification import EvalClassification
from .restoration import EvalRestoration

__all__ = [
    "EvalClassification",
    "EvalRestoration"
]
