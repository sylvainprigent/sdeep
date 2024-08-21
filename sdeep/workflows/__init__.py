"""Deep learning training and testing scheme

Implementation of deep learning training and testing workflow

"""
from .base import SWorkflowBase
from .classification import ClassificationWorkflow
from .restoration import RestorationWorkflow
from .segmentation import SegmentationWorkflow
from .self_supervised import SelfSupervisedWorkflow


__all__ = [
    'SWorkflowBase',
    'ClassificationWorkflow',
    'RestorationWorkflow',
    'SegmentationWorkflow',
    'SelfSupervisedWorkflow'
]
