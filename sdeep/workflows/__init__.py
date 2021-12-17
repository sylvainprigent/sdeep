"""Deep learning training and testing scheme

Implementation of deep learning training and testing workflow

"""
from .base import SWorkflow
from .restoration import RestorationWorkflow

__all__ = ['SWorkflow',
           'RestorationWorkflow']
