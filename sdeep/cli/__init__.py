"""CLI module

Set of tools to use SDeep with a command line interface

"""

from .models_factory import sdeepModels
from .losses_factory import sdeepLosses
from .optimizers_factory import sdeepOptimizers
from .datasets_factory import sdeepDatasets
from .workflows_factory import sdeepWorkflows


__all__ = ['sdeepModels',
           'sdeepLosses',
           'sdeepDatasets',
           'sdeepOptimizers',
           'sdeepWorkflows'
           ]
