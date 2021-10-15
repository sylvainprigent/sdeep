"""CLI module

Set of tools to use SDeep with a command line interface

"""

from .models_factory import sdeepModels
from .losses_factory import sdeepLosses

__all__ = ['sdeepModels',
           'sdeepLosses'
           ]
