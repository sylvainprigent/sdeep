"""Factories module

Factories to instantiate all the available models, losses, optimizers, datasets and workflows

"""

from sdeep.factories.models_factory import sdeepModels
from sdeep.factories.losses_factory import sdeepLosses
from sdeep.factories.optimizers_factory import sdeepOptimizers
from sdeep.factories.datasets_factory import sdeepDatasets
from sdeep.factories.workflows_factory import sdeepWorkflows


__all__ = ['sdeepModels',
           'sdeepLosses',
           'sdeepDatasets',
           'sdeepOptimizers',
           'sdeepWorkflows'
           ]
