"""Factory for Optimizers

This module implements factory to instantiate all the available
optimizers of SDeep

Classes
-------
AdamBuilder


Objects
-------
sdeepLosses

"""
import torch
from .utils import get_arg_int, SDeepOptimizersFactory, SDeepOptimizerBuilder


class AdamBuilder(SDeepOptimizerBuilder):
    """Service builder for the adam optimizer"""
    def get_instance(self, model, args):
        if not self._instance:
            l_r = get_arg_int(args, 'lr', 0.001)
            self._instance = torch.optim.Adam(model.parameters(), lr=l_r)
        return self._instance

    def get_parameters(self):
        return [{'key': 'lr',
                 'default': 0.001,
                 'help': 'Learning rate'}
        ]


sdeepOptimizers = SDeepOptimizersFactory()
sdeepOptimizers.register_builder('Adam', AdamBuilder())
