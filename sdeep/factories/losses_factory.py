"""Factory for Losses

This module implements factory to instantiate all the available
losses of SDeep

Classes
-------
MSELossBuilder


Objects
-------
sdeepLosses

"""
import torch
from sdeep.factories.utils import SDeepModulesFactory, SDeepModuleBuilder


class MSELossBuilder(SDeepModuleBuilder):
    """Service builder for the MSE loss"""
    def get_instance(self, args):
        if not self._instance:
            self._instance = torch.nn.MSELoss()
        return self._instance

    def get_parameters(self):
        return {}


class MAELossBuilder(SDeepModuleBuilder):
    """Service builder for the MAE (mean absolute error) loss"""
    def get_instance(self, args):
        if not self._instance:
            self._instance = torch.nn.L1Loss()
        return self._instance

    def get_parameters(self):
        return {}


sdeepLosses = SDeepModulesFactory()
sdeepLosses.register_builder('MSELoss', MSELossBuilder())
sdeepLosses.register_builder('MAELoss', MAELossBuilder())
