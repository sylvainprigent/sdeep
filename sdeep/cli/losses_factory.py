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
from .utils import SDeepModulesFactory, SDeepModuleBuilder


class MSELossBuilder(SDeepModuleBuilder):
    """Service builder for the MSE loss"""
    def __init__(self):
        super().__init__()

    def get_instance(self, args):
        if not self._instance:
            self._instance = torch.nn.MSELoss()
        return self._instance

    def get_parameters(self):
        return {}


sdeepLosses = SDeepModulesFactory()
sdeepLosses.register_builder('MSELoss', MSELossBuilder())
