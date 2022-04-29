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
from sdeep.factories.utils import get_arg_float, SDeepModulesFactory, SDeepModuleBuilder
from sdeep.losses import SAContrarioMSELoss, VGGL1PerceptualLoss


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


class SAContrarioMSELossBuilder(SDeepModuleBuilder):
    """Service builder for the MAE (mean absolute error) loss"""
    def get_instance(self, args):
        if not self._instance:
            self._instance = SAContrarioMSELoss()
        return self._instance

    def get_parameters(self):
        return {}


class VGGL1PerceptualLossBuilder(SDeepModuleBuilder):
    """Service builder for the VGGL1Perceptual loss"""
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'vggl1_weight',
                            'default': 1.0,
                            'value': 1.0,
                            'help': 'weight to the perceptual part vs L1 part'
                            }
                           ]

    def get_instance(self, args):
        if not self._instance:
            weight = get_arg_float(args, 'vggl1_weight', 1.0)
            self.parameters[0]['value'] = weight
            self._instance = VGGL1PerceptualLoss(weight=weight)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._instance.to(device)
        return self._instance

    def get_parameters(self):
        return self.parameters


sdeepLosses = SDeepModulesFactory()
sdeepLosses.register_builder('MSELoss', MSELossBuilder())
sdeepLosses.register_builder('MAELoss', MAELossBuilder())
sdeepLosses.register_builder('WACMSELoss', SAContrarioMSELossBuilder())
sdeepLosses.register_builder('VGGL1', VGGL1PerceptualLossBuilder())