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
from sdeep.losses import SAContrarioMSELoss, VGGL1PerceptualLoss, FRCLoss, FMSELoss, FRMSELoss


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


class FRCLossBuilder(SDeepModuleBuilder):
    """Service builder for the FRCLoss loss"""
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'frc_len',
                            'default': 20.0,
                            'value': 20.0,
                            'help': 'Diameter of the largest FRC ring'
                            }
                           ]

    def get_instance(self, args):
        if not self._instance:
            patch_size = get_arg_float(args, 'frc_len', 40.0)
            self.parameters[0]['value'] = patch_size
            self._instance = FRCLoss(patch_size=patch_size)
            return self._instance

    def get_parameters(self):
        return self.parameters


class FMSELossBuilder(SDeepModuleBuilder):
    """Service builder for the FRCLoss loss"""
    def __init__(self):
        super().__init__()
        self.parameters = []

    def get_instance(self, args):
        if not self._instance:
            self._instance = FMSELoss()
            return self._instance

    def get_parameters(self):
        return self.parameters


class FRMSELossBuilder(SDeepModuleBuilder):
    """Service builder for the FRMSELoss loss"""
    def __init__(self):
        super().__init__()
        self.parameters = []

    def get_instance(self, args):
        if not self._instance:
            self._instance = FRMSELoss()
            return self._instance

    def get_parameters(self):
        return self.parameters


sdeepLosses = SDeepModulesFactory()
sdeepLosses.register_builder('MSELoss', MSELossBuilder())
sdeepLosses.register_builder('MAELoss', MAELossBuilder())
sdeepLosses.register_builder('WACMSELoss', SAContrarioMSELossBuilder())
sdeepLosses.register_builder('VGGL1', VGGL1PerceptualLossBuilder())
sdeepLosses.register_builder('FRCLoss', FRCLossBuilder())
sdeepLosses.register_builder('FMSELoss', FMSELossBuilder())
sdeepLosses.register_builder('FRMSELoss', FRMSELossBuilder())
