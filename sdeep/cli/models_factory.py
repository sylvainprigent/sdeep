"""SDeep Models factory.

This module implements factory to instantiate all the available
models of SDeep

Classes
-------
SDeepServiceProvider

"""
from sdeep.models import DnCNN
from .utils import SDeepModulesFactory, SDeepModuleBuilder


class DnCNNBuilder(SDeepModuleBuilder):
    """Service builder for the DnCNN model"""
    def get_instance(self, args):
        if not self._instance:
            num_of_layers = self.get_arg_int(args, 'dncnn_layers', 17)
            channels = self.get_arg_int(args, 'dncnn_channels', 1)
            features = self.get_arg_int(args, 'dncnn_features', 64)
            self._instance = DnCNN(num_of_layers=num_of_layers,
                                   channels=channels,
                                   features=features)
        return self._instance

    def get_parameters(self):
        return {'dncnn_layers': 17,
                'dncnn_channels': 1,
                'dncnn_features': 64
                }


sdeepModels = SDeepModulesFactory()
sdeepModels.register_builder('DnCNN', DnCNNBuilder())
