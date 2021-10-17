"""SDeep Models factory.

This module implements factory to instantiate all the available
models of SDeep

Classes
-------
SDeepServiceProvider

"""
from sdeep.models import DnCNN
from .utils import get_arg_int, SDeepModulesFactory, SDeepModuleBuilder


class DnCNNBuilder(SDeepModuleBuilder):
    """Service builder for the DnCNN model"""
    def get_instance(self, args):
        if not self._instance:
            num_of_layers = get_arg_int(args, 'dncnn_layers', 17)
            channels = get_arg_int(args, 'dncnn_channels', 1)
            features = get_arg_int(args, 'dncnn_features', 64)
            self._instance = DnCNN(num_of_layers=num_of_layers,
                                   channels=channels,
                                   features=features)
        return self._instance

    def get_parameters(self):
        return [{'key': 'dncnn_layers',
                 'default': 17,
                 'help': 'Number of convolutional layers'},
                 {'key': 'dncnn_channels',
                 'default': 1,
                 'help': 'Number of input channels'},
                 {'key': 'dncnn_features',
                 'default': 64,
                 'help': 'Number of features per convolutional layers'}
                 ]


sdeepModels = SDeepModulesFactory()
sdeepModels.register_builder('DnCNN', DnCNNBuilder())
