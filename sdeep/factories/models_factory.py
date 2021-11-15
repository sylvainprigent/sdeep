"""SDeep Models factory.

This module implements factory to instantiate all the available
models of SDeep

Classes
-------
SDeepServiceProvider

"""
from sdeep.models import DnCNN
from sdeep.models import UNet
from sdeep.factories.utils import get_arg_int, get_arg_bool, SDeepModulesFactory, SDeepModuleBuilder


class DnCNNBuilder(SDeepModuleBuilder):
    """Service builder for the DnCNN model"""
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'dncnn_layers',
                 'default': 17,
                 'value': 17,
                 'help': 'Number of convolutional layers'},
                 {'key': 'dncnn_channels',
                 'default': 1,
                 'value': 1,
                 'help': 'Number of input channels'},
                 {'key': 'dncnn_features',
                 'default': 64,
                 'value': 64,
                 'help': 'Number of features per convolutional layers'}
                 ]

    def get_instance(self, args):
        if not self._instance:
            num_of_layers = get_arg_int(args, 'dncnn_layers', 17)
            self.parameters[0]['value'] = num_of_layers
            channels = get_arg_int(args, 'dncnn_channels', 1)
            self.parameters[1]['value'] = channels
            features = get_arg_int(args, 'dncnn_features', 64)
            self.parameters[2]['value'] = features
            self._instance = DnCNN(num_of_layers=num_of_layers,
                                   channels=channels,
                                   features=features)
        return self._instance

    def get_parameters(self):
        return self.parameters


class UnetBuilder(SDeepModuleBuilder):
    """Service builder for the DnCNN model"""
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'unet_n_channels_in',
                            'default': 1,
                            'value': 1,
                            'help': 'Number of input channels (or features)'},
                           {'key': 'unet_n_channels_out',
                            'default': 1,
                            'value': 1,
                            'help': 'Number of output channels (or features)'},
                           {'key': 'unet_n_feature_first',
                            'default': 32,
                            'value': 32,
                            'help': 'Number of channels (or features) in the first convolution block'},
                           {'key': 'unet_use_batch_norm',
                            'default': True,
                            'value': True,
                            'help': 'True to use the batch norm layers'},
                           ]

    def get_instance(self, args):
        if not self._instance:
            n_channels_in = get_arg_int(args, 'unet_n_channels_in', 1)
            self.parameters[0]['value'] = n_channels_in
            n_channels_out = get_arg_int(args, 'unet_n_channels_out', 1)
            self.parameters[1]['value'] = n_channels_out
            n_feature_first = get_arg_int(args, 'unet_n_feature_first', 32)
            self.parameters[2]['value'] = n_feature_first
            use_batch_norm = get_arg_bool(args, 'unet_use_batch_norm', True)
            self.parameters[3]['value'] = use_batch_norm
            self._instance = UNet(n_channels_in=n_channels_in,
                                  n_channels_out=n_channels_out,
                                  n_feature_first=n_feature_first,
                                  use_batch_norm=use_batch_norm)
        return self._instance

    def get_parameters(self):
        return self.parameters


sdeepModels = SDeepModulesFactory()
sdeepModels.register_builder('DnCNN', DnCNNBuilder())
sdeepModels.register_builder('UNet', UnetBuilder())
