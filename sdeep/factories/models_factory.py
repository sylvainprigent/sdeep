"""SDeep Models factory.

This module implements factory to instantiate all the available
models of SDeep

Classes
-------
SDeepServiceProvider

"""
from sdeep.models import DnCNN
from sdeep.models import UNet
from sdeep.models import DRUNet
from sdeep.models import RCAN
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


class DRUNetBuilder(SDeepModuleBuilder):
    """Service builder for the DnCNN model"""
    def __init__(self):
        super().__init__()
        self.parameters = [{
                                'key': 'drunet_in_nc',
                                'default': 1,
                                'value': 1,
                                'help': 'Number of input channels (or features)'
                           },
                           {
                                'key': 'drunet_out_nc',
                                'default': 1,
                                'value': 1,
                                'help': 'Number of output channels (or features)'
                           },
                           {
                               'key': 'drunet_nb',
                               'default': 4,
                               'value': 4,
                               'help': 'Number of residual block per level'
                           },
                           {
                                'key': 'drunet_nc',
                                'default': 64,
                                'value': 64,
                                'help': 'Number of channels in the first layer'
                           }
                           ]

    def get_instance(self, args):
        if not self._instance:
            in_nc = get_arg_int(args, 'drunet_in_nc', 1)
            self.parameters[0]['value'] = in_nc
            out_nc = get_arg_int(args, 'drunet_out_nc', 1)
            self.parameters[1]['value'] = out_nc
            nb = get_arg_int(args, 'drunet_nb', 4)
            self.parameters[2]['value'] = nb
            nc = get_arg_int(args, 'drunet_nc', 64)
            self.parameters[3]['value'] = nc
            self._instance = DRUNet(in_nc=in_nc,
                                    out_nc=out_nc,
                                    nc=[nc, 2*nc, 4*nc, 8*nc],
                                    nb=nb)
        return self._instance

    def get_parameters(self):
        return self.parameters


class RCANBuilder(SDeepModuleBuilder):
    """Service builder for the RCAN model"""
    def __init__(self):
        super().__init__()
        self.parameters = [
                            {
                                'key': 'rcan_n_resgroups',
                                'default': 10,
                                'value': 10,
                                'help': 'Number of residual groups'
                            },
                            {
                                'key': 'rcan_n_resblocks',
                                'default': 20,
                                'value': 20,
                                'help': 'Number of residual blocs in each residual group'
                            },
                            {
                                'key': 'rcan_n_feats',
                                'default': 64,
                                'value': 64,
                                'help': 'Number of features'
                            },
                            {
                                'key': 'rcan_reduction',
                                'default': 16,
                                'value': 16,
                                'help': 'Reduction factor for channels downscaling'
                            },
                            {
                                'key': 'rcan_scale',
                                'default': 1,
                                'value': 1,
                                'help': 'Scale factor between the input and output image'
                            },
                           ]

    def get_instance(self, args):
        if not self._instance:
            n_resgroups = get_arg_int(args, 'rcan_n_resgroups', 10)
            self.parameters[0]['value'] = n_resgroups
            n_resblocks = get_arg_int(args, 'rcan_n_resblocks', 20)
            self.parameters[1]['value'] = n_resblocks
            n_feats = get_arg_int(args, 'rcan_n_feats', 64)
            self.parameters[2]['value'] = n_feats
            reduction = get_arg_int(args, 'rcan_reduction', 16)
            self.parameters[3]['value'] = reduction
            scale = get_arg_int(args, 'rcan_scale', 16)
            self.parameters[3]['value'] = scale
            self._instance = RCAN(n_colors=1, n_resgroups=n_resgroups,
                                  n_resblocks=n_resblocks, n_feats=n_feats,
                                  reduction=reduction, scale=scale)
        return self._instance

    def get_parameters(self):
        return self.parameters


sdeepModels = SDeepModulesFactory()
sdeepModels.register_builder('DnCNN', DnCNNBuilder())
sdeepModels.register_builder('UNet', UnetBuilder())
sdeepModels.register_builder('DRUNet', DRUNetBuilder())
sdeepModels.register_builder('RCAN', RCANBuilder())
