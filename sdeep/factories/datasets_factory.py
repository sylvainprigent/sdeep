"""Factory for datasets

This module implements factory to instantiate all the available
datasets of SDeep

Classes
-------
MSELossBuilder


Objects
-------
sdeepLosses
"""
from sdeep.datasets import RestorationPatchDataset, RestorationDataset, RestorationPatchDataset2
from sdeep.factories.utils import (SDeepDatasetsFactory, SDeepDatasetBuilder,
                                   get_arg_str, get_arg_int)


class RestorationDatasetBuilder(SDeepDatasetBuilder):
    """Service builder for the RestorationDataset
    """
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'rd_path_source',
                            'default': '',
                            'value': '',
                            'help': 'Path of the source files'},
                           {'key': 'rd_path_target',
                            'default': '',
                            'value': '',
                            'help': 'Path of the taget files'}
                           ]

    def get_instance(self, args):
        if not self._instance:
            train_source_dir = get_arg_str(args, 'rd_path_source', '')
            self.parameters[0]['value'] = train_source_dir
            train_target_dir = get_arg_str(args, 'rd_path_target', '')
            self.parameters[1]['value'] = train_target_dir
            self._instance = RestorationDataset(train_source_dir,
                                                train_target_dir,
                                                use_data_augmentation=True)
        return self._instance

    def get_parameters(self):
        return self.parameters


class RestorationPatchDatasetBuilder(SDeepDatasetBuilder):
    """Service builder for the RestorationPatchDataset
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'rpd_path_source',
                            'default': '',
                            'value': '',
                            'help': 'Path of the source files'},
                           {'key': 'rpd_path_target',
                            'default': '',
                            'value': '',
                            'help': 'Path of the taget files'},
                           {'key': 'rpd_patch_size',
                            'default': 40,
                            'value': 40,
                            'help': 'Size of a training patch'},
                           {'key': 'rpd_stride',
                            'default': 10,
                            'value': 10,
                            'help': 'Stride between two patches'}
                           ]

    def get_instance(self, args):
        if not self._instance:
            train_source_dir = get_arg_str(args, 'rpd_path_source', '')
            self.parameters[0]['value'] = train_source_dir
            train_target_dir = get_arg_str(args, 'rpd_path_target', '')
            self.parameters[1]['value'] = train_target_dir
            patch_size = get_arg_int(args, 'rpd_patch_size', 40)
            self.parameters[2]['value'] = patch_size
            stride = get_arg_int(args, 'rpd_stride', 10)
            self.parameters[3]['value'] = stride
            self._instance = RestorationPatchDataset(
                                    train_source_dir,
                                    train_target_dir,
                                    patch_size=patch_size,
                                    stride=stride,
                                    use_data_augmentation=True)
        return self._instance

    def get_parameters(self):
        return self.parameters


class RestorationPatchDataset2Builder(SDeepDatasetBuilder):
    """Service builder for the RestorationPatchDataset2
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'rpd2_path_source',
                            'default': '',
                            'value': '',
                            'help': 'Path of the source files'},
                           {'key': 'rpd2_path_target',
                            'default': '',
                            'value': '',
                            'help': 'Path of the target files'},
                           {'key': 'rpd2_patch_size',
                            'default': 40,
                            'value': 40,
                            'help': 'Size of a training patch'},
                           {'key': 'rpd2_stride',
                            'default': 10,
                            'value': 10,
                            'help': 'Stride between two patches'}
                           ]

    def get_instance(self, args):
        if not self._instance:
            train_source_dir = get_arg_str(args, 'rpd2_path_source', '')
            self.parameters[0]['value'] = train_source_dir
            train_target_dir = get_arg_str(args, 'rpd2_path_target', '')
            self.parameters[1]['value'] = train_target_dir
            patch_size = get_arg_int(args, 'rpd2_patch_size', 40)
            self.parameters[2]['value'] = patch_size
            stride = get_arg_int(args, 'rpd2_stride', 10)
            self.parameters[3]['value'] = stride
            self._instance = RestorationPatchDataset2(
                                    train_source_dir,
                                    train_target_dir,
                                    patch_size=patch_size,
                                    stride=stride,
                                    use_data_augmentation=True)
        return self._instance

    def get_parameters(self):
        return self.parameters


sdeepDatasets = SDeepDatasetsFactory()
sdeepDatasets.register_builder('RestorationPatchDataset2', RestorationPatchDataset2Builder())
sdeepDatasets.register_builder('RestorationPatchDataset', RestorationPatchDatasetBuilder())
sdeepDatasets.register_builder('RestorationDataset', RestorationDatasetBuilder())
