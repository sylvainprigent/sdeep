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
import os
from sdeep.datasets import RestorationPatchDataset, RestorationDataset
from .utils import SDeepDatasetsFactory, SDeepDatasetBuilder, get_arg_str

class RestorationDatasetBuilder(SDeepDatasetBuilder):
    """Service builder for the RestorationDataset
    """
    def get_instance(self, dataset_path, args):
        if not self._instance:
            train_source_dir = os.path.join(dataset_path, 'source')
            train_target_dir = os.path.join(dataset_path, 'target')
            self._instance = RestorationDataset(train_source_dir,
                                                train_target_dir,
                                                use_data_augmentation=True)
        return self._instance

    def get_parameters(self):
        return {}

class RestorationPatchDatasetBuilder(SDeepDatasetBuilder):
    """Service builder for the RestorationPatchDataset
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def get_instance(self, dataset_path, args):
        if not self._instance:
            train_source_dir = os.path.join(dataset_path, 'source')
            train_target_dir = os.path.join(dataset_path, 'target')
            patch_size = get_arg_str(args, 'rpd_patch_size', 40)
            stride = get_arg_str(args, 'rpd_stride', 10)
            self._instance = RestorationPatchDataset(
                                    train_source_dir,
                                    train_target_dir,
                                    patch_size=patch_size,
                                    stride=stride,
                                    use_data_augmentation=True)
        return self._instance

    def get_parameters(self):
        return {}


sdeepDatasets = SDeepDatasetsFactory()
sdeepDatasets.register_builder('RestorationPatchDataset', RestorationPatchDatasetBuilder())
sdeepDatasets.register_builder('RestorationDataset', RestorationDatasetBuilder())
