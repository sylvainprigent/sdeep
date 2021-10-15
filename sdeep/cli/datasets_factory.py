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
from sdeep.datasets import RestorationPatchDataset, RestorationDataset
from .utils import SDeepModulesFactory, SDeepModuleBuilder


class RestorationPatchDatasetBuilder(SDeepModuleBuilder):
    """Service builder for the MSE loss"""
    def __init__(self):
        super().__init__()

    def get_instance(self, data_path, args):
        if not self._instance:
            train_source_dir = os.path.join(data_path, 'source')
            train_target_dir = os.path.join(data_path, 'target')
            patch_size = self.get_arg_str(args, 'rpd_patch_size', 40)
            stride = self.get_arg_str(args, 'rpd_stride', 10)
            RestorationPatchDataset(train_source_dir,
                                    train_target_dir,
                                    patch_size=patch_size,
                                    stride=stride,
                                    use_data_augmentation=True)
            self._instance = torch.nn.MSELoss()
        return self._instance

    def get_parameters(self):
        return {}


sdeepLosses = SDeepModulesFactory()
sdeepLosses.register_builder('MSELoss', MSELossBuilder())