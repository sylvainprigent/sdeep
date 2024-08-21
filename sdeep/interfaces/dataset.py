"""Define a dataset as a pytorch dataset plus metadata"""
from torch.utils.data import Dataset

from .transform import STransform


class SDataset:
    """Interface for a dataset compatible with the framework

    The dataset contains the PyTorch dataset and extra metadata for the
    framework to manage the dataset

    :param dataset: PyTorch dataset
    :param args: Arguments to init the PyTorch dataset
    """
    def __init__(self, dataset: Dataset, args: dict[str, any], transform: STransform):
        self.__dataset = dataset
        self.__args = args
        self.__transform = transform

    @property
    def dataset(self) -> Dataset:
        """Get the pytorch dataset"""
        return self.__dataset

    @property
    def transform(self) -> STransform:
        """Get the pytorch dataset"""
        return self.__transform

    @property
    def args(self) -> dict[str, any]:
        """Get the pytorch dataset"""
        args = self.__args
        args["name"] = self.__dataset.__class__.__name__
        return args
