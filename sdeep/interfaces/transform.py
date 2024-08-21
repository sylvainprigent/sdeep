"""Define a transform Callable for dataset data transformation"""
from typing import Callable
import torch


class STransform:
    """Interface for a data transform compatible with the framework

    The dataset contains the PyTorch dataset and extra metadata for the
    framework to manage the dataset

    :param transform: Callable to transform the data
    :param args: Arguments to init the PyTorch dataset
    """
    def __init__(self, transform: Callable, args: dict[str, any]):
        self.__transform = transform
        self.__args = args

    @property
    def transform(self) -> Callable:
        """Get the transform callable"""
        return self.__transform

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the transform

        :param data: Data to transform,
        :return: The transformed data
        """
        return self.__transform(data)

    @property
    def args(self) -> dict[str, any]:
        """Get the pytorch dataset"""
        return self.__args
