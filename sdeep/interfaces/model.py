"""Define a dataset as a pytorch dataset plus metadata"""
import torch


class SModel:
    """Interface for a model compatible with the framework

    The model contains the PyTorch nn.Module and extra metadata for the
    framework to manage the model

    :param model: PyTorch module
    :param args: Arguments to init the PyTorch dataset
    """
    def __init__(self, model: torch.nn.Module, args: dict[str, any]):
        self.__model = model
        self.__args = args

    @property
    def model(self) -> torch.nn.Module:
        """Get the pytorch dataset"""
        return self.__model

    @property
    def args(self) -> dict[str, any]:
        """Get the pytorch dataset"""
        args = self.__args.copy()
        args["name"] = self.__model.__class__.__name__
        return args
