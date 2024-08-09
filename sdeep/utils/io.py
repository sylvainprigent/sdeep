"""IO module to save models"""
from pathlib import Path
import torch


def save_model(model: torch.nn.Module, params: dict[str, any], filename: Path):
    """Save a model and associated model info to a file

    :param model: Model to save,
    :param params: Parameters to instantiate the model
    :param filename: Path of the destination file
    """
    torch.save({
        'model': model.__class__.__name__,
        'model_args': params,
        'model_state_dict': model.state_dict()
    }, filename)
