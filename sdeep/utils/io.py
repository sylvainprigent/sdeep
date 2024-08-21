"""IO module to save models"""
from pathlib import Path

import numpy as np

from skimage.io import imread
from skimage.io import imsave

import torch


def save_model(model: torch.nn.Module,
               params: dict[str, any],
               filename: Path,
               transform: dict[str, any] = None):
    """Save a model and associated model info to a file

    :param model: Model to save,
    :param params: Parameters to instantiate the model
    :param filename: Path of the destination file
    :param transform: name and parameters of the transformation of data before model
    """
    if not transform:
        transform = {}

    torch.save({
        'model': model.__class__.__name__,
        'model_args': params,
        'model_state_dict': model.state_dict(),
        'transform': transform
    }, filename)


def read_data(filename: Path) -> torch.Tensor | None:
    """Read data from file

    This method is a factory to read single data from file.

    TODO: This code may be adapted to a real factory when the number of supported formats grows

    :param filename: Name of the file to load
    :return: a tensor of the loaded data
    """
    ext_check = str(filename).lower()
    if ext_check.endswith('tif') or \
            ext_check.endswith('tiff') or \
            ext_check.endswith('png') or \
            ext_check.endswith('jpg') or \
            ext_check.endswith('jpeg'):
        image = imread(filename)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = np.moveaxis(image, -1, 0)
        return torch.tensor(image).float()

    if ext_check.endswith('npy'):
        return torch.tensor(np.load(filename))
    return None


def write_data(filename: Path, data: torch.Tensor) -> torch.Tensor | None:
    """Write data to file

    This method is a factory to write single data from file.

    TODO: This code may be adapted to a real factory when the number of supported formats grows

    :param filename: Name of the destination file
    :param data: a tensor of the data to save
    """
    ext_check = str(filename).lower()
    if ext_check.endswith('tif') or \
            ext_check.endswith('tiff') or \
            ext_check.endswith('png') or \
            ext_check.endswith('jpg') or \
            ext_check.endswith('jpeg'):
        return imsave(filename, data.numpy())
    if ext_check.endswith('npy'):
        return np.save(filename, data.numpy())
    return None
