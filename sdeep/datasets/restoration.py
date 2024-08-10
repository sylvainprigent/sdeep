"""Default dataset for training image restoration network

Classes
-------
RestorationDataset


"""
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from skimage import io

from torch.utils.data import Dataset


class RestorationDataset(Dataset):
    """Dataset to train from full images

    All the training images must be saved as individual images in source and
    target folders.

    :param source_dir: Path of the noisy training images (or patches)
    :param target_dir: Path of the ground truth images (or patches)
    :param transform: Transformation to apply to the image before model call
    """
    def __init__(self,
                 source_dir: str | Path,
                 target_dir: str | Path,
                 transform: Callable = None):
        super().__init__()
        self.device = None
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))
        if len(self.source_images) != len(self.target_images):
            raise Exception("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):

        img_source_np = \
            np.float32(io.imread(self.source_images[idx]))

        img_target_np = \
            np.float32(io.imread(self.target_images[idx]))

        # data augmentation
        if self.transform:
            img_source_np = self.transform(img_source_np)
            img_target_np = self.transform(img_target_np)

        # numpy continuous array
        img_source_np = np.ascontiguousarray(img_source_np)
        img_target_np = np.ascontiguousarray(img_target_np)

        # to tensor
        source_patch_tensor = torch.from_numpy(img_source_np).\
            view(1, * img_source_np.shape).float()
        target_patch_tensor = torch.from_numpy(img_target_np).\
            view(1, *img_target_np.shape).float()

        return source_patch_tensor, target_patch_tensor, self.source_images[idx].stem


class RestorationPatchDataset(Dataset):
    """Dataset to train from patches

    All the training images must be saved as individual images in source and
    target folders.

    :param source_dir: Path of the noisy training images (or patches)
    :param target_dir: Path of the ground truth images (or patches)
    :param patch_size: Size of the patches (width=height)
    :param stride: Length of the patch overlapping
    :param transform: Transformation to apply to the image before model call

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, source_dir, target_dir, patch_size=40, stride=10,
                 transform: Callable = None):
        super().__init__()
        self.device = None
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))
        if len(self.source_images) != len(self.target_images):
            raise Exception("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)
        image = io.imread(self.source_images[0])
        self.n_patches = self.nb_images * ((image.shape[0] - patch_size) // stride) * \
                                          ((image.shape[1] - patch_size) // stride)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Crop a patch from original image
        nb_patch_per_img = self.n_patches // self.nb_images

        elt = self.source_images[idx // nb_patch_per_img]

        img_source_np = \
            np.float32(io.imread(self.source_dir / elt))
        img_target_np = \
            np.float32(io.imread(self.target_dir / elt))

        nb_patch_w = (img_source_np.shape[1] - self.patch_size) // self.stride
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        source_patch = \
            img_source_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]
        target_patch = \
            img_target_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]

        # data augmentation
        if self.transform:
            source_patch = self.transform(source_patch)
            target_patch = self.transform(target_patch)

        # numpy continuous array
        source_patch = np.ascontiguousarray(source_patch)
        target_patch = np.ascontiguousarray(target_patch)

        # to tensor
        return (torch.from_numpy(source_patch).view(1, *source_patch.shape)
                .float(),
                torch.from_numpy(target_patch).view(1, *target_patch.shape)
                .float(),
                str(idx)
                )


class RestorationPatchDatasetLoad(Dataset):
    """Dataset to train from patches

    All the training images must be saved as individual images in source and
    target folders.
    This version load all the dataset in the CPU

    :param source_dir: Path of the noisy training images (or patches)
    :param target_dir: Path of the ground truth images (or patches)
    :param patch_size: Size of the patches (width=height)
    :param stride: Length of the patch overlapping
    :param transform: Transformation to apply to the image before model call

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, source_dir, target_dir, patch_size=40, stride=10,
                 transform: Callable = None):
        super().__init__()
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))

        if len(self.source_images) != len(self.target_images):
            raise Exception("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)
        image = io.imread(self.source_images[0])
        self.n_patches = self.nb_images * ((image.shape[0] - patch_size) // stride) * \
                                          ((image.shape[1] - patch_size) // stride)
        print('num patches = ', self.n_patches)

        # Load all the images in a list
        self.source_data = []
        for source in self.source_images:
            self.source_data.append(np.float32(io.imread(source)))
        self.target_data = []
        for target in self.target_images:
            self.target_data.append(np.float32(io.imread(target)))

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Crop a patch from original image
        nb_patch_per_img = self.n_patches // self.nb_images

        img_number = idx // nb_patch_per_img

        img_source_np = self.source_data[img_number]
        img_target_np = self.target_data[img_number]

        nb_patch_w = (img_source_np.shape[1] - self.patch_size) // self.stride
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        source_patch = \
            img_source_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]
        target_patch = \
            img_target_np[i * self.stride:i * self.stride + self.patch_size,
            j * self.stride:j * self.stride + self.patch_size]

        # data augmentation
        if self.transform:
            source_patch = self.transform(source_patch)
            target_patch = self.transform(target_patch)

        # numpy continuous array
        source_patch = np.ascontiguousarray(source_patch)
        target_patch = np.ascontiguousarray(target_patch)

        return (torch.from_numpy(source_patch).view(1, *source_patch.shape)
                .float(),
                torch.from_numpy(target_patch).view(1, *target_patch.shape)
                .float(),
                str(idx)
                )


export = [RestorationDataset,
          RestorationPatchDataset,
          RestorationPatchDatasetLoad
          ]
