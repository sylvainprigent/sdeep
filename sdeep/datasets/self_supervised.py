"""Implementation of patch image datasets with random mask for self supervised based learning"""
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from skimage.io import imread

from torch.utils.data import Dataset


class SelfSupervisedPatchDataset(Dataset):
    """Gray scaled image patch dataset for Self supervised learning

    :param images_dir: Directory containing the training images
    :param patch_size: Size of the squared training patches
    :param stride: Stride used to extract overlapping patches from images
    :param transform: Transformation to images before model
    """
    def __init__(self,
                 images_dir: Path,
                 patch_size: int = 40,
                 stride: int = 10,
                 transform: Callable = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        source_images = sorted(self.images_dir.glob('*.*'))

        self.nb_images = len(source_images)
        image = imread(source_images[0])
        self.n_patches = self.nb_images * ((image.shape[0] - patch_size) // stride) * \
                                          ((image.shape[1] - patch_size) // stride)
        print('num patches = ', self.n_patches)

        # Load all the images in a list
        self.images_data = []
        for source in source_images:
            self.images_data.append(np.float32(imread(source)))

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        nb_patch_per_img = self.n_patches // self.nb_images

        img_number = idx // nb_patch_per_img

        img_np = self.images_data[img_number]

        nb_patch_w = (img_np.shape[1] - self.patch_size) // self.stride
        idx = idx % nb_patch_per_img
        i, j = idx // nb_patch_w, idx % nb_patch_w
        img_patch = \
            img_np[i * self.stride:i * self.stride + self.patch_size,
                   j * self.stride:j * self.stride + self.patch_size]

        if self.transform:
            img_patch = self.transform(torch.Tensor(img_patch))

        return (
            img_patch.view(1, *img_patch.shape),
            str(idx)
        )


class SelfSupervisedDataset(Dataset):
    """Gray scaled image patch dataset for Self supervised learning

    :param images_dir: Directory containing the training images
    :param transform: Transformation to images before model
    """
    def __init__(self,
                 images_dir: Path,
                 transform: Callable = None):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.transform = transform

        self.source_images = sorted(self.images_dir.glob('*.*'))

        self.nb_images = len(self.source_images)

        # Load all the images in a list
        self.images_data = []
        for source in self.source_images:
            self.images_data.append(np.float32(imread(source)))

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):

        img_patch = self.images_data[idx]
        if self.transform:
            img_patch = self.transform(torch.Tensor(img_patch))

        return (
            img_patch.view(1, *img_patch.shape),
            self.source_images[idx].stem
        )


export = [SelfSupervisedPatchDataset, SelfSupervisedDataset]
