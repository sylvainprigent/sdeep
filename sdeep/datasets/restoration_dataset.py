"""Default dataset for training image restoration network

Classes
-------
RestorationDataset


"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from skimage import io


class RestorationDataset(Dataset):
    """Dataset to train from full images

    All the training images must be saved as individual images in source and
    target folders.

    Parameters
    ----------
    source_dir: str
        Path of the noisy training images (or patches)
    target_dir: str
        Path of the ground truth images (or patches)
    use_data_augmentation: bool
        True to use data augmentation. False otherwise. The data augmentation is
        90, 180, 270 degrees rotations and flip (horizontal or vertical)
    """

    def __init__(self, source_dir, target_dir, use_data_augmentation=True):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.use_data_augmentation = use_data_augmentation

        self.source_images = natsorted(os.listdir(source_dir))
        self.target_images = natsorted(os.listdir(target_dir))
        if len(self.source_images) != len(self.target_images):
            raise Exception("Source and target dirs are not the same length")

        self.nb_images = len(os.listdir(source_dir))

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):

        img_source_np = \
            np.float32(io.imread(os.path.join(self.source_dir,
                                              self.source_images[idx])))

        img_target_np = \
            np.float32(io.imread(os.path.join(self.target_dir,
                                              self.target_images[idx])))

        # data augmentation
        if self.use_data_augmentation:
            # rotation
            k_1 = np.random.randint(4)
            img_source_np = np.rot90(img_source_np, k_1)
            img_target_np = np.rot90(img_target_np, k_1)
            # flip
            k_2 = np.random.randint(3)
            if k_2 < 2:
                img_source_np = np.flip(img_source_np, k_2)
                img_target_np = np.flip(img_target_np, k_2)

        # numpy continuous array
        img_source_np = np.ascontiguousarray(img_source_np)
        img_target_np = np.ascontiguousarray(img_target_np)

        # to tensor
        source_patch_tensor = torch.from_numpy(img_source_np).\
            view(1, 1, * img_source_np.shape).float()
        target_patch_tensor = torch.from_numpy(img_target_np).\
            view(1, 1, *img_target_np.shape).float()

        return source_patch_tensor, target_patch_tensor


class RestorationPatchDataset(Dataset):
    """Dataset to train from patches

    All the training images must be saved as individual images in source and
    target folders.

    Parameters
    ----------
    source_dir: str
        Path of the noisy training images (or patches)
    target_dir: str
        Path of the ground truth images (or patches)
    patch_size: int
        Size of the patches (width=height)
    stride: int
        Length of the patch overlapping
    use_data_augmentation: bool
        True to use data augmentation. False otherwise. The data augmentation is
        90, 180, 270 degrees rotations and flip (horizontal or vertical)

    """

    def __init__(self, source_dir, target_dir, patch_size=40, stride=10,
                 use_data_augmentation=True):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.stride = stride
        self.use_data_augmentation = use_data_augmentation

        self.source_images = natsorted(os.listdir(source_dir))
        self.target_images = natsorted(os.listdir(target_dir))
        if len(self.source_images) != len(self.target_images):
            raise Exception("Source and target dirs are not the same length")

        self.nb_images = len(os.listdir(source_dir))
        self.n_patches = self.nb_images * ((250 - patch_size) // stride) * \
                                          ((250 - patch_size) // stride)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Crop a patch from original image
        nb_patch_per_img = self.n_patches // self.nb_images

        img_number = idx // nb_patch_per_img
        elt = 'img_' + str(img_number).zfill(4) + '.tif'

        img_source_np = \
            np.float32(io.imread(os.path.join(self.source_dir, elt)))
        img_target_np = \
            np.float32(io.imread(os.path.join(self.target_dir, elt)))

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
        if self.use_data_augmentation:
            # rotation
            k_1 = np.random.randint(4)
            source_patch = np.rot90(source_patch, k_1)
            target_patch = np.rot90(target_patch, k_1)
            # flip
            k_2 = np.random.randint(3)
            if k_2 < 2:
                source_patch = np.flip(source_patch, k_2)
                target_patch = np.flip(target_patch, k_2)

        # numpy continuous array
        source_patch = np.ascontiguousarray(source_patch)
        target_patch = np.ascontiguousarray(target_patch)

        # to tensor
        source_patch_tensor = torch.from_numpy(source_patch).view(1, 1,
                                                                  *source_patch.shape).float()
        target_patch_tensor = torch.from_numpy(target_patch).view(1, 1,
                                                                  *target_patch.shape).float()

        return source_patch_tensor, target_patch_tensor
