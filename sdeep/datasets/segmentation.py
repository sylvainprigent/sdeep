"""Default dataset for training image segmentation network"""
from typing import Callable
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

from skimage import io
from skimage import measure

from torch.utils.data import Dataset


def patch_centroid(image: np.ndarray, patch_size: int) -> list[tuple[int, int]]:
    """Extract the positions of path centered on a label mask object

    :param image: Mask containing the object as binary or label image
    :param patch_size: Dimension of square patch
    :return: The positions (center) of extracted patches
    """
    regions = []
    if np.max(image > 1):
        labels = np.unique(image)
        labels = labels[1:]  # remove 0
        for label_ in labels:
            regions += patch_centroid_single(
                np.uint8(image == label_), patch_size)
    else:
        regions += patch_centroid_single(image, patch_size)
    return regions


def patch_centroid_single(image: np.ndarray, patch_size: int) -> list[tuple[int, int]]:
    """Extract the positions of path centered on a binary mask object

    :param image: Mask containing the object as binary image
    :param patch_size: Dimension of square patch
    :return: The positions (center) of extracted patches
    """
    label_image = measure.label(image)

    regions = []
    for region in measure.regionprops(label_image):
        c_x, c_y = region.centroid
        c_x = int(c_x)
        c_y = int(c_y)
        if c_x - patch_size / 2 > 0 and c_y - patch_size / 2 > 0 \
                and c_x + patch_size / 2 < image.shape[0] \
                and c_y + patch_size / 2 < image.shape[1]:
            regions.append((c_x, c_y))
    return regions


def patch_grid(image: np.ndarray, patch_size: int, patch_overlap: 10) -> list[tuple[int, int]]:
    """Extract the positions of regular grid patches on a mask

    :param image: Mask containing the object as binary image
    :param patch_size: Dimension of square patch
    :param patch_overlap: Overlapping between patches
    :return: The positions (center) of extracted patches
    """
    sy = image.shape[-1]
    sx = image.shape[-2]
    step = patch_size - patch_overlap
    start = int(patch_size/2)+1
    regions = []
    for c_x in range(start, sx, step):
        for c_y in range(start, sy, step):
            if c_x - patch_size / 2 > 0 and c_y - patch_size / 2 > 0 \
                    and c_x + patch_size / 2 < image.shape[0] \
                    and c_y + patch_size / 2 < image.shape[1]:
                regions.append((c_x, c_y))
    return regions


def extract_patches(image: np.ndarray,
                    patch_strategy: str,
                    patch_size: int = 48,
                    patch_overlap: int = 8):
    """Extract patches position from a ground truth mask depending on the given strategy

    :param image: Mask containing the object as binary image
    :param patch_strategy: Strategy used to extract patches (grid, centroid)
    :param patch_size: Dimension of square patch
    :param patch_overlap: Overlapping between patches
    :return: The positions (center) of extracted patches
    """
    if patch_strategy == "grid":
        return patch_grid(image, patch_size, patch_overlap)
    if patch_strategy == "centroid":
        return patch_centroid(image, patch_size)
    raise ValueError('Segmentation dataset patching strategy not found')


def read_source_image(filename: Path) -> np.array:
    """Read and shape a source image

    :param filename: File containing the image,
    :return: The image array
    """
    transform = v2.ToDtype(torch.float32, scale=True)
    image = io.imread(filename).astype(float)
    if image.ndim == 3:
        image = np.moveaxis(image, -1, 0)
    image = transform(image)
    return image


def read_target_image(filename: Path) -> np.array:
    """Read and shape a target image

    :param filename: File containing the image,
    :return: The image array
    """
    image = io.imread(filename)
    if image.ndim == 3:
        image = np.ascontiguousarray(image[:, :, 0])
    return image/255


class SegmentationPatchDataset(Dataset):
    """Dataset for image segmentation where images are partitioned into patches

    All the training images must be saved as individual images in source and
    target folders.

    :param source_dir: Path of the images to segment (or patches)
    :param target_dir: Path of the ground truth segmentation images (or patches)
    :param patch_size: Size of the patches (width=height)
    :param transform: Transformation to apply to the image before model call

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self,
                 source_dir: str | Path,
                 target_dir: str | Path,
                 patch_strategy: str = 'grid',
                 patch_size: int = 48,
                 patch_overlap: int = 8,
                 use_labels: bool = False,
                 preload: bool = True,
                 transform: Callable = True,):
        super().__init__()
        self.device = None
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.patch_size = patch_size
        self.preload = preload
        self.use_labels = use_labels
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))

        if len(self.source_images) != len(self.target_images):
            raise ValueError("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)

        if self.preload:
            self.source_array = []
            self.target_array = []
            for i in range(0, self.nb_images):
                self.source_array.append(read_source_image(self.source_images[i]))
                self.target_array.append(read_target_image(self.target_images[i]))

        self.patches_info = []
        for i in range(0, self.nb_images):
            image = read_target_image(self.target_images[i])
            image_patches = extract_patches(image, patch_strategy, patch_size, patch_overlap)
            for patch in image_patches:
                self.patches_info.append([i, patch[0], patch[1]])

        print('patch num = ', len(self.patches_info))

    def __len__(self):
        return len(self.patches_info)

    def __extract_patch_info(self,
                             patch_info: list[int]
                             ) -> tuple[int, int, int, int, int]:
        image_id = patch_info[0]
        patch_cx = patch_info[1]
        patch_cy = patch_info[2]

        min_x = int(patch_cx - self.patch_size / 2)
        max_x = int(patch_cx + self.patch_size / 2)
        min_y = int(patch_cy - self.patch_size / 2)
        max_y = int(patch_cy + self.patch_size / 2)
        return int(image_id), min_x, max_x, min_y, max_y

    def __getitem__(self, idx):

        # retrieve patch
        image_id, min_x, max_x, min_y, max_y = self.__extract_patch_info(self.patches_info[idx])

        # load source patch
        if self.preload:
            img_source_np = self.source_array[image_id]
        else:
            img_source_np = read_source_image(self.source_images[image_id])

        if img_source_np.ndim > 2:
            source_patch = img_source_np[:, min_x:max_x, min_y:max_y]
        else:
            source_patch = img_source_np[min_x:max_x, min_y:max_y]

        # load target mask
        if self.preload:
            img_target_np = self.target_array[image_id]
        else:
            img_target_np = read_target_image(self.target_images[image_id])

        target_patch = img_target_np[min_x:max_x, min_y:max_y]

        # source to tensor
        source_patch = torch.from_numpy(source_patch).float()
        if source_patch.ndim < 3:
            source_patch = source_patch.view(1, *source_patch.shape)

        # target to tensor
        if not self.use_labels:
            target_tensor = torch.from_numpy(target_patch).float()
        else:
            target_tensor = torch.from_numpy(target_patch).long()
        target_tensor = target_tensor.view(1, *target_tensor.shape)

        # data augmentation
        if self.transform:
            both_images = torch.cat((source_patch, target_tensor), 0)
            transformed_images = self.transform(both_images)
            source_patch = transformed_images[0:-1, ...]
            target_tensor = transformed_images[-1].view(*target_tensor.shape)

        return (source_patch,
                target_tensor,
                str(idx)
                )


class SegmentationDataset(Dataset):
    """Dataset for image segmentation where images are stored in a single directory

    :param source_dir: Directory containing the images to segment
    :param target_dir: Directory containing the segmentation ground truth
    :param use_labels: True if the ground truth is a label image. False if stack of binary images
    :param transform: Transform for data augmentation
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self,
                 source_dir: str | Path,
                 target_dir: str | Path,
                 use_labels: bool = False,
                 transform: Callable = None):
        super().__init__()
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.use_labels = use_labels
        self.transform = transform

        self.source_images = sorted(self.source_dir.glob('*.*'))
        self.target_images = sorted(self.target_dir.glob('*.*'))

        if len(self.source_images) != len(self.target_images):
            raise ValueError("Source and target dirs are not the same length")

        self.nb_images = len(self.source_images)

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):

        img_source_np = read_source_image(self.source_images[idx])
        img_target_np = read_target_image(self.target_images[idx])

        # source to tensor
        source_tensor = torch.from_numpy(img_source_np).float()
        if source_tensor.ndim < 3:
            source_tensor = source_tensor.view(1, *source_tensor.shape)

        # to tensor
        if not self.use_labels:
            target_tensor = torch.from_numpy(img_target_np).float()
        else:
            target_tensor = torch.from_numpy(img_target_np).long()
        target_tensor = target_tensor.view(1, *target_tensor.shape)

        # data augmentation
        if self.transform:
            both_images = torch.cat((source_tensor, target_tensor), 0)
            transformed_images = self.transform(both_images)
            source_tensor = transformed_images[0:-1, ...]
            target_tensor = transformed_images[-1, ...].view(1,
                                                             source_tensor.shape[-2],
                                                             source_tensor.shape[-1])

        return (source_tensor,
                target_tensor,
                self.source_images[idx].stem
                )


export = [SegmentationDataset, SegmentationPatchDataset]
