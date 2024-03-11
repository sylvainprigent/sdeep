"""Default dataset for training image segmentation network"""
from typing import List
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from skimage import io
from skimage.measure import label
from skimage.measure import regionprops


class SegmentationDataset(Dataset):
    """Dataset to train from full images

    All the training images must be saved as individual images in source and
    target folders.

    Parameters
    ----------
    source_dir: str
        Path of the images to segment (or patches)
    target_dir: str
        Path of the ground truth segmentation (or patches)
    use_data_augmentation: bool
        True to use data augmentation. False otherwise. The data augmentation
        is 90, 180, 270 degrees rotations and flip (horizontal or vertical)
    """

    def __init__(self, source_dir, target_dir, use_data_augmentation=True):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.use_data_augmentation = use_data_augmentation

        # load the images
        if not source_dir.endswith(".npy") or not target_dir.endswith(".npy"):
            raise FileNotFoundError('SegmentationDataset can only '
                                    'use .npy file')

        self.source_data = np.load(source_dir)
        self.target_data = np.load(target_dir)
        if self.source_data.shape != self.target_data.shape:
            raise Exception("Source and target data are not the same shape")

        self.nb_images = self.source_data.shape[0]

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):
        img_source_np = self.source_data[idx]
        img_target_np = self.target_data[idx]

        imin = img_source_np.min()
        imax = img_source_np.max()
        img_source_np = (img_source_np - imin) / (imax - imin)
        img_target_np /= 255

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
            view(1, * img_source_np.shape).float()
        target_patch_tensor = torch.from_numpy(img_target_np).\
            view(1, *img_target_np.shape).float()

        return source_patch_tensor, target_patch_tensor, \
               self.source_images[idx]


def find_patches(image: np.ndarray, patch_size: int):
    regions = []
    for i in range(image.shape[0]):
        if np.max(image[i, ...] > 1):
            labels = np.unique(image[i, ...])
            labels = labels[1:] # remove 0
            for label in labels:
                regions += find_patches_single(
                    np.uint8(image[i, ...] == label), patch_size)
        else:
            regions += find_patches_single(image[i, ...], patch_size)
    return regions


def find_patches_single(image: np.ndarray, patch_size: int):
    label_image = label(image)

    regions = []
    for region in regionprops(label_image):
        c_x, c_y = region.centroid
        c_x = int(c_x)
        c_y = int(c_y)
        if c_x - patch_size / 2 > 0 and c_y - patch_size / 2 > 0 \
                and c_x + patch_size / 2 < image.shape[0] \
                and c_y + patch_size / 2 < image.shape[1]:
            regions.append((c_x, c_y))
    return regions


class SegmentationPatchDataset(Dataset):
    """Dataset to train from patches

    All the training images must be saved as individual images in source and
    target folders.

    Parameters
    ----------
    source_dir: str
        Path of the images to segment (or patches)
    target_dir: str
        Path of the ground truth segmentation images (or patches)
    patch_size: int
        Size of the patches (width=height)
    stride: int
        Length of the patch overlapping
    use_data_augmentation: bool
        True to use data augmentation. False otherwise. The data augmentation
        is 90, 180, 270 degrees rotations and flip (horizontal or vertical)

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, source_dir, target_dir, patch_size=48,
                 use_data_augmentation=True,
                 use_labels=False):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.use_data_augmentation = use_data_augmentation
        self.__use_labels = use_labels

        # load the images
        if not source_dir.endswith(".npy") or not target_dir.endswith(".npy"):
            raise FileNotFoundError('SegmentationDataset can only use '
                                    '.npy file')

        self.source_data = np.load(source_dir)
        self.source_data = self.source_data[0:10, ...]
        self.target_data = np.load(target_dir)
        self.target_data = self.target_data[0:10, ...]

        if self.target_data.ndim == 3:
            self.target_data = np.expand_dims(self.target_data, axis=1)

        if self.target_data.ndim != 4 and self.source_data.ndim != 3:
            raise ValueError("source and target data ndim not allowed")    

        if self.source_data.shape[0] != self.target_data.shape[0]:
            raise ValueError("source and target dataset have not same "
                             "sample size")

        if self.source_data.shape[1] != self.target_data.shape[2] or \
                self.source_data.shape[2] != self.target_data.shape[3]:
            raise Exception("Source and target images are not the same shape")

        self.nb_images = self.source_data.shape[0]

        self.patches_info = []
        for i in range(0, self.nb_images):
            image = self.target_data[i, ...]
            image_patches = find_patches(image, patch_size)
            for patch in image_patches:
                self.patches_info.append([i, patch[0], patch[1]])
   
        print('patch num = ', len(self.patches_info))

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):

        patch_info = self.patches_info[idx]
        image_id = patch_info[0]
        patch_cx = patch_info[1]
        patch_cy = patch_info[2]

        min_x = int(patch_cx - self.patch_size / 2)
        max_x = int(patch_cx + self.patch_size / 2)
        min_y = int(patch_cy - self.patch_size / 2)
        max_y = int(patch_cy + self.patch_size / 2)

        img_source_np = self.source_data[image_id, ...]

        mini = np.min(img_source_np)
        maxi = np.max(img_source_np)
        img_source_np = (img_source_np-mini)/(maxi-mini)
        source_patch = img_source_np[min_x:max_x, min_y:max_y]

        img_target_np = self.target_data[image_id, ...]
        target_patch = img_target_np[:, min_x:max_x, min_y:max_y]
        if not self.__use_labels:
            target_patch = target_patch / 255

        # data augmentation
        if self.use_data_augmentation:
            # rotation
            k_1 = np.random.randint(4)
            source_patch = np.rot90(source_patch, k_1, axes=(0, 1))
            target_patch = np.rot90(target_patch, k_1, axes=(1, 2))
            # flip
            k_2 = np.random.randint(3)
            if k_2 < 2:
                source_patch = np.flip(source_patch, k_2)
                target_patch = np.flip(target_patch, k_2+1)

        # numpy continuous array
        if self.__use_labels:        
            target_patch = target_patch[0, ...].astype(np.uint8)
        source_patch = np.ascontiguousarray(source_patch)
        target_patch = np.ascontiguousarray(target_patch)

        # to tensor
        if not self.__use_labels:
            target_tensor = torch.from_numpy(target_patch).float()
        else: 
            target_tensor = torch.from_numpy(target_patch).long()    
        return (torch.from_numpy(source_patch).view(1, *source_patch.shape)
                .float(),
                target_tensor,
                str(idx)
                )


def load_sequence(filenames: str, parent_dir: str) -> np.ndarray:
    """Load 3d array from list of files"""
    data = []
    for file in filenames:
        img = np.float32(io.imread(os.path.join(parent_dir, file)))
        if img.ndim > 2:
            data.append(img[0, ...])
        else:    
            data.append(img)
    return np.array(data)    


def target_dir_files(target_dir: str) -> List[List[str]]:
    """Get the list of files for each layer of target"""
    subfolders = [ f.path for f in os.scandir(target_dir) if f.is_dir() ]
    if len(subfolders) > 0:
        subf_list = []
        for subfolder in subfolders:
            subf_list.append(natsorted(os.listdir(os.path.join(target_dir,
                                                               subfolder))))
        # Check data size
        fil_count = len(subf_list[0])
        for i in range(len(subf_list)):
            if fil_count != len(subf_list[i]):
                raise ValueError("Dataset subfolders are not same size ")
        # invert list
        out_list = []
        for i in range(fil_count):
            layers = []
            for j in range(len(subf_list)):
                layers.append(os.path.join(subfolders[j], subf_list[j][i]))
            out_list.append(layers)    
        return out_list
    else:
        data = natsorted(os.listdir(target_dir))
        out_list = []
        for dat in data:
            out_list.append([dat])
        return out_list


class SegmentationFileDataset(Dataset):

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, source_dir, target_dir,
                 use_data_augmentation=True,
                 use_labels=False):

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.use_data_augmentation = use_data_augmentation
        self.__use_labels = use_labels

        self.source_images = natsorted(os.listdir(source_dir))
        self.target_images = target_dir_files(target_dir)

        if len(self.source_images) != len(self.target_images):
            raise Exception("Source and target dirs are not the same length")

        # Load all the images in a list
        self.source_data = []
        for source in self.source_images:
            self.source_data.append(np.float32(io.imread(
                os.path.join(self.source_dir, source))))
        self.target_data = []
        for target in self.target_images:
            self.target_data.append(load_sequence(target, self.target_dir))

        self.nb_images = len(os.listdir(source_dir))

    def __len__(self):
        return self.nb_images

    def __getitem__(self, idx):

        img_source_np = self.source_data[idx]
        img_target_np = self.source_data[idx]

        mini = np.min(img_source_np)
        maxi = np.max(img_source_np)
        img_source_np = (img_source_np-mini)/(maxi-mini)
        if not self.__use_labels:
            img_target_np = self.target_data[idx] / 255

        # data augmentation
        if self.use_data_augmentation:
            # rotation
            k_1 = np.random.randint(4)
            img_source_np = np.rot90(img_source_np, k_1, axes=(0, 1))
            img_target_np = np.rot90(img_target_np, k_1, axes=(1, 2))
            # flip
            k_2 = np.random.randint(3)
            if k_2 < 2:
                img_source_np = np.flip(img_source_np, k_2)
                img_target_np = np.flip(img_target_np, k_2+1)

        # numpy continuous array
        img_source_np = np.ascontiguousarray(img_source_np)
        img_target_np = np.ascontiguousarray(img_target_np)

        # to tensor
        if not self.__use_labels:
            target_tensor = torch.from_numpy(img_target_np).view(
                *img_target_np.shape).float()
        else: 
            target_tensor = torch.from_numpy(img_target_np).view(
                *img_target_np.shape).long()
        return (torch.from_numpy(img_source_np).view(1, *img_source_np.shape)
                .float(),
                target_tensor,
                self.source_images[idx]
                )


export = [SegmentationDataset, SegmentationPatchDataset,
          SegmentationFileDataset]
