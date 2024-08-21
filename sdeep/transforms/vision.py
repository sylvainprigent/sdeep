"""Implementation of standards data transform for imaging"""

import torch
from torchvision.transforms import v2


class VisionScale:
    """Scale images in [-1, 1]"""

    def __init__(self):
        self.__transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __call__(self, image):
        return self.__transform(image)


class VisionCrop:
    """Crop the images at the center
    
    :param size: Size of the croped region
    """
    def __init__(self, size: tuple[int, int]):
        self.__transform = v2.Compose([
            v2.CenterCrop(size)
        ])

    def __call__(self, image):
        return self.__transform(image)


export = [VisionScale, VisionCrop]
