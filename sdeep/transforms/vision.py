"""Implementation of standards data transform for imaging"""

import torch
from torchvision.transforms import v2


class VisionScale:
    """Data augmentation for image restoration"""

    def __init__(self):
        self.__transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __call__(self, image):
        return self.__transform(image)


export = [VisionScale]
