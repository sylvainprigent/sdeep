"""Data transformation for image restoration workflow"""
import torch
from torchvision.transforms import v2


class RestorationAugmentation:
    """Data augmentation flipping images"""
    def __init__(self):
        self.__transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __call__(self, image):
        return self.__transform(image)


export = [RestorationAugmentation]
