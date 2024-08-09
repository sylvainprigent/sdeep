from typing import Callable
import numpy as np
import torch
from torchvision.datasets import MNIST

from torch.utils.data import Dataset


class MNISTClassif(Dataset):
    def __init__(self, dir_name: str, train: bool = True, transform: Callable = None):
        super().__init__()
        self.mnist = MNIST(dir_name, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img = torch.Tensor(np.asarray(self.mnist[idx][0]).copy().astype(float))
        label = torch.tensor(self.mnist[idx][1])
        if self.transform is not None:
            img = self.transform(img)
        return img.view(1, 28, 28), label, str(idx)


class MNISTAutoencoder(Dataset):
    def __init__(self, dir_name: str, train: bool = True, transform: Callable = None):
        super().__init__()
        self.mnist = MNIST(dir_name, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img = torch.Tensor(np.asarray(self.mnist[idx][0]).copy().astype(float))
        if self.transform is not None:
            img = self.transform(img)
        return img.view(1, 28, 28), img.view(1, 28, 28), str(idx)


export = [MNISTAutoencoder, MNISTClassif]
