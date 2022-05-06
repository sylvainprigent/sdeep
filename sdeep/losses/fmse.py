import numpy as np
import torch
from skimage import draw


class FMSELoss(torch.nn.Module):
    """Define an image reconstruction loss with the MSE in fourier space

    The point of this Loss is only to test the Fourier transform

    Parameters
    ----------
    patch_size: int
        size of the input image patch in it smallest dimension

    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input, target):
        input_fft = torch.fft.fft2(input)
        target_fft = torch.fft.fft2(target)
        return torch.sum(torch.square(torch.abs(input_fft-target_fft)))
