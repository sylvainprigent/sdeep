"""Losses for Noise2Void"""
from pathlib import Path
import torch
from skimage.io import imread

from sdeep.utils import device


class N2XDenoise(torch.nn.Module):
    """MSE Loss with mask for Noise2Void denoising"""
    def forward(self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Calculate forward loss

        :param predict: tensor predicted by the model
        :param target: Reference target tensor
        :param mask: Mask to select pixels of interest
        """
        num = torch.sum((predict*mask - target*mask)**2)
        den = torch.sum(mask)
        return num/den


class N2XDecon(torch.nn.Module):
    """MSE Loss with mask for Noise2Void deconvolution

    :param psf_file: File image containing the Point Spread Function
    :return: Loss tensor
    """
    def __init__(self,
                 psf_file: Path, ):
        super().__init__()

        self.__psf = torch.Tensor(imread(psf_file)).float()
        if self.__psf.ndim > 2:
            raise ValueError('N2VDecon PSF must be a gray scaled 2D image')

        self.__psf = self.__psf.view((1, 1, *self.__psf.shape)).to(device())
        print('psf shape=', self.__psf.shape)
        self.__conv_op = torch.nn.Conv2d(1, 1,
                                         kernel_size=self.__psf.shape[2],
                                         stride=1,
                                         padding=int((self.__psf.shape[2] - 1) / 2),
                                         bias=False)
        with torch.no_grad():
            self.__conv_op.weight = torch.nn.Parameter(self.__psf)

    def forward(self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Calculate forward loss

        :param predict: tensor predicted by the model
        :param target: Reference target tensor
        :param mask: Mask to select pixels of interest
        """
        conv_img = self.__conv_op(predict)

        num = torch.sum((conv_img*mask - target*mask)**2)
        den = torch.sum(mask)
        return num/den


export = [N2XDenoise, N2XDecon]
