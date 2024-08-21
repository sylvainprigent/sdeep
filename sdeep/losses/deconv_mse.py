"""Define a loss as MSE with a (de)convolution filter"""
from pathlib import Path
import torch

from skimage.io import imread


def hv_loss(img: torch.Tensor, weighting: float) -> torch.Tensor:
    """Sparse Hessian regularization term


    :param img: Tensor of shape BCYX containing the estimated image
    :param weighting: Sparse weighting parameter in [0, 1]. 0 sparse, and 1 not sparse
    :return the sparce hessian loss term for the batch
    """
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    h_v = torch.sqrt(weighting * weighting * (dxx2 + dyy2 + 2 * dxy2) +
                     (1 - weighting) * (1 - weighting) * torch.square(img[:, :, 1:-1, 1:-1]))
    return torch.mean(h_v)


def hessian(img: torch.Tensor) -> torch.Tensor:
    """Compute the Hessian norm on a 2D images batch

    :param img: Image batch to process
    :return: The batch norm of hessian
    """
    dxx2 = torch.square(-img[:, :, 2:, 1:-1] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, :-2, 1:-1])
    dyy2 = torch.square(-img[:, :, 1:-1, 2:] + 2 * img[:, :, 1:-1, 1:-1] - img[:, :, 1:-1, :-2])
    dxy2 = torch.square(img[:, :, 2:, 2:] - img[:, :, 2:, 1:-1] - img[:, :, 1:-1, 2:] +
                        img[:, :, 1:-1, 1:-1])
    return dxx2 + dyy2 + 2 * dxy2


class DeconSpitfire(torch.nn.Module):
    """MSE LOSS with a (de)convolution filter and Spitfire regularisation

    :param psf_file: File containing the PSF for deconvolution
    :return: Loss tensor
    """
    def __init__(self,
                 psf_file: Path,
                 regularization: float = 1e-3,
                 weighting: float = 0.6
                 ):
        super().__init__()
        self.psf_file = psf_file
        self.regularization = regularization
        self.weighting = weighting

        self.__psf = torch.Tensor(imread(psf_file)).float()
        if self.__psf.ndim > 2:
            raise ValueError('DeconMSE PSF must be a gray scaled 2D image')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__psf = self.__psf.view((1, 1, *self.__psf.shape)).to(self.device)
        print('psf shape=', self.__psf.shape)
        self.__conv_op = torch.nn.Conv2d(1, 1,
                                         kernel_size=self.__psf.shape[2],
                                         stride=1,
                                         padding=int((self.__psf.shape[2] - 1) / 2),
                                         bias=False)
        with torch.no_grad():
            self.__conv_op.weight = torch.nn.Parameter(self.__psf, requires_grad=False)
        self.__conv_op.requires_grad_(False)

    def forward(self, input_image: torch.Tensor, target: torch.Tensor):
        """Deconvolution L2 data-term

        Compute the L2 error between the original image (input) and the
        convoluted reconstructed image (target)

        :param input_image: Tensor of shape BCYX containing the original blurry image
        :param target: Tensor of shape BCYX containing the estimated deblurred image
        """
        conv_img = self.__conv_op(input_image)

        mse = torch.nn.MSELoss()
        return mse(target, conv_img) + self.regularization*hv_loss(input_image,
                                                                   weighting=self.weighting)


class DeconMSEHessian(torch.nn.Module):
    """MSE LOSS with a (de)convolution filter and Hessian regularization

    :param psf_file: File containing the PSF for deconvolution
    :return: Loss tensor
    """

    def __init__(self,
                 psf_file: Path
                    ):
        super().__init__()
        self.psf_file = psf_file

        self.__psf = torch.Tensor(imread(psf_file)).float()
        if self.__psf.ndim > 2:
            raise ValueError('DeconMSE PSF must be a gray scaled 2D image')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__psf = self.__psf.view((1, 1, *self.__psf.shape)).to(self.device)
        print('psf shape=', self.__psf.shape)
        self.__conv_op = torch.nn.Conv2d(1, 1,
                                         kernel_size=self.__psf.shape[2],
                                         stride=1,
                                         padding=int((self.__psf.shape[2] - 1) / 2),
                                         bias=False)

    def forward(self, input_image: torch.Tensor, target: torch.Tensor):
        """Deconvolution L2 data-term

        Compute the L2 error between the original image (input) and the
        convoluted reconstructed image (target)

        :param input_image: Tensor of shape BCYX containing the original blurry image
        :param target: Tensor of shape BCYX containing the estimated deblurred image
        """

        with torch.no_grad():
            self.__conv_op.weight = torch.nn.Parameter(self.__psf)
        mse = torch.nn.MSELoss()
        conv_x = self.__conv_op(input_image)
        return mse(target, conv_x) + torch.mean(hessian(target-conv_x)**2)


export = [DeconSpitfire, DeconMSEHessian]
