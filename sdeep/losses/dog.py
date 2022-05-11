import math
import numpy as np
import torch


def gaussian_kernel(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    x0 = kernel_size / 2
    y0 = kernel_size / 2
    sigma2 = 0.5 / (sigma * sigma)
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = math.exp(- pow(x - x0, 2) * sigma2
                                    - pow(y - y0, 2) * sigma2)
    return kernel/np.sum(kernel)


def dog_kernel(sigma1, sigma2, kernel_size):
    return gaussian_kernel(sigma1, kernel_size) - gaussian_kernel(sigma2, kernel_size)


def dog_kernels(sigma_pairs, kernel_size):
    """Generate a family of 2D DoG kernels

    Parameters
    ----------
    sigma_pairs: list
        List of sigma pairs tuples: [(0.5, 1), (1, 2),  ...]
    kernel_size: int
        Size of the DoG kernel in X and Y axis

    Returns
    -------
    ndarray: [C, Y, X] kernels stack

    """
    kernels = np.zeros((len(sigma_pairs), kernel_size, kernel_size))
    for i, sigma_pair in enumerate(sigma_pairs):
        kernels[i, ...] = dog_kernel(sigma_pair[0], sigma_pair[1], kernel_size)
    return kernels


class DoGLoss(torch.nn.Module):
    """Define an image reconstruction loss DoG filters + MSE

    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mse = torch.nn.MSELoss()

        # DoG filter
        sigma_pairs = [(0.2, 0.202), (0.2, 0.5), (0.2, 0.7), (0.2, 1.0), (0.2, 1.2), (0.2, 1.4)]
        weights = dog_kernels(sigma_pairs, 7)
        print(weights)
        self.dog_filter = torch.nn.Conv2d(1, len(sigma_pairs), 7, padding=3, bias=False,
                                          padding_mode='reflect', device=self.device)
        self.dog_filter.requires_grad_(False)
        self.dog_filter.weight.data = torch.Tensor(weights).view((len(sigma_pairs), 1, 7, 7))

        # linear filter
        linear_weight = torch.from_numpy(np.ones((len(sigma_pairs)))).to(self.device)
        self.linear = torch.nn.Linear(len(sigma_pairs), 1, bias=False, device=self.device)
        self.linear.requires_grad_(False)
        self.linear.weight.data = linear_weight

    def forward(self, input, target):
        dog_input = self.dog_filter(input)
        dog_output = self.dog_filter(target)
        dog_loss = self.linear(torch.sum(torch.square(dog_input-dog_output)))
        return self.mse(input, target) + dog_loss
