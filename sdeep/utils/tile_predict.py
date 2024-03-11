"""Tiling strategy to run an image prediction on large image"""
from typing import Tuple
from typing import List

import torch
import torch.nn.functional as F


class TilePredict:
    """Predict a model on a data using tiling

    The data is partitioned in overlapping tiles and the model predict
    each tile independently. Then the final image is reconstructed by
    recombining predicted tiles. In the 'crop' mode, overlapping regions are
    discarded and in the mean mode, overlapping regions are averaged

    :param model: nn.Module
        Model used to predict the result
    kernel_size: int
        width of the square kernel. None by default meaning that the kernel
        size is automatically calculated using a heuristic
    stride: int
        Stride between two tiles. None by default, meaning that it is half the
        model receptive field that is used
    """

    def __init__(self,
                 model: torch.nn.Module,
                 kernel_size: int = None,
                 stride: int = None):
        self.model = model
        self.kernel_size = kernel_size
        self.stride = stride
        if kernel_size is None or stride is None:
            self.__estimate_kernel_stride()

    def __estimate_kernel_stride(self):

        if self.model.receptive_field <= 256:
            self.kernel_size = int(256 / self.model.receptive_field) * \
                               self.model.receptive_field
        else:
            self.kernel_size = self.model.receptive_field
        self.stride = int(self.model.receptive_field / 2)

    def __calculate_pad_left_right(self,
                                   width: int,
                                   den: int
                                   ) -> Tuple[int, int]:
        """Calculate left and right padding

        :param width: image width
        :param den: kernel width
        :return: padding values
        """
        n_patch_x = width / den
        margin_x = width - int(n_patch_x) * den
        margin_x_left = int(margin_x / 2)
        margin_x_right = margin_x - margin_x_left
        pad_x_left = self.stride + (
                    self.kernel_size - self.stride) - margin_x_left
        pad_x_right = self.stride + (
                    self.kernel_size - self.stride) - margin_x_right
        return pad_x_left, pad_x_right

    def __calculate_pad_top_bottom(self,
                                   height: int,
                                   den: int
                                   ) -> Tuple[int, int]:
        """Calculate left and right padding

        :param height: image height
        :param den: kernel width
        :return: padding values
        """
        margin_y = height - int(height / den) * den
        margin_y_top = int(margin_y / 2)
        margin_y_bottom = margin_y - margin_y_top
        pad_y_top = self.stride + (
                self.kernel_size - self.stride) - margin_y_top
        pad_y_bottom = self.stride + (
                self.kernel_size - self.stride) - margin_y_bottom
        return pad_y_top, pad_y_bottom

    def __pad(self,
              image: torch.Tensor
              ) -> Tuple[torch.Tensor, List[int]]:
        """Add a padding to the original image

        This padding allows to fit an entire set of tile in the image

        :param image: Image to process. The size must be (batch, channel,
        width, height)
        :return: tuple: The padded image and the padding values

        """
        _, _, dim_w, dim_h = image.shape

        if self.kernel_size == self.stride:
            den = self.kernel_size
        else:
            den = self.kernel_size - self.stride

        pad_x_left, pad_x_right = self.__calculate_pad_left_right(dim_w, den)
        pad_y_top, pad_y_bottom = self.__calculate_pad_top_bottom(dim_h, den)

        return F.pad(image, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom),
                     mode='reflect'), \
            [pad_x_left, pad_x_right, pad_y_top, pad_y_bottom]

    def run(self, image: torch.Tensor):
        """Exec the prediction with tiling

        :param image: Image to process. The size must be (batch, channel,
                      width, height)
        :return: ndarray: The processed image
        """
        pad_image, padding = self.__pad(image)

        # predict
        output_pad = self.run_crop(pad_image)

        # remove pad
        offset = int((self.kernel_size - self.stride) / 2)
        return output_pad[:, :,
               padding[0] - offset:output_pad.shape[2] - padding[1] + offset,
               padding[2] - offset:output_pad.shape[3] - padding[3] + offset]

    def run_crop(self, image: torch.Tensor):
        """Exec the prediction with tiling using the crop mode

        :param image: Image to process. The size must be (batch, channel,
                      width, height)
        :return: torch.Tensor: The processed image
        """
        dim_b, dim_c, dim_w, dim_h = image.shape

        unfold = torch.nn.Unfold(self.kernel_size, dilation=1, padding=0,
                                 stride=self.stride)
        patches = unfold(image)
        dim_l = patches.shape[2]
        patches = patches.contiguous().view(dim_b, dim_c, self.kernel_size,
                                            self.kernel_size, dim_l)

        # perform the operations on each patch
        for i in range(dim_l):
            with torch.no_grad():
                patches[:, :, :, :, i] = self.model(patches[:, :, :, :, i])

        half_overlap = int((self.kernel_size - self.stride) / 2)
        start = half_overlap
        end = self.kernel_size - half_overlap
        crop_size = end - start
        patches = patches[:, :, start:end, start:end, :]

        patches = patches.contiguous().view(dim_b,
                                            dim_c * crop_size * crop_size,
                                            dim_l)
        output = F.fold(
            patches, output_size=(dim_h - 2 * half_overlap,
                                  dim_w - 2 * half_overlap),
            kernel_size=crop_size, stride=crop_size)
        # print(output.shape)  # [B, C, H, W]
        return output
