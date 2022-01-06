import math
import numpy as np
import torch
import torch.nn.functional as F


class TilePredict:
    """Predict a model on a data using tiling

    The data is partitioned in overlapping tiles and the model predict
    each tile independently. Then the final image is reconstructed by recombining
    predicted tiles. The overlapping regions are discarded.

    Parameters
    ----------
    model: nn.Module
        Model used to predict the result
    kernel_size: int
        width of the square kernel
    stride: int
        Stride between two tiles

    """
    def __init__(self, model, kernel_size=256, stride=128):
        self.model = model
        self.kernel_size = kernel_size
        self.stride = stride

    def _pad(self, image, mode):
        B, C, W, H = image.shape

        # calculate padding x
        if self.kernel_size == self.stride:
            den = self.kernel_size
        else:
            den = self.kernel_size - self.stride

        if mode == 'mean':
            #n_patch_x = (W - self.stride) / den
            n_patch_x = W / den
        else:
            n_patch_x = W / den
        margin_x = W - int(n_patch_x) * den
        margin_x_left = int(margin_x / 2)
        margin_x_right = margin_x - margin_x_left
        pad_x_left = self.stride + (self.kernel_size - self.stride) - margin_x_left
        pad_x_right = self.stride + (self.kernel_size - self.stride) - margin_x_right

        # calculate padding y
        if mode == 'mean':
            #n_patch_y = (H - self.stride) / den
            n_patch_y = H / den
        else:
            n_patch_y = H / den
        margin_y = H - int(n_patch_y) * den
        margin_y_top = int(margin_y / 2)
        margin_y_bottom = margin_y - margin_y_top
        pad_y_top = self.stride + (self.kernel_size - self.stride) - margin_y_top
        pad_y_bottom = self.stride + (self.kernel_size - self.stride) - margin_y_bottom

        pad_image = F.pad(image, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom), mode='reflect')
        return pad_image, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom)

    def run(self, image, mode='crop'):

        pad_image, padding = self._pad(image, mode)

        if mode == 'crop':
            # predict
            output_pad = self.run_crop(pad_image)
            # remove pad
            offset = int((self.kernel_size - self.stride)/2)
            return output_pad[:, :, padding[0]-offset:output_pad.shape[2]-padding[1]+offset,
                              padding[2]-offset:output_pad.shape[3]-padding[3]+offset]
        elif mode == 'mean':
            # predict
            output_pad = self.run_mean(pad_image)
            # remove pad
            return output_pad[:, :, padding[0]:output_pad.shape[2]-padding[1],
                              padding[2]:output_pad.shape[3]-padding[3]]

    def run_crop(self, image):

        B, C, W, H = image.shape

        unfold = torch.nn.Unfold(self.kernel_size, dilation=1, padding=0, stride=self.stride)
        patches = unfold(image)
        print("unfold=", patches.shape)
        L = patches.shape[2]
        patches = patches.contiguous().view(B, C, self.kernel_size, self.kernel_size, L)
        print("unfold reshape=", patches.shape)

        # perform the operations on each patch
        for i in range(L):
            with torch.no_grad():
                patches[:, :, :, :, i] = self.model(patches[:, :, :, :, i])

        half_overlap = int((self.kernel_size-self.stride)/2)
        start = half_overlap
        end = self.kernel_size-half_overlap
        crop_size = end-start
        patches = patches[:, :, start:end, start:end, :]
        print("cropped=", patches.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
        print("crop size=", crop_size)

        patches = patches.contiguous().view(B, C*crop_size*crop_size, L)
        output = F.fold(
            patches, output_size=(H-2*half_overlap, W-2*half_overlap), kernel_size=crop_size, stride=crop_size)
        print(output.shape)  # [B, C, H, W]
        return output

    def run_mean(self, image):
        B, C, W, H = image.shape
        print('padded image shape=', image.shape)

        unfold = torch.nn.Unfold(self.kernel_size, dilation=1, padding=0, stride=self.stride)
        patches = unfold(image)
        print("unfold=", patches.shape)
        L = patches.shape[2]
        patches = patches.contiguous().view(B, C, self.kernel_size, self.kernel_size, L)
        print("unfold reshape=", patches.shape)

        # perform the operations on each patch
        for i in range(L):
            with torch.no_grad():
                patches[:, :, :, :, i] = self.model(patches[:, :, :, :, i])

        patches = patches.contiguous().view(B, C*self.kernel_size*self.kernel_size, L)

        output = F.fold(
            patches, output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride)
        print(output.shape)  # [B, C, H, W]

        # transform the sum of nn.Fold into mean using mask
        recovery_mask = F.fold(torch.ones_like(patches), output_size=(
            H, W), kernel_size=self.kernel_size, stride=self.stride)
        return output/recovery_mask
