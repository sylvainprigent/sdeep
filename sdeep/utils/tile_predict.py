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
    tile_shape: tuple
        Shape of one tile
    """
    def __init__(self, model, kernel_size=256, stride=128):
        self.model = model
        self.kernel_size = kernel_size
        self.stride = stride

    def _pad(self, image, mode):
        B, C, W, H = image.shape

        # calculate padding x
        if mode == 'mean':
            n_patch_x = (W - self.stride) / (self.kernel_size - self.stride)
        else:
            n_patch_x = W / (self.kernel_size - self.stride)
        margin_x = W - int(n_patch_x) * (self.kernel_size - self.stride)
        margin_x_left = int(margin_x / 2)
        margin_x_right = margin_x - margin_x_left
        pad_x_left = self.stride + (self.kernel_size - self.stride) - margin_x_left
        pad_x_right = self.stride + (self.kernel_size - self.stride) - margin_x_right

        # calculate padding y
        if mode == 'mean':
            n_patch_y = (H - self.stride) / (self.kernel_size - self.stride)
        else:
            n_patch_y = H / (self.kernel_size - self.stride)
        margin_y = H - int(n_patch_y) * (self.kernel_size - self.stride)
        margin_y_top = int(margin_y / 2)
        margin_y_bottom = margin_y - margin_y_top
        pad_y_top = self.stride + (self.kernel_size - self.stride) - margin_y_top
        pad_y_bottom = self.stride + (self.kernel_size - self.stride) - margin_y_bottom

        padding = torch.nn.ZeroPad2d((pad_x_left, pad_x_right, pad_y_top, pad_y_bottom))
        pad_image = padding(image)
        return pad_image, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom)

    def run(self, image, mode='crop'):

        pad_image, padding = self._pad(image, mode)

        if mode == 'crop':
            # predict
            output_pad = self.run_crop(pad_image)
            # remove pad
            offset = int(self.stride/2)
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
        print('padded image shape=', image.shape)
        patches = image.unfold(3, self.kernel_size, self.stride)
        print(patches.shape)
        patches = patches.unfold(2, self.kernel_size, self.stride)
        print(patches.shape)  # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]

        # perform the operations on each patch
        # ...

        # reshape output to match F.fold input
        patches = torch.transpose(patches, 4, 5)

        # crop to remove the overlap
        start = int(self.stride/2)
        end = self.kernel_size-int(self.stride/2)
        crop_size = end-start
        patches = patches[:, :, :, :, start:end, start:end]
        print("cropped=", patches.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
        print("crop size=", crop_size)

        patches = patches.contiguous().view(B, C, -1, crop_size * crop_size)
        print(patches.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
        patches = patches.permute(0, 1, 3, 2)
        print(patches.shape)  # [B, C, kernel_size*kernel_size, nb_patches_all]
        patches = patches.contiguous().view(B, C * crop_size * crop_size, -1)
        print(patches.shape)  # [B, C*prod(kernel_size), L] as expected by Fold

        output = F.fold(
            patches, output_size=(H-self.stride, W-self.stride), kernel_size=crop_size, stride=crop_size)
        print(output.shape)  # [B, C, H, W]
        return output

    def run_mean(self, image):
        B, C, W, H = image.shape
        print('padded image shape=', image.shape)
        patches = image.unfold(3, self.kernel_size, self.stride)
        print(patches.shape)
        patches = patches.unfold(2, self.kernel_size, self.stride)
        print(patches.shape)  # [B, C, nb_patches_h, nb_patches_w, kernel_size, kernel_size]

        # perform the operations on each patch
        # ...

        # reshape output to match F.fold input
        patches = torch.transpose(patches, 4, 5)
        patches = patches.contiguous().view(B, C, -1, self.kernel_size * self.kernel_size)
        print(patches.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
        patches = patches.permute(0, 1, 3, 2)
        print(patches.shape)  # [B, C, kernel_size*kernel_size, nb_patches_all]
        patches = patches.contiguous().view(B, C * self.kernel_size * self.kernel_size, -1)
        print(patches.shape)  # [B, C*prod(kernel_size), L] as expected by Fold

        output = F.fold(
            patches, output_size=(H, W), kernel_size=self.kernel_size, stride=self.stride)
        print(output.shape)  # [B, C, H, W]

        # transform the sum of nn.Fold into mean using mask
        recovery_mask = F.fold(torch.ones_like(patches), output_size=(
            H, W), kernel_size=self.kernel_size, stride=self.stride)
        return output/recovery_mask


    def predict(self, image):
        """Run the prediction

        Parameters
        ----------
        image: ndarray
            Input image for prediction.

        Returns
        -------
        The predicted output. Same shape as input image
        """
        # if no tiling
        if self.tile_shape is None:
            return self.model.predict(image)

        # predict with tiling
        #stride_x = int(3*self.tile_shape[0]/4)
        #stride_y = int(3*self.tile_shape[1]/4)

        #n_tiles_x = (image.shape[0] - self.tile_shape[0]) // stride_x
        #n_tiles_y = (image.shape[1] - self.tile_shape[1]) // stride_y

        tiles = image.reshape(image.shape[0] // self.tile_shape[0],
                              self.tile_shape[0],
                              image.shape[1] // self.tile_shape[1],
                              self.tile_shape[1]
                              )
        tiles = tiles.swapaxes(1, 2)

        print('tiles shape = ', tiles.shape)
        tiles_out = np.zeros(tiles.shape)
        for i in range(tiles.shape[0]):
            for j in range(tiles.shape[1]):
                tiles_out[i, j, :, :] = self.model.predict(tiles[i, j, :, :])

        output_image = tiles_out.swapaxes(1, 2).reshape(image.shape[0], image.shape[1])


        #output_image = np.zeros(image.shape)
        # for i in range(n_tiles_x):
        #    for j in range(n_tiles_y):
        #        # get the tile
        #        tile = image[i * stride_x:i * stride_x + self.tile_shape[0],
        #                     j * stride_y:j * stride_y + self.tile_shape[1]]
        #        # predict
        #        # todo: manage to(device) and squeeze
        #        p_tile = self.model.predict(tile)
        #        # put the tile in output
        #        # todo: manage stride
        #        output_image[i * stride_x:i * stride_x + self.tile_shape[0],
        #                     j * stride_y:j * stride_y + self.tile_shape[1]] = p_tile
        return output_image