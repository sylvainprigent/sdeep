import torch
import torch.nn.functional as F


class TilePredict:
    """Predict a model on a data using tiling

    The data is partitioned in overlapping tiles and the model predict
    each tile independently. Then the final image is reconstructed by recombining
    predicted tiles. In the 'crop' mode, overlapping regions are discarded and in the mean mode,
    overlapping regions are averaged

    Parameters
    ----------
    model: nn.Module
        Model used to predict the result
    kernel_size: int
        width of the square kernel. None by default meaning that the kernel size is automatically
        calculated using a heuristic
    stride: int
        Stride between two tiles. None by default, meaning that it is half the model receptive
        field that is used

    """
    def __init__(self, model, kernel_size=None, stride=None):
        self.model = model
        self.kernel_size = kernel_size
        self.stride = stride
        if kernel_size is None or stride is None:
            self._estimate_kernel_stride()

        print('kernel_size=', self.kernel_size)
        print('stride=', self.stride)

    def _estimate_kernel_stride(self):

        if self.model.receptive_field <= 256:
            self.kernel_size = int(256/self.model.receptive_field)*self.model.receptive_field
        else:
            self.kernel_size = self.model.receptive_field
        self.stride = int(self.model.receptive_field/2)

    def _pad(self, image):
        """Add a padding to the original image

        This padding allows to fit an entire set of tile in the image

        Parameters
        ----------
        image: ndarray
            Image to process. The size must be (batch, channel, width, height)

        Return
        ------
        tuple: The padded image and the padding values

        """
        B, C, W, H = image.shape

        # calculate padding x
        if self.kernel_size == self.stride:
            den = self.kernel_size
        else:
            den = self.kernel_size - self.stride

        n_patch_x = W / den
        margin_x = W - int(n_patch_x) * den
        margin_x_left = int(margin_x / 2)
        margin_x_right = margin_x - margin_x_left
        pad_x_left = self.stride + (self.kernel_size - self.stride) - margin_x_left
        pad_x_right = self.stride + (self.kernel_size - self.stride) - margin_x_right

        # calculate padding y
        n_patch_y = H / den
        margin_y = H - int(n_patch_y) * den
        margin_y_top = int(margin_y / 2)
        margin_y_bottom = margin_y - margin_y_top
        pad_y_top = self.stride + (self.kernel_size - self.stride) - margin_y_top
        pad_y_bottom = self.stride + (self.kernel_size - self.stride) - margin_y_bottom

        pad_image = F.pad(image, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom), mode='reflect')
        return pad_image, (pad_x_left, pad_x_right, pad_y_top, pad_y_bottom)

    def run(self, image):
        """Exec the prediction with tiling

        Parameters
        ----------
        image: ndarray
            Image to process. The size must be (batch, channel, width, height)

        Return
        ------
        ndarray: The processed image

        """
        pad_image, padding = self._pad(image)

        # predict
        output_pad = self._run_crop(pad_image)

        # remove pad
        offset = int((self.kernel_size - self.stride)/2)
        return output_pad[:, :, padding[0]-offset:output_pad.shape[2]-padding[1]+offset,
                          padding[2]-offset:output_pad.shape[3]-padding[3]+offset]

    def _run_crop(self, image):
        """Exec the prediction with tiling using the crop mode

        Parameters
        ----------
        image: ndarray
            Image to process. The size must be (batch, channel, width, height)

        Return
        ------
        ndarray: The processed image

        """
        B, C, W, H = image.shape

        unfold = torch.nn.Unfold(self.kernel_size, dilation=1, padding=0, stride=self.stride)
        patches = unfold(image)
        L = patches.shape[2]
        patches = patches.contiguous().view(B, C, self.kernel_size, self.kernel_size, L)

        # perform the operations on each patch
        for i in range(L):
            with torch.no_grad():
                patches[:, :, :, :, i] = self.model(patches[:, :, :, :, i])

        half_overlap = int((self.kernel_size-self.stride)/2)
        start = half_overlap
        end = self.kernel_size-half_overlap
        crop_size = end-start
        patches = patches[:, :, start:end, start:end, :]

        patches = patches.contiguous().view(B, C*crop_size*crop_size, L)
        output = F.fold(
            patches, output_size=(H-2*half_overlap, W-2*half_overlap), kernel_size=crop_size, stride=crop_size)
        # print(output.shape)  # [B, C, H, W]
        return output
