"""Workflow to apply Self supervised learning"""
import os
from timeit import default_timer as timer

import numpy as np
from skimage.io import imsave

import torch

from ..utils import device

from ..interfaces import SModel
from ..interfaces import SEval
from ..interfaces import SDataset

from .base import SWorkflowBase


def generate_2d_points(shape: tuple[int, int], n_point: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random 2D coordinates to mask

    :param shape: Shape of the image to mask
    :param n_point: Number of coordinates to mask
    :return: (y, x) coordinates to mask
    """
    idy_msk = np.random.randint(0, int(shape[0]/2), n_point)
    idx_msk = np.random.randint(0, int(shape[1]/2), n_point)

    idy_msk = 2*idy_msk
    idx_msk = 2*idx_msk
    if np.random.randint(2) == 1:
        idy_msk += 1
    if np.random.randint(2) == 1:
        idx_msk += 1

    return idy_msk, idx_msk


def generate_mask(img_shape: tuple[int, int], ratio: float) -> torch.Tensor:
    """Generate a zero spots mask fot the patch image

    :param img_shape: Shape of the image to mask
    :param ratio: Ratio of blind spots for input patch masking
    :return: the transformed image and the mask image
    """
    num_sample = int(img_shape[0] * img_shape[1] * ratio)
    mask = torch.zeros((img_shape[0], img_shape[1]), dtype=torch.float32)

    idy_msk, idx_msk = generate_2d_points((img_shape[0], img_shape[1]), num_sample)
    id_msk = (idy_msk, idx_msk)

    mask[id_msk] = 1.0

    return mask


def generate_mask_n2v(image: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a blind spots mask fot the patch image by randomly switch pixels values

    :param image: Image patch to add blind spots
    :param ratio: Ratio of blind spots for input patch masking
    :return: the transformed image and the mask image
    """
    img_shape = image.shape
    size_window = (5, 5)
    num_sample = int(img_shape[-2] * img_shape[-1] * ratio)

    mask = torch.zeros((img_shape[-2], img_shape[-1]), dtype=torch.float32)
    output = image.clone()

    idy_msk, idx_msk = generate_2d_points((img_shape[-2], img_shape[-1]), num_sample)
    num_sample = len(idy_msk)

    idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                  size_window[0] // 2 + size_window[0] % 2,
                                  num_sample)
    idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                  size_window[1] // 2 + size_window[1] % 2,
                                  num_sample)

    idy_msk_neigh = idy_msk + idy_neigh
    idx_msk_neigh = idx_msk + idx_neigh

    idy_msk_neigh = (idy_msk_neigh + (idy_msk_neigh < 0) * size_window[0] -
                     (idy_msk_neigh >= img_shape[-2]) * size_window[0])
    idx_msk_neigh = (idx_msk_neigh + (idx_msk_neigh < 0) * size_window[1] -
                     (idx_msk_neigh >= img_shape[-1]) * size_window[1])

    id_msk = (idy_msk, idx_msk)
    id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

    output[:, :, *id_msk] = image[:, :, *id_msk_neigh]
    mask[id_msk] = 1.0

    return output, mask


def generate_mask_n2i(image: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a blind spots mask fot the patch image and interpolate the masked pixel by it
       neighborhood

    :param image: Image patch to add blind spots
    :param ratio: Ratio of blind spots for input patch masking
    :return: the transformed image and the mask image
    """
    img_shape = image.shape
    size_window = (5, 5)
    num_sample = int(img_shape[-2] * img_shape[-1] * ratio)

    mask = torch.zeros((img_shape[-2], img_shape[-1]), dtype=torch.float32)
    output = image.clone()
    idy_msk, idx_msk = generate_2d_points((img_shape[-2], img_shape[-1]), num_sample)

    # interpolate the masked pixels
    output[:, :, idy_msk, idx_msk] = 0
    for i in np.arange(-int(size_window[0]/2), int(size_window[0]/2)+1):
        for j in np.arange(-int(size_window[1]/2), int(size_window[1]/2)+1):
            idy = idy_msk + i
            idx = idx_msk + j
            idy = idy + (idy < 0) * size_window[0] - (idy >= img_shape[-2]) * size_window[0]
            idx = idx + (idx < 0) * size_window[1] - (idx >= img_shape[-1]) * size_window[1]
            output[:, :, idy_msk, idx_msk] += image[:, :, idy, idx]

    output[:, :, idy_msk, idx_msk] /= size_window[0]*size_window[1]

    mask[idy_msk, idx_msk] = 1.0

    return output, mask


def generate_mask_n2s(image: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a blind spots mask by setting to zero random pixels

    :param image: Image to mask,
    :param ratio: Ratio of pixels to mask,
    :return: The masked image and the invert mask
    """
    mask = generate_mask((image.shape[-2], image.shape[-1]), ratio)
    return image * (1 - mask), mask


class SelfSupervisedWorkflow(SWorkflowBase):
    """Workflow to train and predict a restoration neural network

    :param model: Neural network model
    :param loss_fn: Training loss function
    :param optimizer: Back propagation optimizer
    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param evaluate: Evaluation method
    :param train_batch_size: Size of a training batch
    :param val_batch_size: Size of a validation batch
    :param epochs: Number of epoch for training
    :param mask_type: Masking strategy (n2v, n2s)
    """
    def __init__(self,
                 model: SModel,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.nn.Module,
                 train_dataset: SDataset,
                 val_dataset: SDataset,
                 evaluate: SEval,
                 train_batch_size: int,
                 val_batch_size: int,
                 epochs: int = 50,
                 num_workers: int = 0,
                 mask_type: str = "n2v"):
        super().__init__(model, loss_fn, optimizer, train_dataset, val_dataset, evaluate,
                         train_batch_size, val_batch_size, epochs, num_workers)
        self.mask_type = mask_type
        self.ratio = 0.1

    def apply_mask(self, x: torch.Tensor, ratio: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate and apply mask to x input

        :param x: input batch to mask
        :param ratio: Ratio of pixels to mask
        :return: the masked input and the mask
        """
        if self.mask_type == 'n2v':
            return generate_mask_n2v(x, ratio)
        if self.mask_type == 'n2s':
            return generate_mask_n2s(x, ratio)
        if self.mask_type == 'n2i':
            return generate_mask_n2i(x, ratio)
        raise ValueError("'mask_type' not recognized must be: 'n2v', 'n2s' or 'n2i'")

    def train_step(self):
        """Runs one step of training"""
        size = len(self.train_data_loader.dataset)
        self.model_torch.train()
        step_loss = 0
        count_step = 0
        tic = timer()
        for batch, (x, _) in enumerate(self.train_data_loader):
            count_step += 1

            masked_x, mask = self.apply_mask(x, self.ratio)
            x, masked_x, mask = x.to(device()), masked_x.to(device()), mask.to(device())

            # Compute prediction error
            prediction = self.model_torch(masked_x)
            loss = self.loss_fn(prediction, x, mask)
            step_loss += loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # count time
            toc = timer()
            full_time = toc - tic
            total_batch = int(size / len(x))
            remains = full_time * (total_batch - (batch+1)) / (batch+1)

            self.after_train_batch({'loss': loss,
                                    'id_batch': batch+1,
                                    'total_batch': total_batch,
                                    'remain_time': int(remains+0.5),
                                    'full_time': int(full_time+0.5)
                                    })

        if count_step > 0:
            step_loss /= count_step
        self.current_loss = step_loss
        return {'train_loss': step_loss}

    def val_step(self):
        """Runs one step of validation

        Returns
        -------
        A dictionary of data to save/log/process
        This dictionary must contain at least the val_loss entry

        """
        num_batches = len(self.val_data_loader)
        self.model_torch.eval()
        print('')
        val_loss = 0
        for x, _ in self.val_data_loader:

            masked_x, mask = self.apply_mask(x, self.ratio)
            x, masked_x, mask = x.to(device()), masked_x.to(device()), mask.to(device())

            with torch.no_grad():
                prediction = self.model_torch(masked_x)
            val_loss += self.loss_fn(prediction, x, mask).item()
        val_loss /= num_batches
        return {'val_loss': val_loss}

    def after_train(self):
        """Instructions runs after the train."""
        SWorkflowBase.after_train(self)

        # create the output dir
        predictions_dir = os.path.join(self.out_dir, 'predictions')
        if os.path.isdir(self.out_dir):
            os.mkdir(predictions_dir)

        # predict on all the test set
        self.model_torch.eval()
        for x, names in self.val_data_loader:
            x = x.to(device())

            with torch.no_grad():
                prediction = self.model_torch(x)
            for i, name in enumerate(names):
                imsave(os.path.join(predictions_dir, name + ".tif"),
                       prediction[i, ...].cpu().numpy())


export = [SelfSupervisedWorkflow]
