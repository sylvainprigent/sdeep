"""Restoration deep learning workflow"""
import os
from skimage.io import imsave
import torch

from sdeep.utils import TilePredict
from .base import SWorkflow


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
class SegmentationWorkflow(SWorkflow):
    """Workflow to train and predict a semantic segmentation neural network

    :param model: Neural network model
    :param loss_fn: Training loss function
    :param optimizer: Back propagation optimizer
    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param train_batch_size: Size of a training batch
    :param val_batch_size: Size of a validation batch
    :param epochs: Number of epoch for training
    :param use_tiling: use tiling or not for prediction
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 train_dataset,
                 val_dataset,
                 train_batch_size,
                 val_batch_size,
                 epochs=50,
                 use_tiling=False):
        super().__init__(model, loss_fn, optimizer, train_dataset,
                         val_dataset, train_batch_size, val_batch_size, epochs)
        self.use_tiling = use_tiling

    def val_step(self):
        """Runs one step of validation

        Returns
        -------
        A dictionary of data to save/log/process
        This dictionary must contain at least the val_loss entry

        """
        num_batches = len(self.val_data_loader)
        self.model.eval()
        # print('val step use tiling=', self.use_tiling)
        print("")
        val_loss = 0
        for x, y, _ in self.val_data_loader:
            x, y = x.to(self.device), y.to(self.device)
            if self.use_tiling:
                tile_predict = TilePredict(self.model)
                pred = tile_predict.run(x)
            else:
                with torch.no_grad():
                    pred = self.model(x)
            val_loss += self.loss_fn(pred, y).item()
        val_loss /= num_batches
        return {'val_loss': val_loss}

    def after_train(self):
        """Instructions runs after the train."""
        SWorkflow.after_train(self)

        # create the output dir
        predictions_dir = os.path.join(self.out_dir, 'predictions')
        if os.path.isdir(self.out_dir):
            os.mkdir(predictions_dir)

        # predict on all the test set
        self.model.eval()
        for x, _, names in self.val_data_loader:
            x = x.to(self.device)
            if self.use_tiling:
                tile_predict = TilePredict(self.model)
                pred = tile_predict.run(x)
            else:
                with torch.no_grad():
                    pred = self.model(x)
            for i, name in enumerate(names):
                imsave(os.path.join(predictions_dir, name),
                       pred[i, :, :].cpu().numpy())


export = [SegmentationWorkflow]
