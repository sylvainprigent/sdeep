"""Restoration deep learning workflow"""
import os
from skimage.io import imsave
import torch


from torch.utils.data import Dataset

from sdeep.utils import device
from sdeep.utils import TilePredict
from sdeep.evals import Eval

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
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 evaluate: Eval,
                 train_batch_size: int,
                 val_batch_size: int,
                 epochs: int = 50,
                 num_workers: int = 0,
                 save_all: bool = False,
                 use_tiling=False):
        super().__init__(model, loss_fn, optimizer, train_dataset,
                         val_dataset, evaluate, train_batch_size, val_batch_size, epochs,
                         num_workers, save_all)
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
            x, y = x.to(device()), y.to(device())
            if self.use_tiling:
                tile_predict = TilePredict(self.model)
                prediction = tile_predict.run(x)
            else:
                with torch.no_grad():
                    prediction = self.model(x)
            val_loss += self.loss_fn(prediction, y).item()
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
            x = x.to(device())
            if self.use_tiling:
                tile_predict = TilePredict(self.model)
                prediction = tile_predict.run(x)
            else:
                with torch.no_grad():
                    prediction = self.model(x)
            for i, name in enumerate(names):
                imsave(os.path.join(predictions_dir, f"{name}.tif"),
                       prediction[i, :, :].cpu().numpy())


export = [SegmentationWorkflow]
