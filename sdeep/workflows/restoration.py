"""Restoration deep learning workflow"""
from pathlib import Path
from skimage.io import imsave

import torch

from ..utils import TilePredict
from ..utils import device

from ..interfaces import SModel
from ..interfaces import SEval
from ..interfaces import SDataset

from .base import SWorkflowBase


class RestorationWorkflow(SWorkflowBase):
    """Workflow to train and predict a restoration neural network

    :param model: Neural network model
    :param loss_fn: Training loss function
    :param optimizer: Back propagation optimizer
    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param train_batch_size: Size of a training batch
    :param val_batch_size: Size of a validation batch
    :param epochs: Number of epoch for training
    :param num_workers: Number of workers for data loading
    :param use_tiling: use tiling or not for prediction
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
                 save_all: bool = False,
                 use_tiling=False):
        super().__init__(model, loss_fn, optimizer, train_dataset,
                         val_dataset, evaluate, train_batch_size, val_batch_size,
                         epochs, num_workers, save_all)
        self.use_tiling = use_tiling

    def val_step(self):
        """Runs one step of validation

        Returns
        -------
        A dictionary of data to save/log/process
        This dictionary must contain at least the val_loss entry
        """
        out_dir = Path(self.out_dir, "evals", f"epoch_{self.current_epoch}")
        out_dir.mkdir(parents=True)

        num_batches = len(self.val_data_loader)
        self.model_torch.eval()
        val_loss = 0
        if self.save_all:
            self.evaluate.clear()

        for x, y, idx in self.val_data_loader:
            x, y = x.to(device()), y.to(device())
            if self.use_tiling:
                tile_predict = TilePredict(self.model_torch)
                prediction = tile_predict.run(x)
            else:
                with torch.no_grad():
                    prediction = self.model_torch(x)
            val_loss += self.loss_fn(prediction, y).item()
            if self.save_all:
                for i, id_ in enumerate(idx):
                    self.evaluate.eval_step(prediction[i, ...], y[i, ...], id_, out_dir)
        val_loss /= num_batches

        if self.save_all:
            self.evaluate.eval(out_dir)

        return {'val_loss': val_loss}

    def after_train(self):
        """Instructions runs after the train."""
        SWorkflowBase.after_train(self)

        # create the output dir
        predictions_dir = Path(self.out_dir, 'predictions')
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # predict on all the test set
        self.model_torch.eval()
        for x, _, names in self.val_data_loader:
            x = x.to(device())
            if self.use_tiling:
                tile_predict = TilePredict(self.model_torch)
                prediction = tile_predict.run(x)
            else:
                with torch.no_grad():
                    prediction = self.model_torch(x)
            for i, name in enumerate(names):
                imsave(Path(predictions_dir, name + ".tif"),
                       prediction[i, ...].cpu().numpy())


export = [RestorationWorkflow]
