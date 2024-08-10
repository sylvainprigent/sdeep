"""Classification of tensor data deep learning workflow"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from sdeep.utils import device
from sdeep.evals import Eval

from .base import SWorkflow


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
class ClassificationWorkflow(SWorkflow):
    """Workflow to train and predict a classification neural network

    :param model: Neural network model
    :param loss_fn: Training loss function
    :param optimizer: Back propagation optimizer
    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param train_batch_size: Size of a training batch
    :param val_batch_size: Size of a validation batch
    :param epochs: Number of epoch for training
    :param num_workers: Number of workers for data loading
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
                 save_all: bool = False):
        super().__init__(model, loss_fn, optimizer, train_dataset,
                         val_dataset, evaluate, train_batch_size, val_batch_size,
                         epochs, num_workers, save_all)

    def after_train(self):
        """Instructions runs after the train."""
        SWorkflow.after_train(self)

        # create the output dir
        predictions_dir = Path(self.out_dir, 'predictions')
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # predict on all the test set
        self.model.eval()

        table_data = "data,y_pred,y_true\n"
        for x, y, names in self.val_data_loader:
            x = x.to(device())
            with torch.no_grad():
                prediction = self.model(x)
            for i, name in enumerate(names):
                p_value = torch.argmax(prediction[i, ...])
                # if prediction.shape[1] > 1:
                #     p_value = torch.argmax(prediction[i, ...])
                # else:
                #     p_value = prediction[i, ...]

                table_data += (f"{name},{p_value.cpu().numpy().item()},"
                               f"{y[i,...].cpu().numpy().item()}\n")

        with open(predictions_dir / "prediction.csv", "w", encoding='utf-8') as file:
            file.write(table_data)


export = [ClassificationWorkflow]
