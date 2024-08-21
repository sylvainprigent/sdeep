"""Interface for training workflow"""
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from ..interfaces import SModel
from ..interfaces import SDataset
from ..interfaces import SEval

from ..utils import device
from ..utils import SProgressObservable
from ..utils import SDataLogger
from ..utils.progress_loggers import SProgressLogger


class SWorkflow(ABC):
    """Default workflow to train and predict a neural network

    :param model: Neural network model
    :param loss_fn: Training loss function
    :param optimizer: Back propagation optimizer
    :param train_dataset: Training dataset
    :param val_dataset: Validation dataset
    :param evaluate: Evaluation method
    :param train_batch_size: Size of a training batch
    :param val_batch_size: Size of a validation batch
    :param epochs: Number of epoch for training
    :param num_workers: Number of workers for data loading
    :param save_all: Save model and run evals to all epoch
    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
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
                 save_all: bool = False):

        self.model = model
        self.model.model.to(device())
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.evaluate = evaluate
        self.val_dataset = val_dataset
        self.train_data_loader = DataLoader(train_dataset.dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=num_workers)
        self.val_data_loader = DataLoader(val_dataset.dataset,
                                          batch_size=val_batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=0)
        self.epochs = epochs
        self.save_all = save_all

        self.logger = None
        self.progress = SProgressObservable()
        self.progress.prefix = 'SWorkflow'

    def set_data_logger(self, logger: SDataLogger):
        """Set the data logger to the workflow

        Parameters
        ----------
        logger: SDataLogger
            Preconfigured data logger

        """
        self.logger = logger

    def set_progress_observable(self, observable):
        """The progress logger observable

        Parameters
        ----------
        observable: SProgressObservable
            The progress observable instance

        """
        self.progress = observable
        self.progress.set_prefix(self.__class__.__name__)

    def add_progress_logger(self, logger: SProgressLogger):
        """Add one progress logger

        :param logger: Instance of a progress logger
        """
        logger.prefix = self.__class__.__name__
        self.progress.add_logger(logger)

    @abstractmethod
    def before_train(self):
        """Instructions runs before the train.

        This method can be used to log data or print console messages
        """

    @abstractmethod
    def after_train(self):
        """Instructions runs after the train.

        This method can be used to log data or print console messages
        """

    @abstractmethod
    def after_train_step(self, data: dict):
        """Instructions runs after one train step.

        This method can be used to log data or print console messages

        :param data: Dictionary of metadata to log or process
        """

    @abstractmethod
    def after_train_batch(self, data: dict[str, any]):
        """Instructions runs after one batch

        :param data: Dictionary of metadata to log or process
        """

    @abstractmethod
    def train_step(self):
        """Runs one step of training"""

    @abstractmethod
    def after_val_step(self, data):
        """Instructions runs after one train step.

        This method can be used to log data or print console messages

        Parameters
        ----------
        data: dict
            Dictionary of metadata to log or process
        """

    @abstractmethod
    def val_step(self):
        """Runs one step of validation

        :returns: A dictionary of data to save/log/process, t
                  This dictionary must contain at least the val_loss entry

        """

    @abstractmethod
    def train(self):
        """Train the model

        Main training loop.

        """

    def fit(self):
        """API function

        For the API it is more readable to use fit than train
        """
        self.train()

    @abstractmethod
    def save_checkpoint(self):
        """Save the model weights as a checkpoint"""

    @abstractmethod
    def save_checkpoint_to_file(self, path):
        """Save a checkpoint at a given epoch to a file

        This method purpose is to save a checkpoint at each epoch in case
        the training crash and restart from the previous epoch

        """

    @abstractmethod
    def load_checkpoint(self, path):
        """Initialize the training for a checkpoint"""
