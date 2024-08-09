"""SDeep workflow interface definition and base workflows implementation

Classes
-------
SWorkflow

"""
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sdeep.utils import device, SProgressObservable, SDataLogger
from sdeep.utils.progress_loggers import SProgressLogger
from sdeep.utils.utils import seconds2str
from sdeep.utils.io import save_model

from sdeep.evals import Eval


class SWorkflow:
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

        self.model = model.to(device())
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.evaluate = evaluate
        self.train_data_loader = DataLoader(train_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=num_workers)
        self.val_data_loader = DataLoader(val_dataset,
                                          batch_size=val_batch_size,
                                          shuffle=False,
                                          drop_last=False,
                                          num_workers=0)
        self.epochs = epochs
        self.save_all = save_all

        self.logger = None
        self.progress = SProgressObservable()
        self.progress.prefix = 'SWorkflow'

        self.out_dir = ''

        self.current_epoch = 0
        self.current_loss = -1

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

    def before_train(self):
        """Instructions runs before the train.

        This method can be used to log data or print console messages
        """
        num_parameters = sum(p.numel() for p in
                             self.model.parameters() if p.requires_grad)

        if hasattr(self.model, 'input_shape'):
            dummy_input = torch.rand([1, 1, *self.model.input_shape]).to(device())
            self.logger.add_graph(self.model, dummy_input)
            self.logger.flush()
        self.progress.message(f"Using {device()} device")
        self.progress.message(f"Model number of parameters: "
                              f"{num_parameters:d}")

        checkpoint_file = Path(self.out_dir, 'checkpoint.ckpt')
        if checkpoint_file.is_file():
            self.progress.message("Initialize training from checkpoint")
            self.load_checkpoint(checkpoint_file)

    def after_train(self):
        """Instructions runs after the train.

        This method can be used to log data or print console messages
        """
        if self.evaluate:
            out_dir = Path(self.out_dir, "evals", "final")
            out_dir.mkdir(parents=True)

            self.model.eval()
            self.evaluate.clear()
            with torch.no_grad():
                for x, y, idx in self.val_data_loader:
                    x, y = x.to(device()), y.to(device())
                    prediction = self.model(x)
                    for i, id_ in enumerate(idx):
                        self.evaluate.eval_step(prediction[i, ...], y[i, ...], id_, out_dir)

            self.evaluate.eval(out_dir)

        self.progress.new_line()
        self.logger.close()

    def after_train_step(self, data: dict):
        """Instructions runs after one train step.

        This method can be used to log data or print console messages

        :param data: Dictionary of metadata to log or process
        """
        self.logger.add_scalar('train_loss',
                               data['train_loss'],
                               self.current_epoch)
        self.save_checkpoint()
        if self.save_all:
            save_model(self.model,
                       self.model.args,
                       Path(self.out_dir, f'model_{self.current_epoch}.ml'))

    def after_train_batch(self, data: dict[str, any]):
        """Instructions runs after one batch

        :param data: Dictionary of metadata to log or process
        """
        prefix = f"Epoch = {self.current_epoch:d}"
        loss_str = f"{data['loss']:.7f}"
        full_time_str = seconds2str(int(data['full_time']))
        remains_str = seconds2str(int(data['remain_time']))
        suffix = str(data['id_batch']) + '/' + str(data['total_batch']) + \
            '   [' + full_time_str + '<' + remains_str + ', loss=' + \
            loss_str + ']     '
        self.progress.progress(data['id_batch'],
                               data['total_batch'],
                               prefix=prefix,
                               suffix=suffix)

    def train_step(self):
        """Runs one step of training"""
        size = len(self.train_data_loader.dataset)
        self.model.train()
        step_loss = 0
        full_time = 0
        count_step = 0
        tic = timer()
        for batch, (x, y, _) in enumerate(self.train_data_loader):
            count_step += 1
            x, y = x.to(device()), y.to(device())

            # Compute prediction error
            prediction = self.model(x)

            loss = self.loss_fn(prediction, y)
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

    def after_val_step(self, data):
        """Instructions runs after one train step.

        This method can be used to log data or print console messages

        Parameters
        ----------
        data: dict
            Dictionary of metadata to log or process
        """
        # just loss by default but add a code here to save the crop results on
        # tensor board or any output folder ?
        self.logger.add_scalar('val_loss', data['val_loss'],
                               self.current_epoch)

    def val_step(self):
        """Runs one step of validation

        :returns: A dictionary of data to save/log/process, t
                  This dictionary must contain at least the val_loss entry

        """
        out_dir = Path(self.out_dir, "evals", f"epoch_{self.current_epoch}")
        out_dir.mkdir(parents=True)

        num_batches = len(self.val_data_loader)
        self.model.eval()
        val_loss = 0
        if self.save_all and self.evaluate:
            self.evaluate.clear()
        with torch.no_grad():
            for x, y, idx in self.val_data_loader:
                x, y = x.to(device()), y.to(device())
                prediction = self.model(x)
                val_loss += self.loss_fn(prediction, y).item()
                if self.save_all and self.evaluate:
                    for i, id_ in enumerate(idx):
                        self.evaluate.eval_step(prediction[i, ...], y[i, ...], id_, out_dir)
        val_loss /= num_batches

        if self.save_all and self.evaluate:
            self.evaluate.eval(out_dir)

        return {'val_loss': val_loss}

    def train(self):
        """Train the model

        Main training loop.

        """
        self.before_train()
        for epoch in range(self.current_epoch, self.epochs, 1):
            self.current_epoch = epoch
            train_data = self.train_step()
            self.after_train_step(train_data)
            if self.val_data_loader:
                val_data = self.val_step()
                self.after_val_step(val_data)
        self.after_train()

    def fit(self):
        """API function

        For the API it is more readable to use fit than train
        """
        self.train()

    def save_checkpoint(self):
        """Save the model weights as a checkpoint"""
        if self.out_dir != '':
            self.save_checkpoint_to_file(Path(self.out_dir, 'checkpoint.ckpt'))

    def save_checkpoint_to_file(self, path):
        """Save a checkpoint at a given epoch to a file

        This method purpose is to save a checkpoint at each epoch in case
        the training crash and restart from the previous epoch

        """
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.current_loss,
            }, path)

    def load_checkpoint(self, path):
        """Initialize the training for a checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_loss = checkpoint['loss']

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a single input

        :param x: input data. The data is considered in CPU and is moved to GPU if needed.
        :return: the prediction
        """
        x_device = x.to(device()).unsqueeze(0).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x_device)
        return prediction


export = [SWorkflow]
