"""SDeep workflow interface definition and base workflows implementation

Classes
-------
SWorkflow

"""

import time
import torch
from torch.utils.tensorboard import SummaryWriter

from sdeep.utils.progress import SProgressBar


class SWorkflow:
    """Default workflow to train and predict a neural network

    Parameters
    ----------
    model: nn.Module
        Neural network model
    loss_fn: nn.Module
        Training loss function
    optimizer: nn.Module
        Optimizer
    train_data_loader: DataLoader
        Data loader to iterate a training dataset
    val_data_loader : DataLoader
        Data loader to iterate a testing dataset

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self, model, loss_fn, optimizer, train_data_loader,
                 val_data_loader, epochs=50):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.epochs = epochs

        self.logger = SummaryWriter()
        self.progress = SProgressBar()
        self.progress.prefix = 'SWorkflow'

        self.current_epoch = -1

    def before_train(self):
        """Instructions runs before the train.

        This method can be used to log data or print console messages
        """
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        dummy_input = torch.rand(1, 1, 40, 40).to(self.device)
        self.logger.add_graph(self.model, dummy_input)
        self.logger.flush()
        self.progress.message(f"Using {self.device} device")
        self.progress.message(f"Model number of parameters: {num_parameters:d}")

    def after_train(self):
        """Instructions runs after the train.

        This method can be used to log data or print console messages
        """
        self.logger.close()

    def after_train_step(self, data):
        """Instructions runs after one train step.

        This method can be used to log data or print console messages

        Parameters
        ----------
        data: dict
            Dictionary of metadata to log or process
        """
        self.logger.add_scalar('train_loss', data['train_loss'],
                               self.current_epoch)

    def after_train_batch(self, data):
        """Instructions runs after one batch

        Parameters
        ----------
        data: dict
            Dictionary of metadata to log or process

        """
        prefix = f"Epoch = {self.current_epoch:d}"
        loss_str = "{data['loss']:.2f}"
        full_time_str = time.strftime("%H:%M:%S", data['full_time'])
        remains_str = time.strftime("%H:%M:%S", data['remain_time'])
        suffix = str(data['id_batch'] + 1) + '/' + str(data['total_batch']) + \
                 '   [' + full_time_str + '<' + remains_str + ', loss=' + \
                 loss_str + ']'
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
        for batch, (x, y) in enumerate(self.train_data_loader):
            tic = time.perf_counter()
            x, y = x.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            step_loss += loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # count time
            toc = time.perf_counter()
            delay = toc - tic
            full_time += delay
            total_batch = size / len(x)
            remains = (total_batch - batch) * delay

            self.after_train_batch({'loss': loss,
                                    'id_batch': batch,
                                    'total_batch': total_batch,
                                    'remain_time': remains,
                                    'full_time': full_time
                                    })
            # if batch % 100 == 0:
            #    loss, current = loss.item(), batch * len(X)
            #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
        # tensor board or any output output folder ?
        self.logger.add_scalar('val_loss', data['val_loss'],
                               self.current_epoch)

    def val_step(self):
        """Runs one step of validation

        Returns
        -------
        A dictionary of data to save/log/process
        This dictionary must contain at least the val_loss entry

        """
        num_batches = len(self.val_data_loader)
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in self.val_data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                val_loss += self.loss_fn(pred, y).item()
        val_loss /= num_batches

        # print(
        #    f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        return {'val_loss': val_loss}

    def train(self):
        """Train the model

        Main training loop.

        """
        self.before_train()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch + 1}\n-------------------------------")
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

    def predict(self, x):
        """Predict a single input

        Parameters
        ----------
        x: Tensor
            input data. The data is considered in CPU and is moved to GPU if
            needed.

        """
        x_device = x.to(self.device).unsqueeze(0).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_device)
        return pred

    def save(self, filename):
        """Save a model to a file

        Parameters
        ----------
        filename: str
            Path to the destination file
        """
        # self.model.save(self.model.state_dict(), filename)
        torch.save(self.model, filename)

    def load(self, filename):
        """Load a model from file

        Parameters
        ----------
        filename: str
            Path of the file containing the model
        """
        # self.model.load_state_dict(torch.load(filename))
        self.model = torch.load(filename)
