"""Set of classes to save data during training

Classes
-------
SDataLogger

"""
from torch.utils.tensorboard import SummaryWriter


class SDataLogger:
    """Interface for data logger during sdeep workflow

    Parameters
    ----------
    output_dir: str
        Directory where the data are stored

    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def add_scalar(self, tag, value, step):
        """Save a data scalar

        Parameters
        ----------
        tag: str
            Name of the data
        value: number
            Value of the scalar
        step: int
            Step at which the scalar have this value
        """
        raise NotImplementedError()

    def add_graph(self, tag, model):
        """Save a data scalar

        Parameters
        ----------
        tag: str
            Name of the data
        model: pytorch.nn.Module
            Value of the scalar
        """

        raise NotImplementedError()

    def flush(self):
        """Flush the recorded data to disk"""
        raise NotImplementedError()

    def close(self):
        """Close the file writer(s)"""
        raise NotImplementedError()


class STensorboardLogger(SDataLogger):
    """Data logger using TensorBoard"""
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.writer = SummaryWriter(self.output_dir)

    def add_scalar(self, tag, value, step):
        """Save a data scalar

        Parameters
        ----------
        tag: str
            Name of the data
        value: number
            Value of the scalar
        step: int
            Step at which the scalar have this value

        """
        self.writer.add_scalar(tag, value, step)

    def add_graph(self, tag, model):
        """Save a data scalar

        Parameters
        ----------
        tag: str
            Name of the data
        model: pytorch.nn.Module
            Value of the scalar

        """
        self.writer.add_graph(tag, model)

    def flush(self):
        """Flush the recorded data to disk"""
        self.writer.flush()

    def close(self):
        """Close the file writer(s)"""
        self.writer.close()
