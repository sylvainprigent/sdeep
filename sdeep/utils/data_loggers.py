"""Set of classes to save data during training"""
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class SDataLogger:
    """Interface for data logger during workflow run

    :param output_dir: Directory where the data are stored
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def add_scalar(self, tag: str, value: float, step: int):
        """Save a data scalar

        :param tag: Name of the data
        :param value: Value of the scalar
        :param step: Step at which the scalar have this value
        """
        raise NotImplementedError()

    def add_graph(self, tag: str, model: torch.nn.Module):
        """Save a data scalar

        :param tag: Name of the data
        :param model: Value of the scalar
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
    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.writer = SummaryWriter(str(self.output_dir))

    def add_scalar(self, tag: str, value: float, step: int):
        """Save a data scalar

        :param tag: Name of the data
        :param value: Value of the scalar
        :param step: Step at which the scalar have this value
        """
        self.writer.add_scalar(tag, value, step)

    def add_graph(self, tag: str, model: torch.nn.Module):
        """Save a data scalar

        :param tag: Name of the data
        :param model: Value of the scalar
        """
        self.writer.add_graph(tag, model)

    def flush(self):
        """Flush the recorded data to disk"""
        self.writer.flush()

    def close(self):
        """Close the file writer(s)"""
        self.writer.close()
