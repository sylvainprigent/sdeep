"""Set of classes to save data during training

Classes
-------
SDataLogger

"""


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
