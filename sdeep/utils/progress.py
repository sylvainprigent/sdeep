"""Set of classes to log a workflow

Classes
-------
SProgressLogger
SProgressBar

"""
COLOR_WARNING = '\033[93m'
COLOR_ERROR = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_ENDC = '\033[0m'


class SProgressLogger:
    """Default logger

    A logger is used by a workflow to print the warnings, errors and progress.
    A logger can be used to print in the console or in a log file

    """
    def __init__(self):
        self.prefix = ''

    def new_line(self):
        """Print a new line in the log"""
        print(f'\n')

    def message(self, message):
        """Log a default message

        Parameters
        ----------
        message: str
            Message to log
        """
        print(f'{self.prefix}: {message}')

    def error(self, message):
        """Log an error message

        Parameters
        ----------
        message: str
            Message to log
        """
        print(f'{COLOR_ERROR}{self.prefix} ERROR: '
              f'{message}{COLOR_ENDC}')

    def warning(self, message):
        """Log a warning

        Parameters
        ----------
        message: str
            Message to log
        """
        print(f'{COLOR_WARNING}{self.prefix} WARNING: '
              f'{message}{COLOR_ENDC}')

    def progress(self, iteration, total, prefix, suffix):
        """Log a progress

        Parameters
        ----------
        iteration: int
            Current iteration
        total: int
            Total number of iteration
        prefix: str
            Text to print before the progress
        suffix: str
            Text to print after the message
        """
        raise NotImplementedError()


class SProgressBar(SProgressLogger):
    """Console logger displaying a progress bar

    The progress bar display the basic information of a batch loop (loss,
    batch id, time/remaining time)

    """
    def __init__(self):
        super().__init__()
        self.decimals = 1
        self.print_end = "\r"
        self.length = 100
        self.fill = '█'

    def progress(self, iteration, total, prefix, suffix):
        percent = ("{0:." + str(self.decimals) + "f}").format(
            100 * (iteration / float(total)))
        filled_length = int(self.length * iteration // total)
        bar_ = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{prefix} {percent}% |{bar_}| {suffix}',
              end=self.print_end)
