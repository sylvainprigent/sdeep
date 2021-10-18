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


class SProgressObservable:
    """Observable pattern

    This pattern allows to set multiple progress logger to
    one workflow

    """
    def __init__(self):
        self._loggers = []

    def set_prefix(self, prefix):
        """Set the prefix for all loggers

        The prefix is a printed str ad the beginning of each
        line of the logger

        Parameters
        ----------
        prefix: str
            Prefix content

        """
        for logger in self._loggers:
            logger.prefix = prefix

    def add_logger(self, logger):
        """Add a logger to the observer

        Parameters
        ----------
        logger: SProgressLogger
            Logger to add to the observer
        """
        self._loggers.append(logger)

    def new_line(self):
        """Print a new line in the loggers"""
        for logger in self._loggers:
            logger.new_line()

    def message(self, message):
        """Log a default message

        Parameters
        ----------
        message: str
            Message to log
        """
        for logger in self._loggers:
            logger.message(message)

    def error(self, message):
        """Log an error message

        Parameters
        ----------
        message: str
            Message to log
        """
        for logger in self._loggers:
            logger.error(message)

    def warning(self, message):
        """Log a warning message

        Parameters
        ----------
        message: str
            Message to log
        """
        for logger in self._loggers:
            logger.warning(message)

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
        for logger in self._loggers:
            logger.progress(iteration, total, prefix, suffix)

    def close(self):
        """Close the loggers"""
        for logger in self._loggers:
            logger.close()


class SProgressLogger:
    """Default logger

    A logger is used by a workflow to print the warnings, errors and progress.
    A logger can be used to print in the console or in a log file

    """
    def __init__(self):
        self.prefix = ''

    def new_line(self):
        """Print a new line in the log"""
        raise NotImplementedError()

    def message(self, message):
        """Log a default message

        Parameters
        ----------
        message: str
            Message to log
        """
        raise NotImplementedError()

    def error(self, message):
        """Log an error message

        Parameters
        ----------
        message: str
            Message to log
        """
        raise NotImplementedError()

    def warning(self, message):
        """Log a warning

        Parameters
        ----------
        message: str
            Message to log
        """
        raise NotImplementedError()

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

    def close(self):
        """Close the logger"""
        raise NotImplementedError()


class SFileLogger(SProgressLogger):
    """Logger that write logs into txt file"""
    def __init__(self, filepath):
        super().__init__()
        self.file = open(filepath, 'a', encoding="utf8")

    def new_line(self):
        self.file.write(f"{self.prefix}:\n")

    def message(self, message):
        self.file.write(f'{self.prefix}: {message}\n')

    def error(self, message):
        self.file.write(f'{COLOR_ERROR}{self.prefix} ERROR: '
              f'{message}{COLOR_ENDC}\n')

    def warning(self, message):
        self.file.write(f'{COLOR_WARNING}{self.prefix} WARNING: '
              f'{message}{COLOR_ENDC}\n')

    def progress(self, iteration, total, prefix, suffix):
        self.file.write(f'{prefix}: iteration {iteration}/{total} ({suffix})\n')

    def close(self):
        self.file.close()


class SConsoleLogger(SProgressLogger):
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

    def new_line(self):
        print(f"{self.prefix}:\n")

    def message(self, message):
        print(f'{self.prefix}: {message}')

    def error(self, message):
        print(f'{COLOR_ERROR}{self.prefix} ERROR: '
              f'{message}{COLOR_ENDC}')

    def warning(self, message):
        print(f'{COLOR_WARNING}{self.prefix} WARNING: '
              f'{message}{COLOR_ENDC}')

    def progress(self, iteration, total, prefix, suffix):
        percent = ("{0:." + str(self.decimals) + "f}").format(
            100 * (iteration / float(total)))
        filled_length = int(self.length * iteration // total)
        bar_ = self.fill * filled_length + ' ' * (self.length - filled_length)
        print(f'\r{prefix} {percent}% |{bar_}| {suffix}',
              end=self.print_end)

    def close(self):
        pass
