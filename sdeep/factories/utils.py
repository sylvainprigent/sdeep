"""SDeep factory utils module.

Generic Interface to Object Factory

Classes
-------
ObjectFactory

"""


def get_arg_int(args, key, default_value):
    """Get the value of a parameter from the args list

    The default value of the parameter is returned if the
    key is not in args

    Parameters
    ----------
    args: dict
        Dictionary of the input args
    key: str
        Name of the parameters
    default_value: int
        Default value of the parameter

    """
    if hasattr(args, key):
        return int(getattr(args, key))
    if isinstance(args, dict) and key in args:
        return int(args[key])
    return default_value


def get_arg_float(args, key, default_value):
    """Get the value of a parameter from the args list

    The default value of the parameter is returned if the
    key is not in args

    Parameters
    ----------
    args: dict
        Dictionary of the input args
    key: str
        Name of the parameters
    default_value: float
        Default value of the parameter

    """
    if hasattr(args, key):
        return float(getattr(args, key))
    if isinstance(args, dict) and key in args:
        return float(args[key])
    return default_value


def get_arg_str(args, key, default_value):
    """Get the value of a parameter from the args list

    The default value of the parameter is returned if the
    key is not in args

    Parameters
    ----------
    args: dict
        Dictionary of the input args
    key: str
        Name of the parameters
    default_value: str
        Default value of the parameter

    """
    if hasattr(args, key):
        return str(getattr(args, key))
    if isinstance(args, dict) and key in args:
        return str(args[key])
    return default_value


class SDeepAbstractFactory:
    """Define the common methods of all factories

    """
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        """Register a new builder to the factory

        Parameters
        ----------
        key: str
            Name of the module to register
        builder: Object
            Builder instance
        """
        self._builders[key] = builder

    def get_parameters(self, key):
        """Get the parameters of the SDeep module

        Returns
        -------
        list: list of dictionary of key, default, help for each parameter

        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder.get_parameters()

    def get_keys(self):
        """Get the names of all the registered modules

        Returns
        -------
        list: list of all the registered modules names

        """
        return self._builders.keys()


class SDeepModulesFactory(SDeepAbstractFactory):
    """Factory for SDeep modules

    """
    def get_instance(self, key, args):
        """Get the instance of the SDeep module

        Parameters
        ----------
        key: str
            Name of the module to load
        args: dict
            Dictionary of CLI args for models parameters (ex: number of channels)

        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder.get_instance(args)


class SDeepModuleBuilder:
    """Interface for a SDeep module builder

    The builder is used by the factory to instantiate a module

    """
    def __init__(self):
        self._instance = None

    def get_instance(self, args):
        """Get the instance of the module

        Returns
        -------
        Object: instance of the SDeep module

        """
        raise NotImplementedError

    def get_parameters(self):
        """Get the parameters of the module

        Returns
        -------
        dict: dictionary of key:value for each parameters

        """
        raise NotImplementedError


class SDeepOptimizersFactory(SDeepAbstractFactory):
    """Factory for SDeep modules

    """
    def get_instance(self, key, model, args):
        """Get the instance of the SDeep module

        Parameters
        ----------
        key: str
            Name of the module to load
        model: nn.Module
            Neural network model
        args: dict
            Dictionary of CLI args for models parameters (ex: number of channels)

        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder.get_instance(model, args)


class SDeepOptimizerBuilder:
    """Interface for a SDeep optimizer builder

    The builder is used by the factory to instantiate an optimizer

    """
    def __init__(self):
        self._instance = None

    def get_instance(self, model, args):
        """Get the instance of the module

        Parameters
        ----------
        model: nn.Module
            Neural network model
        args: dict
            dict of parameters (key:value)

        Returns
        -------
        Object: instance of the SDeep module

        """
        raise NotImplementedError

    def get_parameters(self):
        """Get the parameters of the module

        Returns
        -------
        dict: dictionary of key:value for each parameters

        """
        raise NotImplementedError


class SDeepDatasetsFactory(SDeepAbstractFactory):
    """Factory for SDeep datasets

    """
    def get_instance(self, key, args):
        """Get the instance of the SDeep dataset

        Parameters
        ----------
        key: str
            Name of the module to load
        args: dict
            Dictionary of CLI args for models parameters (ex: number of channels)

        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder.get_instance(args)


class SDeepDatasetBuilder:
    """Interface for a SDeep module builder

    The builder is used by the factory to instantiate a module

    """
    def __init__(self):
        self._instance = None

    def get_instance(self, args):
        """Get the instance of the module

        Returns
        -------
        Object: instance of the SDeep module

        """
        raise NotImplementedError

    def get_parameters(self):
        """Get the parameters of the module

        Returns
        -------
        dict: dictionary of key:value for each parameters

        """
        raise NotImplementedError


class SDeepWorkflowsFactory(SDeepAbstractFactory):
    """Factory for SDeep workflow

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def get_instance(self, key, model, loss_fn, optimizer,
                     train_data_loader, test_data_loader, args):
        """Get the instance of the SDeep dataset

        Parameters
        ----------
        key: str
            Name of the module to load
        model: nn.Module
            Neural network model
        loss_fn: nn.Module
            Loss function
        optimiser: Object
            Neural network train optimization function
        train_data_loader: Object
            Data loader for training set
        test_data_loader: Object
            Data loader for validation set
        args: dict
            Dictionary of CLI args for models parameters (ex: number of iteration)

        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder.get_instance(model, loss_fn, optimizer,
                                    train_data_loader,
                                    test_data_loader, args)


class SDeepWorkflowBuilder:
    """Interface for a SDeep workflow builder

    The builder is used by the factory to instantiate a workflow

    """
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self):
        self._instance = None

    def get_instance(self, model, loss_fn, optimizer, train_data_loader, test_data_loader, args):
        """Get the instance of the module

        Parameters
        ----------
        model: nn.Module
            Neural network model
        loss_fn: nn.Module
            Loss function
        optimiser: Object
            Neural network train optimization function
        train_data_loader: Object
            Data loader for training set
        test_data_loader: Object
            Data loader for validation set
        args: dict
            parameters dictionary

        Returns
        -------
        Object: instance of the SDeep module

        """
        raise NotImplementedError

    def get_parameters(self):
        """Get the parameters of the module

        Returns
        -------
        dict: dictionary of key:value for each parameters

        """
        raise NotImplementedError
