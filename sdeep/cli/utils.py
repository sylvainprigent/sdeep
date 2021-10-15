"""SDeep factory utils module.

Generic Interface to Object Factory

Classes
-------
ObjectFactory

"""


class SDeepModulesFactory:
    """Factory for SDeep modules

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

    def get_parameters(self, key):
        """Get the parameters of the SDeep module

        Returns
        -------
        dict: a dictionary of key:value for each parameter

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

    @staticmethod
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
        if key in args:
            return int(args[key])
        return default_value

    @staticmethod
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
        if key in args:
            return float(args[key])
        return default_value

    @staticmethod
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
        if key in args:
            return str(args[key])
        return default_value
