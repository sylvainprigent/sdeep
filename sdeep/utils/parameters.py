"""Read and access workflow parameters"""
from typing import Tuple
from typing import Dict
from typing import List
from typing import Union

from pathlib import Path
import json


class SParameters:
    """Containers for a workflow parameters"""
    def __init__(self):
        self.__params = {}

    def __extract_data(self, name: str) -> Tuple[str, Dict]:
        """Extract a module parameters

        :param name: name of the module type (model, loss...)
        :return: the module instance name and args
        """
        if name not in self.__params:
            raise ValueError(f'The module {name} is not available in the '
                             f'parameters')
        data = self.__params[name]
        if 'name' not in data:
            raise ValueError('Module name missing in the parameters')
        args = data.copy()
        args.pop('name', None)
        return data["name"], args

    @property
    def data(self) -> Dict:
        """Params as a dictionary"""
        return self.__params

    def set_data(self, data: Dict):
        """Fill the data

        :param data: dict of data
        """
        self.__params = data

    def is_module(self, name) -> False:
        """Check if a module exists in the parameters

        :param name: Name of the module
        :return: True if the module exists, False otherwise
        """
        if name in self.__params:
            return True
        return False

    def module(self, name: str) -> Tuple[str, Dict]:
        """Get a module parameters using it name

        :param name: name of the module type (model, loss...)
        :return: the module instance name and args
        """
        return self.__extract_data(name)

    def parameter(self, name: str) -> Union[str, float, List]:
        """Read a parameter

        :param name: name of the parameter
        :return: the value of the parameter
        """
        if name not in self.__params:
            raise ValueError(f'The module {name} is not available in the '
                             f'parameters')
        return self.__params[name]


class SParametersReader:
    """Read parameters from file"""
    @staticmethod
    def read(filename: Path) -> SParameters:
        """Read parameters from file

        :param filename: Path of the parameters file
        :return: the parameters container
        """
        with open(filename, 'r', encoding='utf8') as f_in:
            data = json.load(f_in)
        params = SParameters()
        params.set_data(data)
        return params

    @staticmethod
    def write(params: SParameters, filename: Path):
        """Write parameters into a file

        :param params: Parameters to save
        :param filename: Destination file
        """
        with open(filename, "w", encoding='utf8') as fp:
            json.dump(params.data, fp, indent=4)
