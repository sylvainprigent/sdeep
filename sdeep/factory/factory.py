"""Factory to instantiate modules"""
from typing import Dict
from typing import List

import os
import importlib

import torch
from torch.utils.data import Dataset

from ..workflows import SWorkflow


class SFactoryError(Exception):
    """Raised when an error happen when a module is built in the factory"""


class SFactory:
    """Factory to instantiate modules"""
    def __init__(self):
        self.__models = self.__register_modules("models")
        self.__losses = self.__register_modules("losses")
        self.__optims = self.__register_modules("optims")
        self.__datasets = self.__register_modules("datasets")
        self.__workflows = self.__register_modules("workflows")

    def __register_modules(self, directory: str):
        modules = self.__find_modules(directory)
        modules_info = {}
        for name in modules:
            mod = importlib.import_module(name)
            for value in mod.export:
                modules_info[value.__name__] = value
        return modules_info

    @staticmethod
    def __find_modules(directory: str) -> List:
        """Search for modules in a specific directory

        :param directory: Directory to parse
        :return: the list of founded modules
        """
        path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.dirname(path)
        modules = []
        for parent in [directory]:
            path_ = os.path.join(path, parent)
            for module_path in os.listdir(path_):
                if str(module_path).endswith(".py") and \
                        'setup' not in module_path and \
                        'utils' not in module_path and \
                        '__init__' not in module_path and not \
                        str(module_path).startswith("_"):
                    module_name = str(module_path).split('.', maxsplit=1)[0]
                    modules.append(f"sdeep.{parent}.{module_name}")
        return modules

    def get_model(self, name: str, args: Dict) -> torch.nn.Module:
        """Instantiate a model

        :param name: name of the model
        :param args: parameters of the model
        :return: an instance of the model
        """
        if name not in self.__models:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__models[name](**args)

    def get_loss(self, name: str, args: Dict) -> torch.nn.Module:
        """Instantiate a loss

        :param name: name of the loss
        :param args: parameters of the loss
        :return: an instance of the loss
        """
        if name not in self.__losses:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__losses[name](**args)

    def get_optim(self, name: str,
                  model: torch.nn.Module,
                  args: Dict) -> torch.nn.Module:
        """Instantiate an optim scheme

        :param name: name of the optim
        :param model: model to optimize
        :param args: parameters of the optim
        :return: an instance of the optim
        """
        if name not in self.__optims:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__optims[name](model.parameters(), **args)

    def get_dataset(self, name: str, args: Dict) -> Dataset:
        """Instantiate a dataset

        :param name: name of the dataset
        :param args: parameters of the dataset
        :return: an instance of the dataset
        """
        if name not in self.__datasets:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__datasets[name](**args)

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def get_workflow(self,
                     name: str,
                     model: torch.nn.Module,
                     loss_fn: torch.nn.Module,
                     optim: torch.nn.Module,
                     train_dataset: Dataset,
                     val_dataset: Dataset,
                     args: Dict
                     ) -> SWorkflow:
        """Instantiate a dataset

        :param name: name of the dataset
        :param model: instance of the model,
        :param loss_fn: instance of the loss function,
        :param optim: instance of the optimization scheme,
        :param train_dataset: instance of the train set data loader,
        :param val_dataset: instance of the validation set data loader,
        :param args: parameters of the workflows
        :return: an instance of the workflow
        """
        if name not in self.__workflows:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__workflows[name](model, loss_fn, optim, train_dataset,
                                      val_dataset, **args)
