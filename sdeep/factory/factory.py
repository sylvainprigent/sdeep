"""Factory to instantiate modules"""
import os
import importlib
import pkgutil

import torch

from ..interfaces import STransform
from ..interfaces import SDataset
from ..interfaces import SModel
from ..interfaces import SEval
from ..interfaces.workflow import SWorkflow


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
        self.__evals = self.__register_modules("evals")
        self.__transforms = self.__register_modules("transforms")

    def __register_modules(self, directory: str):
        modules = self.__find_modules(directory)
        modules += self.__register_plugins(directory)
        modules_info = {}
        for name in modules:
            mod = importlib.import_module(name)
            for value in mod.export:
                modules_info[value.__name__] = value
        return modules_info

    @staticmethod
    def __is_module_name(module_path: str):
        """Check if a module name is a good module candidate"""
        return module_path.endswith(".py") and \
            ("setup" not in module_path) and \
            ("utils" not in module_path) and \
            ("__init__" not in module_path) and \
            (not module_path.startswith("interface")) and \
            (not module_path.startswith("_"))

    @staticmethod
    def __find_modules(directory: str) -> list:
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
                if SFactory.__is_module_name(module_path):
                    module_name = str(module_path).split('.', maxsplit=1)[0]
                    modules.append(f"sdeep.{parent}.{module_name}")
        return modules

    @staticmethod
    def __register_plugins(submodule_name: str):
        """Register compatible plugins installed in the environment to the factory

        :param submodule_name: Name of the submodule (loss, model...)
        """
        discovered_plugins = {
            name: name
            for finder, name, is_pkg
            in pkgutil.iter_modules()
            if name.startswith("sd_")
        }
        modules = []
        for name in discovered_plugins:
            try:
                importlib.import_module(f'{name}.{submodule_name}')
                modules.append(f'{name}.{submodule_name}')
            except ModuleNotFoundError:
                print(f"Warning: no implementation of {submodule_name} in {name}")
        return modules

    def get_model(self, name: str, args: dict[str, any]) -> SModel:
        """Instantiate a model

        :param name: name of the model
        :param args: parameters of the model
        :return: an instance of the model
        """
        if name not in self.__models:
            raise SFactoryError(f'No implementation found for {name}')
        model = SModel(self.__models[name](**args), args)
        return model

    def get_loss(self, name: str, args: dict[str, any]) -> torch.nn.Module:
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
                  args: dict[str, any]) -> torch.nn.Module:
        """Instantiate an optim scheme

        :param name: name of the optim
        :param model: model to optimize
        :param args: parameters of the optim
        :return: an instance of the optim
        """
        if name not in self.__optims:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__optims[name](model.parameters(), **args)

    def get_dataset(self,
                    name: str,
                    args: dict[str, any],
                    transform: STransform = None
                    ) -> SDataset:
        """Instantiate a dataset

        :param name: name of the dataset
        :param args: parameters of the dataset
        :param transform: Data transformation
        :return: an instance of the dataset
        """
        if name not in self.__datasets:
            raise SFactoryError(f'No implementation found for {name}')
        if transform:
            args["transform"] = transform
        dataset = SDataset(self.__datasets[name](**args), args, transform)
        return dataset

    def get_eval(self, name: str, args: dict[str, any]) -> SEval:
        """Instantiate evaluation module

        :param name: name of the evaluation module
        :param args: parameters of the evaluation module
        :return: an instance of the evaluation module
        """
        if name not in self.__evals:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__evals[name](**args)

    def get_transform(self, name: str, args: dict[str, any]) -> STransform:
        """Instantiate transformation module

        :param name: name of the evaluation module
        :param args: parameters of the evaluation module
        :return: an instance of the evaluation module
        """
        if name not in self.__transforms:
            raise SFactoryError(f'No implementation found for {name}')
        args_ = args.copy()
        args_['name'] = name
        transform = STransform(self.__transforms[name](**args), args_)
        return transform

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def get_workflow(self,
                     name: str,
                     model: SModel,
                     loss_fn: torch.nn.Module,
                     optim: torch.nn.Module,
                     train_dataset: SDataset,
                     val_dataset: SDataset,
                     evaluate: SEval,
                     args: dict[str, any]
                     ) -> SWorkflow:
        """Instantiate a dataset

        :param name: name of the dataset
        :param model: instance of the model,
        :param loss_fn: instance of the loss function,
        :param optim: instance of the optimization scheme,
        :param train_dataset: instance of the train set data loader,
        :param val_dataset: instance of the validation set data loader,
        :param evaluate: Evaluation method
        :param args: parameters of the workflows
        :return: an instance of the workflow
        """
        if name not in self.__workflows:
            raise SFactoryError(f'No implementation found for {name}')
        return self.__workflows[name](model, loss_fn, optim, train_dataset,
                                      val_dataset, evaluate, **args)
