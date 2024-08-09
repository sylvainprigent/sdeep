"""Implementation of the sdeep API"""
from pathlib import Path

import numpy as np
import torch

from sdeep.utils import SProgressObservable
from sdeep.utils import SFileLogger
from sdeep.utils import SConsoleLogger
from sdeep.utils import STensorboardLogger
from sdeep.utils import TilePredict
from sdeep.utils import SParameters
from sdeep.utils import device

from sdeep.factory import SFactory


class SDeepAPI:
    """API to instantiate a training or prediction"""
    def __init__(self):
        self.__factory = SFactory()

    @staticmethod
    def __init_observable(out_dir: Path) -> SProgressObservable:
        """Initialize the observers for training

        :param out_dir: destination dir for log file
        :return: the observable instance
        """
        observable = SProgressObservable()
        logger_file = SFileLogger(out_dir / 'log.txt')
        logger_console = SConsoleLogger()
        observable.add_logger(logger_file)
        observable.add_logger(logger_console)
        return observable

    @staticmethod
    def __log_params(params: SParameters,
                     out_dir: Path,
                     observable: SProgressObservable):
        """Log the training initialization

        :param params: training parameters
        :param out_dir: log destination dir
        :param observable: progress observable
        """
        observable.message('Start')

        name, args = params.module("model")
        observable.message(f"Model: {name}")
        for key, value in args.items():
            observable.message(f"    - {key}={value}")

        name, args = params.module("loss")
        observable.message(f"Loss: {name}")
        for key, value in args.items():
            observable.message(f"    - {key}={value}")

        name, args = params.module("optim")
        observable.message(f"Optimizer: {name}")
        for key, value in args.items():
            observable.message(f"    - {key}={value}")

        name, args = params.module("train_dataset")
        observable.message(f"Train dataset: {name}")
        for key, value in args.items():
            observable.message(f"    - {key}={value}")

        if params.is_module("train_transform"):
            name, args = params.module("train_transform")
            observable.message(f"Train transform: {name}")
            for key, value in args.items():
                observable.message(f"    - {key}={value}")

        name, args = params.module("val_dataset")
        observable.message(f"Train dataset: {name}")
        for key, value in args.items():
            observable.message(f"    - {key}={value}")

        if params.is_module("val_transform"):
            name, args = params.module("val_transform")
            observable.message(f"Val transform: {name}")
            for key, value in args.items():
                observable.message(f"    - {key}={value}")

        if params.is_module("eval"):
            name, args = params.module("eval")
            observable.message(f"Eval: {name}")
            for key, value in args.items():
                observable.message(f"    - {key}={value}")

        name, args = params.module("workflow")
        observable.message(f"Workflow: {name}")
        for key, value in args.items():
            observable.message(f"    - {key}={value}")

        observable.message(f"Save directory: {out_dir}")
        observable.new_line()

    def train(self, params: SParameters, out_dir: Path):
        """Instantiate the training workflow from parameters

        :param params: training workflow parameters
        :param out_dir: directory to save training results
        """
        model = self.__factory.get_model(*params.module("model"))
        loss_fn = self.__factory.get_loss(*params.module("loss"))
        optim_module, optim_args = params.module("optim")
        optim = self.__factory.get_optim(optim_module, model, optim_args)

        # Train dataset
        train_transform = None
        if params.is_module("train_transform"):
            train_transform = self.__factory.get_transform(*params.module("train_transform"))
        td_name, td_args = params.module("train_dataset")
        train_dataset = self.__factory.get_dataset(td_name, td_args, train_transform)

        # Test dataset
        val_transform = None
        if params.is_module("val_transform"):
            val_transform = self.__factory.get_transform(*params.module("val_transform"))
        vd_name, vd_args = params.module("val_dataset")
        val_dataset = self.__factory.get_dataset(vd_name, vd_args, val_transform)

        # Eval
        eval_module = None
        if params.is_module("eval"):
            eval_module = self.__factory.get_eval(*params.module("eval"))

        workflow_name, workflow_args = params.module("workflow")
        workflow = self.__factory.get_workflow(workflow_name,
                                               model,
                                               loss_fn,
                                               optim,
                                               train_dataset,
                                               val_dataset,
                                               eval_module,
                                               workflow_args)

        # progress loggers
        observable = self.__init_observable(out_dir)
        workflow.set_progress_observable(observable)
        self.__log_params(params, out_dir, observable)

        # data logger
        data_logger = STensorboardLogger(out_dir)
        workflow.set_data_logger(data_logger)
        workflow.out_dir = out_dir

        workflow.fit()

        torch.save({
            'model': params.module("model")[0],
            'model_args': params.module("model")[1],
            'model_state_dict': model.state_dict()
        }, out_dir / "model.ml")

        observable.message("Done")
        observable.close()

    def load_model(self, filename: Path) -> torch.nn.Module:
        """Load a model from a file

        :param filename: File containing the model
        :return: instance of the model
        """
        params = torch.load(filename,
                            map_location=torch.device(device()))
        model = self.__factory.get_model(params['model'], params['model_args'])
        model.load_state_dict(params['model_state_dict'])
        model.to(device())
        model.eval()
        return model

    @staticmethod
    def predict(model: torch.nn.Module,
                in_array: np.array,
                tiling: bool = False
                ) -> np.array:
        """Run a model prediction

        :param model: Instance of the model to run
        :param in_array: Array to process
        :param tiling: True to use tiling for prediction.
                       False for direct prediction
        """
        # load array to device
        image_torch = torch.from_numpy(in_array).float()
        image_device = image_torch.to(device()).unsqueeze(0).unsqueeze(0)

        # run the model
        if tiling:
            tile_predict = TilePredict(model)
            prediction = tile_predict.run(image_device)
        else:
            with torch.no_grad():
                prediction = model(image_device)

        # convert the output to array
        return prediction[0, 0, ...].cpu().numpy()
