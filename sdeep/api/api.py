"""Implementation of the SDeep API"""
from typing import Callable
from pathlib import Path

import torch

from sdeep.utils import SProgressObservable
from sdeep.utils import SFileLogger
from sdeep.utils import SConsoleLogger
from sdeep.utils import STensorboardLogger
from sdeep.utils import TilePredict
from sdeep.utils import SParameters
from sdeep.utils import device
from sdeep.utils import io

from sdeep.interfaces import SModel
from sdeep.interfaces import STransform
from sdeep.interfaces import SDataset

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

    def __load_dataset(self, params: SParameters,
                       dataset_module_name: str,
                       transform_module_name) -> SDataset:
        train_transform = None
        if params.is_module(transform_module_name):
            train_transform = self.__factory.get_transform(*params.module(transform_module_name))
        td_name, td_args = params.module(dataset_module_name)
        return self.__factory.get_dataset(td_name, td_args, train_transform)

    def train(self, params: SParameters, out_dir: Path):
        """Instantiate the training workflow from parameters

        :param params: training workflow parameters
        :param out_dir: directory to save training results
        """
        model = self.__factory.get_model(*params.module("model"))
        loss_fn = self.__factory.get_loss(*params.module("loss"))
        optim_module, optim_args = params.module("optim")
        optim = self.__factory.get_optim(optim_module, model.model, optim_args)

        # Train dataset
        train_dataset = self.__load_dataset(params,
                                            "train_dataset",
                                            "train_transform"
                                            )

        # Test dataset
        val_dataset = self.__load_dataset(params,
                                          "val_dataset",
                                          "val_transform"
                                          )

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
        workflow.set_data_logger(STensorboardLogger(out_dir))
        workflow.out_dir = out_dir

        workflow.fit()

        # save the models
        io.save_model(model.model,
                      params.module("model")[1],
                      out_dir / "model.ml",
                      SParameters.to_dict(params.module("val_transform")))
        observable.message("Done")
        observable.close()

    def load_model(self, filename: Path) -> tuple[SModel, STransform]:
        """Load a model from a file

        :param filename: File containing the model
        :return: instance of the model
        """
        params = torch.load(filename,
                            map_location=torch.device(device()))

        print('loaded model:')
        print("params['model']=", params['model'])
        print("params['model_args']=", params['model_args'])
        print("params['transform']=", params['transform'])

        model = self.__factory.get_model(params['model'], params['model_args']).model
        model.load_state_dict(params['model_state_dict'])
        model.to(device())

        transform_name = params['transform']['name']
        transform_args = params['transform'].copy()
        transform_args.pop('name')

        transform = self.__factory.get_transform(transform_name, transform_args)

        return SModel(model, params['model_args']), transform

    @staticmethod
    def eval(model: torch.nn.Module,
             data: torch.Tensor,
             transform: Callable = None,
             use_tiling: bool = False,
             is_batch: bool = False) -> torch.Tensor:
        """Evaluate a model

        :param model: Model to run,
        :param data: Tensor to process,
        :param transform: Pre transformation on the data (optional),
        :param use_tiling: To use tiling for image prediction,
        :param is_batch: True if the data is already a batch
        :return: The model prediction
        """
        if is_batch:
            x = data.to(device())
        else:
            x = data.to(device()).unsqueeze(0)
        model.eval()
        if transform:
            x = transform(x)

        if use_tiling:
            tile_predict = TilePredict(model)
            prediction = tile_predict.run(x)
        else:
            with torch.no_grad():
                prediction = model(x)

        if is_batch:
            return prediction.detach().cpu()
        return prediction[0, ...].detach().cpu()

    def predict(self,
                model_path: Path,
                input_path: Path,
                output_path: Path,
                *,
                out_extension: str = "",
                batch_size: int = 1,
                use_tiling: bool = False):
        """Run a model prediction

        TODO: optimize this method using batch for directory

        :param model_path: Path to the model file
        :param input_path: Path to the single input data file or multiple data dir
        :param output_path: Path to the single output data file or multiple data dir
        :param out_extension: Extension of the output file name (for dir input only)
        :param batch_size: Number of files to process in parallel
        :param use_tiling: To use tiling for image prediction
        """
        model, transform = self.load_model(model_path)
        if input_path.is_file():
            self.__predict_single(input_path, output_path, model, transform, use_tiling)
        elif input_path.is_dir() and batch_size == 1:
            self.__predict_dir(input_path, output_path, out_extension, model, transform, use_tiling)
        elif input_path.is_dir() and batch_size > 1:
            self.__predict_batch(input_path, output_path, out_extension, batch_size, model,
                                 transform, use_tiling)

    def __predict_single(self,
                         input_path: Path,
                         output_path: Path,
                         model: SModel,
                         transform: STransform,
                         use_tiling: bool = False
                         ):
        in_data = io.read_data(input_path)
        out_data = self.eval(model.model, in_data, transform,
                             use_tiling=use_tiling,
                             is_batch=False)
        io.write_data(output_path, out_data)

    def __predict_dir(self,
                      input_path: Path,
                      output_path: Path,
                      out_extension: str,
                      model: SModel,
                      transform: STransform,
                      use_tiling: bool = False):
        for file in input_path.rglob("*"):
            print('processing file:', file)
            in_data = io.read_data(file)
            out_data = self.eval(model.model, in_data, transform,
                                 use_tiling=use_tiling,
                                 is_batch=False)
            output_file = Path(str(file).replace(str(input_path), str(output_path)))
            output_file = output_file.with_suffix(out_extension)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            io.write_data(output_file, out_data)

    def __predict_batch(self,
                        input_path: Path,
                        output_path: Path,
                        out_extension: str,
                        batch_size: int,
                        model: SModel,
                        transform: STransform,
                        use_tiling: bool = False):
        batch_files = []
        counter = 0
        files = list(input_path.rglob("*"))
        for i, file in enumerate(files):
            batch_files.append(file)
            if len(batch_files) == batch_size or i == len(files) - 1:
                counter += 1
                print('process batch ', counter)
                self.__predict_batch_item(model.model, transform.transform,
                                          batch_files, input_path, output_path,
                                          out_extension, use_tiling=use_tiling)
                batch_files = []

    def __predict_batch_item(self,
                             model: torch.nn.Module,
                             transform: Callable,
                             batch_files: list[Path],
                             input_path: Path,
                             output_path: Path,
                             out_extension: str,
                             use_tiling: bool = False):
        in_data = []
        for file in batch_files:
            in_data.append(transform(io.read_data(file)))
        in_data = torch.stack(in_data)
        print("in data shape=", in_data.shape)
        out_data = self.eval(model, in_data,
                             transform=None,
                             use_tiling=use_tiling,
                             is_batch=True)
        for i in range(out_data.shape[0]):
            output_file = Path(str(batch_files[i]).replace(str(input_path), str(output_path)))
            output_file = output_file.with_suffix(out_extension)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            io.write_data(output_file, out_data[i, ...])
