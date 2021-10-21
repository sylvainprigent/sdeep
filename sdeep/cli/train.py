import os
import argparse
import torch
from torch.utils.data import DataLoader
from sdeep.factories import sdeepModels, sdeepLosses, sdeepOptimizers, sdeepDatasets, sdeepWorkflows
from sdeep.utils import SProgressObservable, SFileLogger, SConsoleLogger, STensorboardLogger


def add_args_to_parser(parser, factory):
    for name in factory.get_keys():
        params = factory.get_parameters(name)
        for param in params:
            parser.add_argument(f"--{param['key']}", help=param['help'], default=param['default'])


def get_subdir(main_dir):
    run_id = 1
    path = os.path.join(main_dir, f"run_{run_id}")
    while os.path.isdir(path):
        run_id += 1
        path = os.path.join(main_dir, f"run_{run_id}")
    os.mkdir(path)
    return path


def main():

    parser = argparse.ArgumentParser(description='SDeep train')

    parser.add_argument('-m', '--model', help='neural network model', default='DnCNN')
    parser.add_argument('-l', '--loss', help='Loss function', default='MSELoss')
    parser.add_argument('-o', '--optim', help='Optimizer method', default='Adam')
    parser.add_argument('-t', '--train_dataset', help='Training dataset', default='RestorationPatchDataset')
    parser.add_argument('-v', '--val_dataset', help='Validation dataset', default='RestorationDataset')
    parser.add_argument('-w', '--workflow', help='Training workflow', default='SWorkflow')
    parser.add_argument('-s', '--save', help='Save directory', default='./run')

    parser.add_argument('--train_batch_size', help='Size of training batch', default='128')
    parser.add_argument('--val_batch_size', help='Size of validation batch', default='3')
    parser.add_argument('--reuse', help='True to reuse a previous checking point', default='false')

    # parse modules parameters
    add_args_to_parser(parser, sdeepModels)
    add_args_to_parser(parser, sdeepLosses)
    add_args_to_parser(parser, sdeepOptimizers)
    add_args_to_parser(parser, sdeepDatasets)
    add_args_to_parser(parser, sdeepWorkflows)

    args = parser.parse_args()

    # instantiate
    if args.reuse == "true":
        out_dir = args.save
    else:
        out_dir = get_subdir(args.save)
    model = sdeepModels.get_instance(args.model, args)
    loss_fn = sdeepLosses.get_instance(args.loss, args)
    optim = sdeepOptimizers.get_instance(args.optim, model, args)
    train_dataset = sdeepDatasets.get_instance(args.train_dataset, args)
    val_dataset = sdeepDatasets.get_instance(args.val_dataset, args)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=int(args.train_batch_size),
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=0)

    val_data_loader = DataLoader(val_dataset,
                                 batch_size=int(args.val_batch_size),
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=0)

    workflow = sdeepWorkflows.get_instance(args.workflow,
                                           model,
                                           loss_fn,
                                           optim,
                                           train_data_loader,
                                           val_data_loader,
                                           args)

    # progress Loggers
    observable = SProgressObservable()
    logger_file = SFileLogger(os.path.join(out_dir, 'log.txt'))
    logger_console = SConsoleLogger()
    observable.add_logger(logger_file)
    observable.add_logger(logger_console)
    workflow.set_progress_observable(observable)

    # log setup
    observable.message('Start')
    observable.message(f"Model: {args.model}")
    model_args = {}
    for param in sdeepModels.get_parameters(args.model):
        observable.message(f"    - {param['key']}={param['value']}")
        model_args[param['key']] = param['value']
    observable.message(f"Loss: {args.loss}")
    for param in sdeepLosses.get_parameters(args.loss):
        observable.message(f"    - {param['key']}={param['value']}")
    observable.message(f"Optimizer: {args.optim}")
    for param in sdeepOptimizers.get_parameters(args.optim):
        observable.message(f"    - {param['key']}={param['value']}")
    observable.message(f"Train dataset: {args.train_dataset}")
    for param in sdeepDatasets.get_parameters(args.train_dataset):
        observable.message(f"    - {param['key']}={param['value']}")
    observable.message(f"    - train batch size={args.train_batch_size}")
    observable.message(f"Train dataset: {args.val_dataset}")
    for param in sdeepDatasets.get_parameters(args.val_dataset):
        observable.message(f"    - {param['key']}={param['value']}")
    observable.message(f"Workflow: {args.workflow}")
    for param in sdeepWorkflows.get_parameters(args.workflow):
        observable.message(f"    - {param['key']}={param['value']}")
    observable.message(f"Save directory: {args.save}")
    observable.new_line()

    # data logger
    data_logger = STensorboardLogger(out_dir)
    workflow.set_data_logger(data_logger)
    workflow.out_dir = out_dir

    workflow.fit()

    torch.save({
        'model': args.model,
        'model_args': model_args,
        'model_state_dict': model.state_dict()
    }, os.path.join(out_dir, "model.ckpt"))

    observable.message(f"Done")
    logger_file.close()
    logger_console.close()


if __name__ == "__main__":
    main()
