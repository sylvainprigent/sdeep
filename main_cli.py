import os
import argparse
from torch.utils.data import DataLoader
from sdeep.cli import sdeepModels, sdeepLosses, sdeepOptimizers, sdeepDatasets, sdeepWorkflows
from sdeep.utils import SFileLogger, SConsoleLogger

def add_args_to_parser(parser, factory):
    for name in factory.get_keys():
        params = factory.get_parameters(name)
        for param in params:
            parser.add_argument(f"--{param['key']}", help=param['help'], default=param['default'])


if __name__ == "__main__":

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

    # parse modules parameters
    add_args_to_parser(parser, sdeepModels)
    add_args_to_parser(parser, sdeepLosses)
    add_args_to_parser(parser, sdeepOptimizers)
    add_args_to_parser(parser, sdeepDatasets)
    add_args_to_parser(parser, sdeepWorkflows)
    
    args = parser.parse_args()

    # instantiate
    out_dir = args.save
    model = sdeepModels.get_instance(args.model, args)
    loss_fn = sdeepLosses.get_instance(args.loss, args)
    optim = sdeepOptimizers.get_instance(args.optim, model, args)
    train_dataset = sdeepDatasets.get_instance(args.train_dataset, args)
    val_dataset = sdeepDatasets.get_instance(args.val_dataset, args)


    train_data_loader = DataLoader(val_dataset,
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

    logger_file = SFileLogger(os.path.join(out_dir, 'log.txt'))
    logger_console = SConsoleLogger()
    workflow.add_progress_logger(logger_file)
    workflow.add_progress_logger(logger_console)
    workflow.fit()
    workflow.save(os.path.join(args.save, "model.pt"))

    logger_file.close()
    logger_console.close()

    #print('optim form string = ', getattr(args, 'optim'))
