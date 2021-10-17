"""Factory for Workflows

This module implements factory to instantiate all the available
workflows of SDeep

Classes
-------
SWorkflowBuilder


Objects
-------
sdeepWorkflows

"""
from sdeep.workflows import SWorkflow
from .utils import get_arg_int, SDeepWorkflowsFactory, SDeepWorkflowBuilder


class SWorkflowBuilder(SDeepWorkflowBuilder):
    """Service builder for the default sdeep workflow"""
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(self):
        super().__init__()
        self.parameters = [{'key': 'epoch',
                            'default': 50,
                            'value': 50,
                            'help': 'Number of epoch'}
                    ]
    def get_instance(self, model, loss_fn, optimizer,
                     train_data_loader, test_data_loader, args):
        if not self._instance:
            epochs = get_arg_int(args, 'epochs', 50)
            self.parameters[0]['value'] = epochs
            self._instance = SWorkflow(model, loss_fn, optimizer,
                                       train_data_loader,
                                       test_data_loader,
                                       epochs=epochs)
        return self._instance

    def get_parameters(self):
        return self.parameters


sdeepWorkflows = SDeepWorkflowsFactory()
sdeepWorkflows.register_builder('SWorkflow', SWorkflowBuilder())
