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
    def get_instance(self, model, loss_fn, optimizer,
                     train_data_loader, test_data_loader, args):
        if not self._instance:
            epochs = get_arg_int(args, 'epochs', 50)
            self._instance = SWorkflow(model, loss_fn, optimizer,
                                       train_data_loader,
                                       test_data_loader,
                                       epochs=epochs)
        return self._instance

    def get_parameters(self):
        return {"epoch": 50}


sdeepWorkflows = SDeepWorkflowsFactory()
sdeepWorkflows.register_builder('SWorkflow', SWorkflowBuilder())
