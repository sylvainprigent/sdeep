Modules
=======

Most of the modules used by SDeep inherit from `PyTorch` modules. The two modules specific to
`SDeep` are :class:`SWorkflow <sdeep.interfaces.workflow.SWorkflow>`
and :class:`Eval <sdeep.interfaces.SEval>`

.. list-table:: SDeep main modules
   :widths: 25 75

   * - `Transform <https://docs.python.org/3/reference/datamodel.html#emulating-callable-objects>`_
     - Callable to transform a tensor into another tensor of same shape
   * - `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
     - Read and apply transform to data for training or validation
   * - `Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module>`_
     - Neural network as a `PyTorch` module
   * - `Loss <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module>`_
     - Loss function to be minimized. Implemented as a `PyTorch` module
   * - `Optim <https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer>`_
     - Optimizer for back-propagation. Implemented as a `PyTorch` optimizer
   * - :class:`SEval <sdeep.interfaces.SEval>`
     - Implements the evaluation metrics and write evaluation output files
   * - :class:`SWorkflow <sdeep.interfaces.workflow.SWorkflow>`
     - Implements the training strategy

Interfaces
----------

Interfaces are the core of the framework to define modules that store metadata and allows to
efficiently build training and prediction workflows.

The available interfaces are:

.. currentmodule:: sdeep.interfaces

.. autosummary::
    :toctree: generated
    :nosignatures:

    SDataset
    SEval
    SModel
    STransform
    SWorkflow


Workflows
---------

.. currentmodule:: sdeep.workflows

.. autosummary::
    :toctree: generated
    :nosignatures:

    SWorkflowBase
    ClassificationWorkflow
    RestorationWorkflow
    SegmentationWorkflow
    SelfSupervisedWorkflow

Eval
----

.. currentmodule:: sdeep.evals

.. autosummary::
    :toctree: generated
    :nosignatures:

    EvalClassification
    EvalRestoration

Models
------

.. currentmodule:: sdeep.models

.. autosummary::
    :toctree: generated
    :nosignatures:

    Autoencoder
    DnCNN
    DeepFinder
    DRUNet
    MNistClassifier
    UNet

Losses
------

Losses listed in the documentation are the one implemented in `SDeep`. Any `PyTorch` loss can be
made accessible to the `SDeep` framework using the ``export=[MyLoss]`` in the losses module.
By default `PyTorch` available losses are:

- `MSELoss`
- `L1Loss`
- `BCELoss`
- `BCEWithLogitsLoss`
- `CrossEntropyLoss`

.. currentmodule:: sdeep.losses

.. autosummary::
    :toctree: generated
    :nosignatures:

    DeconMSEHessian
    DeconSpitfire
    DiceLoss
    BinaryDiceLoss
    VGGL1PerceptualLoss
    N2XDenoise
    N2XDecon
    TverskyLoss

Datasets
--------

.. currentmodule:: sdeep.datasets

.. autosummary::
    :toctree: generated
    :nosignatures:

    MNISTClassif
    MNISTAutoencoder
    RestorationDataset
    RestorationPatchDataset
    RestorationPatchDatasetLoad
    SegmentationDataset
    SegmentationPatchDataset
    SelfSupervisedDataset
    SelfSupervisedPatchDataset

Transforms
-----------

.. currentmodule:: sdeep.transforms

.. autosummary::
    :toctree: generated
    :nosignatures:

    FlipAugmentation
    RestorationAugmentation
    VisionScale
    VisionCrop

Optims
------

`SDeep` uses optimizers from `PyTorch`. Current available optimizers are:

- Adam
- SGD