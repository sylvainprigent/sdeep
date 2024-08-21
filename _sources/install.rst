Install
=======

This section contains the instructions to install ``SDeep``

Using pip
---------

Releases are available in a GitHub repository. We recommend using virtual environment.
Depending on the GPU and ``PyTorch`` version you are using you may need to install various packages.
For default local usage:

.. code-block:: shell

    python -m venv .venv
    source .env/bin/activate
    pip install https://github.com/sylvainprigent/sdeep/archive/master.zip


From source
-----------

If you plan to develop ``SDeep`` or want to install locally from sources

.. code-block:: shell

    python -m venv .venv
    source .venv/bin/activate
    git clone https://github.com/sylvainprigent/sdeep.git
    cd sdeep
    pip install -e .


Startup
-------

You are now ready to use ``SDeep``
You can run the MNIST example to test your install.

1. Create a working directory with the following `params.json` parameter file:

.. code-block:: javascript

   {
        "model": {
          "name": "MNistClassifier"
        },
        "loss": {
          "name": "CrossEntropyLoss"
        },
        "optim": {
          "name": "Adam",
          "lr": 0.001
        },
        "workflow": {
          "name": "ClassificationWorkflow",
          "epochs": 3,
          "train_batch_size": 64,
          "val_batch_size": 1000,
          "save_all": true
        },
        "train_dataset": {
          "name": "MNISTClassif",
          "dir_name": "mnist_train",
          "train": true
        },
        "train_transform": {
          "name": "VisionScale"
        },
        "val_dataset": {
          "name": "MNISTClassif",
          "dir_name": "mnist_train",
          "train": false
        },
        "val_transform": {
          "name": "VisionScale"
        },
        "eval": {
          "name": "EvalClassification"
        }
    }

2. Run the training command

.. code-block:: shell

    source .venv/bin/activate
    sdtrain -p params.json
