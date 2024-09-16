# sdeep

Set of deep learning models for image restoration and analysis

# Documentation

The documentation is available [here](https://sylvainprigent.github.io/sdeep/about.html)

# Get started

Run a training using the command line:

```bash
sdtrain -p params.json -s /save/dir/path
```

where `params.json` contains the training parameter. Example:

```json
{
  "model": {
    "name": "DnCNN",
    "num_of_layers": 17,
    "channels": 1,
    "features": 64
  },
  "loss": {
    "name": "MSELoss"
  },
  "optim": {
    "name": "Adam",
    "lr": 0.001
  },
  "workflow": {
    "name": "RestorationWorkflow",
    "epoch": 50,
    "train_batch_size": 128,
    "val_batch_size": 4,
    "use_tiling": false
  },
  "train_dataset": {
    "name": "RestorationPatchDataset",
    "source_dir": "C:\\fake\\train\\source\\dir",
    "target_dir": "C:\\fake\\train\\target\\dir",
    "patch_size": 40,
    "stride": 10,
    "use_data_augmentation": true
  },
  "val_dataset": {
    "name": "RestorationDataset",
    "source_dir": "C:\\fake\\val\\source\\dir",
    "target_dir": "C:\\fake\\val\\target\\dir",
    "use_data_augmentation": false
  }
}
```


# Development

## Build the documentation

The documentation is written with Sphinx. To build is run the commands:

```bash
cd docs
pipenv run sphinx-build -b html ./source ./build
```

## Generate the requirements.txt

The `requirements.txt` file is generated from Pipenv with:

```bash
pipenv lock --requirements > requirements.txt
```
