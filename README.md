
# scnndeconv

Set of deep learning models for image denoising and deconvolution

# Documentation

The documentation is available [here](https://bioimageit.github.io/scnndeconv/).

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
