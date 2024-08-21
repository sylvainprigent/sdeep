About
=====

`SDeep` is a python framework to ease the training of light deep learning models locally.
Each training components (transforms, datasets, models, training workflow, evaluation...) are
implemented as independent python classes and a single command line with a composition JSON file
allow to train all the combinations of components and hyper-parameters without rewriting code.

.. image:: images/caroussel/1.png
  :width: 250
  :alt: Alternative text

.. image:: images/caroussel/2.png
  :width: 250
  :alt: Alternative text

|


.. image:: images/caroussel/4.png
  :width: 250
  :alt: Alternative text

.. image:: images/caroussel/5.png
  :width: 250
  :alt: Alternative text

.. image:: images/caroussel/6.png
  :width: 250
  :alt: Alternative text

Why `SDeep`
-----------

- When training a model, we need to test various workflows and hyper-parameters. It become then
  cumbersome to repeat redundant script
- When training scripts are saved as notebooks or independent files, it is hard and need extra work
  to adapt it to another computer or dataset.
- Ensuring traceability (FAIR principles), of data and results is an extra-work
  to include code, models, validations results and predictions

SDeep tends to encapsulate all the training procedure and traceability into one framework.

Principles
----------

Principle 1: Write only once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aim of `SDeep` is to avoid writing over and over training loops, datasets or loss. We write only
once each component and then a workflow to assemble them. We can then test them, version them and
reuse them in various environment and datasets.

Principle 2: Commands not scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Commands based programs are much more easier to maintain and distribute than a collection of
dedicate scripts. With a standardised command line any user known how to install and run the
program. Using scripts need extra work for the user to install dependencies, adapt the inputs,
outputs, and identify hyper-parameters locations.

Principle 3: Keep it simple, stupid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`SDeep` is based on `pyTorch` and reuse all the existing components. No re-invention of loss,
dataset...
`SDeep` only introduce:

1. The *Workflow* class for assembling and monitoring a training
2. A unique API and CLI with a plugin discovery mechanism to run and monitor trainings
