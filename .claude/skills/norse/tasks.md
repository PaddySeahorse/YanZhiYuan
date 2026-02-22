# Running Tasks

Norse arrives with a number of built-in examples called tasks. 
These tasks serve to 1) illustrate what types of tasks *can* be done with Norse
and 2) how to use Norse for specific tasks. Please note that some tasks require
additional dependencies, like OpenAI gym for the cartpole task, which is not included in 
vanilla Norse.

## Parameters

The tasks below use a large number of configurable parameters to control the model/network size, 
epochs and batch size, task load, pytorch device, learning rate, and many more parameters.

Another very important parameter determines which backpropagation model to use.
This is particularly important for spiking neural network models and is described in more 
detail on the page about learning.

All programs below are built with Abseil Python which gives you extensive command line interface (CLI) help descriptions, where you can look up further
descriptions - and find even more - of the parameters above. You can access these by
using the `--help` flag on any tasks below, for instance `python -m norse.task.mnist --help`.

## Cartpole

This task is a balancing exercise where a controller learns to counter the gravitational force
on an upright cartpole. You will need to install OpenAI Gym to provide the simulation environments for the robot.

```bash
pip install gym
python3 -m norse.task.cartpole
```

## Cifar10

Cifar10 is a labeled database of 60'000 32x32 images in 10 classes. The task is to learn to classify each image.

```bash
python3 -m norse.task.cifar10
```

## Correlation experiment

The correlation experiment serves to demonstrate how neurons can learn patterns with a certain probability.

```bash
python3 -m norse.task.correlation_experiment
```

## Memory task

The memory task demonstrates how a recurrent spiking neural network can store a pattern and later recall it.

```bash
python3 -m norse.task.memory
```

## MNIST

MNIST is a database of 70'000 handwritten digits where the task is to learn to classify each image as one of the 10 digits.

```bash
python3 -m norse.task.mnist
```

## MNIST in PyTorch Lightning

This task is similar to the MNIST task above, but built with PyTorch Lightning.
PyTorch Lightning is a library to build, train, scale, and verify a model with little overhead.
It also provides GPU parallelisation, logging with e.g. Tensorboard, model checkpointing, and much more.

**Note** that the task depends on an installation of PyTorch Lightning: `pip install pytorch-lightning`

```bash
python -m norse.task.mnist_pl
```

## Speech Command recognition task

The speech commands dataset serves as an example of a temporal classification task.

```bash
python -m norse.task.speech_commands.run
```
