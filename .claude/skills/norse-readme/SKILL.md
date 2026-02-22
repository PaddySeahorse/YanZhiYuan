---
name: norse-readme
description: Norse - A deep learning library for spiking neural networks. Main README with installation, examples, and overview.
allowed-tools: Bash
---

# Norse - Deep Learning Library for Spiking Neural Networks

## Overview

Norse aims to exploit the advantages of bio-inspired neural components, which are sparse and event-driven - a fundamental difference from artificial neural networks.
Norse expands PyTorch with primitives for bio-inspired neural components, bringing you two advantages: a modern and proven infrastructure based on PyTorch and deep learning-compatible spiking neural network components.

**Documentation**: [docs/index.rst](docs/index.rst)

## Getting Started

The fastest way to try Norse is via the jupyter notebooks in the [norse/notebooks](https://github.com/norse/notebooks) repository.

## Using Norse

Norse presents plug-and-play components for deep learning with spiking neural networks.
Here, we describe how to install Norse and start to apply it in your own work.
[Read more in our documentation](docs/pages/working.md).

### Installation

We assume you are using **Python version 3.8+** and have **installed PyTorch version 1.9 or higher**. 
[Read more about the prerequisites in our documentation](docs/pages/installing.md).

#### Installation Methods

| Method | Instructions | Prerequisites |
|--------|--------------|---------------|
| From PyPi | `pip install norse` | Pip |
| From source | `pip install -qU git+https://github.com/norse/norse` | Pip, PyTorch |
| With Docker | `docker pull quay.io/norse/norse` | Docker |
| From Conda | `conda install norse` | Anaconda or Miniconda |

For troubleshooting, please refer to our [installation guide](docs/pages/installing.md#installation-troubleshooting).

### Running Examples

Norse is bundled with a number of example tasks, serving as short, self contained, correct examples (SSCCE).
They can be run by invoking the `norse` module from the base directory.
More information and tasks are available [in our documentation](docs/pages/tasks.rst) and in your console by typing: `python -m norse.task.<task> --help`.

- To train an MNIST classification network: `python -m norse.task.mnist`
- To train a CIFAR classification network: `python -m norse.task.cifar10`
- To train the cartpole balancing task with Policy gradient: `python -m norse.task.cartpole`

Norse is compatible with PyTorch Lightning, as demonstrated in the [PyTorch Lightning MNIST task variant](norse/task/mnist_pl.py):

```bash
python -m norse.task.mnist_pl --gpus=4
```

### Example: Spiking Convolutional Classifier

This classifier is taken from our tutorial on training a spiking MNIST classifier and achieves >99% accuracy.

```python
import torch, torch.nn as nn
from norse.torch import LICell             # Leaky integrator
from norse.torch import LIFCell            # Leaky integrate-and-fire
from norse.torch import SequentialState    # Stateful sequential layers

model = SequentialState(
    nn.Conv2d(1, 20, 5, 1),      # Convolve from 1 -> 20 channels
    LIFCell(),                   # Spiking activation layer
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),     # Convolve from 20 -> 50 channels
    LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),                # Flatten to 800 units
    nn.Linear(800, 10),
    LICell(),                    # Non-spiking integrator layer
)

data = torch.randn(8, 1, 28, 28) # 8 batches, 1 channel, 28x28 pixels
output, state = model(data)      # Provides a tuple (tensor (8, 10), neuron state)
```

### Example: Long Short-Term Spiking Neural Networks

The long short-term spiking neural networks from the paper by G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass (2018) is another interesting way to apply norse:

```python
import torch
from norse.torch import LSNNRecurrent
# Recurrent LSNN network with 2 input neurons and 10 output neurons
layer = LSNNRecurrent(2, 10)
# Generate data: 20 timesteps with 8 datapoints per batch for 2 neurons
data  = torch.zeros(20, 8, 2)
# Tuple of (output spikes of shape (20, 8, 2), layer state)
output, new_state = layer(data)
```

## Why Norse?

Norse was created for two reasons: 
1. Apply findings from decades of research in practical settings
2. Accelerate our own research within bio-inspired learning

We are passionate about Norse: we strive to follow best practices and promise to maintain this library for the simple reason that we depend on it ourselves.
We have implemented a number of neuron models, synapse dynamics, encoding and decoding algorithms, dataset integrations, tasks, and examples.
Combined with the PyTorch infrastructure and our high coding standards, we have found Norse to be an excellent tool for modelling *scaleable* experiments.

Finally, we are working to keep Norse as performant as possible. 
Preliminary benchmarks suggest that Norse achieves excellent performance on small networks of up to ~5000 neurons per layer.
See [norse/benchmark/README.md](norse/benchmark/README.md) for more details.

[Read more about Norse in our documentation](docs/pages/about.rst).

## Similar Work

We refer to the Neuromorphic Software Guide for a comprehensive list of software for neuromorphic computing.

## Contributing

Contributions are warmly encouraged and always welcome. However, we also have high expectations around the code base so if you wish to contribute, please refer to our [contribution guidelines](contributing.md).

## Credits

Norse is created by:
- Christian Pehle (@GitHub cpehle), PostDoc at University of Heidelberg, Germany
- Jens E. Pedersen (@GitHub jegp), doctoral student at KTH Royal Institute of Technology, Sweden

## Citation

If you use Norse in your work, please cite it as follows:

```BibTex
@software{norse2021,
  author       = {Pehle, Christian and
                  Pedersen, Jens Egholm},
  title        = {{Norse -  A deep learning library for spiking 
                   neural networks}},
  month        = jan,
  year         = 2021,
  note         = {Documentation: https://norse.ai/docs/},
  publisher    = {Zenodo},
  version      = {0.0.7},
  doi          = {10.5281/zenodo.4422025},
  url          = {https://doi.org/10.5281/zenodo.4422025}
}
```

## License

LGPLv3. See [LICENSE](LICENSE) for license details.
