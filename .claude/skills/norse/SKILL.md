# Norse - Deep Learning Library for Spiking Neural Networks

A deep learning library for spiking neural networks (SNN).

## Source Files

This skill is generated from the Norse project documentation. The following source files are available:

- `/tmp/norse_repo/README.md` - Main project README
- `/tmp/norse_repo/contributing.md` - Contribution guidelines
- `/tmp/norse_repo/docs/pages/hardware.rst` - Hardware acceleration documentation
- `/tmp/norse_repo/docs/norse.torch.rst` - norse.torch module documentation
- `/tmp/norse_repo/docs/norse.torch.functional.rst` - norse.torch.functional documentation
- `/tmp/norse_repo/docs/pages/development.md` - Development documentation
- `/tmp/norse_repo/docs/pages/tasks.rst` - Running tasks documentation
- `/tmp/norse_repo/docs/pages/about.rst` - About Norse
- `/tmp/norse_repo/docs/api.rst` - Complete API documentation
- `/tmp/norse_repo/docs/index.rst` - Documentation index
- `/tmp/norse_repo/docs/_toc.yml` - Table of contents
- `/tmp/norse_repo/norse/torch/functional/coba_lif.py` - Conductance based LIF neuron
- `/tmp/norse_repo/norse/benchmark/README.md` - Benchmark documentation

## Quick Links

- **Installation**: See `/tmp/norse_repo/README.md` section 2.1
- **Examples**: See `/tmp/norse_repo/README.md` sections 2.3 and 2.4
- **Tasks**: See `/tmp/norse_repo/docs/pages/tasks.rst`
- **Hardware Acceleration**: See `/tmp/norse_repo/docs/pages/hardware.rst`
- **Development**: See `/tmp/norse_repo/docs/pages/development.md`
- **API Reference**: See `/tmp/norse_repo/docs/norse.torch.rst` and `/tmp/norse_repo/docs/norse.torch.functional.rst`

## Usage

To use Norse for spiking neural networks:

```python
import torch
import norse.torch as snn

# Create a Leaky Integrate-and-Fire (LIF) neuron
layer = snn.LIFCell(input_features=10, hidden_features=20)

# Run on input data
data = torch.randn(8, 10)  # batch_size=8, input_features=10
output, state = layer(data)
```
