---
name: norse-torch
description: Norse torch module API documentation. Building blocks for spiking neural networks including neuron models, encoding, and convolutions.
allowed-tools: Bash
---

# norse.torch

Building blocks for spiking neural networks based on PyTorch.

## Contents

- Containers
- Convolutions
- Encoding
- Neuron models

## Containers

- Lift
- SequentialState
- RegularizationCell

## Convolutions

- LConv2d

## Encoding

- ConstantCurrentLIFEncoder
- PoissonEncoder
- PoissonEncoderStep
- PopulationEncoder
- SignedPoissonEncoder
- SpikeLatencyEncoder
- SpikeLatencyLIFEncoder

## Neuron Models

### Integrate-and-fire

Simple integrators that sums up incoming signals until a threshold.

- IAFFeedForwardState
- IAFParameters
- IAFCell

### Izhikevich

- IzhikevichParameters
- IzhikevichState
- IzhikevichSpikingBehavior
- Izhikevich
- IzhikevichCell
- IzhikevichRecurrent
- IzhikevichRecurrentCell

### Leaky Integrator

- LIState
- LIParameters
- LI
- LICell
- LILinearCell

### Leaky Integrate-and-Fire (LIF)

- LIFParameters
- LIFState
- LIF
- LIFCell
- LIFRecurrent
- LIFRecurrentCell

### LIF, Box Model

- LIFBoxFeedForwardState
- LIFBoxParameters
- LIFBoxCell

### LIF, Conductance Based

- CobaLIFCell (see [norse/torch/functional/coba_lif.py](norse/torch/functional/coba_lif.py))

### LIF, Adaptive Exponential

- LIFAdEx
- LIFAdExCell
- LIFAdExRecurrent
- LIFAdExRecurrentCell

### LIF, Exponential

- LIFEx
- LIFExCell
- LIFExRecurrent
- LIFExRecurrentCell

### LIF, Multicompartmental

- LIFMCRecurrentCell

### LIF, Multicompartmental with Refraction

- LIFMCRefracRecurrentCell

### LIF, Refractory

- LIFRefracCell
- LIFRefracRecurrentCell

### Long Short-Term Memory (LSNN)

- LSNN
- LSNNCell
- LSNNRecurrent
- LSNNRecurrentCell

## Receptive Fields

- SpatialReceptiveField2d
- TemporalReceptiveField

## Related Documentation

- [Functional API](docs/norse.torch.functional.rst)
- [Development Documentation](docs/pages/development.md)
- [Main README](README.md)
