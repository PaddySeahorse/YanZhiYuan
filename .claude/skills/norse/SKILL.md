---
name: norse
description: A deep learning library for spiking neural networks (SNN). Use when working with bio-inspired neural components, event-driven sparse operations, neuron models (LIF, Izhikevich, LSNN), encoding methods (poisson, population, latency), STDP learning, or building stateful SNN architectures.
references:
  - neurons/README.md
  - neurons/lif.md
  - neurons/izhikevich.md
  - neurons/lsnn.md
  - neurons/iaf.md
  - neurons/other.md
  - encoding/README.md
  - encoding/poisson.md
  - encoding/population.md
  - encoding/latency.md
  - learning/README.md
  - learning/stdp.md
  - networks/README.md
  - networks/sequential.md
  - networks/models.md
  - activation/README.md
---

# Norse - Spiking Neural Networks for PyTorch

Norse is a deep learning library for spiking neural networks (SNN) that extends PyTorch with bio-inspired neural components.

## Quick Start

```python
import torch
import norse.torch as snn

# Simple spiking neural network
model = snn.SequentialState(
    snn.Lift(torch.nn.Conv2d(1, 20, 5, 1)),  # Conv layer
    snn.LIFCell(),                             # LIF spiking layer
    torch.nn.MaxPool2d(2, 2),
    snn.LIFCell(),
    torch.nn.Flatten(),
    snn.LICell(),                              # Non-spiking integrator
)

# Run the network
data = torch.randn(8, 1, 28, 28)  # Batch of 8, 1 channel, 28x28
output, state = model(data)
```

## Architecture Overview

Norse follows PyTorch's design pattern with two layers:

1. **Functional layer** (`norse.torch.functional`): Stateless operations, single timestep
2. **Module layer** (`norse.torch.module`): Stateful PyTorch modules, handles sequences

### Cell vs Layer Naming

- **Cell**: Processes single timestep (e.g., `LIFCell`)
- **Layer**: Processes temporal sequences (e.g., `LIF`)

### State Management

Norse uses named tuples for neuron state:

```python
from norse.torch import LIFState, LIFFeedForwardState, LIFParameters

# Recurrent state (with spikes)
state = LIFState(z=torch.zeros(8, 100), v=torch.zeros(8, 100), i=torch.zeros(8, 100))

# Feed-forward state (no recurrent spikes)
ff_state = LIFFeedForwardState(v=torch.zeros(8, 100), i=torch.zeros(8, 100))

# Custom parameters
p = LIFParameters(tau_mem_inv=1/10e-3, v_th=1.0)
```

## Decision Trees

### I need a neuron model

```
What type of neuron?
├── Leaky integrate-and-fire (most common) → LIF family
│   ├── Basic → LIF / LIFCell / LIFRecurrent
│   ├── With adaptation → LIFAdEx / LIFAdExRefrac
│   ├── Exponential synapses → LIFEx
│   ├── With refractory period → LIFRefrac
│   └── Conductance-based → CobaLIF
├── Long Short-Term Memory → LSNN / LSNNRecurrent
├── Izhikevich (rich dynamics) → Izhikevich / IzhikevichRecurrent
├── Integrate-and-fire (no leak) → IAF
└── Simple integrator (no spikes) → LI / LICell
```

### I need to encode data

```
What encoding method?
├── Rate-based (random spikes) → Poisson encoding
│   ├── Standard → poisson_encode / PoissonEncoder
│   └── Signed (bipolar) → signed_poisson_encode / SignedPoissonEncoder
├── Population encoding (distributed representation) → population_encode / PopulationEncoder
├── Latency encoding (timing-based) → spike_latency_encode / SpikeLatencyEncoder
└── Current-based → constant_current_lif_encode / ConstantCurrentLIFEncoder
```

### I need learning/plasticity

```
What learning rule?
├── Spike-timing-dependent plasticity → STDP
│   ├── Functional → stdp_step_linear
│   └── Sensor → stdp_sensor_step
├── Short-term plasticity → Tsodyks-Makram (stp_step)
└── Regularization → regularize_step
```

### I need to build a network

```
What network structure?
├── Sequential layers with state → SequentialState
├── Recurrent SNN → RecurrentSequential
├── Pre-built models → ConvNet, VGG, MobileNetV2
└── Custom layer → lift() for converting regular PyTorch layers
```

## Core API Index

### Neuron Models

| Model | Module | Functional | Description |
|-------|--------|------------|-------------|
| LIF | `LIF`, `LIFCell`, `LIFRecurrent`, `LIFRecurrentCell` | `lif_step`, `lif_feed_forward_step` | Leaky Integrate-and-Fire (most common) |
| LIFAdEx | `LIFAdEx`, `LIFAdExCell` | `lif_adex_step`, `lif_adex_feed_forward_step` | LIF with Adaptive Exponential |
| LIFEx | `LIFEx`, `LIFExCell` | `lif_ex_step`, `lif_ex_feed_forward_step` | LIF with Exponential synapses |
| LIFRefrac | `LIFRefrac`, `LIFRefracCell` | `lif_refrac_step`, `lif_refrac_feed_forward_step` | LIF with Refractory period |
| LSNN | `LSNN`, `LSNNCell`, `LSNNRecurrent` | `lsnn_step`, `lsnn_feed_forward_step` | Long Short-Term SNN |
| Izhikevich | `Izhikevich`, `IzhikevichCell` | `izhikevich_feed_forward_step` | Izhikevich model |
| IAF | `IAF`, `IAFCell` | `iaf_step`, `iaf_feed_forward_step` | Integrate-and-Fire (no leak) |
| LI | `LI`, `LICell` | `li_step`, `li_feed_forward_step` | Leaky Integrator (no spikes) |
| CobaLIF | `CobaLIFCell` | `coba_lif_step`, `coba_lif_feed_forward_step` | Conductance-based LIF |

### Encoders

| Encoder | Module | Functional | Description |
|---------|--------|------------|-------------|
| Poisson | `PoissonEncoder` | `poisson_encode` | Rate-based random encoding |
| Signed Poisson | `SignedPoissonEncoder` | `signed_poisson_encode` | Bipolar rate encoding |
| Population | `PopulationEncoder` | `population_encode` | Distributed population coding |
| Spike Latency | `SpikeLatencyEncoder` | `spike_latency_encode` | Timing-based encoding |
| Constant Current | `ConstantCurrentLIFEncoder` | `constant_current_lif_encode` | Current-based LIF encoding |

### Learning

| Function | Description |
|----------|-------------|
| `stdp_step_linear` | STDP for feed-forward layers |
| `stdp_sensor_step` | STDP for correlation sensors |
| `stp_step` | Short-term plasticity (Tsodyks-Makram) |
| `regularize_step` | Spike/voltage regularization |

### Networks

| Class | Description |
|-------|-------------|
| `SequentialState` | Sequential layers with state management |
| `RecurrentSequential` | Recurrent network with feedback |
| `ConvNet` | Simple convolutional SNN |
| `ConvNet4` | 4-layer convolutional SNN |
| `VGG` | VGG-style SNN |
| `MobileNetV2` | MobileNetV2-style SNN |
| `Lift` | Convert regular PyTorch layers to spiking |

### Activation Functions

| Function | Description |
|----------|-------------|
| `heaviside` | Step function for spikes |
| `super_fn` | SuperSpike surrogate gradient |
| `threshold` | Configurable threshold function |

## Key Concepts

### Biological Parameters

Use `bio_default()` for biologically realistic parameters:

```python
from norse.torch import LIFParameters

# Default technical parameters
p = LIFParameters()

# Biologically realistic parameters
p_bio = LIFParameters.bio_default()
```

### Surrogate Gradients

Control spike behavior with the `method` parameter:

```python
p = LIFParameters(method="super")    # SuperSpike (default)
p = LIFParameters(method="adjoint")  # Adjoint method
p = LIFParameters(method="heaviside") # Straight-through estimator
```

### Time Dimension

SNNs process data over time. First dimension is typically timesteps:

```python
# Input: (timesteps, batch, features)
data = torch.zeros(10, 8, 100)  # 10 timesteps, batch of 8, 100 features

# Output: (timesteps, batch, features)
output, state = layer(data)
```

### NIR Interoperability

Norse supports Neural Intermediate Representation (NIR) for interoperability:

```python
from norse.torch.utils import from_nir, to_nir

# Export to NIR
nir_graph = to_nir(model, example_input)

# Import from NIR
model = from_nir(nir_graph)
```

## Installation

```bash
pip install norse
# or
conda install norse
```

## Common Patterns

### Basic MNIST Classifier

```python
import torch
import torch.nn as nn
import norse.torch as snn

model = snn.SequentialState(
    nn.Conv2d(1, 20, 5, 1),
    snn.LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),
    snn.LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(800, 10),
    snn.LICell(),
)

data = torch.randn(8, 1, 28, 28)
output, state = model(data)
```

### With Encoding Pipeline

```python
import torch
import norse.torch as snn

# Encode input with Poisson encoding
encoder = snn.PoissonEncoder()
data = torch.rand(8, 1, 28, 28)  # Input values in [0, 1]
spikes = encoder(data, seq_length=10)  # (10, 8, 1, 28, 28)

# Process with SNN
layer = snn.LIF(10)
output, state = layer(spikes)
```

## Detailed Topics

* **Neuron Models**: [references/neurons/README.md](references/neurons/README.md)
  * LIF: [references/neurons/lif.md](references/neurons/lif.md)
  * Izhikevich: [references/neurons/izhikevich.md](references/neurons/izhikevich.md)
  * LSNN: [references/neurons/lsnn.md](references/neurons/lsnn.md)
  * IAF: [references/neurons/iaf.md](references/neurons/iaf.md)
  * Other: [references/neurons/other.md](references/neurons/other.md)

* **Encoding**: [references/encoding/README.md](references/encoding/README.md)
  * Poisson: [references/encoding/poisson.md](references/encoding/poisson.md)
  * Population: [references/encoding/population.md](references/encoding/population.md)
  * Latency: [references/encoding/latency.md](references/encoding/latency.md)

* **Learning**: [references/learning/README.md](references/learning/README.md)
  * STDP: [references/learning/stdp.md](references/learning/stdp.md)

* **Networks**: [references/networks/README.md](references/networks/README.md)
  * SequentialState: [references/networks/sequential.md](references/networks/sequential.md)
  * Pre-built models: [references/networks/models.md](references/networks/models.md)

* **Activation Functions**: [references/activation/README.md](references/activation/README.md)
