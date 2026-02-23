# LIF - Leaky Integrate-and-Fire

The most common and fundamental spiking neuron model in Norse. Combines a leaky integrator with a threshold to produce spikes.

## Mathematical Model

The LIF neuron dynamics:

$$\dot{v} = \frac{1}{\tau_{mem}}(v_{leak} - v + i)$$

$$\dot{i} = -\frac{1}{\tau_{syn}} i$$

Spike generation:

$$z = \Theta(v - v_{th})$$

Reset:

$$v = (1-z)v + z \cdot v_{reset}$$

Where:
- $v$: membrane potential
- $i$: synaptic current
- $\tau_{mem}$: membrane time constant
- $\tau_{syn}$: synaptic time constant
- $v_{th}$: spike threshold
- $z$: output spike (0 or 1)

## Usage

### Module Layer (Recommended)

```python
import torch
import norse.torch as snn

# Recurrent LIF with temporal sequence
layer = snn.LIF(input_features=100, hidden_features=200)
data = torch.randn(10, 8, 100)  # (time, batch, features)
output, state = layer(data)

# Single Cell for timestep processing
cell = snn.LIFCell(input_features=100, hidden_features=200)
output, state = cell(input_tensor, state)

# Recurrent variant with feedback
recurrent = snn.LIFRecurrent(input_features=100, hidden_features=200)
output, state = recurrent(data)
```

### Functional Layer

```python
import torch
from norse.torch.functional.lif import (
    lif_step, 
    lif_feed_forward_step,
    LIFState, 
    LIFFeedForwardState,
    LIFParameters
)

# Parameters
p = LIFParameters(
    tau_mem_inv=1/10e-3,  # 10ms membrane time constant
    tau_syn_inv=1/5e-3,    # 5ms synaptic time constant
    v_th=1.0
)

# Initial state
state = LIFState(z=torch.zeros(8, 100), v=torch.zeros(8, 100), i=torch.zeros(8, 100))

# Single step
input_spikes = torch.randn(8, 100)
z, state = lif_step(input_spikes, state, input_weights, recurrent_weights, p)

# Feed-forward single step
ff_state = LIFFeedForwardState(v=torch.zeros(8, 100), i=torch.zeros(8, 100))
z, ff_state = lif_feed_forward_step(input_tensor, ff_state, p)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_syn_inv` | 1/5e-3 | Inverse synaptic time constant (1/ms) |
| `tau_mem_inv` | 1/1e-2 | Inverse membrane time constant (1/ms) |
| `v_leak` | 0.0 | Leak potential (mV) |
| `v_th` | 1.0 | Threshold potential (mV) |
| `v_reset` | 0.0 | Reset potential (mV) |
| `method` | "super" | Surrogate gradient method |
| `alpha` | 100.0 | Surrogate gradient steepness |

## State

### LIFState (Recurrent)
```python
LIFState(
    z: torch.Tensor,  # Output spikes
    v: torch.Tensor,  # Membrane potential
    i: torch.Tensor   # Synaptic current
)
```

### LIFFeedForwardState (Feed-forward)
```python
LIFFeedForwardState(
    v: torch.Tensor,  # Membrane potential
    i: torch.Tensor   # Synaptic current
)
```

## Biological Default

```python
# Use biologically realistic parameters
p_bio = LIFParameters.bio_default()
```

## Variants

| Variant | Description |
|---------|-------------|
| `LIFCell` | Single timestep, recurrent |
| `LIF` | Temporal sequence, recurrent |
| `LIFRecurrentCell` | Alias for LIFCell |
| `LIFRecurrent` | Alias for LIF |

## Examples

### Simple Spiking Layer

```python
import torch
import torch.nn as nn
import norse.torch as snn

model = snn.SequentialState(
    nn.Linear(784, 256),
    snn.LIFCell(),
    nn.Linear(256, 10),
    snn.LICell(),  # Non-spiking output layer
)

x = torch.randn(32, 784)
output, states = model(x)
```

### Convolutional SNN

```python
model = snn.SequentialState(
    nn.Conv2d(1, 32, 3, 1),
    snn.LIFCell(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, 1),
    snn.LIFCell(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1024, 10),
    snn.LICell(),
)
```
