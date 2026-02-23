# LSNN - Long Short-Term Spiking Neural Network

Combines Leaky Integrate-and-Fire neurons with adaptive thresholds to provide long-term memory capabilities in spiking networks.

## Mathematical Model

The LSNN extends LIF with an adaptive threshold:

$$\dot{v} = \frac{1}{\tau_{mem}}(v_{leak} - v + i)$$

$$\dot{i} = -\frac{1}{\tau_{syn}} i$$

$$\dot{b} = -\frac{1}{\tau_b} b$$

Threshold adaptation:

$$v_{th} = v_{th,0} + b$$

Spike generation:

$$z = \Theta(v - v_{th})$$

Where:
- $b$: threshold adaptation variable
- $\tau_b$: adaptation time constant
- $v_{th,0}$: base threshold

## Usage

### Module Layer

```python
import torch
import norse.torch as snn

# Basic LSNN layer
layer = snn.LSNN(input_features=100, hidden_features=200)
data = torch.randn(10, 8, 100)  # (time, batch, features)
output, state = layer(data)

# Recurrent variant
layer = snn.LSNNRecurrent(input_features=100, hidden_features=200)

# Single cell
cell = snn.LSNNCell(input_features=100, hidden_features=200)
```

### Functional Layer

```python
import torch
from norse.torch.functional.lsnn import (
    lsnn_step,
    lsnn_feed_forward_step,
    LSNNState,
    LSNNParameters
)

p = LSNNParameters(
    tau_mem_inv=1/10e-3,
    tau_syn_inv=1/5e-3,
    tau_adapt_inv=1/100e-3,  # Long adaptation time constant
    v_th=1.0,
    v_th_adapt=0.0,  # Initial threshold adaptation
    alpha_adr=0.0,   # Adaptation rate
)

state = LSNNState(
    z=torch.zeros(8, 100),
    v=torch.zeros(8, 100),
    i=torch.zeros(8, 100),
    b=torch.zeros(8, 100)  # Adaptation variable
)

z, state = lsnn_feed_forward_step(input_tensor, state, p)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_adapt_inv` | 1/100e-3 | Inverse adaptation time constant |
| `v_th_adapt` | 0.0 | Initial threshold adaptation |
| `alpha_adr` | 0.0 | Adaptation rate |
| Other LIF params | - | Same as LIF |

## State

```python
LSNNState(
    z: torch.Tensor,  # Output spikes
    v: torch.Tensor,  # Membrane potential
    i: torch.Tensor,   # Synaptic current
    b: torch.Tensor    # Threshold adaptation
)
```

## Key Difference from LIF

LSNN neurons have **memory** through threshold adaptation:
- When a neuron fires, its threshold increases slightly
- This creates competition between neurons
- Longer adaptation time constant (100ms vs typical 10ms) enables temporal integration

## Use Cases

1. **Temporal sequence learning**: Memory over longer timescales
2. **Speech recognition**: Capture temporal dependencies in audio
3. **Neuromorphic computing**: More brain-like dynamics
