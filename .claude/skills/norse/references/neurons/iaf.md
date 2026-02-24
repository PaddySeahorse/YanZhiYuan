# IAF - Integrate-and-Fire

A simple neuron model without the "leak" term, integrating input current until reaching a threshold.

## Mathematical Model

$$\dot{v} = i$$

Spike generation:

$$z = \Theta(v - v_{th})$$

Reset:

$$v = v - z \cdot v_{th}$$

Unlike LIF, there is no leak - the membrane potential only increases with input current.

## Usage

### Module Layer

```python
import torch
import norse.torch as snn

# IAF layer
layer = snn.IAF(input_features=100, hidden_features=200)
data = torch.randn(10, 8, 100)
output, state = layer(data)

# Single cell
cell = snn.IAFCell(input_features=100, hidden_features=200)
```

### Functional Layer

```python
import torch
from norse.torch.functional.iaf import (
    iaf_step,
    iaf_feed_forward_step,
    IAFState,
    IAFFeedForwardState,
    IAFParameters
)

p = IAFParameters(
    v_th=1.0,       # Threshold potential
    v_reset=0.0,    # Reset potential (added after spike)
)

# Recurrent state
state = IAFState(
    z=torch.zeros(8, 100),
    v=torch.zeros(8, 100)
)

# Feed-forward state  
ff_state = IAFFeedForwardState(v=torch.zeros(8, 100))

z, state = iaf_step(input_tensor, state, input_weights, recurrent_weights, p)
z, ff_state = iaf_feed_forward_step(input_tensor, ff_state, p)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_th` | 1.0 | Threshold potential |
| `v_reset` | 0.0 | Reset potential |

## State

```python
# Recurrent
IAFState(
    z: torch.Tensor,  # Output spikes
    v: torch.Tensor   # Membrane potential
)

# Feed-forward
IAFFeedForwardState(
    v: torch.Tensor   # Membrane potential
)
```

## Use Cases

1. **Simple spike generation**: When leak is not needed
2. **Energy-efficient SNN**: Simple dynamics, low computation
3. **Leaky Integrate-and-Fire variant**: If you need leak, use LIF instead
