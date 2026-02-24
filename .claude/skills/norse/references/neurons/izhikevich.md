# Izhikevich Neuron Model

A computationally efficient model that can reproduce rich dynamical behaviors observed in biological neurons.

## Mathematical Model

The Izhikevich model combines quadratic integrate-and-fire dynamics:

$$\dot{v} = 0.04v^2 + 5v + 140 - u$$

$$\dot{u} = a(bv - u)$$

With spike reset:

$$if\ v \ge 30mV:\ then\ v \leftarrow c,\ u \leftarrow u + d$$

Where:
- $v$: membrane potential
- $u$: recovery variable
- $a$: time scale of recovery
- $b$: sensitivity of recovery to subthreshold oscillations
- $c$: after-spike reset value of $v$
- $d$: after-spike reset value of $u$

## Predefined Spiking Behaviors

Norse provides 20+ predefined parameter sets for different behaviors:

### Excitatory Patterns
| Behavior | Parameters |
|----------|-------------|
| Tonic spiking | `tonic_spiking()` |
| Phasic spiking | `phasic_spiking()` |
| Tonic bursting | `tonic_bursting()` |
| Phasic bursting | `phasic_bursting()` |
| Mixed mode | `mixed_mode()` |
| Spike frequency adaptation | `spike_frequency_adaptation()` |
| Class 1 excitable | `class_1_exc()` |
| Class 2 excitable | `class_2_exc()` |
| Spike latency | `spike_latency()` |
| Subthreshold oscillation | `subthreshold_oscillation()` |
| Resonator | `resonator()` |
| Integrator | `integrator()` |
| Rebound spike | `rebound_spike()` |
| Rebound burst | `rebound_burst()` |
| Threshold variability | `threshold_variability()` |
| Bistability | `bistability()` |
| DAP | `dap()` |
| Accommodation | `accomodation()` |

### Inhibitory Patterns
| Behavior | Parameters |
|----------|-------------|
| Inhibition-induced spiking | `inhibition_induced_spiking()` |
| Inhibition-induced bursting | `inhibition_induced_bursting()` |

## Usage

### Module Layer

```python
import torch
import norse.torch as snn

# Basic Izhikevich neuron
layer = snn.Izhikevich(input_features=100, hidden_features=200)
data = torch.randn(10, 8, 100)  # (time, batch, features)
output, state = layer(data)

# With predefined behavior
layer = snn.IzhikevichRecurrent(
    input_features=100, 
    hidden_features=200,
    p=snn.tonic_spiking()  # Use predefined parameters
)
```

### Functional Layer

```python
import torch
from norse.torch.functional.izhikevich import (
    izhikevich_feed_forward_step,
    izhikevich_recurrent_step,
    IzhikevichState,
    IzhikevichParameters,
    tonic_spiking
)

# Use predefined parameters
p = tonic_spiking()

# Or create custom parameters
p = IzhikevichParameters(
    a=0.02,
    b=0.2,
    c=-65.0,
    d=8.0,
    k=0.0,
    v_rest=-65.0,
    v_reset=-65.0,
    v_th=30.0,
    vpeak=30.0,
    method="super"
)

# Initial state
state = IzhikevichState(
    v=torch.zeros(8, 100),  # membrane potential
    u=torch.zeros(8, 100)    # recovery variable
)

# Single step (recurrent)
z, state = izhikevich_recurrent_step(
    input_tensor, state, input_weights, recurrent_weights, p
)

# Feed-forward
z, state = izhikevich_feed_forward_step(input_tensor, state, p)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `a` | Time scale of recovery variable |
| `b` | Sensitivity of recovery |
| `c` | After-spike reset of v |
| `d` | After-spike reset of u |
| `k` | Potassium conductance |
| `v_rest` | Resting potential |
| `v_reset` | Reset potential |
| `v_th` | Threshold potential |
| `vpeak` | Peak spike potential |
| `method` | Surrogate gradient method |

## State

```python
IzhikevichState(
    v: torch.Tensor,  # Membrane potential
    u: torch.Tensor  # Recovery variable
)
```

## Use Cases

1. **Neuroscience research**: Reproduce specific neuron behaviors
2. **Efficient simulation**: More efficient than detailed compartment models
3. **Diversity**: Different neurons with different dynamics same network
 in