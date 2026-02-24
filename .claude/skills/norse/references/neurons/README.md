# Neuron Models Overview

Norse provides a comprehensive set of spiking neuron models. Each model has two representations:

1. **Functional** (`norse.torch.functional`): Stateless functions for single timestep operations
2. **Module** (`norse.torch.module`): Stateful PyTorch modules that handle temporal sequences

## Model Families

| Family | Description | Use Case |
|--------|-------------|----------|
| **LIF** | Leaky Integrate-and-Fire | Most common SNN model |
| **LIFAdEx** | LIF with Adaptive Exponential | Neurons with adaptation |
| **LIFEx** | LIF with Exponential synapses | Fast synaptic dynamics |
| **LIFRefrac** | LIF with Refractory period | Realistic refractoriness |
| **LSNN** | Long Short-Term SNN | Memory in spiking neurons |
| **Izhikevich** | Izhikevich model | Rich dynamical behaviors |
| **IAF** | Integrate-and-Fire | Simple spike generation |
| **LI** | Leaky Integrator | Non-spiking neurons |

## Architecture Pattern

Each neuron model follows this pattern:

### Parameters

```python
from norse.torch import LIFParameters

p = LIFParameters(
    tau_syn_inv=1/5e-3,    # Inverse synaptic time constant
    tau_mem_inv=1/1e-2,    # Inverse membrane time constant  
    v_leak=0.0,            # Leak potential
    v_th=1.0,              # Threshold potential
    v_reset=0.0,           # Reset potential
    method="super",        # Surrogate gradient method
    alpha=100.0            # Surrogate gradient parameter
)
```

### State

Recurrent neurons have three state components:
- `z`: Output spikes
- `v`: Membrane potential
- `i`: Synaptic current

Feed-forward neurons have:
- `v`: Membrane potential
- `i`: Synaptic current

### Cell vs Layer

- **Cell**: Single timestep processing
- **Layer**: Temporal sequence processing (iterates over time dimension)

```python
# Single timestep (Cell)
cell = snn.LIFCell()
output, state = cell(input_timestep, state)

# Multiple timesteps (Layer)
layer = snn.LIF()
output, state = layer(input_sequence)  # input: (time, batch, features)
```

## Choosing a Model

See detailed guides:
- [LIF Family](lif.md) - Most common, start here
- [Izhikevich](izhikevich.md) - Rich dynamics
- [LSNN](lsnn.md) - Long-term memory
- [IAF](iaf.md) - Simple integrate-and-fire
- [Other Models](other.md) - LI, CobaLIF, etc.
