# Conductance Based LIF Neuron (coba_lif.py)

This module provides a conductance-based Leaky Integrate-and-Fire (LIF) neuron model.

## Classes

### CobaLIFState

State of a conductance based LIF neuron.

Parameters:
- `z` (torch.Tensor): recurrent spikes
- `v` (torch.Tensor): membrane potential
- `g_e` (torch.Tensor): excitatory input conductance
- `g_i` (torch.Tensor): inhibitory input conductance

### CobaLIFParameters

Parameters of conductance based LIF neuron.

Parameters:
- `tau_syn_exc_inv` (torch.Tensor): inverse excitatory synaptic input time constant
- `tau_syn_inh_inv` (torch.Tensor): inverse inhibitory synaptic input time constant
- `c_m_inv` (torch.Tensor): inverse membrane capacitance
- `g_l` (torch.Tensor): leak conductance
- `e_rev_I` (torch.Tensor): inhibitory reversal potential
- `e_rev_E` (torch.Tensor): excitatory reversal potential
- `v_rest` (torch.Tensor): rest membrane potential
- `v_reset` (torch.Tensor): reset membrane potential
- `v_thresh` (torch.Tensor): threshold membrane potential
- `method` (str): method to determine the spike threshold (relevant for surrogate gradients)
- `alpha` (float): hyper parameter to use in surrogate gradient computation

Default biological parameters:
```python
default_bio_parameters = CobaLIFParameters(
    tau_syn_exc_inv=1 / 0.3,
    tau_syn_inh_inv=1 / 0.5,
    e_rev_E=0.0,
    e_rev_I=-70.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    v_rest=-65.0,
)
```

### CobaLIFFeedForwardState

State of a conductance based feed forward LIF neuron.

Parameters:
- `v` (torch.Tensor): membrane potential
- `g_e` (torch.Tensor): excitatory input conductance
- `g_i` (torch.Tensor): inhibitory input conductance

## Functions

### coba_lif_step

Euler integration step for a conductance based LIF neuron.

Parameters:
- `input_spikes` (torch.Tensor): the input spikes at the current time step
- `state` (CobaLIFState): current state of the neuron
- `input_weights` (torch.Tensor): input weights (sign determines contribution to inhibitory / excitatory input)
- `recurrent_weights` (torch.Tensor): recurrent weights (sign determines contribution to inhibitory / excitatory input)
- `p` (CobaLIFParameters): parameters of the neuron
- `dt` (float): Integration time step

Returns: Tuple of (output spikes, new state)

### coba_lif_feed_forward_step

Euler integration step for a conductance based feed forward LIF neuron.

Parameters:
- `input_tensor` (torch.Tensor): synaptic input
- `state` (CobaLIFFeedForwardState): current state of the neuron
- `p` (CobaLIFParameters): parameters of the neuron
- `dt` (float): Integration time step

Returns: Tuple of (output spikes, new state)

## Usage Example

```python
import torch
from norse.torch.functional.coba_lif import (
    CobaLIFParameters,
    CobaLIFState,
    coba_lif_feed_forward_step
)

# Define parameters
p = CobaLIFParameters()

# Initial state
state = CobaLIFState(v=-65.0, g_e=0.0, g_i=0.0)

# Input tensor
input_tensor = torch.randn(10)  # 10 input neurons

# Step the neuron
output, new_state = coba_lif_feed_forward_step(input_tensor, state, p)
```

## Source

This skill is generated from `/tmp/norse_repo/norse/torch/functional/coba_lif.py`
