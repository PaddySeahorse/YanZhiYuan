# STDP - Spike-Timing-Dependent Plasticity

Biologically-inspired learning rule where synaptic strength depends on the relative timing of pre- and post-synaptic spikes.

## Mathematical Model

STDP strengthens synapses when pre-synaptic spikes precede post-synaptic spikes (causal), and weakens them otherwise:

$$\Delta w = \begin{cases} 
A_+ \cdot e^{-\Delta t/\tau_+} & \text{if } \Delta t > 0 \\
-A_- \cdot e^{\Delta t/\tau_-} & \text{if } \Delta t < 0
\end{cases}$$

Where:
- $\Delta t = t_{post} - t_{pre}$: Time difference
- $A_+, A_-$: Learning rates
- $\tau_+, \tau_-$: Time constants
- $\Delta t > 0$: Pre before post (potentiation)
- $\Delta t < 0$: Post before pre (depression)

## Usage

### STDP Parameters

```python
from norse.torch.functional.stdp import STDPParameters

p = STDPParameters(
    a_pre=1.0,           # Presynaptic trace contribution
    a_post=1.0,          # Postsynaptic trace contribution
    tau_pre_inv=1/50e-3, # Presynaptic trace time constant
    tau_post_inv=1/50e-3,# Postsynaptic trace time constant
    w_min=0.0,           # Minimum weight
    w_max=1.0,           # Maximum weight
    eta_plus=1e-3,       # Potentiation learning rate
    eta_minus=1e-3,      # Depression learning rate
    stdp_algorithm="additive",  # Algorithm variant
    mu=0.0,              # Exponent for multiplicative
    hardbound=True       # Clip weights to bounds
)
```

### STDP Algorithm Variants

| Algorithm | Description |
|-----------|-------------|
| `"additive"` | Linear weight change (default) |
| `"additive_step"` | Additive with hard bounds |
| `"multiplicative_pow"` | Multiplicative with exponent |
| `"multiplicative_relu"` | Multiplicative with ReLU |

### STDP State

```python
from norse.torch.functional.stdp import STDPState

state = STDPState(
    t_pre=torch.zeros(100, 100),  # Presynaptic traces
    t_post=torch.zeros(100, 100)  # Postsynaptic traces
)
```

### Linear STDP Step

```python
import torch
from norse.torch.functional.stdp import (
    stdp_step_linear,
    STDPParameters,
    STDPState
)

# Forward pass first
pre_layer = snn.LIFRecurrent(784, 500)
post_layer = snn.LIFRecurrent(500, 10)

z_pre, s_pre = pre_layer(input_data)
z_post, s_post = post_layer(z_pre)

# Apply STDP update
w_new, state_new = stdp_step_linear(
    z_pre=z_pre,          # Presynaptic spikes
    z_post=z_post,        # Postsynaptic spikes
    w=weight_matrix,      # Current weights
    state_stdp=stdp_state,
    p_stdp=params,
    dt=0.001
)
```

### STDP Sensor

For correlation-based learning in sensory processing:

```python
from norse.torch.functional.stdp_sensor import (
    stdp_sensor_step,
    STDPSensorParameters,
    STDPSensorState
)

z, state = stdp_sensor_step(
    input_tensor=z_pre,
    state=state,
    p=STDPSensorParameters(),
    dt=0.001
)
```

## Complete Example

```python
import torch
import norse.torch as snn
from norse.torch.functional.stdp import (
    stdp_step_linear,
    STDPParameters,
    STDPState
)

# Two-layer network
pre_neurons = 100
post_neurons = 50

# Initialize weights
weights = torch.rand(pre_neurons, post_neurons) * 0.3

# STDP state
stdp_state = STDPState(
    t_pre=torch.zeros(pre_neurons),
    t_post=torch.zeros(post_neurons)
)

# Parameters
p = STDPParameters(eta_plus=0.01, eta_minus=0.01)

# Training step
z_pre = torch.rand(1, pre_neurons) > 0.8  # Sparse input
z_post = torch.rand(1, post_neurons) > 0.8

weights, stdp_state = stdp_step_linear(
    z_pre=z_pre,
    z_post=z_post,
    w=weights,
    state_stdp=stdp_state,
    p_stdp=p
)
```

## Tsodyks-Makram (Short-term Plasticity)

Models short-term synaptic dynamics with facilitation and depression:

```python
from norse.torch.functional.tsodyks_makram import (
    stp_step,
    TsodyksMakramParameters,
    TsodyksMakramState
)

p = TsodyksMakramParameters(
    tau_f=100e-3,  # Facilitation time constant
    tau_d=200e-3,  # Depression time constant
    U=0.5          # Utilization parameter
)

state = TsodyksMakramState(
    x=torch.ones(100),  # Available synaptic resources
    u=torch.zeros(100)   # Current utilization
)

z, state = stp_step(input_spikes, state, p, dt=0.001)
```

## Regularization

Encourage sparse spiking activity:

```python
from norse.torch.functional.regularization import (
    regularize_step,
    spike_accumulator,
    voltage_accumulator
)

# Add to loss
reg_loss = regularize_step(
    state, 
    method=spike_accumulator,  # or voltage_accumulator
    kernel=5
)
loss = output_loss + 0.01 * reg_loss
```
