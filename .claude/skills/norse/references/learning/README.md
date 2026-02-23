# Learning Overview

Norse provides biologically-inspired learning rules for training spiking neural networks.

## Learning Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **STDP** | Spike-timing-dependent plasticity | Unsupervised learning |
| **STDP Sensor** | Correlation-based STDP | Sensory processing |
| **Tsodyks-Makram** | Short-term plasticity | Synaptic dynamics |
| **Regularization** | Spike/voltage regularization | Network homeostasis |

## Key Concepts

### Surrogate Gradients

For backpropagation through spiking neurons, Norse supports surrogate gradient methods:

```python
from norse.torch import LIFParameters

p = LIFParameters(method="super")    # SuperSpike (default)
p = LIFParameters(method="adjoint")  # Adjoint method
p = LIFParameters(method="heaviside") # Straight-through
```

### STDP

Spike-timing-dependent plasticity strengthens synapses when pre-synaptic spikes occur before post-synaptic spikes (causal), and weakens them otherwise.

See detailed guide: [STDP](stdp.md)

### Short-term Plasticity

Models synaptic dynamics where connection strength changes based on recent activity (facilitation/depression).

See: [Tsodyks-Makram](learning/stdp.md#tsodyks-makram)

## Training Approaches

### 1. Surrogate Gradient (Recommended)

Standard PyTorch backprop with surrogate gradients:

```python
import torch
import norse.torch as snn

model = snn.SequentialState(
    snn.LIFRecurrent(784, 500),
    snn.LIFRecurrent(500, 10),
)

# Standard PyTorch training loop
output, states = model(data)
loss = criterion(output.sum(0), labels)
loss.backward()
optimizer.step()
```

### 2. STDP for Unsupervised Learning

Biologically plausible learning:

```python
from norse.torch.functional.stdp import stdp_step_linear, STDPParameters
# Apply STDP after forward pass
```

### 3. Hybrid Approaches

Combine supervised (surrogate gradients) with unsupervised (STDP) learning.
