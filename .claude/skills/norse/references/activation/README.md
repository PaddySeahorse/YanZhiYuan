# Activation Functions

Norse provides spike activation functions and surrogate gradient implementations.

## Heaviside Step Function

The fundamental spike-generating function:

$$H(x) = \begin{cases} 0 & x \leq 0 \\ 1 & x > 0 \end{cases}$$

```python
import torch
from norse.torch.functional.heaviside import heaviside

x = torch.randn(8, 100)
spikes = heaviside(x)
# Binary spike output
```

## Surrogate Gradient Functions

Used for backpropagation through spiking neurons (since Heaviside has zero gradient everywhere except at 0).

### SuperSpike

The default surrogate gradient:

$$\sigma(x) = \frac{1}{\alpha} \cdot \max(0, 1 - |x \cdot \alpha|)$$

```python
import torch
from norse.torch.functional.superspike import super_fn

x = torch.randn(8, 100, requires_grad=True)
output = super_fn(x, alpha=100)
```

### Threshold Functions

Norse provides configurable threshold functions:

```python
from norse.torch.functional.threshold import (
    threshold,
    heavi_erfc_fn,   # ERFC-based
    heavi_tanh_fn,   # Tanh-based
    heavi_circ_fn,   # Circular
    logistic_fn,     # Logistic
    triangle_fn      # Triangle
)

# All accept (x, method, alpha)
output = heavi_erfc_fn(x, alpha=1.0)
output = heavi_tanh_fn(x, alpha=1.0)
output = heavi_circ_fn(x, alpha=1.0)
output = logistic_fn(x, alpha=1.0)
output = triangle_fn(x, alpha=1.0)

# Configurable method in neuron parameters
from norse.torch import LIFParameters

p = LIFParameters(method="super")    # Default
p = LIFParameters(method="adjoint")  # Adjoint method
p = LIFParameters(method="heaviside") # Straight-through
```

## Reset Methods

After spike generation, the neuron state is reset:

```python
from norse.torch.functional.reset import (
    ResetMethod,
    reset_value,    # Reset to fixed value
    reset_subtract  # Subtract threshold
)

# Default: reset to v_reset
state = reset_value(state, spikes, reset_value=0.0)

# Alternative: subtract threshold
state = reset_subtract(state, spikes)
```

## Spikes to Times Decoder

Convert spike trains back to firing times:

```python
from norse.torch.functional.spikes_to_times_decoder import (
    spikes_to_times_decoder
)

# Input: (time, batch, neurons) spike tensor
# Output: First spike time for each neuron
times = spikes_to_times_decoder(
    spikes,
    num_steps=100
)
```

## Integration with Neurons

These functions are used internally by neuron models:

```python
from norse.torch.functional.lif import lif_feed_forward_step
from norse.torch.functional.threshold import threshold

# Inside a neuron step:
z = threshold(v - v_th, method="super", alpha=100)
```

## Choosing a Surrogate Gradient

| Method | Best For |
|--------|----------|
| `super` (default) | General purpose |
| `adjoint` | Memory efficiency |
| `heaviside` | Straight-through estimator |

The alpha parameter controls gradient steepness:
- Higher alpha: closer to step function
- Lower alpha: smoother gradient
