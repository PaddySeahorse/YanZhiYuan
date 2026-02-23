# Poisson Encoding

Rate-based encoding where spike probability is proportional to input value. Higher values produce more spikes on average.

## Mathematical Model

The probability of generating a spike in each timestep:

$$P(spike) = dt \cdot f_{max} \cdot x$$

where:
- $dt$: simulation timestep
- $f_{max}$: maximum firing rate
- $x$: input value in [0, 1]

The stochastic nature simulates Poisson spike trains.

## Usage

### Module Layer (Recommended)

```python
import torch
import norse.torch as snn

# Create encoder
encoder = snn.PoissonEncoder()

# Input: (batch, features)
data = torch.rand(8, 100)  # 8 samples, 100 features, values in [0, 1]

# Encode to spike trains
# Output: (time, batch, features)
spikes = encoder(data, seq_length=10)

# Use with SNN layer
layer = snn.LIF()
output, state = layer(spikes)
```

### Functional Layer

```python
import torch
from norse.torch.functional.encode import poisson_encode, poisson_encode_step

# Full sequence
spikes = poisson_encode(
    input_values=torch.rand(8, 100),
    seq_length=10,
    f_max=100,  # Maximum firing rate in Hz
    dt=0.001    # Timestep in seconds
)

# Single timestep
spike = poisson_encode_step(
    input_values=torch.rand(8, 100),
    f_max=1000,
    dt=0.001
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `f_max` | 100 | Maximum firing rate (Hz) |
| `dt` | 0.001 | Simulation timestep (s) |

## Signed Poisson Encoding

For data with positive and negative values, use signed Poisson encoding:

```python
import norse.torch as snn

encoder = snn.SignedPoissonEncoder()

# Input: (batch, features) with values in [-1, 1]
data = torch.rand(8, 100) * 2 - 1  # Range [-1, 1]

# Output: (time, batch, features)
spikes = encoder(data, seq_length=10)

# Functional
from norse.torch.functional.encode import signed_poisson_encode
spikes = signed_poisson_encode(data, seq_length=10)
```

Positive values generate excitatory spikes, negative values generate inhibitory spikes.

## Complete Example

```python
import torch
import torch.nn as nn
import norse.torch as snn

# Full pipeline
encoder = snn.PoissonEncoder()
layer = snn.LIF()

# Encode MNIST digits
data = torch.rand(32, 1, 28, 28)  # Batch of MNIST images
spikes = encoder(data, seq_length=10)

# Process with SNN
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

output, states = model(spikes)
```

## Key Properties

1. **Stochastic**: Same input produces different spike trains
2. **Rate-based**: Information in spike rate, not timing
3. **Simple**: Easy to implement and use
4. **Biologically plausible**: Models photoreceptor behavior
