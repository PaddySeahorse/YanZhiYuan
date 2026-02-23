# Latency Encoding

Timing-based encoding where spike timing (not rate) carries information. Earlier spikes encode higher values.

## How It Works

Higher input values produce earlier spikes within the encoding window:

```
Value 1.0:  █ (spike at t=1)
Value 0.5:       █ (spike at t=5)
Value 0.1:             █ (spike at t=9)
```

## Spike Latency Encoding

### Module Layer

```python
import torch
import norse.torch as snn

encoder = snn.SpikeLatencyEncoder()

# Input: (batch, features) with values in [0, 1]
data = torch.rand(8, 100)

# Output: (time, batch, features)
spikes = encoder(data, seq_length=10)
```

### Functional Layer

```python
import torch
from norse.torch.functional.encode import spike_latency_encode

spikes = spike_latency_encode(
    input_values=torch.rand(8, 100),
    seq_length=10
)
```

## Latency LIF Encoding

Uses LIF dynamics to generate latency-encoded spikes:

### Module Layer

```python
import norse.torch as snn

encoder = snn.SpikeLatencyLIFEncoder()

# Uses LIF neuron dynamics
spikes = encoder(data, seq_length=10)
```

### Functional Layer

```python
from norse.torch.functional.encode import spike_latency_lif_encode

spikes = spike_latency_lif_encode(
    input_values=torch.rand(8, 100),
    seq_length=10,
    p=LIFParameters(),  # Optional LIF parameters
    dt=0.001
)
```

## Properties

| Property | Description |
|----------|-------------|
| **Temporal precision** | Information in spike timing |
| **Energy efficient** | One spike per neuron per encoding window |
| **Fast decoding** | Can read value from first spike |
| **Synchronous** | All spikes within encoding window |

## Complete Example

```python
import torch
import torch.nn as nn
import norse.torch as snn

# Encoder for MNIST
encoder = snn.SpikeLatencyEncoder()

# Convert images to latency-encoded spikes
data = torch.rand(8, 1, 28, 28)
spikes = encoder(data, seq_length=20)  # (20, 8, 1, 28, 28)

# Process
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

## Comparison to Poisson

| Aspect | Poisson | Latency |
|--------|---------|---------|
| Information | Rate | Timing |
| Energy | Many spikes | One spike per neuron |
| Latency | All timesteps | Early timesteps |
| Noise sensitivity | Averaging | First spike |
| Implementation | Simple | Moderate |
