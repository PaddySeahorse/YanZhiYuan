# Networks Overview

Norse provides tools for building spiking neural network architectures.

## Network Building Blocks

| Component | Description |
|-----------|-------------|
| `SequentialState` | Sequential layers with state management |
| `RecurrentSequential` | Recurrent networks with feedback |
| `Lift` | Convert regular PyTorch layers to spiking |
| Pre-built models | ConvNet, VGG, MobileNetV2 |

See detailed guides:
- [SequentialState](sequential.md)
- [Pre-built Models](models.md)

## Key Concept: Stateful Layers

Unlike regular PyTorch modules, SNN layers maintain state across timesteps:

```python
# Regular PyTorch
layer = nn.Linear(100, 50)
output = layer(input)  # No state

# Norse (stateful)
layer = snn.LIF(input_features=100, hidden_features=50)
output, state = layer(input_timestep, state)  # Returns state

# For sequences
output, state = layer(sequence)  # Iterates over time
```

## SequentialState

The primary way to build SNN architectures:

```python
import norse.torch as snn

model = snn.SequentialState(
    nn.Conv2d(1, 32, 3),
    snn.LIFCell(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    snn.LIFRecurrent(..., 10),
    snn.LICell(),
)

output, states = model(input)
```

See [SequentialState](sequential.md) for details.

## Lift

Convert any PyTorch module to process spikes:

```python
# Lift standard layers to work with spikes
lifted_conv = snn.Lift(nn.Conv2d(16, 8, 3))

# Input is (batch, channels, height, width)
# Output maintains same shape, but works with spike states
```

See [SequentialState](sequential.md) for more on Lift.
