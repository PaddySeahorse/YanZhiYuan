# SequentialState

A sequential container that manages both layer outputs and neuron states, similar to PyTorch's `Sequential` but designed for spiking neural networks.

## Basic Usage

```python
import torch
import torch.nn as nn
import norse.torch as snn

model = snn.SequentialState(
    nn.Conv2d(1, 20, 5, 1),      # Conv layer
    snn.LIFCell(),                 # Spiking activation
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),
    snn.LIFCell(),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(800, 10),
    snn.LICell(),                  # Non-spiking output
)

# Forward pass
data = torch.randn(8, 1, 28, 28)  # Batch of 8, 1 channel, 28x28
output, states = model(data)
# output: (8, 10)
# states: list of neuron states for each layer
```

## State Management

SequentialState automatically handles state for all stateful (spiking) layers:

```python
# First forward pass - state initialized automatically
output, states = model(data)

# Subsequent passes - state flows through
output2, states2 = model(new_data, states)

# Or reset state
output, states = model(new_data, state=None)  # Re-initialize
```

## Lift - Converting Standard Layers

Use `Lift` to convert regular PyTorch layers to work with spikes:

```python
# Lifted Conv2d maintains spike-compatible interface
model = snn.SequentialState(
    snn.Lift(nn.Conv2d(1, 8, 3)),  # Works with spike states
    snn.LIFCell(),
    snn.Lift(nn.Linear(8*26*26, 10)),
)
```

### What Lift Does

- Wraps a PyTorch module
- Returns output without spike processing
- Makes it compatible with SequentialState state management

## RecurrentSequential

For networks with feedback connections:

```python
import norse.torch as snn

model = snn.RecurrentSequential(
    snn.Lift(nn.Linear(100, 50)),
    snn.LIFRecurrent(50, 50),  # Recurrent layer
    snn.LIFRecurrent(50, 10),
    output_modules=[0, 1, 2]  # Outputs from which layers to feed back
)

# Forward with optional initial state
output, state = model(x, state=None)
```

## Full Examples

### MNIST Classifier

```python
import torch
import torch.nn as nn
import norse.torch as snn

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

# Training loop
for data, labels in dataloader:
    output, state = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
```

### With Encoder

```python
import norse.torch as snn

# Encode input as spikes first
encoder = snn.PoissonEncoder()
spikes = encoder(data, seq_length=10)

# Then process
model = snn.SequentialState(
    snn.LIFRecurrent(784, 500),
    snn.LIFRecurrent(500, 10),
)

output, states = model(spikes)
```

### Custom State Hooks

Monitor state during forward pass:

```python
def my_hook(module, input, output):
    # Access state
    if hasattr(output, 'v'):
        print("Voltage:", output.v.mean())

model = snn.SequentialState(...)
model[1].register_forward_hook(my_hook)  # Hook into LIF layer
```

## API

### SequentialState

```python
class SequentialState(*layers, return_hidden=False)
```

Parameters:
- `*layers`: PyTorch modules (including snn layers)
- `return_hidden`: If True, return hidden states from all layers

### RecurrentSequential

```python
class RecurrentSequential(*modules, output_modules=-1)
```

Parameters:
- `*modules`: PyTorch modules
- `output_modules`: Index or list of indices for feedback outputs
