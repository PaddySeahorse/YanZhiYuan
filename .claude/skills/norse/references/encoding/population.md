# Population Encoding

Distributed encoding where each input value is represented by a population of neurons with overlapping receptive fields.

## Mathematical Model

Uses Gaussian radial basis functions (RBF):

$$K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

Each input value $x$ is encoded as a vector where neurons near $x$ have high activity.

## Usage

### Functional Layer

```python
import torch
from norse.torch.functional.encode import population_encode, gaussian_rbf

# Encode single value
data = torch.tensor([0.5])  # Single value
encoded = population_encode(data, out_features=10)
# Output: (1, 10) - 10 neurons covering range [0, max]

# Encode batch
data = torch.rand(8, 3)  # 8 samples, 3 features
encoded = population_encode(data, out_features=10)
# Output: (8, 3, 10)

# With custom kernel and distance function
def custom_kernel(distances):
    return torch.exp(-distances)

encoded = population_encode(
    data, 
    out_features=10,
    kernel=custom_kernel,
    distance_function=lambda x, y: (x - y).abs()
)
```

### Combined with Poisson

```python
import torch
import norse.torch as snn
from norse.torch.functional.encode import population_encode, poisson_encode

# First population encode
data = torch.rand(8, 10)
pop_encoded = population_encode(data, out_features=20)

# Then convert to spikes
spikes = poisson_encode(pop_encoded, seq_length=10)
# Output: (10, 8, 10, 20)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `input_values` | Input tensor with values to encode |
| `out_features` | Number of neurons per input feature |
| `scale` | Scaling factor for kernel centers (default: max of input) |
| `kernel` | Kernel function (default: gaussian_rbf) |
| `distance_function` | Distance metric (default: euclidean_distance) |

## Kernel Functions

```python
from norse.torch.functional.encode import gaussian_rbf, euclidean_distance

# Gaussian RBF (default)
encoded = population_encode(data, out_features=10, kernel=gaussian_rbf)

# Custom kernel
def laplacian_kernel(distances):
    return torch.exp(-distances)

encoded = population_encode(data, out_features=10, kernel=laplacian_kernel)
```

## Visualization

Population encoding creates smooth, overlapping representations:

```
Input: 0.5

Neuron 0:  ████████████████░░░░░  1.0
Neuron 1:  ██████████████████░░░  0.9
Neuron 2:  ████████████████████░  0.6
Neuron 3:  ░░███████████████████  0.3
Neuron 4:  ░░░░█████████████████  0.1
          |---|---|
         0.3   0.5  0.7  value
```

## Use Cases

1. **Continuous values**: Smooth representation of analog signals
2. **Sensor encoding**: Model biological sensory systems
3. **Feature binding**: Distributed representation across neuron populations
4. **Noise robustness**: Errors distributed, not catastrophic
