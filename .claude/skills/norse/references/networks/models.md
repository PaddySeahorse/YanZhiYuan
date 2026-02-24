# Pre-built SNN Models

Norse provides pre-built spiking neural network architectures for common tasks.

## Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `ConvNet` | Simple 2-layer CNN | MNIST, simple patterns |
| `ConvNet4` | 4-layer CNN | CIFAR-10, more complex |
| `VGG` | VGG-style architecture | Image classification |
| `MobileNetV2` | Efficient mobile architecture | Resource-constrained |

## ConvNet

Simple convolutional spiking network:

```python
import norse.torch as snn
import torch

# Create model
model = snn.ConvNet(
    num_classes=10,
    input_channels=1,
    feature_dim=512
)

# Forward pass
data = torch.randn(8, 1, 28, 28)
output, state = model(data)
# output: (8, 10)
```

### ConvNet4

Deeper 4-layer network:

```python
model = snn.ConvNet4(
    num_classes=10,
    input_channels=1,
    batch_norm=True
)
```

## VGG

VGG-style architectures:

```python
import norse.torch as snn

# VGG11
model = snn.vgg11(num_classes=10, num_channels=3)
model = snn.vgg11_bn(num_classes=10, num_channels=3)

# VGG13
model = snn.vgg13(num_classes=10, num_channels=3)
model = snn.vgg13_bn(num_classes=10, num_channels=3)

# VGG16
model = snn.vgg16(num_classes=10, num_channels=3)
model = snn.vgg16_bn(num_classes=10, num_channels=3)

# VGG19
model = snn.vgg19(num_classes=10, num_channels=3)
model = snn.vgg19_bn(num_classes=10, num_channels=3)

# Generic VGG
model = snn.VGG(
    features=snn.vgg11_features(),  # Custom feature extractor
    num_classes=10
)
```

### Batch Normalization

Variants with `_bn` include batch normalization layers.

## MobileNetV2

Efficient architecture for mobile/embedded devices:

```python
import norse.torch as snn

model = snn.MobileNetV2(
    num_classes=10,
    input_channels=3,
    dropout=0.2
)

output, state = model(data)
```

## Class API

### ConvNet

```python
ConvNet(
    num_classes=10,       # Output classes
    input_channels=1,    # Input channels
    feature_dim=512,     # Feature dimension
    tensorboard=False    # Enable tensorboard hooks
)
```

### ConvNet4

```python
ConvNet4(
    num_classes=10,
    input_channels=1,
    batch_norm=False,
    tensorboard=False
)
```

### VGG

```python
VGG(
    features,             # Feature extraction layers
    num_classes=10,      # Output classes
    tensorboard=False
)

# Or use presets:
vgg11(pretrained=False, num_classes=10, num_channels=3)
```

## Customizing Pre-built Models

### Replace Neuron Types

```python
# Use different neurons
from norse.torch.module.lsnn import LSNNRecurrent

# Not directly supported, but can reconstruct:
custom = snn.SequentialState(
    *original_conv_layers,
    LSNNRecurrent(512, 10)
)
```

### Add Regularization

```python
from norse.torch.module.regularization import RegularizationCell

model = snn.SequentialState(
    *conv_features,
    RegularizationCell(),
    nn.Linear(512, 10)
)
```

## Complete Example

```python
import torch
import norse.torch as snn

# MNIST with ConvNet
model = snn.ConvNet(
    num_classes=10,
    input_channels=1,
    feature_dim=512
)

# Training
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for data, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward
        output, state = model(data)
        
        # Loss (sum over time for spike outputs)
        loss = criterion(output, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
```

## Running Tasks

Norse includes pre-built tasks:

```bash
# MNIST classification
python -m norse.task.mnist

# CIFAR-10
python -m norse.task.cifar10

# Cartpole
python -m norse.task.cartpole
```
