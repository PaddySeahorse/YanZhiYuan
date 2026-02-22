# norse.torch

Building blocks for spiking neural networks based on PyTorch.

## Containers

- `Lift` - Lift module
- `SequentialState` - Stateful sequential layers
- `RegularizationCell` - Regularization cell

## Convolutions

- `LConv2d` - Lifted Conv2d

## Encoding

- `ConstantCurrentLIFEncoder` - Constant current LIF encoder
- `PoissonEncoder` - Poisson encoder
- `PoissonEncoderStep` - Poisson encoder step
- `PopulationEncoder` - Population encoder
- `SignedPoissonEncoder` - Signed Poisson encoder
- `SpikeLatencyEncoder` - Spike latency encoder
- `SpikeLatencyLIFEncoder` - Spike latency LIF encoder

## Neuron models

### Integrate-and-fire (IAF)

Simple integrators that sums up incoming signals until a threshold.

- `IAFFeedForwardState` - IAF feedforward state
- `IAFParameters` - IAF parameters
- `IAFCell` - IAF cell

### Izhikevich

- `IzhikevichParameters` - Izhikevich parameters
- `IzhikevichState` - Izhikevich state
- `IzhikevichSpikingBehavior` - Izhikevich spiking behavior
- `Izhikevich` - Izhikevich module
- `IzhikevichCell` - Izhikevich cell
- `IzhikevichRecurrent` - Izhikevich recurrent
- `IzhikevichRecurrentCell` - Izhikevich recurrent cell

### Leaky integrator

- `LIState` - Leaky integrator state
- `LIParameters` - Leaky integrator parameters
- `LI` - Leaky integrator module
- `LICell` - Leaky integrator cell
- `LILinearCell` - Leaky integrator linear cell

### Leaky integrate-and-fire (LIF)

- `LIFParameters` - LIF parameters
- `LIFState` - LIF state
- `LIF` - LIF module
- `LIFCell` - LIF cell
- `LIFRecurrent` - LIF recurrent
- `LIFRecurrentCell` - LIF recurrent cell

### LIF, box model

- `LIFBoxFeedForwardState` - LIF box feedforward state
- `LIFBoxParameters` - LIF box parameters
- `LIFBoxCell` - LIF box cell

### LIF, conductance based

- `CobaLIFCell` - Conductance based LIF cell

### LIF, adaptive exponential

- `LIFAdEx` - LIF adaptive exponential
- `LIFAdExCell` - LIF adaptive exponential cell
- `LIFAdExRecurrent` - LIF adaptive exponential recurrent
- `LIFAdExRecurrentCell` - LIF adaptive exponential recurrent cell

### LIF, exponential

- `LIFEx` - LIF exponential
- `LIFExCell` - LIF exponential cell
- `LIFExRecurrent` - LIF exponential recurrent
- `LIFExRecurrentCell` - LIF exponential recurrent cell

### LIF, multicompartmental

- `LIFMCRecurrentCell` - LIF multicompartmental recurrent cell

### LIF, multicompartmental with refraction

- `LIFMCRefracRecurrentCell` - LIF multicompartmental refractory recurrent cell

### LIF, refractory

- `LIFRefracCell` - LIF refractory cell
- `LIFRefracRecurrentCell` - LIF refractory recurrent cell

### Long short-term memory (LSNN)

- `LSNN` - LSNN module
- `LSNNCell` - LSNN cell
- `LSNNRecurrent` - LSNN recurrent
- `LSNNRecurrentCell` - LSNN recurrent cell

## Receptive fields

- `SpatialReceptiveField2d` - Spatial receptive field 2D
- `TemporalReceptiveField` - Temporal receptive field
