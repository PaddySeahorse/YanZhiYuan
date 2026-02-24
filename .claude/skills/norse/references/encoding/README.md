# Encoding Overview

Encoding transforms real-valued input data into spike trains that can be processed by spiking neural networks.

## Encoding Methods

| Method | Module | Functional | Description |
|--------|--------|------------|-------------|
| Poisson | `PoissonEncoder` | `poisson_encode` | Rate-based random spikes |
| Signed Poisson | `SignedPoissonEncoder` | `signed_poisson_encode` | Bipolar encoding |
| Population | `PopulationEncoder` | `population_encode` | Distributed representation |
| Spike Latency | `SpikeLatencyEncoder` | `spike_latency_encode` | Timing-based |
| Latency LIF | `SpikeLatencyLIFEncoder` | `spike_latency_lif_encode` | LIF-based latency |
| Constant Current | `ConstantCurrentLIFEncoder` | `constant_current_lif_encode` | Current-based |

## Common Pattern

All encoders add a time dimension to the input:

```python
# Input: (batch, features)
# Output: (time, batch, features)

encoder = snn.PoissonEncoder()
spikes = encoder(data, seq_length=10)  # 10 timesteps
```

## Choosing an Encoder

### Rate-based (random)
- **Poisson**: Good for general use, stochastic
- **Signed Poisson**: For data with positive and negative values

### Timing-based
- **Spike Latency**: Fast events encode high values
- **Population**: Good for continuous values

### Current-based
- **Constant Current**: Direct current injection to LIF neurons

See detailed guides:
- [Poisson Encoding](poisson.md)
- [Population Encoding](population.md)
- [Latency Encoding](latency.md)
