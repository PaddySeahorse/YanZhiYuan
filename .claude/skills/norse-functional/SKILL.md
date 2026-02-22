---
name: norse-functional
description: Norse torch.functional API documentation. Functional implementations of neuron models, encoding, and plasticity.
allowed-tools: Bash
---

# norse.torch.functional

Functional implementations for spiking neural networks.

## Encoding

- constant_current_lif_encode
- gaussian_rbf
- euclidean_distance
- population_encode
- poisson_encode
- poisson_encode_step
- signed_poisson_encode
- signed_poisson_encode_step
- spike_latency_lif_encode
- spike_latency_encode
- lif_current_encoder
- lif_adex_current_encoder
- lif_ex_current_encoder

## Logical

- logical_and
- logical_xor
- logical_or
- muller_c
- posedge_detector

## Regularization

- regularize_step
- spike_accumulator
- voltage_accumulator

## Threshold Functions

- heaviside
- heavi_erfc_fn
- heavi_tanh_fn
- logistic_fn
- heavi_circ_fn
- circ_dist_fn
- triangle_fn
- super_fn

## Temporal Operations

- lift

## Neuron Models

### Integrate-and-F- IAFParametersire (IAF)


- IAFFeedForwardState
- iaf_feed_forward_step

### Izhikevich

- IzhikevichParameters
- IzhikevichSpikingBehavior
- tonic_spiking
- tonic_bursting
- phasic_spiking
- phasic_bursting
- mixed_mode
- spike_frequency_adaptation
- class_1_exc
- class_2_exc
- spike_latency
- subthreshold_oscillation
- resonator
- integrator
- rebound_spike
- rebound_burst
- threshhold_variability
- bistability
- dap
- accomodation
- inhibition_induced_spiking
- inhibition_induced_bursting
- izhikevich_feed_forward_step

### Leaky Integrator

- LIParameters
- LIState
- li_feed_forward_step

### Leaky Integrate-and-Fire (LIF)

- LIFParameters
- LIFFeedForwardState
- lif_feed_forward_integral
- lif_feed_forward_step
- lif_feed_forward_step_sparse
- lif_feed_forward_adjoint_step
- lif_feed_forward_adjoint_step_sparse

### LIF, Box Model

A simplified version of the popular leaky integrate-and-fire neuron model that combines a leaky integrator with spike thresholds to produce events (spikes).
Compared to the LIF modules, this model leaves out the current term, making it computationally simpler.

- LIFBoxFeedForwardState
- LIFBoxParameters
- lif_box_feed_forward_step

### LIF, Conductance Based

- CobaLIFParameters (see [norse/torch/functional/coba_lif.py](norse/torch/functional/coba_lif.py))
- CobaLIFFeedForwardState
- coba_lif_feed_forward_step

### LIF, Adaptive Exponential

- LIFAdExParameters
- LIFAdExFeedForwardState
- lif_adex_feed_forward_step
- lif_adex_current_encoder

### LIF, Exponential

- LIFExParameters
- LIFExFeedForwardState
- lif_ex_feed_forward_step
- lif_ex_current_encoder

### LIF, Multicompartmental (MC)

- lif_mc_feed_forward_step
- lif_mc_refrac_feed_forward_step

### LIF, Refractory

- LIFRefraxParameters
- LIFRefraxFeedForwardState
- lif_refrac_feed_forward_step
- lif_refrac_feed_forward_adjoint_step

### Long Short-Term Memory (LSNN)

- LSNNParameters
- LSNNFeedForwardState
- lsnn_feed_forward_step
- lsnn_feed_forward_adjoint_step

## Receptive Fields

- gaussian_kernel
- spatial_receptive_field
- spatial_receptive_fields_with_derivatives
- temporal_scale_distribution

## Plasticity Models

### Spike-Time Dependent Plasticity (STDP)

- STDPSensorParameters
- STDPSensorState
- stdp_sensor_step

### Tsodyks-Markram Timing-Dependent Plasticity (TDP)

- TsodyksMakramParameters
- TsodyksMakramState
- stp_step

## Related Documentation

- [Module API](docs/norse.torch.rst)
- [Development Documentation](docs/pages/development.md)
- [Hardware Acceleration](docs/pages/hardware.rst)
