# norse.torch.functional

Functional implementations for spiking neural networks.

## Encoding

- `constant_current_lif_encode` - Constant current LIF encoding
- `gaussian_rbf` - Gaussian RBF
- `euclidean_distance` - Euclidean distance
- `population_encode` - Population encoding
- `poisson_encode` - Poisson encoding
- `poisson_encode_step` - Poisson encoding step
- `signed_poisson_encode` - Signed Poisson encoding
- `signed_poisson_encode_step` - Signed Poisson encoding step
- `spike_latency_lif_encode` - Spike latency LIF encoding
- `spike_latency_encode` - Spike latency encoding
- `lif_current_encoder` - LIF current encoder
- `lif_adex_current_encoder` - LIF adaptive exponential current encoder
- `lif_ex_current_encoder` - LIF exponential current encoder

## Logical

- `logical_and` - Logical AND
- `logical_xor` - Logical XOR
- `logical_or` - Logical OR
- `muller_c` - Muller C
- `posedge_detector` - Positive edge detector

## Regularization

- `regularize_step` - Regularization step
- `spike_accumulator` - Spike accumulator
- `voltage_accumulator` - Voltage accumulator

## Threshold functions

- `heaviside` - Heaviside step function
- `heavi_erfc_fn` - Heaviside ERFC function
- `heavi_tanh_fn` - Heaviside tanh function
- `logistic_fn` - Logistic function
- `heavi_circ_fn` - Heaviside circular function
- `circ_dist_fn` - Circular distance function
- `triangle_fn` - Triangle function
- `super_fn` - Super function (supergradient)

## Temporal operations

- `lift` - Lift function for temporal operations

## Neuron models

### Integrate-and-fire (IAF)

- `IAFParameters` - IAF parameters
- `IAFFeedForwardState` - IAF feedforward state
- `iaf_feed_forward_step` - IAF feedforward step

### Izhikevich

- `IzhikevichParameters` - Izhikevich parameters
- `IzhikevichSpikingBehavior` - Izhikevich spiking behavior
- `tonic_spiking` - Tonic spiking
- `tonic_bursting` - Tonic bursting
- `phasic_spiking` - Phasic spiking
- `phasic_bursting` - Phasic bursting
- `mixed_mode` - Mixed mode
- `spike_frequency_adaptation` - Spike frequency adaptation
- `class_1_exc` - Class 1 excitatory
- `class_2_exc` - Class 2 excitatory
- `spike_latency` - Spike latency
- `subthreshold_oscillation` - Subthreshold oscillation
- `resonator` - Resonator
- `integrator` - Integrator
- `rebound_spike` - Rebound spike
- `rebound_burst` - Rebound burst
- `threshhold_variability` - Threshold variability
- `bistability` - Bistability
- `dap` - Depolarizing afterpotential
- `accomodation` - Accommodation
- `inhibition_induced_spiking` - Inhibition induced spiking
- `inhibition_induced_bursting` - Inhibition induced bursting
- `izhikevich_feed_forward_step` - Izhikevich feedforward step

### Leaky integrator

- `LIParameters` - Leaky integrator parameters
- `LIState` - Leaky integrator state
- `li_feed_forward_step` - Leaky integrator feedforward step

### Leaky integrate-and-fire (LIF)

- `LIFParameters` - LIF parameters
- `LIFFeedForwardState` - LIF feedforward state
- `lif_feed_forward_integral` - LIF feedforward integral
- `lif_feed_forward_step` - LIF feedforward step
- `lif_feed_forward_step_sparse` - LIF feedforward step sparse
- `lif_feed_forward_adjoint_step` - LIF feedforward adjoint step
- `lif_feed_forward_adjoint_step_sparse` - LIF feedforward adjoint step sparse

### LIF, box model

A simplified version of the popular leaky integrate-and-fire neuron model that combines a leaky integrator with spike thresholds to produce events (spikes).

- `LIFBoxFeedForwardState` - LIF box feedforward state
- `LIFBoxParameters` - LIF box parameters
- `lif_box_feed_forward_step` - LIF box feedforward step

### LIF, conductance based

- `CobaLIFParameters` - Conductance based LIF parameters
- `CobaLIFFeedForwardState` - Conductance based LIF feedforward state
- `coba_lif_feed_forward_step` - Conductance based LIF feedforward step

### LIF, adaptive exponential

- `LIFAdExParameters` - LIF adaptive exponential parameters
- `LIFAdExFeedForwardState` - LIF adaptive exponential feedforward state
- `lif_adex_feed_forward_step` - LIF adaptive exponential feedforward step
- `lif_adex_current_encoder` - LIF adaptive exponential current encoder

### LIF, exponential

- `LIFExParameters` - LIF exponential parameters
- `LIFExFeedForwardState` - LIF exponential feedforward state
- `lif_ex_feed_forward_step` - LIF exponential feedforward step
- `lif_ex_current_encoder` - LIF exponential current encoder

### LIF, multicompartmental (MC)

- `lif_mc_feed_forward_step` - LIF multicompartmental feedforward step
- `lif_mc_refrac_feed_forward_step` - LIF multicompartmental refractory feedforward step

### LIF, refractory

- `LIFRefracParameters` - LIF refractory parameters
- `LIFRefracFeedForwardState` - LIF refractory feedforward state
- `lif_refrac_feed_forward_step` - LIF refractory feedforward step
- `lif_refrac_feed_forward_adjoint_step` - LIF refractory feedforward adjoint step

### Long short-term memory (LSNN)

- `LSNNParameters` - LSNN parameters
- `LSNNFeedForwardState` - LSNN feedforward state
- `lsnn_feed_forward_step` - LSNN feedforward step
- `lsnn_feed_forward_adjoint_step` - LSNN feedforward adjoint step

## Receptive fields

- `gaussian_kernel` - Gaussian kernel
- `spatial_receptive_field` - Spatial receptive field
- `spatial_receptive_fields_with_derivatives` - Spatial receptive fields with derivatives
- `temporal_scale_distribution` - Temporal scale distribution

## Plasticity models

### Spike-time dependent plasticity (STDP)

- `STDPSensorParameters` - STDP sensor parameters
- `STDPSensorState` - STDP sensor state
- `stdp_sensor_step` - STDP sensor step

### Tsodyks-Markram timing-dependent plasticity (TDP)

- `TsodyksMakramParameters` - Tsodyks-Markram parameters
- `TsodyksMakramState` - Tsodyks-Markram state
- `stp_step` - Short-term plasticity step
