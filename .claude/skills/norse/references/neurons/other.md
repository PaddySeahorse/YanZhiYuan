# Other Neuron Models

This document covers additional neuron models in Norse beyond the main LIF, Izhikevich, and LSNN families.

## LI - Leaky Integrator

A non-spiking neuron model that outputs continuous values instead of spikes. Useful for output layers or as a base component.

```python
import norse.torch as snn

# Module
layer = snn.LI(input_features=100, hidden_features=200)
cell = snn.LICell(input_features=100, hidden_features=200)
linear_cell = snn.LILinearCell(100, 200)  # Linear transformation + LI

# Functional
from norse.torch.functional.leaky_integrator import (
    li_step, li_feed_forward_step,
    LIState, LIParameters
)
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_mem_inv` | 1/1e-2 | Inverse membrane time constant |
| `v_leak` | 0.0 | Leak potential |

## CobaLIF - Conductance-Based LIF

LIF with conductance-based synapses for more biologically realistic dynamics.

```python
import norse.torch as snn

cell = snn.CobaLIFCell(input_features=100, hidden_features=200)

from norse.torch.functional.coba_lif import (
    coba_lif_step, coba_lif_feed_forward_step,
    CobaLIFState, CobaLIFParameters
)
```

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_l` | 1/10e-3 | Leak conductance |
| `e_leak` | -65.0 | Leak reversal potential |
| `tau_syn_inv` | 1/5e-3 | Synaptic time constant |

## LIFEx - LIF with Exponential Synapses

LIF using exponential synaptic dynamics for faster rise times.

```python
import norse.torch as snn

layer = snn.LIFEx(input_features=100, hidden_features=200)
cell = snn.LIFExCell(input_features=100, hidden_features=200)
recurrent = snn.LIFExRecurrent(input_features=100, hidden_features=200)

from norse.torch.functional.lif_ex import (
    lif_ex_step, lif_ex_feed_forward_step,
    LIFExState, LIFExParameters
)
```

## LIFRefrac - LIF with Refractory Period

LIF with absolute refractory period after spike generation.

```python
import norse.torch as snn

layer = snn.LIFRefrac(input_features=100, hidden_features=200)
cell = snn.LIFRefracCell(input_features=100, hidden_features=200)
recurrent = snn.LIFRefracRecurrent(input_features=100, hidden_features=200)

from norse.torch.functional.lif_refrac import (
    lif_refrac_step, lif_refrac_feed_forward_step,
    LIFRefracState, LIFRefracParameters
)
```

### Additional Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_refrac_inv` | 1/5e-3 | Inverse refractory time constant |

## LIFAdEx - LIF with Adaptive Exponential

LIF combined with exponential adaptation for neurons that exhibit spike frequency adaptation.

```python
import norse.torch as snn

layer = snn.LIFAdEx(input_features=100, hidden_features=200)
cell = snn.LIFAdExCell(input_features=100, hidden_features=200)
recurrent = snn.LIFAdExRecurrent(input_features=100, hidden_features=200)

from norse.torch.functional.lif_adex import (
    lif_adex_step, lif_adex_feed_forward_step,
    LIFAdExState, LIFAdExParameters
)
```

### Additional Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `a` | 0.0 | Subthreshold adaptation |
| `b` | 0.0 | Spike-triggered adaptation |
| `delta_T` | 0.0 | Exponential adaptation |
| `tau_adapt_inv` | 1/100e-3 | Adaptation time constant |

## LIFAdExRefrac - LIF with Adaptation and Refractory

Combines adaptive exponential and refractory period.

```python
import norse.torch as snn

layer = snn.LIFAdExRefracRecurrent(input_features=100, hidden_features=200)
cell = snn.LIFAdExRefracCell(input_features=100, hidden_features=200)
```

## LIBox - Leaky Integrator Box

A "box-based" leaky integrator with discrete voltage bins.

```python
import norse.torch as snn

cell = snn.LIBoxCell(input_features=100, hidden_features=200)

from norse.torch.functional.leaky_integrator_box import (
    li_box_step, li_box_feed_forward_step,
    LIBoxState, LIBoxParameters
)
```

## LIFBox - LIF Box

Box-based LIF neuron (discrete voltage states).

```python
import norse.torch as snn

cell = snn.LIFBoxCell(input_features=100, hidden_features=200)

from norse.torch.functional.lif_box import (
    lif_box_feed_forward_step,
    LIFBoxState, LIFBoxParameters
)
```

## LIFMC - Multi-Compartment LIF

Multi-compartment LIF for modeling neurons with spatially separated compartments.

```python
from norse.torch.functional.lif_mc import lif_mc_step, lif_mc_feed_forward_step
from norse.torch.functional.lif_mc_refrac import lif_mc_refrac_step, lif_mc_refrac_feed_forward_step

from norse.torch.module.lif_mc import LIFMCRecurrentCell
from norse.torch.module.lif_mc_refrac import LIFMCRefracRecurrentCell
```

## LIFCorrelation - LIF with Correlation

LIF neurons that also track correlation between pre- and postsynaptic activity.

```python
import norse.torch as snn

layer = snn.LIFCorrelation(input_features=100, hidden_features=200)

from norse.torch.functional.lif_correlation import (
    lif_correlation_step,
    LIFCorrelationState, LIFCorrelationParameters
)
```
