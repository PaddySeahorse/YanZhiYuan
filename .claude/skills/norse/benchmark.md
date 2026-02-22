# Norse Benchmarking

This benchmarking package provides preliminary comparisons to other spike-based deep learning approaches.

## LIF benchmark comparison with Norse, BindsNET, and GeNN

BindsNET and GeNN are two of the closest competitors in the SNN space - at least in terms of performance.
The graph shows a benchmark between Norse, BindsNET, and GeNN simulating poisson encoded input to a linearly weighted layer of LIF neurons. The simulation ran on an AMD Ryzen Threadripper 3960X 24-Core machine with a NVIDIA RTX 3090 24 GB RAM GPU for 1000 timesteps with a time-delta of 0.001 seconds and a batch-size of 32.

The benchmark indicates that for a single layer of <= 5000 LIF neurons, Norse outperforms BindsNET by a factor of >10, rivals GeNN for smaller layers, but fails to keep up with GeNN's precompiled GPU code for larger layers.

## LIF benchmark between Norse versions

We continuously strive to improve the performance of Norse.
The graph shows how long it takes to simulate a single layer of LIF neurons - the smaller the better.

## Performance Summary

- **Norse vs BindsNET**: Norse is >10x faster for <= 5000 neurons
- **Norse vs GeNN**: Comparable for smaller layers; GeNN leads for larger layers
- **GPU Acceleration**: Full support via PyTorch CUDA integration

## Source

This skill is generated from `/tmp/norse_repo/norse/benchmark/README.md`
