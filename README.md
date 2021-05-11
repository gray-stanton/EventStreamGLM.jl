# EventStreamGLM
This package modifies IRLS to use the method of conjugate gradients as fitting algorithm for generalized linear models. In particular, the package uses the algorithms implemented in `EventStreamMatrix` to minimize the memory footprint required to fit a discretized GLMs corresponding to dynamical models for neural ensembles. Additional code for point process simulation is included.

## Source files
* ConjGradLinPred.jl - Implements necessary wrappers around EventStreamMatrix objects so that existing CG and IRLS fitting code from the IterativeSolvers and GLM packages can be used
* simulation.jl - Implements Ogata thinning approach to point process simulation, as well as Shedler-Lewis simulation methods for homogenous and inhomogenous Poisson processes.

## Script files
* simulate_eventstream_data.jl - Point process simulation code. Requires a configuration YAML file, see conf.yaml as example.
* fit_eventstream_data.jl - Fit to a simulated dataset, and write out relevant coefficient estimates and other metadata.
* fit_eventstream_data_legacy.R - Uses existing bdmiso and bdglm packages to fit same dynamical model as a comparison.

# Examples
Example call to simulate_eventstream_data.jl

```
simulate_eventstream_data --outfolder ./out --name testfit --nsims 100 --maxtime 1e6 --seed 5324 --config ./conf.yaml
```
Control of number of neurons and associated first-order Volterra kernels are specificed in the configation YAML file.

Example call to fit_eventstream_data.jl

```
fit_eventstream_data --infolder ./out --outfolder ./res --seed 5345 --name testfit --include_lag --fineness 0.5
```

