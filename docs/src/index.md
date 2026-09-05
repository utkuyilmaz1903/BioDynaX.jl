# BioDynaX.jl

Hybrid models of biochemical networks: compiled known kinetics plus one
neural destruction term, recovered symbolically.

BioDynaX fits hybrid models of small biochemical networks. You give it a
known interaction graph and known kinetics (mass action, linear decay, Hill,
Michaelis-Menten saturation, competitive binding, or a custom rate); it
compiles those into a production-destruction ODE

```math
\frac{du_i}{dt} = P_i(u) - D_i(u)\,u_i .
```

Exactly one destruction term may be marked unknown. That term is replaced by
a small neural network (a universal differential equation), trained on
time-series data from one or more initial conditions, and then approximated
symbolically by sparse rational regression (implicit SINDy) over a library
built only from that node's graph neighbours.

The package reports three things: whether the unknown term is practically
identifiable from the data (a Fisher-information and scale-collinearity
diagnostic), how well the hybrid model reproduces observed and held-out
trajectories, and which symbolic terms are recovered. It is a research tool
for small networks with a known graph (the benchmarks cover 2- to 6-state
networks), not a general-purpose network-inference tool or reaction-network
solver.

Version 0.10. The public API may still change before 1.0; see the
[changelog](changelog.md).

## Where to go

- [Getting started](getting-started.md): install the package and run a
  complete fit in a few minutes.
- [Tutorial](tutorial.md): the unknown-inhibition example, step by step.
- [Concepts](concepts.md): the model form, the identifiability diagnostic,
  symbolic discovery, and the reference protocol with its train/holdout split.
- [How-to recipes](howto.md): CSV data, marking an edge unknown, resimulating,
  the SciML solve surface, checkpoints, execution backends.
- [Benchmarks](benchmarks.md): what the recovery benchmarks run and what they
  show.
- [API reference](api.md): every exported name.
- [Extensions](extensions.md): experimental GPU, plotting, ModelingToolkit,
  SBML, and DataDrivenSparse integrations.
- [Scope and limitations](limitations.md): what the package does not do.

## Requirements

Julia 1.10 or newer. Installing the dependencies needs network access to the
Julia package server or to GitHub.
