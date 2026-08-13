# Architecture

BioDynaX separates biological semantics, numerical execution, optimization and
equation discovery.

## Data flow

1. `BiologicalNetwork` defines typed nodes, interactions and reactions.
2. `compile_network` assembles non-negative production and destruction fluxes.
3. `ExperimentSet` carries replicates, irregular samples and observation masks.
4. `train_ude` or `train_experiments` returns a versioned `TrainingResult`.
5. `local_basis` derives candidate variables from each target's graph parents
   (`scope = :graph`, or `:global` for ablations).
6. `discover_equations` fits `D(z)ẋ-N(z)=0`, validates denominators and reports
   bootstrap term-selection frequencies. Failures set `DiscoveryRetcode`.

## Positivity and constraints

The default UDE uses

```math
\dot{x}_i = P_i(x,p,t) - D_i(x,p,t)x_i,\qquad P_i,D_i\geq 0.
```

This points inward at `x_i=0`. Non-structural inequalities use a smooth
Powell–Hestenes–Rockafellar Augmented Lagrangian.

## Rational discovery

For each target and graph-local regulator set, BioDynaX identifies:

```math
D(z)\dot{x}_i-N(z)=0.
```

The constant denominator coefficient is anchored to one. Numerator and
denominator coefficients are selected jointly with QR-based STLSQ. Contiguous
hold-out blocks and bootstrap consensus reject unstable supports.

The product path (`sample_unknown_destruction` → `discover_unknown_rate`)
applies that implicit problem to the **unknown destruction rate**, not the
full state derivative. Known production and linear decay stay in compiled IR.
`compose_hybrid_rhs` stitches the recovered rate back into the ODE.
`local_basis(...; scope=:graph)` versus `scope=:global` is the same solver
with a different library — that is the graph-prior ablation.

## Scaling

Libraries are generated per target from graph parents. For bounded indegree `k`,
the library scales with `Σ O(k_i^d)` rather than global `O(n^d)`.

Streaming chunks (`each_library_chunk`) and blocked STLSQ avoid one dense design
matrix. Implicit candidates are stress-tested on train, validation and an
orthant grid (`domain_samples`).

Raw trajectories can enter discovery via `estimate_derivatives` and
`discover_equations(X, times, network)`. Recovered candidates export to LaTeX
or callable RHS closures (`export_rhs`).

## Execution

Serial, threaded, and distributed backends are supported. The `:gpu` backend is
**experimental array transfer only** (see [Experimental](experimental.md)).
