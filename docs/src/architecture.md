# Architecture

BioDynaX separates biological semantics, numerical execution, optimization and
equation discovery.

## Data flow

1. `BiologicalNetwork` defines typed nodes, interactions and reactions.
2. `compile_network` assembles non-negative production and destruction fluxes.
3. `ExperimentSet` carries replicates, irregular samples and observation masks.
4. `train_ude` or `train_experiments` returns a versioned `TrainingResult`.
5. `local_basis` derives candidate variables from each target's graph parents.
6. `discover_equations` fits `D(z)ẋ-N(z)=0`, validates denominators and reports
   bootstrap term-selection frequencies.

## Positivity and constraints

The default UDE uses

```math
\dot{x}_i = P_i(x,p,t) - D_i(x,p,t)x_i,\qquad P_i,D_i\geq 0.
```

This points inward at `x_i=0`. Non-structural inequalities use a smooth
Powell–Hestenes–Rockafellar Augmented Lagrangian. Constraint residuals retain
their sign, dual variables are projected only in the outer loop, and the
penalty parameter is updated from primal progress; fixed large multipliers are
not part of the default API.

## Rational discovery

For each target and graph-local regulator set, BioDynaX identifies:

```math
D(z)\dot{x}_i-N(z)=0.
```

The constant denominator coefficient is anchored to one, resolving scale
ambiguity. Numerator and denominator coefficients are selected jointly with
QR-based sequential thresholded least squares. Contiguous hold-out blocks,
moving-block bootstrap and consensus refitting prevent time leakage and reject
unstable supports. This natively represents Michaelis–Menten, Hill and
competitive-inhibition kinetics.

## Scaling

Candidate libraries are generated per target from graph parents, then bounded
by derivative-correlation screening. For bounded biological indegree `k`, the
library scales with `Σ O(k_i^d)` rather than global `O(n^d)`.

Execution is backend-neutral (`:serial`, `:threads`, `:distributed`, `:gpu`).
CUDA support is loaded only when CUDA.jl is present.
