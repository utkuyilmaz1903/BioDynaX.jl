# Tutorial: unknown inhibition recovery

This is the only golden path the package is built around:

```text
CSV / time series
  → BiologicalNetwork (known kinetics + one unknown edge)
  → compile / build_ude_model
  → train_ude
  → sample_unknown_destruction
  → discover_unknown_rate
  → compose_hybrid_rhs
  → resimulate vs data
```

A runnable copy is `examples/unknown_inhibition.jl` at the repository root.
It uses the same **multi-IC** protocol as the recovery CI job. SBML, GPU,
multi-head NNs, and custom kinetics are **not** on this path.

The present claim is: correct graph parents (true-monomial **recall**) and a
hybrid RHS that resimulates **versus data**, for **Hill-class** unknown
destruction. Combined support F1 on a trained NN rate is **not** gated at the
analytical Hill value; extra `1` and `r` terms can remain. Canonical Hill
combined F1 ≥ 0.99 is gated on analytical `D` samples after Occam prune
(`RECOVERY_THRESHOLDS.support_f1_clean`). MM unknown edges are not that claim.

## Why this example

You know the interaction graph: species `S` is produced from regulator `R` and
degraded by `R`, while `R` is produced from `S` and decays linearly. The
**degradation law is unknown**. BioDynaX keeps the known mass-action and linear
terms mechanistic, replaces the unknown edge with a positivity-preserving neural
destruction term, fits the hybrid UDE, then recovers a **graph-local** rational
expression only on that edge.

That is the claim this package has to prove. Global SINDy on all states, or a
bare Lux UDE with no graph, is not the product.

## 1. Observations

Columns are time `t` then one column per dynamic state.

```julia
using BioDynaX

experiment, state_names = experiment_from_csv("examples/data/unknown_inhibition.csv")
times = experiment.times
data = experiment.observations
u0 = experiment.u0
```

`Experiment` already supports irregular grids and missing-data masks. CSV import
is the usual entry; `generate_data` is only for fixtures.

## 2. Network: known graph, one unknown edge

```julia
network = build_hill_recovery_network(; known = false, hill_order = 2)
```

Internally this is four reactions:

| Reaction | Role | In the UDE |
|----------|------|------------|
| `R → S` production | known mass action | `MassActionProductionTerm` |
| `R` degrades `S` | **unknown** | `NeuralDestructionTerm` |
| `S → R` production | known mass action | mechanistic |
| `R` linear decay | known | `LinearDestructionTerm` |

Set `known = true` to compile the Hill edge as `HillDestructionTerm` (the
parameter-recovery fixtures do this). The tutorial keeps it unknown.

## 3. Compile and train

The CI protocol is multi-IC (`examples/unknown_inhibition.jl`: 9 ICs, adam
100 / bfgs 50, `train_experiments`). The snippet below is a **single-IC
sketch**, not that protocol.

```julia
using Random
rng = MersenneTwister(7)
model, params = build_ude_model(rng, network)
tspan = (first(times), last(times))
trained = train_ude(
    params, data, times, u0, tspan, model;
    adam_iters = 100, bfgs_iters = 50, verbose = false)
```

`train_ude` is Adam then optional BFGS. Adjoints are SciMLSensitivity /
Zygote. The RHS is `duᵢ = Pᵢ − Dᵢ·uᵢ` with non-negative rates, so trajectories
cannot leave the positive orthant through the boundary.

If training diverges, shorten the horizon (`HorizonCurriculum`) or lower
`adam_lr` before touching the discovery library. Expanding the sparse dictionary
will not fix an unidentified neural edge.

## 4. Discover a rational law on the unknown edge

Discovery targets the unknown destruction rate, not the full `ẋ`. The compiler
keeps known `P` and linear `D`; STLSQ only sees the neural edge.

```julia
X_traj = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X_traj)
discovery = discover_unknown_rate(R, times, D; verbose = false, strict = true)
rhs = compose_hybrid_rhs(
    model, trained.params, term,
    equation_to_function(discovery.candidates[1]))
```

`strict = true` throws. With `strict = false`, check `discovery.retcode`:

| `retcode` | Meaning |
|-----------|---------|
| `DiscoverySuccess` | support recovered; hybrid RHS is allowed |
| `DenominatorUnsafe` | `D(z)` changed sign or hit the floor |
| `EmptySupport` | STLSQ wiped every term |
| `InsufficientSamples` | not enough trajectory points |
| `SingularLibrary` | design matrix was singular |
| `DiscoveryFailed` | anything else |

Do not plot a string and call it a model. The recovered object is a rate that
must resimulate.

## 4b. Report the production–destruction scale warning

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = true)
```

`unidentifiable_edge` is a **practical** collinearity flag (Fisher + trajectory
cosine), not structural identifiability. Pinning `k_prod` with
`TrainingConfig(frozen_phys = [:k_prod])` or normalizing sampled `D` does not
remove the Jacobian tradeoff. The recovery CI requires this flag to be `true`
on the Hill UDE fixture.

## 5. Resimulate the recovered ODE

```julia
using OrdinaryDiffEq, SciMLBase, Statistics
prob = ODEProblem(rhs, u0, tspan)
sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
residual = sqrt(mean(abs2, Array(sol) .- data))
```

Gates for unknown-edge **Hill** recovery (NN fit, monomial recall, support F1
at the UDE skeleton floor, data residual, `unidentifiable_edge`) live in
`RECOVERY_THRESHOLDS` and `test/test_recovery_hard.jl`. MM unknown uses the
same training/residual protocol without the Hill recall gate. A green recovery
job is necessary, not sufficient, for v1.0. See
[Recovery benchmarks](benchmarks.md) and [API stability](stability.md).

## What this package is not

Do not start here for SBML/COPASI (Catalyst + SBMLToolkit), general sparse
regression (DataDrivenDiffEq), or structural identifiability
(StructuralIdentifiability.jl). BioDynaX owns the graph-constrained hybrid UDE
and local rational discovery path.

Next: [How-to recipes](howto.md), [SciML integration](sciml.md).
