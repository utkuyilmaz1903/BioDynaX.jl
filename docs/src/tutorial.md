# Tutorial: unknown inhibition recovery

This is the only golden path the package is built around:

```text
CSV / time series
  → BiologicalNetwork (known kinetics + one unknown edge)
  → compile / build_ude_model
  → train_ude
  → discover_equations
  → export_rhs
  → resimulate vs data
```

A runnable copy is `examples/unknown_inhibition.jl` at the repository root.
SBML, GPU, multi-head NNs, and custom kinetics are **not** on this path.

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

```julia
using Random
rng = MersenneTwister(7)
model, params = build_ude_model(rng, network)
tspan = (first(times), last(times))
trained = train_ude(
    params, data, times, u0, tspan, model;
    adam_iters = 80, bfgs_iters = 20, verbose = false)
```

`train_ude` is Adam then optional BFGS. Adjoints are SciMLSensitivity /
Zygote. The RHS is `duᵢ = Pᵢ − Dᵢ·uᵢ` with non-negative rates, so trajectories
cannot leave the positive orthant through the boundary.

If training diverges, shorten the horizon (`HorizonCurriculum`) or lower
`adam_lr` before touching the discovery library. Expanding the sparse dictionary
will not fix an unidentified neural edge.

## 4. Discover a rational law on the unknown edge

```julia
discovery = discover_equations(
    trained.params, model;
    u0 = u0, tspan = tspan, n_samples = 80,
    verbose = false, strict = true)
rhs = export_rhs(discovery)
```

`discover_equations` samples the **trained UDE**, not the CSV derivatives, and
fits `D(z)ẋ − N(z) = 0` on graph parents only (`local_basis(...; scope=:graph)`).

`strict = true` throws. With `strict = false`, check `discovery.retcode`:

| `retcode` | Meaning |
|-----------|---------|
| `DiscoverySuccess` | support recovered; `export_rhs` is allowed |
| `DenominatorUnsafe` | `D(z)` changed sign or hit the floor |
| `EmptySupport` | STLSQ wiped every term |
| `InsufficientSamples` | not enough trajectory points |
| `SingularLibrary` | design matrix was singular |
| `DiscoveryFailed` | anything else |

`export_rhs` refuses a failed result. Do not plot a string and call it a model.

## 5. Resimulate the recovered ODE

```julia
using OrdinaryDiffEq, SciMLBase, Statistics
prob = ODEProblem((u, p, t) -> rhs(u), u0, tspan)
sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
residual = sqrt(mean(abs2, Array(sol) .- data))
```

The recovered RHS is useful when it tracks the trained UDE (and the data) on the
observed window. Exact Hill coefficients from a short neural fit are **not**
guaranteed; that recovery is gated on compiled (NN-free) fixtures and on
analytical-derivative discovery in the test suite. See
[Recovery benchmarks](benchmarks.md).

## What this package is not

Do not start here for SBML/COPASI (Catalyst + SBMLToolkit), general sparse
regression (DataDrivenDiffEq), or structural identifiability
(StructuralIdentifiability.jl). BioDynaX owns the graph-constrained hybrid UDE
and local rational discovery path.

Next: [How-to recipes](howto.md), [SciML integration](sciml.md).
