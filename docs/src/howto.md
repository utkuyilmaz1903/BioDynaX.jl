# How-to recipes

Research-preview recipes around the [tutorial](tutorial.md). GPU, SBML, and
Fisher identifiability are not on this path; see [Experimental](experimental.md).

The golden path is the **multi-IC** protocol in
`examples/unknown_inhibition.jl` (same ICs, horizon, and residual gate as the
recovery CI job). A single-IC `train_ude` snippet below is a sketch, not the
CI protocol.

## Load a CSV experiment

```julia
experiment, names = experiment_from_csv("examples/data/unknown_inhibition.csv")
times = experiment.times
data = experiment.observations
u0 = experiment.u0
```

Write one with `write_experiment_csv`. `Experiment.mask` can hide a state or
time subset; `train_experiments` already uses that mask.

## Mark an edge as unknown

```julia
network = build_hill_recovery_network(; known = false, hill_order = 2)
```

Known production and linear decay stay mechanistic. The Hill degradation edge
compiles to a `NeuralDestructionTerm`.

## Train, discover, resimulate

Prefer `generate_experiment_set` + `train_experiments` as in the example. A
one-trajectory sketch:

```julia
model, params = build_ude_model(rng, network)
trained = train_ude(params, data, times, u0, tspan, model;
                    adam_iters = 100, bfgs_iters = 50, verbose = false)
X_traj = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X_traj)
discovery = discover_unknown_rate(R, times, D; strict = true)
rhs = compose_hybrid_rhs(
    model, trained.params, term,
    equation_to_function(discovery.candidates[1]))
```

If discovery cannot be trusted, `strict = false` returns
`DiscoveryResult(success=false, retcode=...)` instead of throwing.

After a fit, report the practical scale warning (not exported):

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = true)
```

`TrainingConfig(frozen_phys = [:k_prod])` pins a known production rate. It does
not identify `D(z)` scale in the Jacobian sense.

## Run the recovery suite

```bash
julia --project=. benchmark/recovery_suite.jl
julia --project=. benchmark/sindy_baseline.jl
```

CI thresholds live in `RECOVERY_THRESHOLDS` (`src/Recovery.jl`). Fast checks
are `test/test_recovery.jl`; the closed-loop UDE job is
`test/run_recovery_hard.jl`. Loosening a threshold is breaking.

## Optional SciML backends

```julia
using DataDrivenSparse   # DiscoveryConfig(backend = DataDrivenSparseSTLSQ())
using ModelingToolkit    # export_mtk_system(model)  # known terms; NN is nn_i(t)
using SBMLToolkit        # import_sbmltoolkit_network(path)
```

DataDrivenSparse is never a CI dependency. The graph vs global table is
produced by `benchmark/sindy_baseline.jl`.
