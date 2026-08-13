# How-to recipes

Short recipes around the [tutorial](tutorial.md). GPU, SBML, and Fisher
identifiability are not on this path; see [Experimental](experimental.md).

## Load a CSV experiment

```julia
experiment, names = experiment_from_csv("examples/data/unknown_inhibition.csv")
times = experiment.times
data = experiment.observations
u0 = experiment.u0
```

Write one with `write_experiment_csv`.

## Mark an edge as unknown

```julia
network = build_hill_recovery_network(; known = false, hill_order = 2)
```

Known production and linear decay stay mechanistic. The Hill degradation edge
compiles to a `NeuralDestructionTerm`.

## Train, discover, resimulate

```julia
model, params = build_ude_model(rng, network)
trained = train_ude(params, data, times, u0, tspan, model;
                    adam_iters = 80, bfgs_iters = 20, verbose = false)
X_traj = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X_traj)
discovery = discover_unknown_rate(R, times, D; strict = true)
rhs = compose_hybrid_rhs(
    model, trained.params, term,
    equation_to_function(discovery.candidates[1]))
```

If discovery cannot be trusted, `strict = false` returns
`DiscoveryResult(success=false, retcode=...)` instead of throwing.

## Run the recovery suite

```bash
julia --project=. benchmark/recovery_suite.jl
```

CI thresholds live in `RECOVERY_THRESHOLDS` (`src/Recovery.jl`). Fast checks
are `test/test_recovery.jl`; the closed-loop UDE job is
`test/run_recovery_hard.jl`.

## Optional SciML backends

```julia
using DataDrivenSparse   # DiscoveryConfig(backend = DataDrivenSparseSTLSQ())
using ModelingToolkit    # export_mtk_system(model)  # known terms; NN is nn_i(t)
using SBMLToolkit        # import_sbmltoolkit_network(path)
```
