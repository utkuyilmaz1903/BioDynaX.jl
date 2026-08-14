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
time subset; `train_experiments` already uses that mask. Masked **training**
on missing states is not a claimed UDE path.

## Mark an edge as unknown

Build the network with public constructors (`ReactionSpec`, `HillMetadata`).
Do not call `BioDynaX.build_hill_recovery_network` from user code; that name
is an internal fixture.

```julia
network = BiologicalNetwork(
    [NodeSpec(name = :S), NodeSpec(name = :R)],
    EdgeSpec[];
    reactions = [
        ReactionSpec(name = :produce_s,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg,
                     stoichiometry = Dict(1 => -1.0), regulators = [2],
                     known = false, family = HILL,
                     metadata = HillMetadata(
                         vmax_param = :vmax, k_param = :K, hill_order = 2)),
        ReactionSpec(name = :produce_r,
                     stoichiometry = Dict(2 => 1.0), regulators = [1],
                     metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_r)),
    ])
```

Known production and linear decay stay mechanistic. The Hill degradation edge
compiles to a `NeuralDestructionTerm`.

## Train, discover, resimulate

Prefer `generate_experiment_set` + `train_experiments` as in the example.
Adam may be minibatched; BFGS refines the joint loss over every IC.

```julia
model, params = build_ude_model(rng, network)
trained = train_experiments(params, set, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 50))
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
julia --project=. benchmark/recovery_seeds.jl
julia --project=. benchmark/noise_grid.jl
```

CI thresholds live in `RECOVERY_THRESHOLDS` (`src/Recovery.jl`). Fast checks
are `test/test_recovery.jl`; the closed-loop UDE job is
`test/run_recovery_hard.jl`. Loosening a threshold is breaking.

## Optional SciML backends

```julia
using DataDrivenSparse   # BioDynaX.DataDrivenSparseSTLSQ (not exported)
using ModelingToolkit    # BioDynaX.export_mtk_system  # known terms; NN is nn_i(t)
using SBMLToolkit        # BioDynaX.import_sbmltoolkit_network
```

These are experimental. DataDrivenSparse is never a CI dependency.
