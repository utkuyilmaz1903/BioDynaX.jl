# How-to recipes

Research-preview recipes around the [tutorial](tutorial.md). GPU, SBML, and
Fisher identifiability are not on this path; see [Experimental](experimental.md).

The golden path is the **multi-IC** protocol in
`examples/unknown_inhibition.jl` (seed 103, same ICs, horizon, regulator-grid
discovery, residual gate, and identifiability as the first printed block).
`BIODYNAX_SMOKE=1` is a 1-IC / 8-point fast check and is not that protocol.
A single-IC `train_ude` snippet below is a sketch, not the CI protocol.

## Load a CSV experiment

```julia
experiment, names = experiment_from_csv("examples/data/unknown_inhibition.csv")
times = experiment.times
data = experiment.observations
u0 = experiment.u0
```

That committed CSV is a **static demo table**. `examples/unknown_inhibition.jl`
writes a generated copy to a temp directory and does not overwrite it.
Write a new file with `write_experiment_csv`. `Experiment.mask` can hide a
state or time subset; `train_experiments` already uses that mask. Masked
**training** on missing states is not a claimed UDE path.

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

Prefer `BioDynaX.unique_claim_experiment_set` + `train_experiments` as
in the example (fingerprint ICs and point counts). That path calls
`compile_ground_truth_model` once. A raw
`generate_experiment_set` snippet below is a sketch; it now also
compiles once and then uses `generate_experiment_set_from_compiled_model`.
Adam may be minibatched; BFGS refines the joint loss over every IC.

```julia
model, params = build_ude_model(rng, network)
trained = train_experiments(params, set, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 50))
X_traj = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X_traj)
discovery = discover_unknown_rate(R, times, D; strict = true)
# Protocol path (seed 103 / 9 ICs): sample_unknown_destruction_grid
# over unique_claim_fingerprint(); this snippet is not that job.
rhs = compose_hybrid_rhs(
    model, trained.params, term,
    equation_to_function(discovery.candidates[1]))
```

If discovery cannot be trusted, `strict = false` returns
`DiscoveryResult(success=false, retcode=...)` instead of throwing.

Library evaluation reuses grow-only buffers. `evaluate_library!` writes
in place. `_fit_implicit` streams implicit design chunks through
`_stlsq_blocked!` so a bootstrap draw does not rebuild the Gram. See
[Discovery streaming](discovery-streaming.md).

After a fit, print the protocol block (identifiability first; not exported).
`unidentifiable_edge` is the product warning: coefficients are not
biological constants. `BioDynaX.assert_single_unknown_destruction(model)`
errors unless there is exactly one unknown `D(z)`. `validate_network`
does not enforce that single hole; zero- and two-hole networks still
compile. See [Unique claim](unique-claim.md).

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = false)
println(BioDynaX.format_protocol_result(ident; residual = residual))
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

`run_recovery_suite` admits unique-claim sections through
`admit_recovery_suite_network`. Other sections have an explicit hole
policy and stay open; see [Compiled experiment path](compiled-path.md).
A skipped recovery-suite section does not call `_train_unknown_edge`.
See [Recovery suite skip](recovery-suite-skip.md).

`SolveSurfaceRow` compares `ude_system`, `ODEFunction`, `ODEProblem`,
`remake`, inplace cache, `SciMLBase.solve`, and `predict_ude` on the
same fixtures. See [SciML solve surface](sciml-solve-surface.md).

`TrainingSolveSession` remakes one `ODEProblem` across ICs. Unique-claim
training (`_train_unknown_edge`) calls `train_experiments_with_warmup` so
the first-IC Adam state is not discarded. See
[Training reuse](training-reuse.md). The AD constraint `predict_ude`
call must pass the compiled model.

`experiment_fingerprint` hashes times, observations, mask, and `u0`.
Metadata is not identity. `experiment_batches` covers every IC.
`resume_training` reuses the compiled `UDEModel`. Remapped multi-head
generate and `train_experiments` share one compiled tree. See
[Experiment fingerprint and checkpoint](experiment-checkpoint.md).

`DiscoveryRetcode` names insufficient samples, unsafe denominators,
empty support, a singular library, a generic failure, and success.
See [Failure modes](failure-modes.md).

`compose_hybrid_rhs` with the neural destruction rate recovers
`ude_system`. See [Hybrid compose path](hybrid-compose.md).

`hybrid_data_residual` agrees with `SciMLBase.solve` of
`compose_hybrid_rhs`. Smoke residual (1 IC / 8 points) is not the
seed-103 / 9-IC protocol residual. See
[Hybrid residual versus solver](hybrid-residual.md).

`production_destruction_tradeoff` joins `UniqueClaimProtocolRow`
through `identifiability_product`.
`coefficients_are_biological_constants` is false exactly when
`unidentifiable_edge` is true. See
[Identifiability product rows](identifiability-product.md).

`local_basis` `scope=:graph` uses graph parents; `scope=:global` is
the ablation. `local_has_true_parent_gate` is the recovered-support
membership check. See
[Graph-local library and ablation](graph-local-library.md).
