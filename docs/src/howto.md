# How-to recipes

Short recipes for common tasks. Blocks marked `@example` run in the
documentation build; the others are illustrative and assume the `model`,
`trained`, `set`, and `term` objects from the [Tutorial](tutorial.md).

## Run the whole workflow in one call

`discover_unknown_term(network, experiments)` builds the hybrid model,
trains it (a warm-up on the first experiment, then Adam 100 and BFGS 50 on
the training experiments), samples the learned rate on the regulator grid,
discovers a rational rate, computes the identifiability diagnostic and the
residuals, and prints the four-section report. The last `holdout`
experiments (default 2) are held out of training and reported separately.

```julia
result = discover_unknown_term(ude_net, set; rng = MersenneTwister(0), holdout = 2)
result.params            # trained parameters
result.discovery         # DiscoveryResult
result.residuals         # (data_residual, data_residual_train, data_residual_holdout)
report_unknown_term(result)   # the report as a string
```

`training = TrainingConfig(...)`, `discovery = DiscoveryConfig(...)`,
`stability_selection = StabilitySelection()`, `warmup = false`, and
`phys_init` change the individual steps; the [Tutorial](tutorial.md) shows
the call on the reference protocol and then the steps one by one.

## Load an experiment from CSV

`experiment_from_csv` reads a table whose first column is time and whose
remaining columns are the observed states. It returns the `Experiment` and
the column names. The file shipped in `examples/data/` is a synthetic
fixture generated from the tutorial network, not a measured series.

```@example howto
using BioDynaX
path = joinpath(pkgdir(BioDynaX), "examples", "data", "unknown_inhibition.csv")
experiment, names = experiment_from_csv(path)
(names, length(experiment.times), size(experiment.observations))
```

Several experiments become an `ExperimentSet`:

```@example howto
set = ExperimentSet([experiment], names)
length(set.experiments)
```

`write_experiment_csv(path, experiment; state_names)` writes one back.
`Experiment.mask` can hide a state or time subset from the loss;
`train_experiments` respects it. Training on states that are never observed
is not supported.

## Mark an edge as unknown

Build the network with the public constructors and set `known = false` on the
one reaction (or edge) whose kinetics you do not trust. It compiles to a
`NeuralDestructionTerm` whose inputs are that reaction's regulators.

```@example howto
using Random
network = BiologicalNetwork(
    [NodeSpec(name = :S), NodeSpec(name = :R)],
    EdgeSpec[];
    reactions = [
        ReactionSpec(name = :produce_s, stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg, stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = false, family = HILL,
            metadata = HillMetadata(vmax_param = :vmax, k_param = :K, hill_order = 2)),
        ReactionSpec(name = :produce_r, stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r, stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))])
model, params = build_ude_model(MersenneTwister(0), network)
BioDynaX.count_unknown_destructions(model)
```

The recovery workflow requires exactly one unknown destruction term;
`BioDynaX.assert_single_unknown_destruction(model)` raises an error
otherwise. `validate_network` itself does not enforce the count.

## Generate synthetic data

`generate_experiment_set` compiles the ground-truth model once and
integrates every initial condition from it:

```@example howto
known = BiologicalNetwork(
    [NodeSpec(name = :A), NodeSpec(name = :B)],
    EdgeSpec[];
    reactions = [
        ReactionSpec(name = :b_drives_a, stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_ba)),
        ReactionSpec(name = :a_decay, stoichiometry = Dict(1 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_a)),
        ReactionSpec(name = :b_decay, stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_b))])
synthetic = generate_experiment_set(MersenneTwister(2); network = known,
    truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5),
    initial_conditions = [[0.2, 0.1], [0.5, 0.4]], tspan = (0.0, 4.0),
    n_points = 9, noise_σ = 0.02)
(length(synthetic.experiments), synthetic.metadata[:compiled_once])
```

## Train, discover, resimulate

```julia
trained = train_experiments(params, set, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 50))
X = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X)
discovery = discover_unknown_rate(R, times, D; strict = false)
if discovery.success
    rate_fn = equation_to_function(discovery.candidates[1])
    rhs = compose_hybrid_rhs(model, trained.params, term, rate_fn)
    residual = hybrid_data_residual(model, trained.params, term, rate_fn,
        u0, tspan, times, data)
end
```

`TrainingConfig(frozen_phys = [:k_prod])` pins a known production rate during
training. Use it when the rate is known from a separate assay; it does not
by itself remove the collinearity between the production rate and the scale
of the unknown term.

With `strict = false`, check `discovery.retcode` instead of catching an error:

| `retcode` | Meaning |
|---|---|
| `DiscoverySuccess` | a support was recovered; the hybrid right-hand side can be built |
| `InsufficientSamples` | fewer than 20 sample columns |
| `DenominatorUnsafe` | the denominator changed sign or approached zero on a validation set |
| `EmptySupport` | thresholding removed every term |
| `SingularLibrary` | the design matrix was singular |
| `DiscoveryFailed` | any other error; see `discovery.message` |

`export_rhs` refuses a failed result.

## Discover from raw trajectories

Trajectory data can enter discovery without a trained model. Derivatives are
estimated by central differences, so this path is only reliable at low noise
(the analytical benchmarks succeed up to 2% noise and fail at 5%).

```@example howto
e = first(synthetic.experiments)
dX = estimate_derivatives(e.observations, e.times)
size(dX)
```

```julia
result = discover_equations(X, times, network; derivatives = dX)
rhs = export_rhs(result)
```

## Print the identifiability warning and the report

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = true)
println(BioDynaX.format_protocol_result(ident; residual = residual,
    equations = discovery.equations))
```

`format_protocol_result` accepts the values to print as keyword arguments;
anything not supplied is printed as `NA` or as "not scored".

```@example howto
ident = (; unidentifiable_edge = true, production_param = :k_prod, collinearity = 0.997)
print(BioDynaX.format_protocol_result(ident; residual = 0.0017769, seed = 103, n_ics = 9))
```

## Use the SciML solve surface

A `UDEModel` behaves as an `ODEProblem` factory. `remake` works on `p`, `u0`,
and `tspan`, and an in-place problem with a preallocated cache avoids
allocations in the forward pass:

```@example howto
using SciMLBase, OrdinaryDiffEq
p = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5),
    build_ude_model(MersenneTwister(0), known)[2].nn)
m = build_ude_model(MersenneTwister(0), known)[1]
prob = ODEProblem(m, [0.2, 0.1], (0.0, 4.0), p)
cache = allocate_cache(m, Float64)
inplace = ODEProblem(m, [0.2, 0.1], (0.0, 4.0), p; inplace = true, cache = cache)
sol = solve(remake(prob; u0 = [0.5, 0.4]), Tsit5(); saveat = [0.0, 2.0, 4.0])
round.(sol[end]; digits = 4)
```

Pair the in-place problem with `ProductionAD()` for training:

```julia
solver_config = default_solver_config(model; ad_policy = ProductionAD())
prediction = predict_ude(params, u0, tspan, times, model;
    solver_config = solver_config, cache = cache)
```

`auto_sensealg(model)` returns the adjoint that `train_ude` will use. A
one-shot Optimization.jl path is available as an unexported alternative to
`train_ude`:

```julia
prob, objective = BioDynaX.build_optimization_problem(
    model, params, data, times, u0, tspan; config = TrainingConfig())
result = BioDynaX.train_via_optimization(
    model, params, data, times, u0, tspan; maxiters = 50)
```

## Checkpoint and resume

`train_ude` writes a checkpoint every `checkpoint_every` Adam iterations when
`checkpoint_path` is given. A checkpoint stores the parameters, the Optimisers
state, the iteration counter, and the augmented-Lagrangian state, serialized
with Julia's `Serialization`. `resume_training` continues from it without
recompiling the model; BFGS is a terminal stage and restarts after
resumption.

```julia
trained = train_ude(p_init, data, times, u0, tspan, model;
    config = TrainingConfig(adam_iterations = 200),
    checkpoint_path = "run.jls", checkpoint_every = 50)
checkpoint = BioDynaX.load_checkpoint("run.jls")
resumed = BioDynaX.resume_training(checkpoint, data, times, u0, tspan, model;
    config = TrainingConfig(adam_iterations = 300))
```

`BioDynaX.save_result` and `BioDynaX.load_result` do the same for a finished
`TrainingResult`.

## Run experiments on several threads or processes

`BioDynaX.execute_experiments(f, set; config = BioDynaX.ExecutionConfig(backend = :threads))`
maps `f` over the experiments with the serial, threaded, or distributed
backend and returns the results in input order. The `:gpu` backend needs the
CUDA extension and only transfers arrays (see [Extensions](extensions.md)).

## Run part of the recovery suite

`BioDynaX.run_recovery_suite` takes a `sections` tuple. Sections that are not
requested are not run, so the known-kinetics checks can be run without the
trained-model protocol:

```julia
report = BioDynaX.run_recovery_suite(MersenneTwister(1);
    sections = (:linear, :mm, :hill, :competitive),
    linear_adam = 1, linear_bfgs = 0, mm_adam = 1, mm_bfgs = 0,
    hill_adam = 1, hill_bfgs = 0, competitive_adam = 1, competitive_bfgs = 0)
haskey(report, :ude_discovery)   # false
```

See [Benchmarks](benchmarks.md) for the list of sections.
