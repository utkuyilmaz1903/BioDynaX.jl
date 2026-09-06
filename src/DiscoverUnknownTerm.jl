###############################################################################
# One-call entry point: discover_unknown_term (exported).
#
# Runs the reference workflow of examples/unknown_inhibition.jl with the same
# functions, in the same order, with the same defaults: build the hybrid
# model, warm up on the first experiment, train jointly, sample the learned
# rate on the regulator grid of the training experiments, discover a rational
# rate, compute the identifiability diagnostic and the hybrid residuals, and
# bundle everything in one result whose `report_unknown_term` is the four-section
# report.
###############################################################################

"""
    UnknownTermResult

Everything `discover_unknown_term` computes:

- `network`, `model`, `params`, `training`: the hybrid model, its trained
  parameters, and the `TrainingResult` of the joint fit;
- `term`: the neural destruction term that was discovered;
- `identifiability`: the production/destruction trade-off report on the
  first training experiment;
- `discovery`: the `DiscoveryResult` of the rational-rate regression;
- `samples`: the regulator grid `R` and the learned rate `D` the discovery
  used;
- `residuals`: `data_residual` (hybrid model against the first training
  experiment), `data_residual_train` (mean over the training experiments),
  and `data_residual_holdout` (mean over the held-out experiments, `NaN`
  when none were held out); `Inf` when the hybrid model could not be built;
- `training_indices`, `holdout_indices`: which experiments trained the
  model and which were held out;
- `extras`: the recovered monomials outside `known_support`, or `nothing`
  when no truth was given;
- `settings`: the values printed in the reproduction section of the report.

`report_unknown_term(result)` returns the four-section report as a string
and `show` prints it.
"""
struct UnknownTermResult{M, P, T, I, D, S}
    network::BiologicalNetwork
    model::M
    params::P
    training::TrainingResult
    term::T
    identifiability::I
    discovery::D
    samples::S
    residuals::NamedTuple{(:data_residual, :data_residual_train, :data_residual_holdout),
        NTuple{3, Float64}}
    training_indices::Vector{Int}
    holdout_indices::Vector{Int}
    extras::Union{Nothing, Vector{String}}
    settings::NamedTuple
end

"""
    report_unknown_term(result::UnknownTermResult) -> String

The four-section report (identifiability, fit, discovery, reproduction) of a
`discover_unknown_term` result, as printed by the reference example. The fit
section also lists the mean residual over the training experiments and, when
experiments were held out, the mean residual over them.
"""
function report_unknown_term(result::UnknownTermResult)
    settings = result.settings
    residuals = result.residuals
    holdout = isempty(result.holdout_indices) ? nothing : residuals.data_residual_holdout
    return format_protocol_result(result.identifiability;
        residual = residuals.data_residual,
        residual_train = residuals.data_residual_train,
        residual_holdout = holdout,
        equations = result.discovery.equations,
        extras = result.extras,
        unknown_holes = settings.unknown_holes,
        seed = settings.seed,
        n_ics = settings.n_ics,
        n_points = settings.n_points,
        adam_iters = settings.adam_iters,
        bfgs_iters = settings.bfgs_iters,
        bootstrap = settings.bootstrap,
        discovery_seed = settings.discovery_seed)
end

function Base.show(io::IO, ::MIME"text/plain", result::UnknownTermResult)
    print(io, report_unknown_term(result))
end

function Base.show(io::IO, result::UnknownTermResult)
    print(io, "UnknownTermResult(experiments = ", result.settings.n_ics,
        ", held out = ", length(result.holdout_indices),
        ", discovery ", result.discovery.success ? "succeeded" : "failed",
        ", hybrid_data_residual = ",
        _format_protocol_value(result.residuals.data_residual), ")")
end

function _unknown_term_subset(set::ExperimentSet, indices)
    return ExperimentSet(
        [set.experiments[i] for i in indices], set.state_names;
        units = set.units, metadata = set.metadata)
end

function _unknown_term_choice(model::UDEModel, term)
    terms = neural_destruction_terms(model)
    term === nothing && return only(terms)
    term isa Integer && return terms[term]
    term isa NeuralDestructionTerm && return term
    throw(ArgumentError("term must be nothing, an index, or a NeuralDestructionTerm"))
end

"""
    discover_unknown_term(network, experiments; term=nothing,
                          training=TrainingConfig(adam_iterations=100, bfgs_iterations=50, log_every=10^6),
                          discovery=rate_discovery_config(), holdout=2,
                          rng=MersenneTwister(0), phys_init=nothing, warmup=true,
                          known_support=nothing, stability_selection=nothing,
                          strict=false, seed=nothing, regulator_grid=nothing,
                          verbose=true)

Train the hybrid model of `network` on `experiments`, discover a rational
expression for its one unknown destruction term, and return an
`UnknownTermResult`. The steps are those of `examples/unknown_inhibition.jl`,
in the same order and with the same defaults:

1. `build_ude_model(rng, network)`; the network must have exactly one
   unknown destruction term (`term` selects it when several exist).
2. Physical parameters start from a flat guess of 0.8 (`phys_init` overrides).
3. A warm-up `train_ude` on the first training experiment with the training
   config's settings (Adam iterations and learning rate, gradient clip,
   constraint, solver, frozen parameters), no BFGS, and the horizon
   curriculum 35%, 70%, 100% (`warmup = false` skips it).
4. `train_experiments` on the training experiments with `training`
   (default: Adam 100 then BFGS 50).
5. The learned rate is sampled on the regulator grid of the training
   experiments (`sample_unknown_destruction_grid`) and
   `discover_unknown_rate` fits a rational rate with `discovery` (default:
   the reference protocol's configuration, bootstrap 8, discovery seed 3).
   `stability_selection` and `strict` are passed through. `regulator_grid`
   replaces the grid: a vector or range of regulator values, or a function
   `(model, params, training_set, term) -> grid` called after training,
   which is the way to sample a regulator that is never observed (its
   observations are `NaN` and masked); `nothing` keeps the observed grid.
6. `report_production_destruction_tradeoff` on the first training experiment.
7. `hybrid_data_residual` of the model with the discovered rate against the
   first training experiment, the mean over the training experiments, and
   the mean over the held-out experiments.

`holdout` is the number of experiments at the end of `experiments` that are
held out of training and used only for the held-out residual (the reference
protocol holds out 2 of 9). Residuals and the identifiability diagnostic use
each experiment's observation mask, so unobserved entries do not count.
`known_support`, the true implicit support when
the data are synthetic (for example `BioDynaX.hill_rate_support(2)`), is used
only to list the extra terms in the report. `seed` is recorded in the report
and not used otherwise; `verbose` prints training progress and the report.
"""
function discover_unknown_term(network::BiologicalNetwork, experiments::ExperimentSet;
        term = nothing,
        training::TrainingConfig = TrainingConfig(
            adam_iterations = REFERENCE_PROTOCOL.adam_iterations,
            bfgs_iterations = REFERENCE_PROTOCOL.bfgs_iterations,
            log_every = 10^6),
        discovery::DiscoveryConfig = rate_discovery_config(),
        holdout::Integer = 2,
        rng::AbstractRNG = MersenneTwister(0),
        phys_init = nothing,
        warmup::Bool = true,
        known_support = nothing,
        stability_selection::Union{Nothing, StabilitySelection} = nothing,
        strict::Bool = false,
        seed = nothing,
        regulator_grid = nothing,
        verbose::Bool = true)
    n = length(experiments.experiments)
    n ≥ 1 || throw(ArgumentError("experiments must contain at least one experiment"))
    0 ≤ holdout ≤ n - 1 || throw(ArgumentError(
        "holdout must be between 0 and $(n - 1) for $(n) experiments; got $(holdout)"))
    training_indices = collect(1:(n - holdout))
    holdout_indices = collect((n - holdout + 1):n)
    train_set = holdout == 0 ? experiments :
                _unknown_term_subset(experiments, training_indices)
    holdout_set = holdout == 0 ? nothing :
                  _unknown_term_subset(experiments, holdout_indices)

    model, p0 = build_ude_model(rng, network)
    assert_single_unknown_destruction(model)
    phys_names = Tuple(parameter_schema(model).phys_names)
    guess = phys_init === nothing ?
            NamedTuple{phys_names}(ntuple(_ -> 0.8, length(phys_names))) : phys_init
    ude_init = pack_parameters(guess, p0.nn)
    first_exp = first(train_set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    start = ude_init
    if warmup
        warm = train_ude(
            ude_init, first_exp.observations, first_exp.times, first_exp.u0, tspan, model;
            config = TrainingConfig(training; bfgs_iterations = 0,
                horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0])),
            verbose = verbose)
        start = warm.params
    end
    trained = train_experiments(start, train_set, model; config = training,
        verbose = verbose)

    chosen = _unknown_term_choice(model, term)
    r_range = regulator_grid === nothing ? _regulator_grid(train_set, chosen) :
              regulator_grid isa Function ?
              regulator_grid(model, trained.params, train_set, chosen) : regulator_grid
    R, D, chosen = sample_unknown_destruction_grid(model, trained.params, chosen;
        r_range = r_range)
    times_grid = collect(range(0.0, 1.0; length = size(R, 2)))
    found = discover_unknown_rate(R, times_grid, D;
        config = discovery, verbose = false, strict = strict,
        stability_selection = stability_selection)
    ident = report_production_destruction_tradeoff(
        model, trained.params, first_exp.observations, first_exp.times,
        first_exp.u0, tspan; term = chosen, verbose = false, mask = first_exp.mask)

    residual = Inf
    residual_train = Inf
    residual_holdout = holdout == 0 ? NaN : Inf
    extras = nothing
    if found.success && !isempty(found.candidates)
        candidate = found.candidates[1]
        rate_fn = equation_to_function(candidate)
        function residual_of(e)
            hybrid_data_residual(model, trained.params, chosen, rate_fn,
                e.u0, (first(e.times), last(e.times)), e.times, e.observations;
                mask = e.mask)
        end
        residual = residual_of(first_exp)
        residual_train = mean(residual_of(e) for e in train_set.experiments)
        holdout == 0 || (residual_holdout = mean(residual_of(e)
        for e in holdout_set.experiments))
        known_support === nothing || (extras = discovered_support_extras(
            candidate, known_support.numerator, known_support.denominator))
    end
    backend = discovery.backend
    settings = (;
        unknown_holes = count_unknown_destructions(model),
        seed = seed,
        n_ics = n,
        n_points = size(first_exp.observations, 2),
        adam_iters = training.adam_iterations,
        bfgs_iters = training.bfgs_iterations,
        bootstrap = backend isa ImplicitSINDyPI ? backend.bootstrap_samples : nothing,
        discovery_seed = Int(discovery.seed),
        holdout = holdout,
        warmup = warmup,
        regulator_grid = regulator_grid === nothing ? :observed :
                         regulator_grid isa Function ? :function : :given)
    result = UnknownTermResult(
        network, model, trained.params, trained, chosen, ident, found, (; R, D),
        (; data_residual = Float64(residual),
            data_residual_train = Float64(residual_train),
            data_residual_holdout = Float64(residual_holdout)),
        training_indices, holdout_indices, extras, settings)
    verbose && println(report_unknown_term(result))
    return result
end
