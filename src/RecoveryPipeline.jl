###############################################################################
# Internal unique-claim recovery result (not exported).
#
# This file does not change RECOVERY_THRESHOLDS, public exports, or
# run_recovery_suite control flow beyond named generate / dummy-RNG /
# fit / sample / evaluate / report helpers. ExperimentSplit is the
# locked 7/2 view of an already-generated set. `_train_unknown_edge`
# fits on `split.train` and still returns the original 9-IC set.
# Held-out evaluation lives here as an evaluator, not a suite step.
# Functional identifiability and DestructionSamples are out of scope.
###############################################################################

"""
    MechanismRecoveryResult

Internal unique-claim recovery surface. Field access matches the current
suite row: `result.nn_correlation`, `result[:locked_kpis]`,
`haskey(result, :identifiability)`, and `keys(result)`.

`getindex` / `haskey` / `keys` follow `hasproperty`. `protocol_result`
keeps `PROTOCOL_RESULT_FIELDS` order when present.

Not exported. This is not a held-out, functional-identifiability,
hypothesis, uncertainty, or destruction-sample object.
"""
struct MechanismRecoveryResult
    nn_correlation::Float64
    nn_rate_rmse::Float64
    success::Bool
    retcode::DiscoveryRetcode
    message::String
    support_f1::Float64
    support_recall::Float64
    discovered_rate_rmse::Float64
    data_residual::Float64
    denominator_violations::Int
    normalized_support_f1::Float64
    normalized_support_recall::Float64
    extras
    extras_denominator
    discovery
    term
    identifiability
    locked_kpis
    protocol_result
    model
    params
    experiments
end

function MechanismRecoveryResult(;
        nn_correlation,
        nn_rate_rmse,
        success::Bool,
        retcode::DiscoveryRetcode,
        message,
        support_f1,
        support_recall,
        discovered_rate_rmse,
        data_residual,
        denominator_violations,
        normalized_support_f1,
        normalized_support_recall,
        extras,
        extras_denominator = nothing,
        discovery = nothing,
        term = nothing,
        identifiability = nothing,
        locked_kpis = nothing,
        protocol_result = nothing,
        model = nothing,
        params = nothing,
        experiments = nothing)
    return MechanismRecoveryResult(
        float(nn_correlation),
        float(nn_rate_rmse),
        success,
        retcode,
        String(message),
        float(support_f1),
        float(support_recall),
        float(discovered_rate_rmse),
        float(data_residual),
        Int(denominator_violations),
        float(normalized_support_f1),
        float(normalized_support_recall),
        extras,
        extras_denominator,
        discovery,
        term,
        identifiability,
        locked_kpis,
        protocol_result,
        model,
        params,
        experiments)
end

function Base.getindex(result::MechanismRecoveryResult, key::Symbol)
    hasproperty(result, key) || throw(KeyError(key))
    return getproperty(result, key)
end

function Base.haskey(result::MechanismRecoveryResult, key::Symbol)
    return hasproperty(result, key)
end

function Base.keys(result::MechanismRecoveryResult)
    return propertynames(result)
end

# Test seams for the unique-claim generate → split → fit → domain path.
# Production training and discovery are unchanged unless a test observer
# is installed.
const GENERATE_RECOVERY_EXPERIMENTS_OBSERVER = Ref{Any}(nothing)
const UNIQUE_CLAIM_EXPERIMENT_SPLIT_OBSERVER = Ref{Any}(nothing)
const FIT_UNKNOWN_DESTRUCTION_OBSERVER = Ref{Any}(nothing)
const EVALUATE_UNKNOWN_RATE_RECOVERY_RANGE_OBSERVER = Ref{Any}(nothing)
const SAMPLE_UNKNOWN_DESTRUCTION_GRID_OBSERVER = Ref{Any}(nothing)
const DISCOVER_UNKNOWN_RATE_OBSERVER = Ref{Any}(nothing)
const DISCOVER_EQUATIONS_OBSERVER = Ref{Any}(nothing)

function _note_generate_recovery_experiments(set)
    observer = GENERATE_RECOVERY_EXPERIMENTS_OBSERVER[]
    observer === nothing && return nothing
    observer(set)
    return nothing
end

function _note_unique_claim_experiment_split(split)
    observer = UNIQUE_CLAIM_EXPERIMENT_SPLIT_OBSERVER[]
    observer === nothing && return nothing
    observer(split)
    return nothing
end

function _note_fit_unknown_destruction(set)
    observer = FIT_UNKNOWN_DESTRUCTION_OBSERVER[]
    observer === nothing && return nothing
    return observer(set)
end

function _note_evaluate_unknown_rate_recovery_range(r_range)
    observer = EVALUATE_UNKNOWN_RATE_RECOVERY_RANGE_OBSERVER[]
    observer === nothing && return nothing
    return observer(r_range)
end

function _note_sample_unknown_destruction_grid(r_range)
    observer = SAMPLE_UNKNOWN_DESTRUCTION_GRID_OBSERVER[]
    observer === nothing && return nothing
    observer(r_range)
    return nothing
end

function _note_rate_discovery_entry(R, times, D, config)
    observer = DISCOVER_UNKNOWN_RATE_OBSERVER[]
    observer === nothing && return nothing
    return observer(R, times, D, config)
end

function _note_equation_discovery_entry(X, times, derivatives)
    observer = DISCOVER_EQUATIONS_OBSERVER[]
    observer === nothing && return nothing
    return observer(X, times, derivatives)
end

function with_generate_recovery_experiments_observer(f::Function, observer)
    previous = GENERATE_RECOVERY_EXPERIMENTS_OBSERVER[]
    GENERATE_RECOVERY_EXPERIMENTS_OBSERVER[] = observer
    try
        return f()
    finally
        GENERATE_RECOVERY_EXPERIMENTS_OBSERVER[] = previous
    end
end

function with_unique_claim_experiment_split_observer(f::Function, observer)
    previous = UNIQUE_CLAIM_EXPERIMENT_SPLIT_OBSERVER[]
    UNIQUE_CLAIM_EXPERIMENT_SPLIT_OBSERVER[] = observer
    try
        return f()
    finally
        UNIQUE_CLAIM_EXPERIMENT_SPLIT_OBSERVER[] = previous
    end
end

function with_fit_unknown_destruction_observer(f::Function, observer)
    previous = FIT_UNKNOWN_DESTRUCTION_OBSERVER[]
    FIT_UNKNOWN_DESTRUCTION_OBSERVER[] = observer
    try
        return f()
    finally
        FIT_UNKNOWN_DESTRUCTION_OBSERVER[] = previous
    end
end

function with_evaluate_unknown_rate_recovery_range_observer(f::Function, observer)
    previous = EVALUATE_UNKNOWN_RATE_RECOVERY_RANGE_OBSERVER[]
    EVALUATE_UNKNOWN_RATE_RECOVERY_RANGE_OBSERVER[] = observer
    try
        return f()
    finally
        EVALUATE_UNKNOWN_RATE_RECOVERY_RANGE_OBSERVER[] = previous
    end
end

function with_sample_unknown_destruction_grid_observer(f::Function, observer)
    previous = SAMPLE_UNKNOWN_DESTRUCTION_GRID_OBSERVER[]
    SAMPLE_UNKNOWN_DESTRUCTION_GRID_OBSERVER[] = observer
    try
        return f()
    finally
        SAMPLE_UNKNOWN_DESTRUCTION_GRID_OBSERVER[] = previous
    end
end

function with_discover_unknown_rate_observer(f::Function, observer)
    previous = DISCOVER_UNKNOWN_RATE_OBSERVER[]
    DISCOVER_UNKNOWN_RATE_OBSERVER[] = observer
    try
        return f()
    finally
        DISCOVER_UNKNOWN_RATE_OBSERVER[] = previous
    end
end

function with_discover_equations_observer(f::Function, observer)
    previous = DISCOVER_EQUATIONS_OBSERVER[]
    DISCOVER_EQUATIONS_OBSERVER[] = observer
    try
        return f()
    finally
        DISCOVER_EQUATIONS_OBSERVER[] = previous
    end
end

"""
    generate_recovery_experiments(rng, truth_net, truth_params; tspan, n_points,
                                 noise_σ, initial_conditions)

Nine-IC synthetic set used by unique-claim training. This is not
`unique_claim_experiment_set` and does not attach fingerprint metadata.
"""
function generate_recovery_experiments(rng, truth_net, truth_params;
        tspan, n_points, noise_σ,
        initial_conditions = _unknown_edge_ics())
    set = generate_experiment_set(
        rng; network = truth_net, initial_conditions = initial_conditions,
        tspan = tspan, n_points = n_points, noise_σ = noise_σ,
        truth_params = truth_params)
    _note_generate_recovery_experiments(set)
    return set
end

"""Locked unique-claim train indices. Not a `UNIQUE_CLAIM_PROTOCOL` field."""
const UNIQUE_CLAIM_TRAIN_INDICES = (1, 2, 3, 4, 5, 6, 7)

"""Locked unique-claim holdout indices. Not a `UNIQUE_CLAIM_PROTOCOL` field."""
const UNIQUE_CLAIM_HOLDOUT_INDICES = (8, 9)

"""
    ExperimentSplit

Locked unique-claim 7/2 view of an already-generated 9-IC `ExperimentSet`.
Train indices are `(1, 2, 3, 4, 5, 6, 7)`; holdout indices are `(8, 9)`.
The wrapped `Experiment` objects are the original generated objects.
Not a second generated set. Not exported.
"""
struct ExperimentSplit
    train_indices::NTuple{7,Int}
    holdout_indices::NTuple{2,Int}
    train::ExperimentSet
    holdout::ExperimentSet
end

"""
    unique_claim_experiment_split(set::ExperimentSet) -> ExperimentSplit

Partition a 9-IC unique-claim `ExperimentSet` into the locked 7/2 view.
Requires `length(set) == 9`. Consumes the already-generated set: it does
not generate experiments and does not mutate `set`.
"""
function unique_claim_experiment_split(set::ExperimentSet)
    length(set) == 9 || throw(ArgumentError(
        "unique_claim_experiment_split requires exactly 9 experiments; got $(length(set))"))
    train = ExperimentSet(
        [set.experiments[i] for i in UNIQUE_CLAIM_TRAIN_INDICES],
        set.state_names;
        units = set.units,
        metadata = set.metadata)
    holdout = ExperimentSet(
        [set.experiments[i] for i in UNIQUE_CLAIM_HOLDOUT_INDICES],
        set.state_names;
        units = set.units,
        metadata = set.metadata)
    split = ExperimentSplit(
        UNIQUE_CLAIM_TRAIN_INDICES,
        UNIQUE_CLAIM_HOLDOUT_INDICES,
        train,
        holdout)
    _note_unique_claim_experiment_split(split)
    return split
end

"""
    consume_shared_suite_rng!(rng, truth_net)

Discarded `build_ude_model(rng, truth_net)` consume used by
`:ude_discovery` / `:mm_unknown` so the shared suite RNG stays aligned
with known-kinetics fixtures. Not a per-section seed.
"""
function consume_shared_suite_rng!(rng, truth_net)
    return build_ude_model(rng, truth_net)
end

"""
    fit_unknown_destruction(ude_model, ude_p0, set; adam, bfgs, frozen_phys,
                           phys_init)

Unique-claim UDE fit: physics init, locked training config, and
`train_experiments_with_warmup`. Numerical training configuration is
unchanged from `_train_unknown_edge`.
"""
function fit_unknown_destruction(ude_model, ude_p0, set;
        adam, bfgs,
        frozen_phys::Vector{Symbol} = Symbol[],
        phys_init = nothing)
    observed = _note_fit_unknown_destruction(set)
    observed isa TrainingResult && return observed
    names = Tuple(parameter_schema(ude_model).phys_names)
    guess = phys_init === nothing ?
            NamedTuple{names}(ntuple(_ -> 0.8, length(names))) : phys_init
    ude_init = pack_parameters(guess, ude_p0.nn)
    config = unique_claim_training_config(
        model = ude_model,
        adam_iterations = adam,
        bfgs_iterations = bfgs,
        frozen_phys = frozen_phys)
    return train_experiments_with_warmup(
        ude_init, set, ude_model;
        config = lock_training_config(ude_model, config),
        verbose = false)
end

"""
    sample_destruction(model, params, term; r_range, fill_value)

Thin wrapper around `sample_unknown_destruction_grid`. Returns the
existing `(R, D, term)` representation. Not exported. Not a
`DestructionSamples` object and not `sample_learned_function`.
"""
function sample_destruction(model, params, term;
        r_range = range(0.05, 2.0; length = 80),
        fill_value = 0.3)
    return sample_unknown_destruction_grid(
        model, params, term; r_range = r_range, fill_value = fill_value)
end

"""
    evaluate_recovery(R_grid, D_nn, discovery, discovery_norm, truth_rate,
                      truth_support, data_residual_fn)

Metric-only unique-claim evaluation. Computes the current Q1 / Q2 / Q5
fields from already-run raw and normalized `DiscoveryResult`s. Does not
discover, normalize samples, decide `training_ok`, construct times, or
own success / retcode / message. Not exported. Not a held-out or
functional-identifiability diagnostic.
"""
function evaluate_recovery(R_grid, D_nn, discovery, discovery_norm, truth_rate,
                           truth_support, data_residual_fn)
    r = vec(R_grid)
    D_true = truth_rate(r)
    f1 = 0.0
    recall = 0.0
    rate_rmse = Inf
    residual = Inf
    den_violations = typemax(Int)
    extras = String[]
    extras_denominator = nothing
    if discovery.success
        candidate = discovery.candidates[1]
        metrics = support_f1(candidate, truth_support.numerator,
                             truth_support.denominator)
        f1 = metrics.combined.f1
        recall = metrics.combined.recall
        extras = discovered_support_extras(
            candidate, truth_support.numerator, truth_support.denominator)
        d_hat = equation_to_function(candidate)
        D_hat = [d_hat([rj]) for rj in r]
        rate_rmse = rate_rel_rmse(D_hat, D_true)
        den_violations = denominator_violation_count(candidate, R_grid)
        extras_denominator = ude_extras_denominator_row(
            candidate, R_grid; extras = extras)
        residual = data_residual_fn(d_hat)
    end
    norm_f1 = 0.0
    norm_recall = 0.0
    if discovery_norm.success
        metrics_n = support_f1(discovery_norm.candidates[1],
                               truth_support.numerator, truth_support.denominator)
        norm_f1 = metrics_n.combined.f1
        norm_recall = metrics_n.combined.recall
    end
    return (;
        support_f1 = f1,
        support_recall = recall,
        discovered_rate_rmse = rate_rmse,
        data_residual = residual,
        denominator_violations = den_violations,
        normalized_support_f1 = norm_f1,
        normalized_support_recall = norm_recall,
        extras,
        extras_denominator,
    )
end

"""
    report_recovery(evaled, ident; model, params, experiments)

Typed unique-claim report. Builds an internal `MechanismRecoveryResult`
from composer output and the existing identifiability object. Always
fills `locked_kpis` and `protocol_result`. Not exported. Not a held-out,
functional-identifiability, hypothesis, or uncertainty object.
"""
function report_recovery(evaled, ident;
        model = nothing, params = nothing, experiments = nothing)
    extras_denominator = hasproperty(evaled, :extras_denominator) ?
                         getproperty(evaled, :extras_denominator) : nothing
    discovery = hasproperty(evaled, :discovery) ? getproperty(evaled, :discovery) :
                nothing
    term = hasproperty(evaled, :term) ? getproperty(evaled, :term) : nothing
    kpi_src = (;
        data_residual = evaled.data_residual,
        support_recall = evaled.support_recall,
        support_f1 = evaled.support_f1,
        extras = evaled.extras,
        identifiability = ident)
    return MechanismRecoveryResult(;
        nn_correlation = evaled.nn_correlation,
        nn_rate_rmse = evaled.nn_rate_rmse,
        success = evaled.success,
        retcode = evaled.retcode,
        message = evaled.message,
        support_f1 = evaled.support_f1,
        support_recall = evaled.support_recall,
        discovered_rate_rmse = evaled.discovered_rate_rmse,
        data_residual = evaled.data_residual,
        denominator_violations = evaled.denominator_violations,
        normalized_support_f1 = evaled.normalized_support_f1,
        normalized_support_recall = evaled.normalized_support_recall,
        extras = evaled.extras,
        extras_denominator = extras_denominator,
        discovery = discovery,
        term = term,
        identifiability = ident,
        locked_kpis = locked_ude_kpis(kpi_src),
        protocol_result = build_protocol_result(kpi_src),
        model = model,
        params = params,
        experiments = experiments)
end

"""
    HoldoutEvidence

Four-scalar unique-claim held-out evidence. Not a gate, not a discovery
result, and not a functional-identifiability object. Not exported.
"""
struct HoldoutEvidence
    data_residual_train::Float64
    data_residual_holdout::Float64
    d_rmse_holdout::Float64
    d_rmse_holdout_domain::Float64
end

function _holdout_observed_regulators(holdout::ExperimentSet, term)
    return reduce(vcat, (exp.observations[term.regulator, :]
                         for exp in holdout.experiments))
end

function _unique_claim_external_regulator_band(train::ExperimentSet, term)
    r_train = reduce(vcat, (exp.observations[term.regulator, :]
                            for exp in train.experiments))
    r_lo_train, r_hi_train = extrema(r_train)
    span_train = max(r_hi_train - r_lo_train, 0.1)
    r_lo_external = r_hi_train + 0.15 * span_train
    r_hi_external = r_hi_train + 0.35 * span_train
    n_external = 80
    return range(r_lo_external, r_hi_external; length = n_external)
end

function _finite_rate_rel_rmse(estimate, truth)
    estimate_vec = vec(Float64.(estimate))
    truth_vec = vec(Float64.(truth))
    if !all(isfinite, estimate_vec) || !all(isfinite, truth_vec)
        return Inf
    end
    return rate_rel_rmse(estimate_vec, truth_vec)
end

function _mean_hybrid_residual(experiments, model, params, term, D_hat_fn)
    n = length(experiments)
    total = 0.0
    for exp in experiments
        ρ = hybrid_data_residual(
            model, params, term, D_hat_fn,
            exp.u0, (first(exp.times), last(exp.times)),
            exp.times, exp.observations;
            mask = exp.mask)
        ρ === Inf && return Inf
        total += ρ
    end
    return total / n
end

"""
    evaluate_holdout

Evaluate an already-fitted unique-claim model on the locked 7/2 split.
Always returns HoldoutEvidence. Does not train, discover, or generate.
Not exported.
"""
function evaluate_holdout(split::ExperimentSplit, evaled, model, params, term,
        truth_rate)
    D_hat_fn = neural_identity_rate(model, params, term)
    data_residual_train = _mean_hybrid_residual(
        split.train.experiments, model, params, term, D_hat_fn)
    data_residual_holdout = _mean_hybrid_residual(
        split.holdout.experiments, model, params, term, D_hat_fn)
    r_holdout = _holdout_observed_regulators(split.holdout, term)
    (R, D_hat_vals, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_holdout, fill_value = 0.3)
    d_rmse_holdout = _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
    r_band_external = _unique_claim_external_regulator_band(split.train, term)
    (R, D_hat_vals, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_band_external, fill_value = 0.3)
    d_rmse_holdout_domain = _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
    return HoldoutEvidence(
        Float64(data_residual_train),
        Float64(data_residual_holdout),
        Float64(d_rmse_holdout),
        Float64(d_rmse_holdout_domain))
end
