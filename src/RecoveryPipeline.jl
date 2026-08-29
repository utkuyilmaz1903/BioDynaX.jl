###############################################################################
# Internal unique-claim recovery result (not exported).
#
# This file does not change RECOVERY_THRESHOLDS, public exports, or
# run_recovery_suite control flow beyond named generate / dummy-RNG /
# fit / sample / evaluate / report helpers. Held-out, functional
# identifiability, and DestructionSamples are out of scope.
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

"""
    generate_recovery_experiments(rng, truth_net, truth_params; tspan, n_points,
                                 noise_σ, initial_conditions)

Nine-IC synthetic set used by unique-claim training. This is not
`unique_claim_experiment_set` and does not attach fingerprint metadata.
"""
function generate_recovery_experiments(rng, truth_net, truth_params;
        tspan, n_points, noise_σ,
        initial_conditions = _unknown_edge_ics())
    return generate_experiment_set(
        rng; network = truth_net, initial_conditions = initial_conditions,
        tspan = tspan, n_points = n_points, noise_σ = noise_σ,
        truth_params = truth_params)
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
