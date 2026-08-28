###############################################################################
# Internal unique-claim recovery result (not exported).
#
# This file does not change RECOVERY_THRESHOLDS, public exports, or
# run_recovery_suite control flow beyond named generate / dummy-RNG /
# fit helpers. Held-out, functional identifiability, and
# DestructionSamples are out of scope.
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
