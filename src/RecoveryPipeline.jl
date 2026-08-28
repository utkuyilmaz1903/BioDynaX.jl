###############################################################################
# Internal unique-claim recovery result (not exported).
#
# This file does not change RECOVERY_THRESHOLDS, public exports, or
# run_recovery_suite. Held-out, functional identifiability, and
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
