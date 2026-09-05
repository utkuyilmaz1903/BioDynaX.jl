"""
    TrainingRetcode

SciML-style training outcome codes returned in `TrainingResult.retcode`.
"""
@enum TrainingRetcode begin
    Success
    NotConverged
    BFGSFailure
    GradientFailure
    ODEFailure
end

Base.convert(::Type{Symbol}, code::TrainingRetcode) = Symbol(code)

"""
    TrainingDiagnostics

Structured convergence diagnostics attached to `TrainingResult`.
"""
Base.@kwdef mutable struct TrainingDiagnostics
    mse::Float64 = NaN
    constraint::Float64 = NaN
    primal_residual::Float64 = NaN
    final_gradient_norm::Float64 = NaN
    gradient_norm_history::Vector{Float64} = Float64[]
    bfgs_attempted::Bool = false
    bfgs_success::Bool = false
    bfgs_retcode::Symbol = :none
    bfgs_message::String = ""
    experiment_count::Int = 0
    dual::Vector{Float64} = Float64[]
    ρ::Float64 = NaN
end

"""
    ParameterUncertainty

Asymptotic or bootstrap parameter uncertainty for physical kinetic constants.
"""
struct ParameterUncertainty
    parameter_names::Vector{Symbol}
    estimates::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    level::Float64
    method::Symbol
end

"""
    DiscoveryUncertaintyReport

Public summary of graph-local discovery stability for one target state.
"""
struct DiscoveryUncertaintyReport
    target::Int
    support_frequency::Vector{Float64}
    validation_error::Float64
    stable_terms::Vector{String}
    coefficient_intervals::Vector{Tuple{String, Float64, Float64}}
end

"""
    uncertainty_reports(result::DiscoveryResult)

Extract typed uncertainty summaries from implicit discovery candidates.
"""
function uncertainty_reports(result::DiscoveryResult)
    result.success || return DiscoveryUncertaintyReport[]
    if !isempty(result.candidates) &&
       first(result.candidates) isa ExplicitCandidate
        return map(result.candidates) do candidate
            stable = String[]
            for (coefficient, term) in zip(
                candidate.coefficients, candidate.specification.numerator)
                abs(coefficient) > 0 && push!(stable, term.label)
            end
            DiscoveryUncertaintyReport(
                candidate.target, Float64[], candidate.validation_error,
                stable, Tuple{String, Float64, Float64}[])
        end
    end
    reports = DiscoveryUncertaintyReport[]
    for candidate in result.candidates
        candidate isa ImplicitCandidate || continue
        spec = candidate.specification
        terms = vcat(spec.numerator, spec.denominator)
        coefficients = vcat(
            candidate.numerator_coefficients,
            candidate.denominator_coefficients)
        stable = String[]
        intervals = Tuple{String, Float64, Float64}[]
        for (coefficient, term, frequency) in zip(
            coefficients, terms, candidate.selection_frequency)
            frequency ≥ 0.5 || continue
            push!(stable, term.label)
            push!(intervals, (term.label, coefficient, coefficient))
        end
        push!(reports,
            DiscoveryUncertaintyReport(
                candidate.target, candidate.selection_frequency,
                candidate.validation_error, stable, intervals))
    end
    return reports
end
