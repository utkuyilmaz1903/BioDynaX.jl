"""
    IdentifiabilityReport

**Practical** Fisher-information summary for compiled physical parameters at a
fitted trajectory. This is not structural identifiability
(see StructuralIdentifiability.jl). Rank and credible intervals are local,
finite-difference Gauss–Newton estimates.
"""
struct IdentifiabilityReport
    parameter_names::Vector{Symbol}
    fisher_information::Matrix{Float64}
    condition_number::Float64
    identifiable::BitVector
    correlation_matrix::Matrix{Float64}
    residual_variance::Float64
end

function _perturb_phys_parameter(p, name::Symbol, delta::Real)
    raw = getproperty(p.phys, name)
    pairs = [(sym, sym == name ? raw + delta : getproperty(p.phys, sym))
             for sym in propertynames(p.phys)]
    return ComponentVector(phys = NamedTuple(pairs), nn = p.nn)
end

"""
    trajectory_jacobian(model, p, u0, tspan, times; rel_step=1e-4)

Finite-difference Jacobian of the predicted trajectory with respect to
physical kinetic parameters.
"""
function trajectory_jacobian(model::UDEModel, p, u0, tspan, times;
                             rel_step::Real = 1e-4,
                             solver_config::SolverConfig = SolverConfig())
    schema = parameter_schema(model)
    names = schema.phys_names
    base = predict_ude(
        p, u0, tspan, times, model; solver_config = solver_config)
    n_obs = length(times) * size(base, 1)
    jacobian = Matrix{Float64}(undef, n_obs, length(names))
    for (col, name) in pairs(names)
        raw = getproperty(p.phys, name)
        delta = max(abs(raw), oneunit(raw)) * rel_step
        plus = _perturb_phys_parameter(p, name, delta)
        minus = _perturb_phys_parameter(p, name, -delta)
        forward = predict_ude(
            plus, u0, tspan, times, model; solver_config = solver_config)
        backward = predict_ude(
            minus, u0, tspan, times, model; solver_config = solver_config)
        jacobian[:, col] = vec((forward .- backward) ./ (2delta))
    end
    return jacobian, names
end

"""
    fisher_information_matrix(model, p, data, t_data, u0, tspan; kwargs...)

Gauss–Newton Fisher information matrix `J'J / σ²` for physical parameters.
"""
function fisher_information_matrix(model::UDEModel, p, data, t_data, u0, tspan;
                                   mask = trues(size(data)),
                                   residual_variance = nothing,
                                   kwargs...)
    jacobian, names = trajectory_jacobian(
        model, p, u0, tspan, t_data; kwargs...)
    residual = ifelse.(mask, data .- predict_ude(
        p, u0, tspan, t_data, model; kwargs...), zero(eltype(data)))
    σ² = residual_variance === nothing ?
        max(eps(), sum(abs2, residual) / max(1, count(mask))) :
        float(residual_variance)
    information = (jacobian' * jacobian) ./ σ²
    return information, names, σ²
end

"""
    assess_identifiability(model, p, data, t_data, u0, tspan; threshold=1e-8)

Rank-based **practical** identifiability from the Fisher information at this
fit. Neural parameters are excluded. Not a substitute for structural analysis.
"""
function assess_identifiability(model::UDEModel, p, data, t_data, u0, tspan;
                                threshold::Real = 1e-8, kwargs...)
    information, names, σ² = fisher_information_matrix(
        model, p, data, t_data, u0, tspan; kwargs...)
    eigenvalues = eigvals(Symmetric(information))
    positive = eigenvalues[eigenvalues .> threshold * maximum(eigenvalues)]
    condition = isempty(positive) ? Inf :
        maximum(positive) / max(minimum(positive), threshold)
    identifiable = begin
        covariance = pinv(information)
        variances = diag(covariance)
        .!isinf.(variances) .& (variances .< 1e8)
    end
    correlation = _correlation_from_information(information)
    return IdentifiabilityReport(
        names, information, condition, BitVector(identifiable),
        correlation, σ²)
end

function _correlation_from_information(information::AbstractMatrix)
    n = size(information, 1)
    correlation = Matrix{Float64}(I, n, n)
    for i in 1:n, j in (i + 1):n
        denom = sqrt(information[i, i] * information[j, j])
        value = denom == 0 ? zero(denom) : information[i, j] / denom
        correlation[i, j] = value
        correlation[j, i] = value
    end
    return correlation
end

function _z_score(level::Real)
    level ≈ 0.90 && return 1.6448536269514722
    level ≈ 0.95 && return 1.959963984540054
    level ≈ 0.99 && return 2.5758293035489004
    throw(ArgumentError(
        "unsupported credible level $level; use 0.90, 0.95, or 0.99"))
end

"""
    parameter_credible_intervals(report, level=0.95)

Asymptotic normal credible intervals from the inverse Fisher information.
"""
function parameter_credible_intervals(report::IdentifiabilityReport,
                                      estimate;
                                      level::Real = 0.95)
    z = _z_score(level)
    information = report.fisher_information
    variances = diag(pinv(information))
    intervals = Dict{Symbol,Tuple{Float64,Float64}}()
    for (name, variance) in zip(report.parameter_names, variances)
        center = positive_parameter(getproperty(estimate.phys, name))
        half = z * sqrt(max(variance, zero(variance)))
        intervals[name] = (max(0.0, center - half), center + half)
    end
    return intervals
end

"""
    estimate_parameter_uncertainty(model, params, data, t_data, u0, tspan; level=0.95)

Asymptotic uncertainty for physical parameters from the Fisher information matrix.
"""
function estimate_parameter_uncertainty(model::UDEModel, params, data, t_data,
                                        u0, tspan; level::Real = 0.95, kwargs...)
    report = assess_identifiability(
        model, params, data, t_data, u0, tspan; kwargs...)
    intervals = parameter_credible_intervals(report, params; level = level)
    names = report.parameter_names
    estimates = Float64[positive_parameter(getproperty(params.phys, name))
                        for name in names]
    lower = Float64[intervals[name][1] for name in names]
    upper = Float64[intervals[name][2] for name in names]
    return ParameterUncertainty(names, estimates, lower, upper, float(level), :fisher)
end
