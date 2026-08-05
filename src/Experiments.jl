"""
    Experiment

One biological experiment with irregular sampling, missing-data mask and
arbitrary metadata. Observations are stored as `(states, times)`.
"""
struct Experiment{T<:AbstractFloat,M<:AbstractMatrix{T},V<:AbstractVector{T}}
    name::Symbol
    times::V
    observations::M
    mask::BitMatrix
    u0::V
    metadata::Dict{Symbol,Any}
end

"""Device-resident experiment view used by optional accelerator extensions."""
struct DeviceExperiment{TA,OA,MA,UA}
    name::Symbol
    times::TA
    observations::OA
    mask::MA
    u0::UA
    metadata::Dict{Symbol,Any}
end

function Experiment(name::Symbol, times::AbstractVector{T},
                    observations::AbstractMatrix{T}, u0::AbstractVector{T};
                    mask = isfinite.(observations),
                    metadata = Dict{Symbol,Any}()) where {T<:AbstractFloat}
    issorted(times) || throw(ArgumentError("experiment times must be sorted"))
    length(times) == size(observations, 2) ||
        throw(DimensionMismatch("observations must have one column per time"))
    length(u0) == size(observations, 1) ||
        throw(DimensionMismatch("u0 and observations state dimensions differ"))
    size(mask) == size(observations) ||
        throw(DimensionMismatch("mask and observations must have equal size"))
    all(diff(times) .> zero(T)) ||
        throw(ArgumentError("experiment times must be strictly increasing"))
    return Experiment{T,Matrix{T},Vector{T}}(
        name, collect(times), Matrix(observations), BitMatrix(mask),
        collect(u0), Dict{Symbol,Any}(metadata))
end

struct ExperimentSet{T<:AbstractFloat,E<:Experiment}
    experiments::Vector{E}
    state_names::Vector{Symbol}
    units::Vector{Symbol}
    metadata::Dict{Symbol,Any}
end

function ExperimentSet(experiments::AbstractVector{<:Experiment},
                       state_names::Vector{Symbol};
                       units = fill(:dimensionless, length(state_names)),
                       metadata = Dict{Symbol,Any}())
    isempty(experiments) &&
        throw(ArgumentError("ExperimentSet cannot be empty"))
    T = eltype(first(experiments).times)
    E = typeof(first(experiments))
    T <: AbstractFloat ||
        throw(ArgumentError("experiment time type must be floating point"))
    all(experiment -> experiment isa E, experiments) ||
        throw(ArgumentError("all experiments must share storage type"))
    all(eltype(experiment.times) == T for experiment in experiments) ||
        throw(ArgumentError("all experiments must share numeric type"))
    nstates = length(state_names)
    length(units) == nstates ||
        throw(DimensionMismatch("units and state_names must align"))
    all(size(experiment.observations, 1) == nstates
        for experiment in experiments) ||
        throw(DimensionMismatch("all experiments must share state dimension"))
    return ExperimentSet{T,E}(
        E[experiments...], state_names, collect(units),
        Dict{Symbol,Any}(metadata))
end

Base.length(set::ExperimentSet) = length(set.experiments)
Base.iterate(set::ExperimentSet, state...) = iterate(set.experiments, state...)
Base.getindex(set::ExperimentSet, i::Integer) = set.experiments[i]

function experiment_fingerprint(set::ExperimentSet)
    payload = Any[set.state_names, set.units]
    for exp in set
        append!(payload, (exp.times, exp.observations, exp.mask, exp.u0))
    end
    return data_fingerprint(payload...)
end

function as_experiment_set(data, times, u0;
                           name::Symbol = :experiment,
                           state_names = [Symbol("x$i") for i in axes(data, 1)])
    exp = Experiment(name, times, data, u0)
    return ExperimentSet([exp], collect(state_names))
end

function experiment_batches(set::ExperimentSet, batch_size::Integer;
                            shuffle::Bool = false,
                            rng::AbstractRNG = Random.default_rng())
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    indices = collect(eachindex(set.experiments))
    shuffle && Random.shuffle!(rng, indices)
    return [set.experiments[indices[first:last]]
            for first in 1:batch_size:length(indices)
            for last in (min(first + batch_size - 1, length(indices)),)]
end
