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

"""
    ExperimentSet

A named collection of `Experiment`s that share state dimension and labels.
This is the multi-IC training input for `train_experiments`.
"""
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
    stored = Vector{E}(undef, length(experiments))
    for (index, experiment) in pairs(experiments)
        stored[index] = experiment
    end
    return ExperimentSet{T,E}(
        stored, copy(state_names), collect(Symbol, units),
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

"""Relative experiment weight from metadata (`:weight`, default `1.0`)."""
experiment_weight(experiment::Experiment) =
    Float64(get(experiment.metadata, :weight, 1.0))

"""Observation noise scale for heteroskedastic weighting (`:noise_σ`, default `1.0`)."""
experiment_noise_scale(experiment::Experiment) =
    Float64(get(experiment.metadata, :noise_σ, 1.0))

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

"""
    experiment_from_csv(path; time_column=1, u0=nothing, name=:experiment,
                        state_names=nothing, delim=',')

Load a time-series table into an `Experiment`. The first numeric column is time
unless `time_column` is set; remaining columns are states (rows of `observations`).
"""
function experiment_from_csv(path::AbstractString;
                             time_column::Int = 1,
                             u0 = nothing,
                             name::Symbol = :experiment,
                             state_names = nothing,
                             delim::Char = ',',
                             header::Bool = true)
    raw = readdlm(path, delim, Float64; header = header)
    if header
        body, header_row = raw
        names = vec(string.(header_row))
    else
        body = raw
        names = ["t"; ["x$i" for i in 1:(size(body, 2) - 1)]]
    end
    ndims = size(body, 2)
    1 ≤ time_column ≤ ndims ||
        throw(ArgumentError("time_column $time_column is out of range"))
    times = vec(body[:, time_column])
    state_cols = [i for i in 1:ndims if i != time_column]
    observations = permutedims(body[:, state_cols])
    initial = u0 === nothing ? observations[:, 1] : Float64.(u0)
    labels = if state_names === nothing
        header ?
            [Symbol(replace(names[i], r"[^A-Za-z0-9_]" => "_")) for i in state_cols] :
            [Symbol("x$i") for i in eachindex(state_cols)]
    else
        collect(Symbol, state_names)
    end
    length(labels) == size(observations, 1) ||
        throw(DimensionMismatch("state_names must match observation rows"))
    experiment = Experiment(name, times, observations, initial)
    return experiment, labels
end

"""
    write_experiment_csv(path, experiment; state_names, delim=',')

Write `times` plus one row per state of an `Experiment` to CSV.
"""
function write_experiment_csv(path::AbstractString, experiment::Experiment;
                              state_names = [Symbol("x$i")
                                             for i in axes(experiment.observations, 1)],
                              delim::Char = ',')
    length(state_names) == size(experiment.observations, 1) ||
        throw(DimensionMismatch("state_names must match observation rows"))
    header = ["t"; string.(state_names)]
    body = hcat(experiment.times, permutedims(experiment.observations))
    open(path, "w") do io
        println(io, join(header, delim))
        for row in eachrow(body)
            println(io, join(row, delim))
        end
    end
    return path
end

"""Return a copy of `experiment` with a replacement observation mask."""
function mask_observations(experiment::Experiment, mask::AbstractMatrix)
    return Experiment(
        experiment.name, experiment.times, experiment.observations,
        experiment.u0; mask = BitMatrix(mask),
        metadata = copy(experiment.metadata))
end

"""Zero a fraction of one state's observation mask (column 1 is kept)."""
function subsample_state_mask(experiment::Experiment, state::Int,
                              keep_fraction::Real, rng::AbstractRNG)
    mask = copy(experiment.mask)
    n = size(mask, 2)
    nkeep = max(2, round(Int, keep_fraction * n))
    idx = randperm(rng, n)
    mask[state, :] .= false
    mask[state, idx[1:nkeep]] .= true
    mask[state, 1] = true
    return mask_observations(experiment, mask)
end
