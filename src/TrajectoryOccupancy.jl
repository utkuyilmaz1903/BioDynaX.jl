###############################################################################
# Observed-trajectory occupancy (not exported).
#
# Occupancy: concatenate Experiment.observations in the original experiment
# and sample order. Duplicates are kept. This is an additional sampling
# context, not a functional-identifiability domain and not a composer replacement.
###############################################################################

"""Allowed occupancy provenances. Mixed train∪holdout construction is not supported."""
const TRAJECTORY_OCCUPANCY_PROVENANCES = (
    :train_observed_states,
    :holdout_observed_states)

"""
    TrajectoryOccupancy

Observed full-state occupancy columns. `X` is `nstates × n_points` from
`Experiment.observations`. `times` are provenance only and are not a
discovery / sample-index input. Not exported.
"""
struct TrajectoryOccupancy
    X::Matrix{Float64}
    experiment_index::Vector{Int}
    sample_index_in_exp::Vector{Int}
    times::Vector{Float64}
    provenance::Symbol
    split_indices::Any
    n_points::Int
    construction::Symbol
    function TrajectoryOccupancy(
            X::AbstractMatrix,
            experiment_index::AbstractVector,
            sample_index_in_exp::AbstractVector,
            times::AbstractVector,
            provenance::Symbol,
            split_indices,
            n_points::Integer,
            construction::Symbol)
        provenance in TRAJECTORY_OCCUPANCY_PROVENANCES || throw(ArgumentError(
            "provenance must be :train_observed_states or :holdout_observed_states"))
        construction === :train_obs_union_holdout_obs && throw(ArgumentError(
            "construction must correspond to occupancy provenance"))
        construction === provenance || throw(ArgumentError(
            "construction must correspond to provenance"))
        Xmat = Matrix{Float64}(X)
        exp_idx = collect(Int, experiment_index)
        samp_idx = collect(Int, sample_index_in_exp)
        tvec = collect(Float64, times)
        n = Int(n_points)
        n >= 1 || throw(ArgumentError("n_points must be positive"))
        size(Xmat, 1) >= 1 || throw(ArgumentError(
            "occupancy X must contain at least one state row"))
        size(Xmat, 2) == n || throw(DimensionMismatch(
            "X columns must equal n_points"))
        length(exp_idx) == n || throw(DimensionMismatch(
            "experiment_index length must equal n_points"))
        length(samp_idx) == n || throw(DimensionMismatch(
            "sample_index_in_exp length must equal n_points"))
        length(tvec) == n || throw(DimensionMismatch(
            "times length must equal n_points"))
        stored_indices = Tuple(Int(i) for i in split_indices)
        isempty(stored_indices) && throw(ArgumentError(
            "split_indices must be nonempty"))
        allowed = Set(stored_indices)
        all(i -> i in allowed, exp_idx) || throw(ArgumentError(
            "experiment_index must use values from split_indices"))
        return new(Xmat, exp_idx, samp_idx, tvec, provenance,
            stored_indices, n, construction)
    end
end

function _occupancy_column_index(experiments, split_indices)
    n_points = sum(size(exp.observations, 2) for exp in experiments)
    experiment_index = Vector{Int}(undef, n_points)
    sample_index_in_exp = Vector{Int}(undef, n_points)
    times = Vector{Float64}(undef, n_points)
    col = 1
    for (local_i, exp) in enumerate(experiments)
        n = size(exp.observations, 2)
        dest = col:(col + n - 1)
        experiment_index[dest] .= Int(split_indices[local_i])
        sample_index_in_exp[dest] = 1:n
        times[dest] = Float64.(exp.times)
        col += n
    end
    return experiment_index, sample_index_in_exp, times
end

function _leading_split_indices(indices, n::Integer)
    length(indices) >= n || throw(DimensionMismatch(
        "split_indices is shorter than the selected experiment count"))
    return ntuple(i -> Int(indices[i]), n)
end

function _validated_occupancy_indices(
        n_experiments::Integer, provenance::Symbol, split_indices)
    stored = Tuple(Int(i) for i in split_indices)
    length(stored) == n_experiments || throw(DimensionMismatch(
        "split_indices length must equal the number of selected experiments"))
    train_ids = REFERENCE_PROTOCOL_TRAIN_INDICES
    hold_ids = REFERENCE_PROTOCOL_HOLDOUT_INDICES
    protocol_n = length(train_ids) + length(hold_ids)
    n_experiments == protocol_n && throw(ArgumentError(
        "occupancy does not concatenate a full reference-protocol experiment collection; collect split.train and split.holdout separately"))
    has_train = any(in(train_ids), stored)
    has_hold = any(in(hold_ids), stored)
    has_train && has_hold &&
        throw(ArgumentError(
            "occupancy does not concatenate train and holdout; collect split.train and split.holdout separately"))
    if provenance === :train_observed_states
        all(in(train_ids), stored) || throw(ArgumentError(
            ":train_observed_states requires train split_indices"))
        if n_experiments == length(train_ids) && stored != train_ids
            throw(ArgumentError(
                ":train_observed_states with the locked train count requires indices (1, 2, 3, 4, 5, 6, 7)"))
        end
    elseif provenance === :holdout_observed_states
        all(in(hold_ids), stored) || throw(ArgumentError(
            ":holdout_observed_states requires holdout split_indices"))
        if n_experiments == length(hold_ids) && stored != hold_ids
            throw(ArgumentError(
                ":holdout_observed_states with the locked holdout count requires indices (8, 9)"))
        end
    else
        throw(ArgumentError(
            "provenance must be :train_observed_states or :holdout_observed_states"))
    end
    return stored
end

"""
    collect_observed_occupancy(experiments, provenance; split_indices)

Build occupancy from observed state columns of the selected experiments.
`X = hcat(experiment.observations ...)`. Original experiment order, sample
order, and duplicate columns are kept. A full reference-protocol collection is
rejected unless it arrives through `ExperimentSplit` as train or holdout.
"""
function collect_observed_occupancy(
        experiments::AbstractVector{<:Experiment},
        provenance::Symbol;
        split_indices = ntuple(identity, length(experiments)))
    isempty(experiments) && throw(ArgumentError(
        "occupancy requires at least one experiment"))
    stored_indices = _validated_occupancy_indices(
        length(experiments), provenance, split_indices)
    X = reduce(hcat, (Float64.(exp.observations) for exp in experiments))
    experiment_index, sample_index_in_exp, times = _occupancy_column_index(
        experiments, stored_indices)
    return TrajectoryOccupancy(
        X, experiment_index, sample_index_in_exp, times,
        provenance, stored_indices, size(X, 2), provenance)
end

"""
    collect_observed_occupancy(set, provenance; split_indices)

Occupancy from an already-selected `ExperimentSet`. The vector entry is
the single validated implementation; a full 9-IC set is rejected so
train and holdout occupancy stay separate.
"""
function collect_observed_occupancy(
        set::ExperimentSet,
        provenance::Symbol;
        split_indices = ntuple(identity, length(set.experiments)))
    return collect_observed_occupancy(set.experiments, provenance;
        split_indices = split_indices)
end

"""
    collect_observed_occupancy(split, provenance)

Reference-protocol occupancy. `:train_observed_states` uses `split.train` only.
`:holdout_observed_states` uses `split.holdout` only. The two are never
concatenated.
"""
function collect_observed_occupancy(split::ExperimentSplit, provenance::Symbol)
    if provenance === :train_observed_states
        set = split.train
        indices = _leading_split_indices(split.train_indices, length(set))
        return collect_observed_occupancy(set.experiments, provenance;
            split_indices = indices)
    elseif provenance === :holdout_observed_states
        set = split.holdout
        indices = _leading_split_indices(split.holdout_indices, length(set))
        return collect_observed_occupancy(set.experiments, provenance;
            split_indices = indices)
    end
    throw(ArgumentError(
        "provenance must be :train_observed_states or :holdout_observed_states"))
end

"""
    sample_destruction_occupancy(model, params, term, occupancy)

Primary occupancy sample path:

    sample_unknown_destruction(model, params, occupancy.X; term)

Occupancy `times` are not passed. Dummy-time / sample-index discovery
semantics stay unchanged.
"""
function sample_destruction_occupancy(
        model, params, term, occupancy::TrajectoryOccupancy)
    return sample_unknown_destruction(model, params, occupancy.X; term)
end
