###############################################################################
# Practical functional-identifiability primitives (not exported).
#
# Observed train-then-holdout regulator coordinates and pairwise
# destruction / trajectory metrics. Restart training and diagnostic
# assembly are not in this file.
###############################################################################

"""
    FunctionalIdentifiabilityDomain

Observed-coordinate Q4 domain: train regulator samples, then holdout
regulator samples, original experiment and sample order. Duplicates are
kept. Not a grid, not a restart product, and not a structural certificate.
"""
struct FunctionalIdentifiabilityDomain
    regulator_index::Int
    z::Vector{Float64}
    n_train_points::Int
    n_holdout_points::Int
    fill_value::Float64
    construction::Symbol
    function FunctionalIdentifiabilityDomain(
            regulator_index::Integer,
            z::AbstractVector,
            n_train_points::Integer,
            n_holdout_points::Integer,
            fill_value::Real,
            construction::Symbol)
        construction === :train_obs_union_holdout_obs || throw(ArgumentError(
            "construction must be :train_obs_union_holdout_obs"))
        z_vec = collect(Float64, z)
        n_train = Int(n_train_points)
        n_hold = Int(n_holdout_points)
        (n_train >= 0 && n_hold >= 0) || throw(ArgumentError(
            "point counts must be nonnegative"))
        length(z_vec) == n_train + n_hold || throw(DimensionMismatch(
            "z length must equal n_train_points + n_holdout_points"))
        return new(Int(regulator_index), z_vec, n_train, n_hold,
            Float64(fill_value), construction)
    end
end

function _observed_regulator_coordinates(set::ExperimentSet, regulator::Integer)
    idx = Int(regulator)
    return reduce(vcat, (Float64.(exp.observations[idx, :])
                         for exp in set.experiments))
end

"""
    functional_identifiability_domain(split, regulator; fill_value=0.3)

Concatenate observed regulator coordinates from `split.train`, then
`split.holdout`, preserving exact values, order, and duplicates.
Does not take destruction samples, fitted parameters, or restart results.
"""
function functional_identifiability_domain(
        split::ExperimentSplit, regulator::Integer;
        fill_value::Real = 0.3)
    r_train = _observed_regulator_coordinates(split.train, regulator)
    r_holdout = _observed_regulator_coordinates(split.holdout, regulator)
    z = vcat(r_train, r_holdout)
    return FunctionalIdentifiabilityDomain(
        regulator, z, length(r_train), length(r_holdout),
        fill_value, :train_obs_union_holdout_obs)
end

"""
    scale_align_destruction(D_i, D_j) -> (; alpha, D_j_aligned)

Least-squares scale of `D_j` onto `D_i`:

    alpha = dot(D_i, D_j) / dot(D_j, D_j)

when `dot(D_j, D_j) > 0`. A zero `D_j` yields `alpha === NaN` and a
NaN aligned vector. The pair is not dropped.
"""
function scale_align_destruction(D_i::AbstractVector, D_j::AbstractVector)
    vi = Float64.(D_i)
    vj = Float64.(D_j)
    length(vi) == length(vj) || throw(DimensionMismatch(
        "destruction samples must share a common domain length"))
    denom = dot(vj, vj)
    alpha = denom > 0 ? (dot(vi, vj) / denom) : NaN
    return (; alpha, D_j_aligned = alpha .* vj)
end

function _destruction_correlation(D_i::AbstractVector, D_j::AbstractVector)
    value = cor(D_i, D_j)
    return isnan(value) ? 0.0 : Float64(value)
end

"""
    pairwise_destruction_metrics(D_i, D_j)

Pairwise destruction metrics on a shared domain. Both vectors must already
be aligned to the same `z`. This helper does not construct a domain.
"""
function pairwise_destruction_metrics(D_i::AbstractVector, D_j::AbstractVector)
    aligned = scale_align_destruction(D_i, D_j)
    return (;
        d_rmse_raw = rate_rel_rmse(D_i, D_j),
        d_rmse_scale_normalized = rate_rel_rmse(D_i, aligned.D_j_aligned),
        d_correlation = _destruction_correlation(Float64.(D_i), Float64.(D_j)),
        scale_alpha = aligned.alpha)
end

function _mean_experiment_rate_rel_rmse(pred_i, pred_j)
    if pred_i isa AbstractVector && !(eltype(pred_i) <: Number)
        length(pred_i) == length(pred_j) || throw(DimensionMismatch(
            "trajectory collections must have the same experiment count"))
        isempty(pred_i) && throw(ArgumentError(
            "trajectory collections must be nonempty"))
        return mean(rate_rel_rmse(vec(left), vec(right))
                    for (left, right) in zip(pred_i, pred_j))
    end
    return rate_rel_rmse(vec(pred_i), vec(pred_j))
end

"""
    pairwise_trajectory_metrics(pred_i_train, pred_j_train,
                               pred_i_holdout, pred_j_holdout)

Per-experiment `rate_rel_rmse` of vectorized predicted states, then the
arithmetic mean on the train collection and on the holdout collection.
A vector of arrays is one prediction per experiment. A numeric array is
a single trajectory. Does not derive reporting flags or status.
"""
function pairwise_trajectory_metrics(
        pred_i_train, pred_j_train, pred_i_holdout, pred_j_holdout)
    return (;
        traj_rmse_train = _mean_experiment_rate_rel_rmse(
            pred_i_train, pred_j_train),
        traj_rmse_holdout = _mean_experiment_rate_rel_rmse(
            pred_i_holdout, pred_j_holdout))
end
