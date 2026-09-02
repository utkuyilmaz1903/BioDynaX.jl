###############################################################################
# Practical functional-identifiability primitives (not exported).
#
# M3-A: observed train-then-holdout regulator coordinates and pairwise
# destruction / trajectory metrics.
# M3-B: five independent restart fits on split.train.
# M3-C: live pairwise assembly and derived diagnostic flags / status.
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

# -- M3-B independent restart training ----------------------------------------

"""Locked functional-identifiability restart seeds. Not M4 robustness seeds."""
const FUNCTIONAL_ID_RESTART_SEEDS = (201, 202, 203, 204, 205)

"""A-priori M3 restart training configuration. Not selected from holdout."""
const FUNCTIONAL_ID_TRAINING_CONFIG = (
    adam = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
    bfgs = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
    frozen_phys = Symbol[],
    phys_init = nothing)

"""Reporting cutoffs for the practical Q4 diagnostic. Not a success gate."""
const FUNCTIONAL_ID_REPORTING_CUTOFFS = (
    min_successful_restarts = 3,
    n_attempted_restarts = 5,
    traj_agree_rel_rmse = 0.05,
    d_disagree_scale_norm_rel_rmse = 0.20)

const FUNCTIONAL_ID_STATUS_VOCABULARY = (
    :incomplete,
    :traj_disagree,
    :scale_ambiguity,
    :function_agree,
    :trajectory_agree_function_disagree)

const FUNCTIONAL_ID_FAILURE_REASONS = (
    :none, :fit_threw, :nonfinite_D, :nonfinite_trajectory, :predict_threw)

"""
    FunctionalIdentifiabilityRestart

One attempted restart. `included` is an M3-B-local validity flag, not a
holdout score and not a best-of-N choice.
"""
struct FunctionalIdentifiabilityRestart
    seed::Int
    included::Bool
    training_retcode::Union{TrainingRetcode,Nothing}
    failure_reason::Symbol
    message::String
    nn_init_fingerprint::UInt64
    nn_final_fingerprint::UInt64
    function FunctionalIdentifiabilityRestart(
            seed::Integer,
            included::Bool,
            training_retcode::Union{TrainingRetcode,Nothing},
            failure_reason::Symbol,
            message::AbstractString,
            nn_init_fingerprint::UInt64,
            nn_final_fingerprint::UInt64)
        failure_reason in FUNCTIONAL_ID_FAILURE_REASONS || throw(ArgumentError(
            "unsupported functional-identifiability failure_reason"))
        if included
            failure_reason === :none || throw(ArgumentError(
                "included restart must use failure_reason === :none"))
        else
            failure_reason === :none && throw(ArgumentError(
                "excluded restart must record a failure_reason"))
            isempty(message) && throw(ArgumentError(
                "excluded restart must record a nonempty message"))
        end
        return new(Int(seed), included, training_retcode, failure_reason,
            String(message), nn_init_fingerprint, nn_final_fingerprint)
    end
end

"""
    nn_parameter_fingerprint(nn_params) -> UInt64

Numerical fingerprint of NN parameter values. Not `objectid`, not a seed
label, and identical for `deepcopy` of the same numbers.
"""
function nn_parameter_fingerprint(nn_params)::UInt64
    return hash(_nn_fingerprint_values(nn_params))
end

function _nn_fingerprint_values(nn_params)
    if nn_params isa AbstractArray && eltype(nn_params) <: Number
        return collect(Float64, vec(nn_params))
    end
    buf = Float64[]
    _append_nn_fingerprint_values!(buf, nn_params)
    return buf
end

function _append_nn_fingerprint_values!(buf, x)
    if x isa Number
        push!(buf, Float64(x))
    elseif x isa AbstractArray && eltype(x) <: Number
        append!(buf, Float64.(vec(x)))
    elseif x isa AbstractArray || x isa Tuple || x isa NamedTuple
        for child in x
            _append_nn_fingerprint_values!(buf, child)
        end
    elseif x isa AbstractDict
        for key in sort!(collect(keys(x)); by = string)
            _append_nn_fingerprint_values!(buf, x[key])
        end
    end
    return buf
end

function _params_nn_fingerprint(params)
    hasproperty(params, :nn) || return UInt64(0)
    return nn_parameter_fingerprint(getproperty(params, :nn))
end

function _require_functional_id_restart_seeds(restart_seeds)
    seeds = Tuple(restart_seeds)
    103 in seeds && throw(ArgumentError(
        "seed 103 is not a functional-identifiability restart seed"))
    length(seeds) == 5 || throw(ArgumentError(
        "functional-identifiability restarts require exactly 5 seeds; got $(length(seeds))"))
    length(unique(seeds)) == 5 || throw(ArgumentError(
        "functional-identifiability restart seeds must be unique"))
    seeds === FUNCTIONAL_ID_RESTART_SEEDS || throw(ArgumentError(
        "functional-identifiability restart seeds must be $(FUNCTIONAL_ID_RESTART_SEEDS)"))
    return FUNCTIONAL_ID_RESTART_SEEDS
end

function _functional_id_regulator(ude_net::BiologicalNetwork)
    terms = neural_destruction_terms(ude_net)
    length(terms) == 1 || throw(ArgumentError(
        "functional-identifiability restarts require exactly one unknown destruction"))
    return only(terms).regulator
end

function _finite_numeric_tree(x)
    if x isa Number
        return isfinite(Float64(x))
    elseif x isa AbstractArray && eltype(x) <: Number
        return all(isfinite, x)
    elseif x isa AbstractArray || x isa Tuple || x isa NamedTuple
        return all(_finite_numeric_tree, x)
    elseif hasproperty(x, :nn) && hasproperty(x, :phys)
        return _finite_numeric_tree(getproperty(x, :nn)) &&
               _finite_numeric_tree(getproperty(x, :phys))
    end
    return true
end

function _finite_training_output(fit::TrainingResult)
    isfinite(Float64(fit.initial_loss)) || return false
    isfinite(Float64(fit.final_loss)) || return false
    return _finite_numeric_tree(fit.params)
end

function _finite_predictions(preds)
    preds isa AbstractVector || return false
    isempty(preds) && return false
    return all(pred -> pred isa AbstractArray && all(isfinite, pred), preds)
end

function _predict_one_experiment(model, params, exp::Experiment)
    tspan = (Float64(first(exp.times)), Float64(last(exp.times)))
    return predict_ude(params, Float64.(exp.u0), tspan, Float64.(exp.times),
        model)
end

function _predict_experiment_set(model, params, set::ExperimentSet)
    return [_predict_one_experiment(model, params, exp)
            for exp in set.experiments]
end

function _nonempty_failure_message(err)
    message = sprint(showerror, err)
    return isempty(message) ? "restart failed" : message
end

function _empty_restart_raw(seed::Integer)
    return (;
        seed = Int(seed),
        model = nothing,
        params = nothing,
        fit = nothing,
        D = nothing,
        pred_train = nothing,
        pred_holdout = nothing,
        attempt_count = 1)
end

function _restart_raw(; seed, model, params, fit, D, pred_train, pred_holdout)
    return (;
        seed = Int(seed),
        model,
        params,
        fit,
        D,
        pred_train,
        pred_holdout,
        attempt_count = 1)
end

function _isolated_restart_failure(
        seed, reason, err, init_fp, final_fp, retcode, raw)
    return (
        FunctionalIdentifiabilityRestart(
            seed, false, retcode, reason, _nonempty_failure_message(err),
            init_fp, final_fp),
        merge(raw, (; seed = Int(seed), attempt_count = 1)))
end

"""
    fit_functional_identifiability_restart(split, ude_net, seed, domain; ...)

One restart: `MersenneTwister(seed)` → `build_ude_model` → one
`fit_unknown_destruction(..., split.train)`. No retry. Called only by
`train_functional_identifiability_restarts`.
"""
function fit_functional_identifiability_restart(
        split::ExperimentSplit,
        ude_net::BiologicalNetwork,
        seed::Integer,
        domain::FunctionalIdentifiabilityDomain;
        adam = FUNCTIONAL_ID_TRAINING_CONFIG.adam,
        bfgs = FUNCTIONAL_ID_TRAINING_CONFIG.bfgs,
        frozen_phys = FUNCTIONAL_ID_TRAINING_CONFIG.frozen_phys,
        phys_init = FUNCTIONAL_ID_TRAINING_CONFIG.phys_init)
    init_fp = UInt64(0)
    final_fp = UInt64(0)
    retcode = nothing
    raw = _empty_restart_raw(seed)
    model = nothing
    p0 = nothing
    try
        model, p0 = build_ude_model(MersenneTwister(Int(seed)), ude_net)
        init_fp = nn_parameter_fingerprint(p0.nn)
    catch err
        return _isolated_restart_failure(
            seed, :fit_threw, err, init_fp, final_fp, retcode, raw)
    end
    fit = try
        fit_unknown_destruction(model, p0, split.train;
            adam = adam, bfgs = bfgs,
            frozen_phys = frozen_phys, phys_init = phys_init)
    catch err
        return _isolated_restart_failure(
            seed, :fit_threw, err, init_fp, final_fp, retcode, raw)
    end
    params = fit.params
    retcode = fit.retcode
    final_fp = _params_nn_fingerprint(params)
    raw = _restart_raw(;
        seed, model, params, fit, D = nothing,
        pred_train = nothing, pred_holdout = nothing)
    if !_finite_training_output(fit)
        return _isolated_restart_failure(
            seed, :nonfinite_trajectory,
            ErrorException("training output was non-finite"),
            init_fp, final_fp, retcode, raw)
    end
    D = try
        term = only_unknown_destruction(model)
        _, D_matrix, _ = sample_unknown_destruction_grid(
            model, params, term;
            r_range = domain.z, fill_value = domain.fill_value)
        vec(D_matrix)
    catch err
        return _isolated_restart_failure(
            seed, :nonfinite_D, err, init_fp, final_fp, retcode, raw)
    end
    raw = merge(raw, (; D))
    if !(length(D) == length(domain.z) && all(isfinite, D))
        return _isolated_restart_failure(
            seed, :nonfinite_D,
            ErrorException("destruction samples were non-finite or mis-sized"),
            init_fp, final_fp, retcode, raw)
    end
    pred_train = nothing
    pred_holdout = nothing
    try
        pred_train = _predict_experiment_set(model, params, split.train)
        pred_holdout = _predict_experiment_set(model, params, split.holdout)
    catch err
        return _isolated_restart_failure(
            seed, :predict_threw, err, init_fp, final_fp, retcode,
            merge(raw, (; pred_train, pred_holdout)))
    end
    raw = merge(raw, (; pred_train, pred_holdout))
    if !(_finite_predictions(pred_train) && _finite_predictions(pred_holdout))
        return _isolated_restart_failure(
            seed, :nonfinite_trajectory,
            ErrorException("predicted trajectories were non-finite"),
            init_fp, final_fp, retcode, raw)
    end
    # NotConverged is not an automatic exclusion when outputs are finite.
    record = FunctionalIdentifiabilityRestart(
        seed, true, retcode, :none, "", init_fp, final_fp)
    return record, raw
end

"""
    train_functional_identifiability_restarts(split, ude_net; ...)

M3-B owner of restart fits. Attempts each locked seed exactly once,
independently, on `split.train`. Failures are isolated. There is no retry
and no best-of-N selection.
"""
function train_functional_identifiability_restarts(
        split::ExperimentSplit,
        ude_net::BiologicalNetwork;
        restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS,
        adam = FUNCTIONAL_ID_TRAINING_CONFIG.adam,
        bfgs = FUNCTIONAL_ID_TRAINING_CONFIG.bfgs,
        frozen_phys = FUNCTIONAL_ID_TRAINING_CONFIG.frozen_phys,
        phys_init = FUNCTIONAL_ID_TRAINING_CONFIG.phys_init,
        fill_value::Real = 0.3)
    seeds = _require_functional_id_restart_seeds(restart_seeds)
    regulator = _functional_id_regulator(ude_net)
    domain = functional_identifiability_domain(
        split, regulator; fill_value = fill_value)
    restarts = FunctionalIdentifiabilityRestart[]
    raw = NamedTuple[]
    sizehint!(restarts, length(seeds))
    sizehint!(raw, length(seeds))
    for seed in seeds
        record, payload = try
            fit_functional_identifiability_restart(
                split, ude_net, seed, domain;
                adam = adam, bfgs = bfgs,
                frozen_phys = frozen_phys, phys_init = phys_init)
        catch err
            _isolated_restart_failure(
                seed, :fit_threw, err, UInt64(0), UInt64(0), nothing,
                _empty_restart_raw(seed))
        end
        push!(restarts, record)
        push!(raw, payload)
    end
    n_attempted = length(seeds)
    n_successful = count(restart -> restart.included, restarts)
    n_failed = n_attempted - n_successful
    return (;
        restart_seeds = seeds,
        n_attempted,
        n_successful,
        n_failed,
        domain,
        restarts,
        raw)
end

# -- M3-C diagnostic assembly -------------------------------------------------

"""
    FunctionalIdentifiabilityPair

One unordered successful-restart pair (`seed_i < seed_j`) on the common
domain. Metrics are stored; reporting flags are not.
"""
struct FunctionalIdentifiabilityPair
    seed_i::Int
    seed_j::Int
    d_rmse_raw::Float64
    d_rmse_scale_normalized::Float64
    d_correlation::Float64
    scale_alpha::Float64
    traj_rmse_train::Float64
    traj_rmse_holdout::Float64
    function FunctionalIdentifiabilityPair(
            seed_i::Integer,
            seed_j::Integer,
            d_rmse_raw::Real,
            d_rmse_scale_normalized::Real,
            d_correlation::Real,
            scale_alpha::Real,
            traj_rmse_train::Real,
            traj_rmse_holdout::Real)
        Int(seed_i) < Int(seed_j) || throw(ArgumentError(
            "functional-identifiability pairs require seed_i < seed_j"))
        return new(Int(seed_i), Int(seed_j),
            Float64(d_rmse_raw), Float64(d_rmse_scale_normalized),
            Float64(d_correlation), Float64(scale_alpha),
            Float64(traj_rmse_train), Float64(traj_rmse_holdout))
    end
end

function _pair_median(pairs::Vector{FunctionalIdentifiabilityPair}, field::Symbol)
    isempty(pairs) && return NaN
    return Float64(median(getfield.(pairs, field)))
end

function _require_included_restart_raw(raw, domain::FunctionalIdentifiabilityDomain)
    raw.D === nothing && throw(ArgumentError(
        "included restart is missing destruction samples"))
    length(raw.D) == length(domain.z) || throw(DimensionMismatch(
        "destruction samples must match the common functional-identifiability domain"))
    raw.pred_train === nothing && throw(ArgumentError(
        "included restart is missing train trajectories"))
    raw.pred_holdout === nothing && throw(ArgumentError(
        "included restart is missing holdout trajectories"))
    return nothing
end

function _validate_functional_id_pairs(
        pairs::Vector{FunctionalIdentifiabilityPair},
        restarts::Vector{FunctionalIdentifiabilityRestart})
    included_seeds = Set(restart.seed for restart in restarts if restart.included)
    expected = binomial(length(included_seeds), 2)
    length(pairs) == expected || throw(ArgumentError(
        "pair count must be binomial(n_successful, 2) = $expected; got $(length(pairs))"))
    seen = Set{Tuple{Int,Int}}()
    for pair in pairs
        pair.seed_i < pair.seed_j || throw(ArgumentError(
            "functional-identifiability pairs require seed_i < seed_j"))
        (pair.seed_i in included_seeds && pair.seed_j in included_seeds) ||
            throw(ArgumentError(
                "pair ($(pair.seed_i), $(pair.seed_j)) is not an included restart pair"))
        key = (pair.seed_i, pair.seed_j)
        key in seen && throw(ArgumentError("duplicate functional-identifiability pair"))
        push!(seen, key)
    end
    return nothing
end

function _derive_functional_identifiability_fields(
        family::Symbol,
        restart_seeds,
        domain::FunctionalIdentifiabilityDomain,
        restarts::Vector{FunctionalIdentifiabilityRestart},
        pairs::Vector{FunctionalIdentifiabilityPair})
    seeds = _require_functional_id_restart_seeds(restart_seeds)
    length(restarts) == length(seeds) || throw(ArgumentError(
        "restart records must cover the five locked seeds"))
    for (k, seed) in enumerate(seeds)
        restarts[k].seed == seed || throw(ArgumentError(
            "restarts[$k].seed must be $seed"))
    end
    _validate_functional_id_pairs(pairs, restarts)
    n_attempted = length(seeds)
    n_successful = count(restart -> restart.included, restarts)
    n_failed = n_attempted - n_successful
    complete = n_attempted == FUNCTIONAL_ID_REPORTING_CUTOFFS.n_attempted_restarts &&
               n_successful >= FUNCTIONAL_ID_REPORTING_CUTOFFS.min_successful_restarts
    median_d_rmse_raw = _pair_median(pairs, :d_rmse_raw)
    median_d_rmse_scale_normalized = _pair_median(pairs, :d_rmse_scale_normalized)
    median_d_correlation = _pair_median(pairs, :d_correlation)
    median_traj_rmse_train = _pair_median(pairs, :traj_rmse_train)
    median_traj_rmse_holdout = _pair_median(pairs, :traj_rmse_holdout)
    traj_cut = FUNCTIONAL_ID_REPORTING_CUTOFFS.traj_agree_rel_rmse
    d_cut = FUNCTIONAL_ID_REPORTING_CUTOFFS.d_disagree_scale_norm_rel_rmse
    trajectory_agree = complete &&
                       median_traj_rmse_train <= traj_cut &&
                       median_traj_rmse_holdout <= traj_cut
    function_disagree = complete &&
                        n_successful >= 2 &&
                        median_d_rmse_scale_normalized >= d_cut
    trajectory_agree_function_disagree = trajectory_agree && function_disagree
    status = if !complete
        :incomplete
    elseif !trajectory_agree
        :traj_disagree
    elseif function_disagree
        :trajectory_agree_function_disagree
    elseif median_d_rmse_raw >= d_cut
        :scale_ambiguity
    else
        :function_agree
    end
    status in FUNCTIONAL_ID_STATUS_VOCABULARY || throw(ArgumentError(
        "unsupported functional-identifiability status"))
    return (;
        family,
        restart_seeds = seeds,
        n_attempted,
        n_successful,
        n_failed,
        complete,
        domain,
        restarts,
        pairs,
        median_d_rmse_raw,
        median_d_rmse_scale_normalized,
        median_d_correlation,
        median_traj_rmse_train,
        median_traj_rmse_holdout,
        trajectory_agree,
        function_disagree,
        trajectory_agree_function_disagree,
        status,
        practical_not_structural = true)
end

function _reject_overridden_derived_fields(derived; overrides...)
    for (name, given) in pairs(overrides)
        given === missing && continue
        actual = getfield(derived, name)
        given == actual || throw(ArgumentError(
            "$name is derived from restart/pair data and cannot be overridden"))
    end
    return nothing
end

"""
    FunctionalIdentifiabilityDiagnostic

Practical Q4 diagnostic assembled from five locked restarts. Flags and
`status` are derived; they are not a structural certificate or a gate.
"""
struct FunctionalIdentifiabilityDiagnostic
    family::Symbol
    restart_seeds::NTuple{5,Int}
    n_attempted::Int
    n_successful::Int
    n_failed::Int
    complete::Bool
    domain::FunctionalIdentifiabilityDomain
    restarts::Vector{FunctionalIdentifiabilityRestart}
    pairs::Vector{FunctionalIdentifiabilityPair}
    median_d_rmse_raw::Float64
    median_d_rmse_scale_normalized::Float64
    median_d_correlation::Float64
    median_traj_rmse_train::Float64
    median_traj_rmse_holdout::Float64
    trajectory_agree::Bool
    function_disagree::Bool
    trajectory_agree_function_disagree::Bool
    status::Symbol
    practical_not_structural::Bool
    function FunctionalIdentifiabilityDiagnostic(
            family::Symbol,
            restart_seeds,
            domain::FunctionalIdentifiabilityDomain,
            restarts::Vector{FunctionalIdentifiabilityRestart},
            pairs::Vector{FunctionalIdentifiabilityPair};
            function_disagree = missing,
            trajectory_agree = missing,
            trajectory_agree_function_disagree = missing,
            status = missing,
            complete = missing,
            practical_not_structural = missing)
        derived = _derive_functional_identifiability_fields(
            family, restart_seeds, domain, restarts, pairs)
        _reject_overridden_derived_fields(derived;
            function_disagree = function_disagree,
            trajectory_agree = trajectory_agree,
            trajectory_agree_function_disagree = trajectory_agree_function_disagree,
            status = status,
            complete = complete,
            practical_not_structural = practical_not_structural)
        return new(
            derived.family,
            derived.restart_seeds,
            derived.n_attempted,
            derived.n_successful,
            derived.n_failed,
            derived.complete,
            derived.domain,
            derived.restarts,
            derived.pairs,
            derived.median_d_rmse_raw,
            derived.median_d_rmse_scale_normalized,
            derived.median_d_correlation,
            derived.median_traj_rmse_train,
            derived.median_traj_rmse_holdout,
            derived.trajectory_agree,
            derived.function_disagree,
            derived.trajectory_agree_function_disagree,
            derived.status,
            derived.practical_not_structural)
    end
end

function _functional_identifiability_pairs(
        restarts::Vector{FunctionalIdentifiabilityRestart},
        raw,
        domain::FunctionalIdentifiabilityDomain)
    length(restarts) == length(raw) || throw(DimensionMismatch(
        "restart records and raw payloads must align"))
    included = Int[]
    for i in eachindex(restarts)
        if restarts[i].included
            _require_included_restart_raw(raw[i], domain)
            push!(included, i)
        end
    end
    pairs = FunctionalIdentifiabilityPair[]
    sizehint!(pairs, binomial(length(included), 2))
    for a in eachindex(included)
        for b in (a + 1):length(included)
            left = included[a]
            right = included[b]
            if restarts[left].seed > restarts[right].seed
                left, right = right, left
            end
            dmet = pairwise_destruction_metrics(raw[left].D, raw[right].D)
            tmet = pairwise_trajectory_metrics(
                raw[left].pred_train, raw[right].pred_train,
                raw[left].pred_holdout, raw[right].pred_holdout)
            push!(pairs, FunctionalIdentifiabilityPair(
                restarts[left].seed, restarts[right].seed,
                dmet.d_rmse_raw, dmet.d_rmse_scale_normalized,
                dmet.d_correlation, dmet.scale_alpha,
                tmet.traj_rmse_train, tmet.traj_rmse_holdout))
        end
    end
    return pairs
end

"""
    assemble_functional_identifiability_diagnostic(family, trained)
    assemble_functional_identifiability_diagnostic(family, restart_seeds, domain, restarts, pairs)

Derive the practical diagnostic from restart records and pairwise metrics.
Does not accept `function_disagree`, `trajectory_agree`, or `status` as
authoritative inputs.
"""
function assemble_functional_identifiability_diagnostic(
        family::Symbol, trained::NamedTuple;
        function_disagree = missing,
        trajectory_agree = missing,
        trajectory_agree_function_disagree = missing,
        status = missing,
        complete = missing,
        practical_not_structural = missing)
    pairs = _functional_identifiability_pairs(
        trained.restarts, trained.raw, trained.domain)
    return FunctionalIdentifiabilityDiagnostic(
        family, trained.restart_seeds, trained.domain,
        trained.restarts, pairs;
        function_disagree = function_disagree,
        trajectory_agree = trajectory_agree,
        trajectory_agree_function_disagree = trajectory_agree_function_disagree,
        status = status,
        complete = complete,
        practical_not_structural = practical_not_structural)
end

function assemble_functional_identifiability_diagnostic(
        family::Symbol,
        restart_seeds,
        domain::FunctionalIdentifiabilityDomain,
        restarts::Vector{FunctionalIdentifiabilityRestart},
        pairs::Vector{FunctionalIdentifiabilityPair};
        function_disagree = missing,
        trajectory_agree = missing,
        trajectory_agree_function_disagree = missing,
        status = missing,
        complete = missing,
        practical_not_structural = missing)
    return FunctionalIdentifiabilityDiagnostic(
        family, restart_seeds, domain, restarts, pairs;
        function_disagree = function_disagree,
        trajectory_agree = trajectory_agree,
        trajectory_agree_function_disagree = trajectory_agree_function_disagree,
        status = status,
        complete = complete,
        practical_not_structural = practical_not_structural)
end

"""
    assess_functional_identifiability(split, ude_net; ...)

Q4 owner: M3-B restarts, pairwise destruction / trajectory metrics on the
M3-A domain, then derived flags and status. Not a unique-claim gate.
"""
function assess_functional_identifiability(
        split::ExperimentSplit,
        ude_net::BiologicalNetwork;
        restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS,
        family::Symbol = :hill,
        adam = FUNCTIONAL_ID_TRAINING_CONFIG.adam,
        bfgs = FUNCTIONAL_ID_TRAINING_CONFIG.bfgs,
        frozen_phys = FUNCTIONAL_ID_TRAINING_CONFIG.frozen_phys,
        phys_init = FUNCTIONAL_ID_TRAINING_CONFIG.phys_init,
        fill_value::Real = 0.3)
    trained = train_functional_identifiability_restarts(
        split, ude_net;
        restart_seeds = restart_seeds,
        adam = adam,
        bfgs = bfgs,
        frozen_phys = frozen_phys,
        phys_init = phys_init,
        fill_value = fill_value)
    return assemble_functional_identifiability_diagnostic(family, trained)
end
