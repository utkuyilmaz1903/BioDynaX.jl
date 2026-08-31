using BioDynaX: ExperimentSplit,
    unique_claim_experiment_split,
    UNIQUE_CLAIM_TRAIN_INDICES,
    UNIQUE_CLAIM_HOLDOUT_INDICES,
    generate_recovery_experiments,
    MechanismRecoveryResult,
    recovery_suite_section_body,
    LOCKED_PUBLIC_EXPORTS,
    UNIQUE_CLAIM_PROTOCOL,
    RECOVERY_THRESHOLDS,
    experiment_fingerprint,
    public_export_list_holds,
    build_hill_recovery_network,
    with_generate_recovery_experiments_observer,
    with_unique_claim_experiment_split_observer,
    with_fit_unknown_destruction_observer,
    with_evaluate_unknown_rate_recovery_range_observer,
    with_sample_unknown_destruction_grid_observer,
    with_discover_unknown_rate_observer,
    with_discover_equations_observer,
    HoldoutEvidence,
    evaluate_holdout,
    _holdout_observed_regulators,
    _unique_claim_external_regulator_band,
    _finite_rate_rel_rmse,
    _mean_hybrid_residual,
    _train_unknown_edge,
    _regulator_grid,
    _unique_claim_rate_recovery,
    _evaluate_unknown_rate_recovery,
    only_unknown_destruction,
    DiscoveryFailed,
    DiscoverySuccess,
    DiscoveryResult,
    UDEModel,
    report_recovery,
    sample_unknown_destruction_grid,
    neural_identity_rate,
    hybrid_data_residual,
    unique_claim_discovery_config,
    hill_rate_truth,
    Experiment,
    ExperimentSet

const _M2A_FORBIDDEN_MUTATORS = (
    "splice!", "deleteat!", "pop!", "push!", "insert!",
    "append!", "resize!", "setindex!", "replace!")

const _M2A_INTERNAL_NAMES = (
    :ExperimentSplit,
    :HoldoutEvidence,
    :evaluate_holdout,
    :unique_claim_experiment_split,
    :UNIQUE_CLAIM_TRAIN_INDICES,
    :UNIQUE_CLAIM_HOLDOUT_INDICES,
    :_holdout_observed_regulators,
    :_unique_claim_external_regulator_band,
    :_finite_rate_rel_rmse,
    :_mean_hybrid_residual)

const _M2B_INTERNAL_NAMES = (
    :with_generate_recovery_experiments_observer,
    :with_unique_claim_experiment_split_observer,
    :with_fit_unknown_destruction_observer,
    :_note_generate_recovery_experiments,
    :_note_unique_claim_experiment_split,
    :_note_fit_unknown_destruction,
    :GENERATE_RECOVERY_EXPERIMENTS_OBSERVER,
    :UNIQUE_CLAIM_EXPERIMENT_SPLIT_OBSERVER,
    :FIT_UNKNOWN_DESTRUCTION_OBSERVER)

const _M2C_INTERNAL_NAMES = (
    :with_evaluate_unknown_rate_recovery_range_observer,
    :_note_evaluate_unknown_rate_recovery_range,
    :EVALUATE_UNKNOWN_RATE_RECOVERY_RANGE_OBSERVER,
    :_unique_claim_rate_recovery)

const _M2C_DOMAIN_TOKEN = "r_range = _regulator_grid(split.train, term)"

const _M2C_WRONG_DOMAIN_TOKENS = (
    "_regulator_grid(ude_set, term)",
    "_regulator_grid(set, term)",
    "_regulator_grid(split.holdout, term)",
    "_regulator_grid(holdout, term)")

const _M2C_HOLDOUT_SENTINEL = 50.0
const _M2C_TRAIN_SENTINEL_LO = 0.0
const _M2C_TRAIN_SENTINEL_HI = 8.0

const _M2B_APPROVED_EDGE_HELPERS = (
    "_note_train_unknown_edge",)

const _M2B_SECOND_TRAINER_TOKENS = (
    "train_ude(",
    "train_experiments(",
    "train_experiments_with_warmup(",
    "_polish_full(",
    "_hidden_full_fit(",
    "warmup_first_experiment(")

const _M2B_GENERATE_TOKENS = (
    "generate_recovery_experiments(",
    "generate_experiment_set(",
    "generate_data(")

function _m2a_source_function_body(path, name)
    src = read(path, String)
    start = findfirst("function " * name, src)
    start === nothing && return nothing
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function _m2a_lookup_local_function(name)
    root = joinpath(@__DIR__, "..", "src")
    for file in ("RecoveryPipeline.jl", "Recovery.jl")
        body = _m2a_source_function_body(joinpath(root, file), name)
        body !== nothing && return body
    end
    return nothing
end

function _m2a_reachable_split_bodies(entry::String)
    queue = [entry]
    seen = Set{String}()
    bodies = Dict{String,String}()
    while !isempty(queue)
        name = popfirst!(queue)
        name in seen && continue
        push!(seen, name)
        body = _m2a_lookup_local_function(name)
        body === nothing && continue
        bodies[name] = body
        for match in eachmatch(r"\b([A-Za-z_][A-Za-z0-9_!]*)\(", body)
            callee = match.captures[1]
            startswith(callee, "_") || continue
            callee in seen && continue
            push!(queue, callee)
        end
    end
    return bodies
end

function _m2a_train_unknown_edge_body()
    body = _m2a_source_function_body(
        joinpath(@__DIR__, "..", "src", "Recovery.jl"),
        "_train_unknown_edge")
    return body === nothing ? "" : body
end

function _m2a_nine_ic_set(seed)
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    return generate_recovery_experiments(
        MersenneTwister(seed), truth_net, truth;
        tspan = (0.0, 1.0), n_points = 5, noise_σ = 0.0)
end

function _m2b_fit_unknown_destruction_body()
    body = _m2a_source_function_body(
        joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"),
        "fit_unknown_destruction")
    return body === nothing ? "" : body
end

function _m2b_reachable_unknown_edge_helpers()
    queue = ["_train_unknown_edge"]
    seen = Set{String}()
    bodies = Dict{String,String}()
    approved = Set(_M2B_APPROVED_EDGE_HELPERS)
    while !isempty(queue)
        name = popfirst!(queue)
        name in seen && continue
        push!(seen, name)
        body = _m2a_lookup_local_function(name)
        body === nothing && continue
        bodies[name] = body
        for match in eachmatch(r"\b([A-Za-z_][A-Za-z0-9_!]*)\(", body)
            callee = String(match.captures[1])
            startswith(callee, "_") || continue
            callee in approved && continue
            callee in seen && continue
            push!(queue, callee)
        end
    end
    delete!(bodies, "_train_unknown_edge")
    return bodies
end

function _m2b_dummy_training_result(set)
    return TrainingResult(
        Float64[], Float64[], 0.0, 0.0,
        RunMetadata(seed = 0),
        (; experiment_count = length(set)),
        true, BioDynaX.Success)
end

function _m2b_mark_holdout!(set)
    for i in 8:9
        set.experiments[i].metadata[:m2b_holdout_sentinel] = i
    end
    return set
end

function _m2b_probe_train_unknown_edge(; seed = 17, mark_holdout::Bool = false)
    generated = Any[]
    splits = Any[]
    fit_sets = Any[]
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    rng = MersenneTwister(seed)
    ude_model, ude_p0 = build_ude_model(rng, ude_net)
    result = with_generate_recovery_experiments_observer(set -> begin
            push!(generated, set)
            mark_holdout && _m2b_mark_holdout!(set)
            nothing
        end) do
        with_unique_claim_experiment_split_observer(split -> push!(splits, split)) do
            with_fit_unknown_destruction_observer(set -> begin
                    push!(fit_sets, set)
                    _m2b_dummy_training_result(set)
                end) do
                _train_unknown_edge(
                    rng, ude_model, ude_p0, truth_net, truth;
                    adam = 0, bfgs = 0, noise_σ = 0.0,
                    tspan = (0.0, 1.0), n_points = 5)
            end
        end
    end
    return (; generated, splits, fit_sets, result)
end

function _m2c_composer_body()
    body = _m2a_source_function_body(
        joinpath(@__DIR__, "..", "src", "Recovery.jl"),
        "_evaluate_unknown_rate_recovery")
    return body === nothing ? "" : body
end

function _m2c_rate_recovery_body()
    body = _m2a_source_function_body(
        joinpath(@__DIR__, "..", "src", "Recovery.jl"),
        "_unique_claim_rate_recovery")
    return body === nothing ? "" : body
end

function _m2c_composer_call(body)
    start = findfirst("_evaluate_unknown_rate_recovery(", body)
    start === nothing && return ""
    stop = findfirst("report_production_destruction_tradeoff", body)
    stop === nothing && return body[first(start):end]
    return body[first(start):(first(stop) - 1)]
end

function _m2c_probe_models()
    rng = MersenneTwister(1)
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    model, params = build_ude_model(rng, net)
    term = only_unknown_destruction(model)
    return model, params, term
end

function _m2c_dummy_evaled(term)
    return (;
        nn_correlation = 0.0,
        nn_rate_rmse = Inf,
        success = false,
        retcode = DiscoveryFailed,
        message = "m2c domain probe",
        support_f1 = 0.0,
        support_recall = 0.0,
        discovered_rate_rmse = Inf,
        data_residual = Inf,
        denominator_violations = typemax(Int),
        normalized_support_f1 = 0.0,
        normalized_support_recall = 0.0,
        extras = String[],
        discovery = nothing,
        term = term)
end

function _m2c_apply_holdout_sentinel!(set, term)
    for i in 8:9
        set.experiments[i].observations[term.regulator, :] .= _M2C_HOLDOUT_SENTINEL
    end
    return set
end

function _m2c_apply_train_sentinel!(set, term)
    set.experiments[1].observations[term.regulator, :] .= _M2C_TRAIN_SENTINEL_LO
    set.experiments[7].observations[term.regulator, :] .= _M2C_TRAIN_SENTINEL_HI
    return set
end

function _m2c_consumed_discovery_range(set, term, model, params)
    captured = Ref{Any}()
    with_evaluate_unknown_rate_recovery_range_observer(r_range -> begin
            captured[] = collect(r_range)
            _m2c_dummy_evaled(term)
        end) do
        _unique_claim_rate_recovery(
            model, params, term, _ -> 0.0, set;
            order = 2, family = :hill, noise_σ = 0.0,
            data_residual_fn = _ -> 0.0)
    end
    return captured[]
end

function _m2c_assert_domain_source(body)
    @test count(r"r_range\s*=", body) == 1
    @test occursin(_M2C_DOMAIN_TOKEN, body)
    assign_at = findfirst(_M2C_DOMAIN_TOKEN, body)
    @test assign_at !== nothing
    after = body[(last(assign_at) + 1):end]
    @test !occursin(r"r_range\s*=", after)
    @test !occursin("union(", body)
    for token in _M2C_WRONG_DOMAIN_TOKENS
        @test !occursin(token, body)
    end
    @test !occursin("evaluate_holdout(", body)
    @test !occursin("HoldoutEvidence", body)
    @test !occursin("d_rmse_holdout", body)
    @test !occursin("generate_recovery_experiments(", body)
    @test !occursin("generate_experiment_set(", body)
    @test !occursin("generate_data(", body)
    for mutator in _M2A_FORBIDDEN_MUTATORS
        @test !occursin(mutator * "(", body)
    end
end

const _M2D_INTERNAL_NAMES = (
    :HoldoutEvidence,
    :evaluate_holdout,
    :_holdout_observed_regulators,
    :_unique_claim_external_regulator_band,
    :_finite_rate_rel_rmse,
    :_mean_hybrid_residual,
    :with_sample_unknown_destruction_grid_observer,
    :with_discover_unknown_rate_observer,
    :with_discover_equations_observer)

const _M2D_FORBIDDEN_DISCOVERY = (
    "discover_unknown_rate(",
    "discover_unknown(",
    "discover_equations(",
    "discover_unknown_destruction(",
    "_peek_holdout",
    "normalize_destruction_samples(",
    "equation_to_function(",
    "sample_learned_function(",
    "evaluate_recovery(",
    "_ensure_holdout")

const _M2D_FORBIDDEN_TRAINING = (
    "train_ude(",
    "train_experiments(",
    "train_experiments_with_warmup(",
    "fit_unknown_destruction(",
    "_polish_full(")

const _M2D_FORBIDDEN_GENERATE = (
    "generate_recovery_experiments(",
    "generate_experiment_set(",
    "generate_data(")

const _M2D_FORBIDDEN_M34 = (
    :functional_identifiability,
    :restart_agreement,
    :uncertainty,
    :hypothesis,
    :occupancy,
    :q4,
    :q7,
    :samples,
    :domain)

struct _M2DMemorizationNN end

function (::_M2DMemorizationNN)(input, p, st)
    r = Float64(input[1])
    d = 1.0 + 2.0 * exp(-((r - 1.75) / 0.05)^2)
    return [d], st
end

function _m2d_evaluate_holdout_body()
    body = _m2a_source_function_body(
        joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"),
        "evaluate_holdout")
    return body === nothing ? "" : body
end

function _m2d_pipeline_source()
    return read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String)
end

function _m2d_experiment(name, r_vals, s_vals)
    n = length(r_vals)
    times = collect(range(0.0, 1.0; length = n))
    observations = Matrix{Float64}(undef, 2, n)
    observations[1, :] .= s_vals
    observations[2, :] .= r_vals
    return Experiment(name, times, observations,
        [Float64(s_vals[1]), Float64(r_vals[1])])
end

function _m2d_synthetic_set(;
        train_r_extrema = (0.50, 1.50),
        holdout_r = (1.75, 1.75),
        holdout_s = (0.40, 1.20),
        n_points = 5)
    r_lo, r_hi = train_r_extrema
    train_r = range(r_lo, r_hi; length = 7)
    train = [_m2d_experiment(Symbol(:train, i),
        fill(Float64(train_r[i]), n_points),
        fill(0.20 + 0.05 * i, n_points)) for i in 1:7]
    hold = [_m2d_experiment(Symbol(:hold, j),
        fill(Float64(rj), n_points), fill(Float64(sj), n_points))
            for (j, (rj, sj)) in enumerate(zip(holdout_r, holdout_s))]
    return ExperimentSet(vcat(train, hold), [:S, :R])
end

function _m2d_evaled(term; success = true, discovery = :ok,
        data_residual = 0.01)
    disc = if discovery === :ok
        DiscoveryResult(true, "m2d ok", "", nothing, nothing, [], (;))
    elseif discovery === :failed
        DiscoveryResult(false, "m2d fail", "", nothing, nothing, [], (;))
    else
        nothing
    end
    return (;
        nn_correlation = 0.95,
        nn_rate_rmse = 0.05,
        success = success,
        retcode = success ? DiscoverySuccess : DiscoveryFailed,
        message = "m2d evaled",
        support_f1 = 0.57,
        support_recall = 1.0,
        discovered_rate_rmse = 0.10,
        data_residual = data_residual,
        denominator_violations = 0,
        normalized_support_f1 = 0.60,
        normalized_support_recall = 1.0,
        extras = String[],
        discovery = disc,
        term = term)
end

function _m2d_experiment_residual(model, params, term, exp)
    D_hat_fn = neural_identity_rate(model, params, term)
    return hybrid_data_residual(
        model, params, term, D_hat_fn,
        exp.u0, (first(exp.times), last(exp.times)),
        exp.times, exp.observations;
        mask = exp.mask)
end

function _m2d_legacy_ic1_residual(model, params, term, set)
    return _m2d_experiment_residual(model, params, term, first(set))
end

function _m2d_memorization_model(model)
    return UDEModel(
        model.network, _M2DMemorizationNN(), model.st, model.compiled,
        model.state_ids, model.impl, model.nstates, model.n_neural,
        model.max_nn_in, model.is_linear_ab, model.k_ba_idx,
        model.k_a_idx, model.k_b_idx)
end

function _m2d_dummy_discovery()
    return DiscoveryResult(false, "m2d probe", "", nothing, nothing, [], (;))
end

_m2d_unit_truth(r) = fill(1.0, length(r))

function _m2d_matching_truth(model, params, term)
    return function (r)
        (_, D, _) = sample_unknown_destruction_grid(
            model, params, term; r_range = r, fill_value = 0.3)
        return vec(D)
    end
end

function _m2d_capture_grid_ranges(f)
    consumed = Any[]
    result = with_sample_unknown_destruction_grid_observer(r_range ->
            push!(consumed, r_range)) do
        f()
    end
    return result, consumed
end

@testset "L-FIELDS ExperimentSplit surface is exactly 7/2" begin
    @test fieldnames(ExperimentSplit) ==
          (:train_indices, :holdout_indices, :train, :holdout)
    @test length(fieldnames(ExperimentSplit)) == 4
    @test :validation_indices ∉ fieldnames(ExperimentSplit)
    @test :seed ∉ fieldnames(ExperimentSplit)
    @test :provenance ∉ fieldnames(ExperimentSplit)
    @test :functional_identifiability ∉ fieldnames(ExperimentSplit)
    @test :occupancy ∉ fieldnames(ExperimentSplit)
    @test :uncertainty ∉ fieldnames(ExperimentSplit)
    @test :hypothesis ∉ fieldnames(ExperimentSplit)
    @test :data_residual_holdout ∉ fieldnames(ExperimentSplit)
    @test :d_rmse_holdout ∉ fieldnames(ExperimentSplit)
end

@testset "L-API M2-A names stay unexported" begin
    @test isdefined(BioDynaX, :ExperimentSplit)
    @test isdefined(BioDynaX, :unique_claim_experiment_split)
    @test isdefined(BioDynaX, :UNIQUE_CLAIM_TRAIN_INDICES)
    @test isdefined(BioDynaX, :UNIQUE_CLAIM_HOLDOUT_INDICES)
    @test !isdefined(BioDynaX, :split_experiments)
    for name in _M2A_INTERNAL_NAMES
        @test !(name in names(BioDynaX))
        @test !(name in LOCKED_PUBLIC_EXPORTS)
    end
    @test public_export_list_holds()
end

@testset "L-SPLIT-ID locked 7/2 preserves Experiment identity" begin
    set = _m2a_nine_ic_set(7)
    @test length(set) == 9
    ids = [set.experiments[i] for i in 1:9]
    for i in 1:9
        set.experiments[i].metadata[:probe] = i
    end
    split = unique_claim_experiment_split(set)
    @test split.train_indices === UNIQUE_CLAIM_TRAIN_INDICES === (1, 2, 3, 4, 5, 6, 7)
    @test split.holdout_indices === UNIQUE_CLAIM_HOLDOUT_INDICES === (8, 9)
    @test propertynames(split) ==
          (:train_indices, :holdout_indices, :train, :holdout)
    @test length(split.train) == 7
    @test length(split.holdout) == 2
    @test 1 ∈ split.train_indices
    @test split.train !== set
    @test split.holdout !== set
    @test split.train[1] === set.experiments[1] === first(set)
    for i in 1:7
        @test split.train[i] === set.experiments[i]
        @test split.train[i] === set.experiments[split.train_indices[i]]
        @test split.train[i] === ids[i]
    end
    for i in 1:2
        @test split.holdout[i] === set.experiments[7 + i]
        @test split.holdout[i] === set.experiments[split.holdout_indices[i]]
        @test split.holdout[i] === ids[7 + i]
    end
    @test [split.train[i].metadata[:probe] for i in 1:7] == 1:7
    @test [split.holdout[i].metadata[:probe] for i in 1:2] == [8, 9]
    train_ids = objectid.([split.train[i] for i in 1:7])
    holdout_ids = objectid.([split.holdout[i] for i in 1:2])
    @test isempty(train_ids ∩ holdout_ids)
    @test issetequal(objectid.(ids), train_ids ∪ holdout_ids)
    @test length(unique(objectid.(ids))) == 9
    regen = _m2a_nine_ic_set(7)
    @test regen.experiments[1].observations == set.experiments[1].observations
    @test regen.experiments[8].observations == set.experiments[8].observations
    @test regen.experiments[1] !== set.experiments[1]
    @test regen.experiments[8] !== set.experiments[8]
    @test split.train[1] === set.experiments[1]
    @test split.train[1] !== regen.experiments[1]
    @test split.holdout[1] === set.experiments[8]
    @test split.holdout[1] !== regen.experiments[8]
    @test split.holdout[2] === set.experiments[9]
    @test split.holdout[2] !== regen.experiments[9]
end

@testset "L-SPLIT-META / L-SET-META split is not hidden on the set" begin
    set = _m2a_nine_ic_set(11)
    split = unique_claim_experiment_split(set)
    @test !haskey(set.metadata, :train)
    @test !haskey(set.metadata, :holdout)
    @test !haskey(set.metadata, :split)
    @test !haskey(set.metadata, :train_indices)
    @test !haskey(set.metadata, :holdout_indices)
    @test !haskey(split.train.metadata, :train)
    @test !haskey(split.train.metadata, :holdout)
    @test !haskey(split.holdout.metadata, :train)
    @test !haskey(split.holdout.metadata, :holdout)
    @test split.train.metadata !== set.metadata
    @test split.holdout.metadata !== set.metadata
    @test split.train.metadata !== split.holdout.metadata
end

@testset "L-SET-INTACT original ExperimentSet stays untouched" begin
    set = _m2a_nine_ic_set(13)
    vec_before = set.experiments
    ids = [set.experiments[i] for i in 1:9]
    names_before = copy(set.state_names)
    units_before = copy(set.units)
    meta_before = deepcopy(set.metadata)
    fp_before = experiment_fingerprint(set)
    obs_before = [set.experiments[i].observations for i in 1:9]
    times_before = [set.experiments[i].times for i in 1:9]
    u0_before = [set.experiments[i].u0 for i in 1:9]
    mask_before = [set.experiments[i].mask for i in 1:9]
    name_before = [set.experiments[i].name for i in 1:9]
    public_fields_before = fieldnames(typeof(set))
    split = unique_claim_experiment_split(set)
    @test set.experiments === vec_before
    @test length(set.experiments) == 9
    @test length(set) == 9
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test set.state_names == names_before
    @test set.units == units_before
    @test set.metadata == meta_before
    @test experiment_fingerprint(set) == fp_before
    @test fieldnames(typeof(set)) == public_fields_before
    @test fieldnames(typeof(set)) ==
          (:experiments, :state_names, :units, :metadata)
    @test !hasfield(ExperimentSet, :train)
    @test !hasfield(ExperimentSet, :holdout)
    @test !haskey(set.metadata, :train)
    @test !haskey(set.metadata, :holdout)
    for i in 1:9
        @test set.experiments[i].observations === obs_before[i]
        @test set.experiments[i].times === times_before[i]
        @test set.experiments[i].u0 === u0_before[i]
        @test set.experiments[i].mask === mask_before[i]
        @test set.experiments[i].name === name_before[i]
    end
    for i in 1:7
        @test split.train[i].observations === set.experiments[i].observations
        @test split.train[i].times === set.experiments[i].times
        @test split.train[i].u0 === set.experiments[i].u0
        @test split.train[i].mask === set.experiments[i].mask
    end
    for i in 1:2
        @test split.holdout[i].observations === set.experiments[7 + i].observations
        @test split.holdout[i].times === set.experiments[7 + i].times
        @test split.holdout[i].u0 === set.experiments[7 + i].u0
        @test split.holdout[i].mask === set.experiments[7 + i].mask
    end
    short = ExperimentSet(
        collect(set.experiments[1:8]), set.state_names; units = set.units)
    @test_throws ArgumentError unique_claim_experiment_split(short)
    @test set.experiments === vec_before
    @test length(set) == 9
    again = unique_claim_experiment_split(set)
    @test set.experiments === vec_before
    @test again.train[1] === split.train[1] === set.experiments[1]
    @test again.holdout[2] === split.holdout[2] === set.experiments[9]
end

@testset "L-SET-INTACT splitter call graph does not mutate the original set" begin
    bodies = _m2a_reachable_split_bodies("unique_claim_experiment_split")
    @test haskey(bodies, "unique_claim_experiment_split")
    for (name, body) in bodies
        for mutator in _M2A_FORBIDDEN_MUTATORS
            @test !occursin(mutator * "(", body)
        end
        @test !occursin(r"\.experiments\s*\[[^\]]+\]\s*=", body)
        @test !occursin("generate_recovery_experiments(", body)
        @test !occursin("generate_experiment_set(", body)
        @test !occursin("generate_data(", body)
    end
    @test !occursin("function split_experiments",
        read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String))
end

@testset "M2-D evaluator exists and stays off the unique-claim suite path" begin
    @test isdefined(BioDynaX, :HoldoutEvidence)
    @test isdefined(BioDynaX, :evaluate_holdout)
    @test isdefined(BioDynaX, :_holdout_observed_regulators)
    @test isdefined(BioDynaX, :_unique_claim_external_regulator_band)
    @test isdefined(BioDynaX, :_finite_rate_rel_rmse)
    @test isdefined(BioDynaX, :_mean_hybrid_residual)
    @test :split ∉ fieldnames(MechanismRecoveryResult)
    @test :holdout ∉ fieldnames(MechanismRecoveryResult)
    @test :data_residual_holdout ∉ fieldnames(MechanismRecoveryResult)
    @test :d_rmse_holdout ∉ fieldnames(MechanismRecoveryResult)
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test !haskey(RECOVERY_THRESHOLDS, :data_residual_holdout)
    @test !haskey(RECOVERY_THRESHOLDS, :d_rmse_holdout)
    @test !haskey(UNIQUE_CLAIM_PROTOCOL, :train_indices)
    @test !haskey(UNIQUE_CLAIM_PROTOCOL, :holdout_indices)
    @test !haskey(UNIQUE_CLAIM_PROTOCOL, :split)
    train_body = _m2a_train_unknown_edge_body()
    @test occursin("generate_recovery_experiments(", train_body)
    @test occursin("fit_unknown_destruction(", train_body)
    @test occursin("return fit, set", train_body)
    @test count("generate_recovery_experiments(", train_body) == 1
    @test !occursin("evaluate_holdout(", train_body)
    @test !occursin("generate_experiment_set(", train_body)
    @test !occursin("generate_data(", train_body)
    composer = _m2c_composer_body()
    @test !occursin("evaluate_holdout(", composer)
    @test !occursin("HoldoutEvidence", composer)
    report_body = _m2a_source_function_body(
        joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"),
        "report_recovery")
    @test report_body !== nothing
    @test !occursin("evaluate_holdout(", report_body)
    for section in (:ude_discovery, :mm_unknown)
        body = recovery_suite_section_body(section)
        @test occursin("_train_unknown_edge", body)
        @test occursin("_evaluate_unknown_rate_recovery(", body)
        @test occursin("report_recovery(", body)
        @test occursin("unique_claim_experiment_split(ude_set)", body)
        @test !occursin("evaluate_holdout(", body)
        @test !occursin("HoldoutEvidence", body)
    end
end

@testset "L-API M2-B seams stay unexported" begin
    for name in _M2B_INTERNAL_NAMES
        @test isdefined(BioDynaX, name)
        @test !(name in names(BioDynaX))
        @test !(name in LOCKED_PUBLIC_EXPORTS)
    end
    @test !(:_train_unknown_edge in names(BioDynaX))
    @test public_export_list_holds()
end

@testset "L-FIT-A unique-claim path has one fit_unknown_destruction(split.train)" begin
    body = _m2a_train_unknown_edge_body()
    @test count("fit_unknown_destruction(", body) == 1
    @test occursin("unique_claim_experiment_split(set)", body)
    @test count("unique_claim_experiment_split(", body) == 1
    @test occursin(r"fit_unknown_destruction\(\s*ude_model,\s*ude_p0,\s*split\.train;",
        body)
    @test !occursin(r"fit_unknown_destruction\(\s*ude_model,\s*ude_p0,\s*set;", body)
    @test occursin("return fit, set", body)
    gen_at = findfirst("generate_recovery_experiments(", body)
    split_at = findfirst("unique_claim_experiment_split(", body)
    fit_at = findfirst("fit_unknown_destruction(", body)
    return_at = findfirst("return fit, set", body)
    @test gen_at !== nothing && split_at !== nothing && fit_at !== nothing
    @test return_at !== nothing
    @test first(gen_at) < first(split_at) < first(fit_at) < first(return_at)
    after_fit = body[last(fit_at):end]
    for token in _M2B_SECOND_TRAINER_TOKENS
        @test !occursin(token, body)
        @test !occursin(token, after_fit)
    end
    @test !occursin("fit_unknown_destruction(", after_fit)
    helpers = _m2b_reachable_unknown_edge_helpers()
    for (name, helper) in helpers
        for token in _M2B_SECOND_TRAINER_TOKENS
            @test !occursin(token, helper)
        end
        @test !occursin("fit_unknown_destruction(", helper)
        for token in _M2B_GENERATE_TOKENS
            @test !occursin(token, helper)
        end
    end
    fit_body = _m2b_fit_unknown_destruction_body()
    @test count("train_experiments_with_warmup(", fit_body) == 1
    @test occursin(r"train_experiments_with_warmup\(\s*ude_init,\s*set,\s*ude_model;",
        fit_body)
    @test !occursin("unique_claim_experiment_split(", fit_body)
    @test !occursin("generate_recovery_experiments(", fit_body)
    @test !occursin("generate_experiment_set(", fit_body)
    @test !occursin("generate_data(", fit_body)
    for section in (:ude_discovery, :mm_unknown)
        section_body = recovery_suite_section_body(section)
        train_at = findfirst("_train_unknown_edge", section_body)
        @test train_at !== nothing
        after = section_body[last(train_at):end]
        for token in ("train_ude(", "train_experiments(",
                      "train_experiments_with_warmup",
                      "fit_unknown_destruction(", "_polish_full(")
            @test !occursin(token, section_body)
            @test !occursin(token, after)
        end
    end
end

@testset "L-FIT-A / L-FIT-B trainer receives exactly experiments 1:7" begin
    probe = _m2b_probe_train_unknown_edge()
    @test length(probe.generated) == 1
    @test length(probe.splits) == 1
    @test length(probe.fit_sets) == 1
    set = probe.result[2]
    split = probe.splits[1]
    fit_set = probe.fit_sets[1]
    @test probe.result[1] isa TrainingResult
    @test set isa ExperimentSet
    @test set === probe.generated[1]
    @test length(set) == 9
    @test split.train_indices === UNIQUE_CLAIM_TRAIN_INDICES === (1, 2, 3, 4, 5, 6, 7)
    @test split.holdout_indices === UNIQUE_CLAIM_HOLDOUT_INDICES === (8, 9)
    @test 1 ∈ split.train_indices
    @test fit_set === split.train
    @test length(fit_set) == 7
    @test fit_set !== set
    ids = [set.experiments[i] for i in 1:9]
    @test length(unique(objectid.(ids))) == 9
    for i in 1:7
        @test fit_set[i] === set.experiments[i] === ids[i] === split.train[i]
    end
    @test first(fit_set) === set.experiments[1]
    @test !(fit_set[1] === set.experiments[8])
    @test !(fit_set[1] === set.experiments[9])
    fit_ids = objectid.([fit_set[i] for i in 1:7])
    @test !(objectid(set.experiments[8]) in fit_ids)
    @test !(objectid(set.experiments[9]) in fit_ids)
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test set.experiments === probe.generated[1].experiments
end

@testset "L-FIT-B holdout sentinel cannot enter the trainer input" begin
    probe = _m2b_probe_train_unknown_edge(; mark_holdout = true)
    set = probe.result[2]
    split = probe.splits[1]
    fit_set = probe.fit_sets[1]
    ids_train = [set.experiments[i] for i in 1:7]
    @test fit_set === split.train
    @test length(fit_set) == 7
    @test all(fit_set[i] === ids_train[i] === set.experiments[i] for i in 1:7)
    @test set.experiments[8].metadata[:m2b_holdout_sentinel] == 8
    @test set.experiments[9].metadata[:m2b_holdout_sentinel] == 9
    @test !any(haskey(fit_set[i].metadata, :m2b_holdout_sentinel) for i in 1:7)
    @test !any(haskey(split.train[i].metadata, :m2b_holdout_sentinel) for i in 1:7)
    @test haskey(split.holdout[1].metadata, :m2b_holdout_sentinel)
    @test haskey(split.holdout[2].metadata, :m2b_holdout_sentinel)
    @test !(fit_set[7] === set.experiments[8])
    @test objectid(set.experiments[8]) ∉ objectid.([fit_set[i] for i in 1:7])
    @test objectid(set.experiments[9]) ∉ objectid.([fit_set[i] for i in 1:7])
end

@testset "L-RNG _train_unknown_edge generates the 9-IC set once" begin
    body = _m2a_train_unknown_edge_body()
    @test count("generate_recovery_experiments(", body) == 1
    @test !occursin("generate_experiment_set(", body)
    @test !occursin("generate_data(", body)
    @test occursin("return fit, set", body)
    @test !occursin("return fit, generate_recovery_experiments", body)
    @test !occursin("return fit, generate_experiment_set", body)
    @test !occursin("return fit, generate_data", body)
    @test !occursin("evaluate_holdout", body)
    @test !occursin("HoldoutEvidence", body)
    @test !occursin("d_rmse_holdout", body)
    @test !occursin("data_residual_holdout", body)
    for mutator in _M2A_FORBIDDEN_MUTATORS
        @test !occursin(mutator * "(", body)
    end
    @test !occursin(r"\.experiments\s*\[[^\]]+\]\s*=", body)
    helpers = _m2b_reachable_unknown_edge_helpers()
    for (_, helper) in helpers
        for token in _M2B_GENERATE_TOKENS
            @test !occursin(token, helper)
        end
    end
    for section in (:ude_discovery, :mm_unknown)
        section_body = recovery_suite_section_body(section)
        @test !occursin("generate_recovery_experiments(", section_body)
        @test !occursin("generate_experiment_set(", section_body)
        @test !occursin("generate_data(", section_body)
    end
    probe = _m2b_probe_train_unknown_edge()
    @test length(probe.generated) == 1
    ids_gen = [probe.generated[1].experiments[i] for i in 1:9]
    @test length(unique(objectid.(ids_gen))) == 9
    fit, set = probe.result
    @test fit isa TrainingResult
    @test set === probe.generated[1]
    @test all(set.experiments[i] === ids_gen[i] for i in 1:9)
    @test all(probe.fit_sets[1][i] === ids_gen[i] for i in 1:7)
    @test probe.splits[1].train[1] === ids_gen[1]
    @test probe.splits[1].holdout[1] === ids_gen[8]
    @test probe.splits[1].holdout[2] === ids_gen[9]
end

@testset "L-SET-INTACT M2-B leaves the original nine Experiment objects" begin
    probe = _m2b_probe_train_unknown_edge()
    set = probe.result[2]
    vec_before = probe.generated[1].experiments
    ids = [probe.generated[1].experiments[i] for i in 1:9]
    @test set.experiments === vec_before
    @test length(set.experiments) == 9
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test !hasfield(ExperimentSet, :train)
    @test !hasfield(ExperimentSet, :holdout)
    @test !haskey(set.metadata, :train)
    @test !haskey(set.metadata, :holdout)
    split = probe.splits[1]
    for i in 1:7
        @test split.train[i] === set.experiments[i]
        @test split.train[i].observations === set.experiments[i].observations
    end
    for i in 1:2
        @test split.holdout[i] === set.experiments[7 + i]
        @test split.holdout[i].observations === set.experiments[7 + i].observations
    end
end

@testset "L-API M2-C domain seam stays unexported" begin
    for name in _M2C_INTERNAL_NAMES
        @test isdefined(BioDynaX, name)
        @test !(name in names(BioDynaX))
        @test !(name in LOCKED_PUBLIC_EXPORTS)
    end
    @test !(:_regulator_grid in names(BioDynaX))
    @test !(:_evaluate_unknown_rate_recovery in names(BioDynaX))
    @test public_export_list_holds()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.nn_correlation == 0.90
    @test :train_indices ∉ keys(UNIQUE_CLAIM_PROTOCOL)
    @test :holdout_indices ∉ keys(UNIQUE_CLAIM_PROTOCOL)
end

@testset "L-DOM-A unique-claim production domain is _regulator_grid(split.train, term)" begin
    helper = _m2c_rate_recovery_body()
    _m2c_assert_domain_source(helper)
    @test occursin("unique_claim_experiment_split(set)", helper)
    @test occursin("_evaluate_unknown_rate_recovery(", helper)
    @test occursin(_M2C_DOMAIN_TOKEN, helper)
    @test count("_evaluate_unknown_rate_recovery(", helper) == 1
    @test !occursin("holdout =", helper)
    @test !occursin("split = split", helper)
    for section in (:ude_discovery, :mm_unknown)
        body = recovery_suite_section_body(section)
        _m2c_assert_domain_source(body)
        @test occursin("split = unique_claim_experiment_split(ude_set)", body)
        @test count("unique_claim_experiment_split(", body) == 1
        @test count("_evaluate_unknown_rate_recovery(", body) == 1
        call = _m2c_composer_call(body)
        @test occursin(_M2C_DOMAIN_TOKEN, call)
        @test count(r"r_range\s*=", call) == 1
        @test !occursin(r"\bholdout\s*=", call)
        @test !occursin(r"\bsplit\s*=", call)
        @test !occursin("ExperimentSplit", call)
        @test !occursin("HoldoutEvidence", call)
        @test occursin("order =", call)
        @test occursin("family =", call)
        @test occursin("noise_σ =", call)
        @test occursin("data_residual_fn =", call)
        @test occursin("ref_exp = first(ude_set.experiments)", body)
        @test !occursin("split.holdout", body)
        @test !occursin("ude_set.experiments[8]", body)
        @test !occursin("ude_set.experiments[9]", body)
        train_at = findfirst("_train_unknown_edge", body)
        split_at = findfirst("unique_claim_experiment_split(ude_set)", body)
        eval_at = findfirst("_evaluate_unknown_rate_recovery(", body)
        @test train_at !== nothing && split_at !== nothing && eval_at !== nothing
        @test first(train_at) < first(split_at) < first(eval_at)
    end
end

@testset "L-DISC-B-1 composer call site has no split/holdout argument" begin
    composer = _m2c_composer_body()
    @test occursin("function _evaluate_unknown_rate_recovery(ude_model, ude_params, term, truth_rate;",
        composer)
    @test occursin("r_range = range(0.05, 2.0; length = 80)", composer)
    @test !occursin("split=", composer)
    @test !occursin("holdout=", composer)
    @test !occursin("ExperimentSplit", composer)
    @test !occursin("HoldoutEvidence", composer)
    @test !occursin("evaluate_holdout", composer)
    @test !occursin("split.holdout", composer)
    @test !occursin("split.train", composer)
    @test !occursin(".holdout", composer)
    @test occursin("times = collect(range(0.0, 1.0; length = length(r)))", composer)
    @test count("discover_unknown_rate(", composer) == 2
    @test occursin("normalize_destruction_samples", composer)
    @test occursin("evaluate_recovery(", composer)
    @test occursin("if !training_ok", composer)
    @test occursin("sample_unknown_destruction_grid(", composer)
    note_at = findfirst("_note_evaluate_unknown_rate_recovery_range(r_range)", composer)
    sample_at = findfirst("sample_unknown_destruction_grid(", composer)
    @test note_at !== nothing && sample_at !== nothing
    @test first(note_at) < first(sample_at)
    for section in (:ude_discovery, :mm_unknown)
        call = _m2c_composer_call(recovery_suite_section_body(section))
        @test occursin(_M2C_DOMAIN_TOKEN, call)
        @test !occursin(r"\bholdout\s*=", call)
        @test !occursin(r"\bsplit\s*=", call)
    end
end

@testset "L-DOM-B holdout sentinel cannot change consumed discovery domain" begin
    model, params, term = _m2c_probe_models()
    set = _m2a_nine_ic_set(23)
    ids = [set.experiments[i] for i in 1:9]
    obs = [set.experiments[i].observations for i in 1:9]
    split0 = unique_claim_experiment_split(set)
    @test split0.train_indices === UNIQUE_CLAIM_TRAIN_INDICES === (1, 2, 3, 4, 5, 6, 7)
    @test split0.holdout_indices === UNIQUE_CLAIM_HOLDOUT_INDICES === (8, 9)
    @test length(split0.train) == 7
    @test length(split0.holdout) == 2
    r0 = _m2c_consumed_discovery_range(set, term, model, params)
    @test r0 == collect(_regulator_grid(split0.train, term))
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test all(set.experiments[i].observations === obs[i] for i in 1:9)

    _m2c_apply_holdout_sentinel!(set, term)
    split1 = unique_claim_experiment_split(set)
    r1 = _m2c_consumed_discovery_range(set, term, model, params)
    @test r1 == r0
    @test r1 == collect(_regulator_grid(split1.train, term))
    r_full = collect(_regulator_grid(set, term))
    @test r_full != collect(_regulator_grid(split1.train, term))
    @test r_full != r1
    @test r_full != r0
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test all(set.experiments[i].observations === obs[i] for i in 1:9)
    @test !any(==( _M2C_HOLDOUT_SENTINEL),
        reduce(vcat, (exp.observations[term.regulator, :] for exp in split1.train)))
    @test all(==( _M2C_HOLDOUT_SENTINEL),
        reduce(vcat, (exp.observations[term.regulator, :] for exp in split1.holdout)))
end

@testset "L-DOM-B train sentinel must change consumed discovery domain" begin
    model, params, term = _m2c_probe_models()
    baseline = _m2a_nine_ic_set(23)
    r0 = _m2c_consumed_discovery_range(baseline, term, model, params)
    set = _m2a_nine_ic_set(23)
    _m2c_apply_train_sentinel!(set, term)
    split = unique_claim_experiment_split(set)
    r2 = _m2c_consumed_discovery_range(set, term, model, params)
    @test r2 != r0
    @test r2 == collect(_regulator_grid(split.train, term))
    holdout_vals = reduce(vcat,
        (exp.observations[term.regulator, :] for exp in split.holdout))
    baseline_holdout = unique_claim_experiment_split(baseline)
    @test holdout_vals == reduce(vcat,
        (exp.observations[term.regulator, :] for exp in baseline_holdout.holdout))
end

@testset "L-SET-INTACT M2-C domain path leaves the original nine experiments" begin
    model, params, term = _m2c_probe_models()
    set = _m2a_nine_ic_set(29)
    vec_before = set.experiments
    ids = [set.experiments[i] for i in 1:9]
    obs_before = [set.experiments[i].observations for i in 1:9]
    times_before = [set.experiments[i].times for i in 1:9]
    meta_before = deepcopy(set.metadata)
    r0 = _m2c_consumed_discovery_range(set, term, model, params)
    @test r0 !== nothing
    @test set.experiments === vec_before
    @test length(set.experiments) == 9
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test all(set.experiments[i].observations === obs_before[i] for i in 1:9)
    @test all(set.experiments[i].times === times_before[i] for i in 1:9)
    @test set.metadata == meta_before
    @test !hasfield(ExperimentSet, :train)
    @test !hasfield(ExperimentSet, :holdout)
    @test !haskey(set.metadata, :train)
    @test !haskey(set.metadata, :holdout)
    split = unique_claim_experiment_split(set)
    for i in 1:7
        @test split.train[i] === set.experiments[i]
        @test split.train[i].observations === set.experiments[i].observations
    end
    for i in 1:2
        @test split.holdout[i] === set.experiments[7 + i]
        @test split.holdout[i].observations === set.experiments[7 + i].observations
    end
end

@testset "L-RNG / L-DISC-B M2-C does not regenerate or evaluate holdout" begin
    helper = _m2c_rate_recovery_body()
    @test occursin("return _evaluate_unknown_rate_recovery(", helper) ||
          occursin("_evaluate_unknown_rate_recovery(", helper)
    @test !occursin("generate_recovery_experiments(", helper)
    @test !occursin("generate_experiment_set(", helper)
    @test !occursin("generate_data(", helper)
    @test !occursin("evaluate_holdout(", helper)
    @test !occursin("HoldoutEvidence", helper)
    @test !occursin("d_rmse_holdout", helper)
    @test !occursin("data_residual_holdout", helper)
    @test !occursin("_unique_claim_external_regulator_band", helper)
    composer = _m2c_composer_body()
    @test !occursin("evaluate_holdout(", composer)
    @test !occursin("HoldoutEvidence", composer)
    @test !occursin("discover_equations(", composer)
    @test !occursin("discover_unknown_destruction(", composer)
    train_body = _m2a_train_unknown_edge_body()
    @test occursin("return fit, set", train_body)
    @test !occursin("return fit, split", train_body)
    for section in (:ude_discovery, :mm_unknown)
        body = recovery_suite_section_body(section)
        @test !occursin("generate_recovery_experiments(", body)
        @test !occursin("generate_experiment_set(", body)
        @test !occursin("generate_data(", body)
        @test !occursin("evaluate_holdout(", body)
        @test !occursin("HoldoutEvidence", body)
        @test !occursin("d_rmse_holdout", body)
    end
end

@testset "L-FIELDS HoldoutEvidence surface is exactly four scalars" begin
    @test fieldnames(HoldoutEvidence) == (
        :data_residual_train, :data_residual_holdout,
        :d_rmse_holdout, :d_rmse_holdout_domain)
    @test length(fieldnames(HoldoutEvidence)) == 4
    for name in _M2D_FORBIDDEN_M34
        @test name ∉ fieldnames(HoldoutEvidence)
    end
    @test :data_residual ∉ fieldnames(HoldoutEvidence)
    @test :success ∉ fieldnames(HoldoutEvidence)
    @test :discovery ∉ fieldnames(HoldoutEvidence)
end

@testset "L-API M2-D names stay unexported" begin
    for name in _M2D_INTERNAL_NAMES
        @test isdefined(BioDynaX, name)
        @test !(name in names(BioDynaX))
        @test !(name in LOCKED_PUBLIC_EXPORTS)
    end
    @test public_export_list_holds()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
end

@testset "L-DISC-A evaluate_holdout has no discovery/training/generation path" begin
    body = _m2d_evaluate_holdout_body()
    @test occursin("function evaluate_holdout", body)
    @test occursin("_holdout_observed_regulators(split.holdout, term)", body)
    @test occursin("_unique_claim_external_regulator_band(split.train, term)", body)
    @test occursin("r_range = r_holdout", body)
    @test occursin("r_range = r_band_external", body)
    @test occursin("fill_value = 0.3", body)
    @test occursin("_mean_hybrid_residual", body)
    @test occursin("_finite_rate_rel_rmse", body)
    @test occursin("split.train.experiments", body)
    @test occursin("split.holdout.experiments", body)
    for token in _M2D_FORBIDDEN_DISCOVERY
        @test !occursin(token, body)
    end
    for token in _M2D_FORBIDDEN_TRAINING
        @test !occursin(token, body)
    end
    for token in _M2D_FORBIDDEN_GENERATE
        @test !occursin(token, body)
    end
    @test !occursin("evaled.success", body)
    @test !occursin("discovery.success", body)
    @test !occursin("RECOVERY_THRESHOLDS", body)
    @test !occursin("> 0.30", body)
    @test !occursin("<= 0.30", body)
    bodies = _m2a_reachable_split_bodies("evaluate_holdout")
    @test haskey(bodies, "evaluate_holdout")
    @test haskey(bodies, "_mean_hybrid_residual")
    @test haskey(bodies, "_holdout_observed_regulators")
    @test haskey(bodies, "_unique_claim_external_regulator_band")
    @test haskey(bodies, "_finite_rate_rel_rmse")
    for (_, helper) in bodies
        for token in _M2D_FORBIDDEN_DISCOVERY
            @test !occursin(token, helper)
        end
        for token in _M2D_FORBIDDEN_TRAINING
            @test !occursin(token, helper)
        end
        for token in _M2D_FORBIDDEN_GENERATE
            @test !occursin(token, helper)
        end
        for mutator in _M2A_FORBIDDEN_MUTATORS
            @test !occursin(mutator * "(", helper)
        end
        @test !occursin(r"\.experiments\s*\[[^\]]+\]\s*=", helper)
    end
    src = _m2d_pipeline_source()
    @test count("discover_equations(", src) == 0
    @test count("discover_unknown_rate(", src) == 0
    @test count("discover_unknown_destruction(", src) == 0
    @test count("_peek_holdout", src) == 0
    @test count("evaluate_holdout(", src) == 1
end

@testset "L-DISC-B-2 composer discovery graph does not take holdout" begin
    bodies = _m2a_reachable_split_bodies("_evaluate_unknown_rate_recovery")
    @test haskey(bodies, "_evaluate_unknown_rate_recovery")
    for (_, helper) in bodies
        @test !occursin("ExperimentSplit", helper)
        @test !occursin("HoldoutEvidence", helper)
        @test !occursin("evaluate_holdout", helper)
        @test !occursin("split.holdout", helper)
        @test !occursin(".holdout", helper)
        @test !occursin("holdout=", helper)
        @test !occursin("split=", helper)
        @test !occursin("discover_unknown_destruction(", helper)
        @test !occursin("discover_equations(", helper)
    end
end

@testset "L-SET-INTACT evaluate_holdout call graph does not mutate the set" begin
    bodies = _m2a_reachable_split_bodies("evaluate_holdout")
    for (_, helper) in bodies
        for mutator in _M2A_FORBIDDEN_MUTATORS
            @test !occursin(mutator * "(", helper)
        end
        @test !occursin(r"\.experiments\s*\[[^\]]+\]\s*=", helper)
    end
end

@testset "L-SPLIT-ID / HOLDOUT IDENTITY evaluate_holdout uses original IC 8 and 9" begin
    model, params, term = _m2c_probe_models()
    set = _m2a_nine_ic_set(41)
    ids = [set.experiments[i] for i in 1:9]
    vec_before = set.experiments
    for i in 1:9
        set.experiments[i].metadata[:probe] = i
    end
    split = unique_claim_experiment_split(set)
    @test split.holdout[1] === set.experiments[8] === ids[8]
    @test split.holdout[2] === set.experiments[9] === ids[9]
    generated = Any[]
    ev = with_generate_recovery_experiments_observer(s -> push!(generated, s)) do
        evaluate_holdout(split, _m2d_evaled(term), model, params, term,
            r -> fill(1.0, length(r)))
    end
    @test ev isa HoldoutEvidence
    @test isempty(generated)
    @test split.holdout[1] === set.experiments[8] === ids[8]
    @test split.holdout[2] === set.experiments[9] === ids[9]
    @test split.holdout[1].observations === set.experiments[8].observations
    @test split.holdout[2].observations === set.experiments[9].observations
    @test set.experiments === vec_before
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test [split.holdout[i].metadata[:probe] for i in 1:2] == [8, 9]
end

@testset "L-RES-HOLD train/holdout residuals use exact arithmetic means" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    rhos = [_m2d_experiment_residual(model, params, term, set.experiments[i])
            for i in 1:9]
    @test all(isfinite, rhos)
    @test rhos[8] != rhos[9]
    @test rhos[1] != sum(rhos[1:7]) / 7
    ev = evaluate_holdout(split, _m2d_evaled(term), model, params, term, _m2d_unit_truth)
    @test ev.data_residual_holdout === (rhos[8] + rhos[9]) / 2
    @test ev.data_residual_train === sum(rhos[1:7]) / 7
    @test ev.data_residual_holdout != sqrt((rhos[8]^2 + rhos[9]^2) / 2)
    @test ev.data_residual_holdout != rhos[8]
    @test ev.data_residual_holdout != rhos[9]
end

@testset "L-RES-LEGACY IC[1] residual stays distinct from the train mean" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    legacy = _m2d_legacy_ic1_residual(model, params, term, set)
    evaled = _m2d_evaled(term; data_residual = legacy)
    ev = evaluate_holdout(split, evaled, model, params, term, _m2d_unit_truth)
    result = report_recovery(evaled, (; unidentifiable_edge = true))
    @test result.data_residual === legacy
    @test result.data_residual != ev.data_residual_train
    @test result.data_residual === _m2d_experiment_residual(
        model, params, term, set.experiments[1])
end

@testset "L-RES-HOLD sentinel changes only the intended residual" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    evaled = _m2d_evaled(term;
        data_residual = _m2d_legacy_ic1_residual(model, params, term, set))
    ev0 = evaluate_holdout(split, evaled, model, params, term, _m2d_unit_truth)
    legacy0 = evaled.data_residual
    set.experiments[8].observations[1, :] .+= 1.5
    ev8 = evaluate_holdout(split, evaled, model, params, term, _m2d_unit_truth)
    @test ev8.data_residual_holdout != ev0.data_residual_holdout
    @test ev8.data_residual_train === ev0.data_residual_train
    @test evaled.data_residual === legacy0
    @test _m2d_legacy_ic1_residual(model, params, term, set) === legacy0
    set.experiments[1].observations[1, :] .+= 1.5
    ev1 = evaluate_holdout(split, evaled, model, params, term, _m2d_unit_truth)
    legacy1 = _m2d_legacy_ic1_residual(model, params, term, set)
    @test legacy1 != legacy0
    @test ev1.data_residual_holdout === ev8.data_residual_holdout
end

@testset "L-D-OCC d_rmse_holdout is the returned production value" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    truth_rate = r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2)
    ev, consumed = _m2d_capture_grid_ranges() do
        evaluate_holdout(split, _m2d_evaled(term), model, params, term, truth_rate)
    end
    @test length(consumed) == 2
    r_holdout = _holdout_observed_regulators(split.holdout, term)
    @test collect(consumed[1]) == collect(r_holdout)
    @test split.holdout[1] === set.experiments[8]
    @test split.holdout[2] === set.experiments[9]
    (R, D_hat_vals, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_holdout, fill_value = 0.3)
    expected = _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
    @test ev.d_rmse_holdout === expected
end

@testset "L-BAND / L-D-OCC domain RMSE uses the train-derived band" begin
    model, params, term = _m2c_probe_models()
    set_a = _m2d_synthetic_set(; train_r_extrema = (0.50, 1.50))
    split_a = unique_claim_experiment_split(set_a)
    truth_rate = r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2)
    ev_a, consumed_a = _m2d_capture_grid_ranges() do
        evaluate_holdout(split_a, _m2d_evaled(term), model, params, term, truth_rate)
    end
    r_consumed_a = consumed_a[2]
    @test collect(r_consumed_a) == collect(range(1.65, 1.85; length = 80))
    (R_a, D_a, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_consumed_a, fill_value = 0.3)
    expected_a = _finite_rate_rel_rmse(D_a, truth_rate(vec(R_a)))
    @test ev_a.d_rmse_holdout_domain === expected_a

    set_b = _m2d_synthetic_set(; train_r_extrema = (1.0, 3.0),
        holdout_r = (1.75, 1.75))
    split_b = unique_claim_experiment_split(set_b)
    ev_b, consumed_b = _m2d_capture_grid_ranges() do
        evaluate_holdout(split_b, _m2d_evaled(term), model, params, term, truth_rate)
    end
    r_consumed_b = consumed_b[2]
    @test collect(r_consumed_b) == collect(range(3.3, 3.7; length = 80))
    @test collect(r_consumed_b) != collect(r_consumed_a)
    (R_b, D_b, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_consumed_b, fill_value = 0.3)
    expected_b = _finite_rate_rel_rmse(D_b, truth_rate(vec(R_b)))
    @test ev_b.d_rmse_holdout_domain === expected_b
end

@testset "L-BAND holdout extrema cannot change the consumed domain" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    truth_rate = _m2d_unit_truth
    ev_before, consumed_before = _m2d_capture_grid_ranges() do
        evaluate_holdout(split, _m2d_evaled(term), model, params, term, truth_rate)
    end
    _m2c_apply_holdout_sentinel!(set, term)
    ev_after, consumed_after = _m2d_capture_grid_ranges() do
        evaluate_holdout(split, _m2d_evaled(term), model, params, term, truth_rate)
    end
    @test collect(consumed_after[2]) == collect(consumed_before[2])
    @test ev_after.d_rmse_holdout_domain === ev_before.d_rmse_holdout_domain
    @test ev_after.d_rmse_holdout != ev_before.d_rmse_holdout ||
          ev_after.data_residual_holdout != ev_before.data_residual_holdout
end

@testset "L-OVERFIT memorization is visible on returned holdout D fields" begin
    model0, params, term = _m2c_probe_models()
    model = _m2d_memorization_model(model0)
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    truth_rate = r -> fill(1.0, length(r))
    ev, consumed = _m2d_capture_grid_ranges() do
        evaluate_holdout(split, _m2d_evaled(term), model, params, term, truth_rate)
    end
    r_holdout = _holdout_observed_regulators(split.holdout, term)
    r_band = consumed[2]
    r_train_grid = range(0.50, 1.50; length = 80)
    (R_h, D_h, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_holdout, fill_value = 0.3)
    (R_d, D_d, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_band, fill_value = 0.3)
    (R_t, D_t, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_train_grid, fill_value = 0.3)
    d_train = _finite_rate_rel_rmse(D_t, truth_rate(vec(R_t)))
    @test ev.d_rmse_holdout === _finite_rate_rel_rmse(D_h, truth_rate(vec(R_h)))
    @test ev.d_rmse_holdout_domain ===
          _finite_rate_rel_rmse(D_d, truth_rate(vec(R_d)))
    @test d_train < 1e-8
    @test ev.d_rmse_holdout > 0.5
    @test ev.d_rmse_holdout_domain > 0.5
    @test collect(r_holdout) == fill(1.75, length(r_holdout))
end

@testset "L-EARLY Case B still evaluates holdout without Inf-as-failure" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    evaled_b = _m2d_evaled(term; success = false, discovery = :failed)
    @test evaled_b.discovery !== nothing
    @test evaled_b.discovery.success == false
    @test evaled_b.success == false
    ev_b = evaluate_holdout(split, evaled_b, model, params, term, _m2d_unit_truth)
    @test ev_b isa HoldoutEvidence
    @test ev_b !== nothing
    @test isfinite(ev_b.data_residual_train)
    @test isfinite(ev_b.data_residual_holdout)
    @test isfinite(ev_b.d_rmse_holdout)
    @test isfinite(ev_b.d_rmse_holdout_domain)
    evaled_c = _m2d_evaled(term; success = true, discovery = :ok)
    ev_c = evaluate_holdout(split, evaled_c, model, params, term, _m2d_unit_truth)
    @test ev_b.data_residual_train === ev_c.data_residual_train
    @test ev_b.data_residual_holdout === ev_c.data_residual_holdout
    @test ev_b.d_rmse_holdout === ev_c.d_rmse_holdout
    @test ev_b.d_rmse_holdout_domain === ev_c.d_rmse_holdout_domain
end

@testset "L-GATE holdout residual above 0.30 is observational" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    set.experiments[8].observations .+= 8.0
    set.experiments[9].observations .+= 9.0
    split = unique_claim_experiment_split(set)
    rho8 = _m2d_experiment_residual(model, params, term, set.experiments[8])
    rho9 = _m2d_experiment_residual(model, params, term, set.experiments[9])
    ev = evaluate_holdout(split, _m2d_evaled(term), model, params, term, _m2d_unit_truth)
    @test ev.data_residual_holdout === (rho8 + rho9) / 2
    @test ev.data_residual_holdout > 0.30
    @test isfinite(ev.data_residual_holdout)
    @test ev isa HoldoutEvidence
end

@testset "L-DISC-A live evaluate_holdout enters no discovery or trainer" begin
    model, params, term = _m2c_probe_models()
    set = _m2d_synthetic_set()
    split = unique_claim_experiment_split(set)
    ev = with_discover_unknown_rate_observer(
            (_...) -> error("discover_unknown_rate entered")) do
        with_discover_equations_observer(
                (_...) -> error("discover_equations entered")) do
            with_fit_unknown_destruction_observer(
                    _ -> error("fit_unknown_destruction entered")) do
                with_generate_recovery_experiments_observer(
                        _ -> error("generate_recovery_experiments entered")) do
                    evaluate_holdout(
                        split, _m2d_evaled(term), model, params, term, _m2d_unit_truth)
                end
            end
        end
    end
    @test ev isa HoldoutEvidence
    @test isfinite(ev.data_residual_train)
end

@testset "L-DISC-B-3 holdout sentinel cannot change composer discovery inputs" begin
    model0, params, term = _m2c_probe_models()
    model = _m2d_memorization_model(model0)
    set = _m2d_synthetic_set()
    truth_rate = _m2d_matching_truth(model, params, term)
    function capture_discovery(current)
        captured = Any[]
        with_discover_unknown_rate_observer((R, times, D, config) -> begin
                push!(captured, (;
                    R = copy(R), times = copy(times), D = copy(D), config))
                return _m2d_dummy_discovery()
            end) do
            _unique_claim_rate_recovery(
                model, params, term, truth_rate, current;
                order = 2, family = :hill, noise_σ = 0.0,
                data_residual_fn = _ -> 0.0)
        end
        return captured
    end
    captured0 = capture_discovery(set)
    @test length(captured0) == 2
    dummy = collect(range(0.0, 1.0; length = size(captured0[1].R, 2)))
    @test captured0[1].times == dummy
    @test captured0[2].times == dummy
    @test captured0[1].config.seed == unique_claim_discovery_config().seed
    _m2c_apply_holdout_sentinel!(set, term)
    captured1 = capture_discovery(set)
    @test length(captured1) == 2
    @test captured1[1].R == captured0[1].R
    @test captured1[1].times == captured0[1].times
    @test captured1[1].D == captured0[1].D
    @test captured1[2].R == captured0[2].R
    @test captured1[2].times == captured0[2].times
    @test captured1[2].D == captured0[2].D
end

@testset "L-SET-INTACT evaluate_holdout leaves the original nine experiments" begin
    model, params, term = _m2c_probe_models()
    set = _m2a_nine_ic_set(43)
    vec_before = set.experiments
    ids = [set.experiments[i] for i in 1:9]
    obs_before = [set.experiments[i].observations for i in 1:9]
    times_before = [set.experiments[i].times for i in 1:9]
    meta_before = deepcopy(set.metadata)
    fp_before = experiment_fingerprint(set)
    split = unique_claim_experiment_split(set)
    ev = evaluate_holdout(split, _m2d_evaled(term), model, params, term, _m2d_unit_truth)
    @test ev isa HoldoutEvidence
    @test set.experiments === vec_before
    @test length(set.experiments) == 9
    @test all(set.experiments[i] === ids[i] for i in 1:9)
    @test all(set.experiments[i].observations === obs_before[i] for i in 1:9)
    @test all(set.experiments[i].times === times_before[i] for i in 1:9)
    @test set.metadata == meta_before
    @test experiment_fingerprint(set) == fp_before
    @test !hasfield(ExperimentSet, :train)
    @test !hasfield(ExperimentSet, :holdout)
    @test split.holdout[1] === set.experiments[8]
    @test split.holdout[2] === set.experiments[9]
end
