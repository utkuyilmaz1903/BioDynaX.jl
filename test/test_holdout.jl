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
    _train_unknown_edge

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

@testset "M2-A/B do not add M2-C/D holdout evaluation or change M1 locks" begin
    @test !isdefined(BioDynaX, :HoldoutEvidence)
    @test !isdefined(BioDynaX, :evaluate_holdout)
    @test !isdefined(BioDynaX, :_holdout_observed_regulators)
    @test !isdefined(BioDynaX, :_unique_claim_external_regulator_band)
    @test !isdefined(BioDynaX, :_finite_rate_rel_rmse)
    @test !isdefined(BioDynaX, :_mean_hybrid_residual)
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
    for section in (:ude_discovery, :mm_unknown)
        body = recovery_suite_section_body(section)
        @test occursin("_train_unknown_edge", body)
        @test occursin("_evaluate_unknown_rate_recovery(", body)
        @test occursin("report_recovery(", body)
        @test !occursin("unique_claim_experiment_split(", body)
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
