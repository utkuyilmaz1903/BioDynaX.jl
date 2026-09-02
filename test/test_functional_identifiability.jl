function _m3a_experiment(name::Symbol, regulator_samples;
        regulator::Int = 2, nstates::Int = 2)
    samples = collect(Float64, regulator_samples)
    n = length(samples)
    times = n == 1 ? [10.0] : collect(range(10.0, 10.0 + (n - 1); length = n))
    observations = zeros(Float64, nstates, n)
    observations[1, :] .= 0.25
    observations[regulator, :] .= samples
    return Experiment(name, times, observations, observations[:, 1])
end

function _m3a_split(train_exps, hold_exps; state_names = [:S, :R])
    return ExperimentSplit(
        UNIQUE_CLAIM_TRAIN_INDICES,
        UNIQUE_CLAIM_HOLDOUT_INDICES,
        ExperimentSet(collect(train_exps), state_names),
        ExperimentSet(collect(hold_exps), state_names))
end

function _m3a_sentinel_split()
    return _m3a_split(
        (_m3a_experiment(:T1, [0.1]), _m3a_experiment(:T2, [0.5])),
        (_m3a_experiment(:H1, [0.1]), _m3a_experiment(:H2, [0.8])))
end

function _m3a_independent_z(split::ExperimentSplit, regulator::Integer)
    r_train = reduce(vcat, (Float64.(exp.observations[regulator, :])
                            for exp in split.train.experiments))
    r_holdout = reduce(vcat, (Float64.(exp.observations[regulator, :])
                              for exp in split.holdout.experiments))
    return vcat(r_train, r_holdout), r_train, r_holdout
end

function _m3a_independent_ls(D_i, D_j)
    vi = Float64.(D_i)
    vj = Float64.(D_j)
    denom = dot(vj, vj)
    alpha = denom > 0 ? (dot(vi, vj) / denom) : NaN
    return (;
        alpha,
        D_j_aligned = alpha .* vj,
        d_rmse_raw = rate_rel_rmse(vi, vj),
        d_rmse_scale_normalized = rate_rel_rmse(vi, alpha .* vj),
        d_correlation = let c = cor(vi, vj)
            isnan(c) ? 0.0 : Float64(c)
        end)
end

@testset "M3-A helpers stay unexported" begin
    @test :FunctionalIdentifiabilityDomain ∉ names(BioDynaX)
    @test :functional_identifiability_domain ∉ names(BioDynaX)
    @test :scale_align_destruction ∉ names(BioDynaX)
    @test :pairwise_destruction_metrics ∉ names(BioDynaX)
    @test :pairwise_trajectory_metrics ∉ names(BioDynaX)
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test :assess_functional_identifiability ∉ names(BioDynaX)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
end

@testset "DOMAIN_ORDER" begin
    split = _m3a_sentinel_split()
    regulator = 2
    z_expected, r_train, r_holdout = _m3a_independent_z(split, regulator)
    @test z_expected == [0.1, 0.5, 0.1, 0.8]
    @test r_train == [0.1, 0.5]
    @test r_holdout == [0.1, 0.8]
    domain = functional_identifiability_domain(split, regulator)
    @test domain.z == z_expected
    @test domain.z == [0.1, 0.5, 0.1, 0.8]
    @test domain.z != sort([0.1, 0.5, 0.1, 0.8])
    @test domain.z != unique([0.1, 0.5, 0.1, 0.8])
    @test domain.z != [0.1, 0.8, 0.1, 0.5]
    @test domain.n_train_points == length(r_train)
    @test domain.n_holdout_points == length(r_holdout)
    @test length(domain.z) ==
          domain.n_train_points + domain.n_holdout_points
    @test domain.regulator_index == regulator
    @test domain.fill_value == 0.3
    @test domain.construction === :train_obs_union_holdout_obs
end

@testset "DOMAIN_INDEPENDENCE" begin
    split = _m3a_sentinel_split()
    D_a = [1.0, 2.0, 3.0, 4.0]
    D_b = [9.0, -1.0, 0.5, 8.0]
    @test D_a != D_b
    domain_a = functional_identifiability_domain(split, 2)
    domain_b = functional_identifiability_domain(split, 2)
    @test domain_a.z == domain_b.z == [0.1, 0.5, 0.1, 0.8]
    @test !hasmethod(functional_identifiability_domain,
        Tuple{ExperimentSplit,Integer,AbstractVector})
    @test first(methods(functional_identifiability_domain)).nargs == 3
end

@testset "DOMAIN_RESTART_INDEPENDENCE" begin
    split = _m3a_sentinel_split()
    restarts_two = [(seed = 201, included = true), (seed = 202, included = true)]
    restarts_five = [(seed = s, included = s != 203) for s in 201:205]
    @test length(restarts_two) != length(restarts_five)
    domain_two = functional_identifiability_domain(split, 2)
    domain_five = functional_identifiability_domain(split, 2)
    @test domain_two.z == domain_five.z
    @test domain_two.z == [0.1, 0.5, 0.1, 0.8]
    @test !hasmethod(functional_identifiability_domain,
        Tuple{ExperimentSplit,Integer,AbstractVector})
    @test !hasmethod(functional_identifiability_domain,
        Tuple{ExperimentSplit,Integer,Vector})
end

@testset "LS_DIRECTION" begin
    D1 = [1.0, 2.0, 3.0]
    D2 = [2.0, 4.0, 9.0]
    expected_alpha = dot(D1, D2) / dot(D2, D2)
    @test expected_alpha == 37 / 101
    aligned = scale_align_destruction(D1, D2)
    @test aligned.alpha == expected_alpha
    @test aligned.alpha == 37 / 101
    @test aligned.D_j_aligned == expected_alpha .* D2
end

@testset "LS_NOT_REVERSED" begin
    D1 = [1.0, 2.0, 3.0]
    D2 = [2.0, 4.0, 9.0]
    reversed = dot(D1, D2) / dot(D1, D1)
    @test reversed == 37 / 14
    aligned = scale_align_destruction(D1, D2)
    @test aligned.alpha != reversed
    @test aligned.alpha != 37 / 14
end

@testset "LS_NOT_MAXABS" begin
    D1 = [1.0, 2.0, 3.0]
    D2 = [2.0, 4.0, 9.0]
    maxabs = maximum(abs, D1) / maximum(abs, D2)
    @test maxabs == 3 / 9
    aligned = scale_align_destruction(D1, D2)
    @test aligned.alpha != maxabs
    @test aligned.alpha != 3 / 9
end

@testset "ZERO_DENOMINATOR" begin
    D1 = [1.0, 2.0, 3.0]
    D2 = zeros(3)
    expected = _m3a_independent_ls(D1, D2)
    @test expected.alpha === NaN
    aligned = scale_align_destruction(D1, D2)
    @test aligned.alpha === NaN
    @test all(isnan, aligned.D_j_aligned)
    metrics = pairwise_destruction_metrics(D1, D2)
    @test metrics.scale_alpha === NaN
    @test metrics.d_rmse_scale_normalized === NaN
    @test metrics.d_rmse_scale_normalized === expected.d_rmse_scale_normalized
end

@testset "COMMON_DOMAIN" begin
    split = _m3a_sentinel_split()
    domain = functional_identifiability_domain(split, 2)
    z_expected, _, _ = _m3a_independent_z(split, 2)
    @test domain.z == z_expected
    D_i = [1.0, 2.0, 1.0, 3.0]
    D_j = [2.0, 4.0, 2.0, 7.0]
    @test length(D_i) == length(D_j) == length(domain.z)
    expected = _m3a_independent_ls(D_i, D_j)
    metrics = pairwise_destruction_metrics(D_i, D_j)
    @test metrics.scale_alpha == expected.alpha
    @test metrics.d_rmse_raw == expected.d_rmse_raw
    @test metrics.d_rmse_scale_normalized == expected.d_rmse_scale_normalized
    @test metrics.d_correlation == expected.d_correlation
    @test_throws DimensionMismatch pairwise_destruction_metrics(D_i, D_j[1:3])
    @test_throws DimensionMismatch scale_align_destruction(D_i, D_j[1:2])
    @test !hasmethod(pairwise_destruction_metrics,
        Tuple{ExperimentSplit,AbstractVector,AbstractVector})
    @test !hasmethod(pairwise_destruction_metrics,
        Tuple{FunctionalIdentifiabilityDomain,AbstractVector,AbstractVector})
end

@testset "SCALE_ONLY_CHANGE" begin
    D1 = [1.0, 2.0, 3.0]
    D2 = 2 .* D1
    expected = _m3a_independent_ls(D1, D2)
    @test expected.alpha == 0.5
    @test expected.d_rmse_scale_normalized == 0
    @test expected.d_rmse_raw > 0
    metrics = pairwise_destruction_metrics(D1, D2)
    @test metrics.scale_alpha == expected.alpha
    @test metrics.d_rmse_scale_normalized == expected.d_rmse_scale_normalized
    @test metrics.d_rmse_scale_normalized ≈ 0 atol = 1e-15
    @test metrics.d_rmse_raw == expected.d_rmse_raw
    @test metrics.d_rmse_raw > 0
end

@testset "SHAPE_CHANGE" begin
    D1 = [1.0, 2.0, 3.0]
    D2 = [2.0, 4.0, 9.0]
    expected = _m3a_independent_ls(D1, D2)
    @test expected.alpha == 37 / 101
    @test expected.d_rmse_scale_normalized > 0
    metrics = pairwise_destruction_metrics(D1, D2)
    @test metrics.scale_alpha == expected.alpha
    @test metrics.d_rmse_scale_normalized == expected.d_rmse_scale_normalized
    @test metrics.d_rmse_scale_normalized > 0
    @test metrics.d_rmse_raw == expected.d_rmse_raw
end

@testset "TRAJECTORY_METRIC" begin
    pred_i_train = [
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2),
        reshape([0.5, 1.5], 2, 1)]
    pred_j_train = [
        reshape([1.1, 1.8, 3.2, 3.9], 2, 2),
        reshape([0.4, 1.7], 2, 1)]
    pred_i_holdout = [reshape([2.0, 3.0, 4.0], 3, 1)]
    pred_j_holdout = [reshape([2.5, 2.5, 4.5], 3, 1)]
    expected_train = mean((
        rate_rel_rmse(vec(pred_i_train[1]), vec(pred_j_train[1])),
        rate_rel_rmse(vec(pred_i_train[2]), vec(pred_j_train[2]))))
    expected_holdout = mean((
        rate_rel_rmse(vec(pred_i_holdout[1]), vec(pred_j_holdout[1])),))
    got = pairwise_trajectory_metrics(
        pred_i_train, pred_j_train, pred_i_holdout, pred_j_holdout)
    @test got.traj_rmse_train == expected_train
    @test got.traj_rmse_holdout == expected_holdout
    single = pairwise_trajectory_metrics(
        pred_i_train[1], pred_j_train[1],
        pred_i_holdout[1], pred_j_holdout[1])
    @test single.traj_rmse_train ==
          rate_rel_rmse(vec(pred_i_train[1]), vec(pred_j_train[1]))
    @test single.traj_rmse_holdout == expected_holdout
end

@testset "M3-A source stays a pure metric layer" begin
    src = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    forbidden = (
        "evaluate_holdout",
        "discover_unknown_rate",
        "discover_equations",
        "run_recovery_suite",
        "RECOVERY_THRESHOLDS",
        "_regulator_grid",
        "_unique_claim_external_regulator_band",
        "range(0.05, 2.0",
        "range(0.0, 1.0")
    for token in forbidden
        @test !occursin(token, src)
    end
    @test occursin("function functional_identifiability_domain", src)
    @test occursin("function scale_align_destruction", src)
    @test occursin("function pairwise_destruction_metrics", src)
    @test occursin("function pairwise_trajectory_metrics", src)
    @test occursin("rate_rel_rmse", src)
end

function _m3b_protocol_split(; holdout_obs_scale = 1.0)
    ics = [
        [0.25, 0.20], [0.80, 0.35], [0.40, 1.10], [1.20, 0.70], [0.15, 0.90],
        [0.50, 0.15], [0.90, 1.50], [0.20, 0.50], [1.50, 1.20],
    ]
    experiments = map(enumerate(ics)) do (i, u0)
        times = [0.0, 0.4]
        obs = zeros(2, 2)
        obs[:, 1] .= u0
        scale = i <= 7 ? 1.0 : holdout_obs_scale
        obs[1, 2] = u0[1] * (0.9 * scale)
        obs[2, 2] = u0[2] * (1.1 * scale)
        Experiment(Symbol(:E, i), times, obs, copy(u0))
    end
    return unique_claim_experiment_split(ExperimentSet(experiments, [:S, :R]))
end

function _m3b_shift_params(p0, seed)
    shifted = deepcopy(p0)
    δ = 0.013 * Float64(seed)
    nn = shifted.nn
    @inbounds for i in eachindex(nn)
        nn[i] = nn[i] + δ
    end
    return shifted
end

function _m3b_fake_fit(params, retcode = BioDynaX.Success)
    return TrainingResult(
        params, Float64[], 1.0, 0.4,
        RunMetadata(seed = 0),
        (;),
        retcode === BioDynaX.Success, retcode)
end

function _m3b_independent_p0_fingerprint(seed, ude_net)
    _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
    return nn_parameter_fingerprint(p0.nn), p0
end

function _m3b_holdout_optimal_adam(_split)
    # Test-only decoy. Production must not call this.
    return 50
end

function _m3b_run_restarts(split, ude_net;
        throw_seed = nothing,
        predict_throw_seed = nothing,
        retcode = BioDynaX.Success)
    entries = Any[]
    fit_results = Any[]
    samples = Any[]
    predicts = Any[]
    holdout_events = Any[]
    order = Symbol[]
    fit_calls = Ref(0)
    seed_attempts = Dict{Int,Int}()
    predict_throw_fp = predict_throw_seed === nothing ? nothing :
        nn_parameter_fingerprint(_m3b_shift_params(
            last(_m3b_independent_p0_fingerprint(predict_throw_seed, ude_net)),
            predict_throw_seed).nn)
    result = with_fit_unknown_destruction_entry_observer(obs -> begin
            push!(order, :fit_entry)
            push!(entries, obs)
        end) do
        with_fit_unknown_destruction_observer(set -> begin
                fit_calls[] += 1
                k = fit_calls[]
                seed = k <= length(FUNCTIONAL_ID_RESTART_SEEDS) ?
                    FUNCTIONAL_ID_RESTART_SEEDS[k] : 1000 + k
                seed_attempts[seed] = get(seed_attempts, seed, 0) + 1
                if throw_seed === seed
                    error("injected fit failure for seed $seed")
                end
                _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
                fit = _m3b_fake_fit(_m3b_shift_params(p0, seed), retcode)
                push!(fit_results, fit)
                return fit
            end) do
            with_sample_unknown_destruction_result_observer(obs -> begin
                    push!(order, :sample)
                    push!(samples, obs)
                end) do
                with_predict_ude_observer(obs -> begin
                        push!(order, :predict)
                        if predict_throw_fp !== nothing &&
                                nn_parameter_fingerprint(obs.params.nn) ==
                                predict_throw_fp
                            error("injected predict failure for seed $(predict_throw_seed)")
                        end
                        push!(predicts, obs)
                    end) do
                    with_evaluate_holdout_observer((args...) -> begin
                            push!(order, :holdout)
                            push!(holdout_events, args)
                            return nothing
                        end) do
                        train_functional_identifiability_restarts(split, ude_net)
                    end
                end
            end
        end
    end
    return (;
        result, entries, fit_results, samples, predicts, holdout_events,
        order, fit_calls = fit_calls[], seed_attempts)
end

@testset "M3-B helpers stay unexported" begin
    for name in (
            :FUNCTIONAL_ID_RESTART_SEEDS,
            :FUNCTIONAL_ID_TRAINING_CONFIG,
            :FunctionalIdentifiabilityRestart,
            :nn_parameter_fingerprint,
            :train_functional_identifiability_restarts,
            :fit_functional_identifiability_restart,
            :with_fit_unknown_destruction_entry_observer,
            :with_sample_unknown_destruction_result_observer,
            :with_predict_ude_observer,
            :FIT_UNKNOWN_DESTRUCTION_ENTRY_OBSERVER,
            :SAMPLE_UNKNOWN_DESTRUCTION_RESULT_OBSERVER,
            :PREDICT_UDE_OBSERVER)
        @test isdefined(BioDynaX, name)
        @test name ∉ names(BioDynaX)
    end
    @test :assess_functional_identifiability ∉ names(BioDynaX)
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
end

@testset "T-B-SEEDS" begin
    @test FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
    @test length(FUNCTIONAL_ID_RESTART_SEEDS) == 5
    @test 103 ∉ FUNCTIONAL_ID_RESTART_SEEDS
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    @test_throws ArgumentError train_functional_identifiability_restarts(
        split, ude_net; restart_seeds = (201, 202, 203))
    @test_throws ArgumentError train_functional_identifiability_restarts(
        split, ude_net; restart_seeds = (201, 202, 203, 204, 103))
    @test_throws ArgumentError train_functional_identifiability_restarts(
        split, ude_net; restart_seeds = (205, 204, 203, 202, 201))
end

@testset "T-B-P0 T-B-INIT T-B-TRAIN T-B-ONEFIT T-B-HP" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3b_run_restarts(split, ude_net)
    @test live.fit_calls == 5
    @test length(live.entries) == 5
    @test live.result.n_attempted == 5
    @test live.result.restart_seeds === (201, 202, 203, 204, 205)
    expected_p0 = Dict{Int,UInt64}()
    for seed in FUNCTIONAL_ID_RESTART_SEEDS
        expected_p0[seed] = first(_m3b_independent_p0_fingerprint(seed, ude_net))
    end
    @test expected_p0[201] != expected_p0[202]
    live_p0 = UInt64[]
    for (k, seed) in enumerate(FUNCTIONAL_ID_RESTART_SEEDS)
        entry = live.entries[k]
        live_fp = nn_parameter_fingerprint(entry.p0.nn)
        push!(live_p0, live_fp)
        @test live_fp == expected_p0[seed]
        @test entry.fit_set === split.train
        @test entry.fit_set_length == 7
        @test length(entry.fit_set) == 7
        @test length(entry.fit_experiments_identity) == 7
        @test all(entry.fit_experiments_identity[i] ===
                  split.train.experiments[i] for i in 1:7)
        @test all(entry.fit_set[i] === split.train[i] for i in 1:7)
        holdout_ids = objectid.(split.holdout.experiments)
        @test all(objectid(entry.fit_set[i]) ∉ holdout_ids for i in 1:7)
        @test entry.adam == 100
        @test entry.bfgs == 50
        @test entry.frozen_phys == Symbol[]
        @test entry.phys_init === nothing
        @test entry.adam != _m3b_holdout_optimal_adam(split)
    end
    @test live_p0[1] != live_p0[2]
    @test length(unique(live_p0)) == 5
    @test !(:holdout in live.order)
    @test isempty(live.holdout_events)
    @test count(==(:fit_entry), live.order) == 5
    @test findfirst(==(:holdout), live.order) === nothing
end

@testset "T-B-HP-SENTINEL holdout cannot choose training config" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split_a = _m3b_protocol_split()
    split_b = _m3b_protocol_split(; holdout_obs_scale = 1e6)
    @test split_a.holdout.experiments[1].observations !=
          split_b.holdout.experiments[1].observations
    live_a = _m3b_run_restarts(split_a, ude_net)
    live_b = _m3b_run_restarts(split_b, ude_net)
    frozen = (100, 50, Symbol[], nothing)
    for live in (live_a, live_b)
        @test length(live.entries) == 5
        @test isempty(live.holdout_events)
        @test !(:holdout in live.order)
        for entry in live.entries
            @test (entry.adam, entry.bfgs, entry.frozen_phys, entry.phys_init) ==
                  frozen
            @test entry.adam != _m3b_holdout_optimal_adam(live.result.domain)
        end
    end
    configs_a = [(e.adam, e.bfgs, e.frozen_phys, e.phys_init)
                 for e in live_a.entries]
    configs_b = [(e.adam, e.bfgs, e.frozen_phys, e.phys_init)
                 for e in live_b.entries]
    @test configs_a == configs_b
    src = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    @test !occursin("evaluate_holdout", src)
    @test !occursin("for adam", src)
    @test !occursin("for bfgs", src)
    @test !occursin("_select_training_by_holdout", src)
end

@testset "T-B-PARAMS final params are restart-specific" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3b_run_restarts(split, ude_net)
    @test length(live.fit_results) == 5
    @test length(live.result.raw) == 5
    @test length(live.samples) == 5
    live_final = UInt64[]
    spy_final = UInt64[]
    sample_final = UInt64[]
    live_D = Vector{Float64}[]
    independent_D = Vector{Float64}[]
    z_expected, _, _ = _m3a_independent_z(split, 2)
    for (k, seed) in enumerate(FUNCTIONAL_ID_RESTART_SEEDS)
        fit = live.fit_results[k]
        raw = live.result.raw[k]
        sample = live.samples[k]
        @test raw.seed == seed
        @test raw.attempt_count == 1
        @test raw.fit === fit
        spy_fp = nn_parameter_fingerprint(fit.params.nn)
        live_fp = nn_parameter_fingerprint(raw.params.nn)
        sample_fp = nn_parameter_fingerprint(sample.params.nn)
        push!(spy_final, spy_fp)
        push!(live_final, live_fp)
        push!(sample_final, sample_fp)
        @test live_fp == spy_fp
        @test sample_fp == spy_fp
        @test sample.params_nn_fingerprint == spy_fp
        @test collect(sample.r_range) == z_expected
        @test collect(sample.r_range) == live.result.domain.z
        term = only_unknown_destruction(raw.model)
        _, D_matrix, _ = sample_unknown_destruction_grid(
            raw.model, fit.params, term;
            r_range = z_expected, fill_value = 0.3)
        push!(live_D, vec(sample.D))
        push!(independent_D, vec(D_matrix))
        @test vec(sample.D) == vec(D_matrix)
        @test length(vec(sample.D)) == length(z_expected)
        p0_fp = first(_m3b_independent_p0_fingerprint(seed, ude_net))
        @test live_fp != p0_fp
    end
    @test length(unique(live_final)) == 5
    @test length(unique(sample_final)) == 5
    @test length(unique(nn_parameter_fingerprint.(live_D))) == 5
    shared = deepcopy(live.fit_results[1].params)
    attack_fps = [nn_parameter_fingerprint(deepcopy(shared).nn) for _ in 1:5]
    @test length(unique(attack_fps)) == 1
    @test live_final != attack_fps
    @test live_D == independent_D
end

@testset "T-B-NC NotConverged is not an automatic exclusion" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3b_run_restarts(split, ude_net; retcode = BioDynaX.NotConverged)
    @test live.result.n_attempted == 5
    @test live.result.n_successful == 5
    @test live.result.n_failed == 0
    for restart in live.result.restarts
        @test restart.training_retcode === BioDynaX.NotConverged
        @test restart.included
        @test restart.failure_reason === :none
    end
end

@testset "T-B-FAIL203 T-B-NORETRY T-B-MSG" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3b_run_restarts(split, ude_net; throw_seed = 203)
    @test live.fit_calls == 5
    @test length(live.entries) == 5
    @test live.seed_attempts[203] == 1
    @test live.result.n_attempted == 5
    @test live.result.n_successful == 4
    @test live.result.n_failed == 1
    @test live.result.n_failed == live.result.n_attempted - live.result.n_successful
    seeds = [restart.seed for restart in live.result.restarts]
    @test seeds == [201, 202, 203, 204, 205]
    @test count(==(203), seeds) == 1
    failed = live.result.restarts[3]
    @test failed.seed == 203
    @test !failed.included
    @test failed.failure_reason === :fit_threw
    @test !isempty(failed.message)
    @test occursin("203", failed.message)
    for restart in live.result.restarts
        restart.seed == 203 && continue
        @test restart.included
        @test restart.failure_reason === :none
    end
    @test live.result.raw[3].seed == 203
    @test live.result.raw[3].attempt_count == 1
end

@testset "T-B-PRED203 predict throw is isolated" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3b_run_restarts(split, ude_net; predict_throw_seed = 203)
    @test live.fit_calls == 5
    @test live.seed_attempts[203] == 1
    @test live.result.n_attempted == 5
    @test live.result.n_successful == 4
    @test live.result.n_failed == 1
    failed = live.result.restarts[3]
    @test failed.seed == 203
    @test !failed.included
    @test failed.failure_reason === :predict_threw
    @test !isempty(failed.message)
    @test occursin("203", failed.message)
    @test live.result.raw[3].params !== nothing
    @test live.result.restarts[1].included
    @test live.result.restarts[5].included
end

@testset "T-B-INC-HOLD holdout performance does not decide inclusion" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split(; holdout_obs_scale = 1e6)
    live = _m3b_run_restarts(split, ude_net)
    @test live.result.n_successful == 5
    holdout_errors = Float64[]
    for raw in live.result.raw
        errors = [sqrt(mean(abs2, vec(pred) .- vec(exp.observations)))
                  for (pred, exp) in zip(raw.pred_holdout, split.holdout.experiments)]
        push!(holdout_errors, mean(errors))
    end
    @test all(err -> err ≥ 1e3, holdout_errors)
    @test all(restart -> restart.included, live.result.restarts)
end

@testset "T-B-ZLIVE domain is the M3-A observed union" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    live = _m3b_run_restarts(split, ude_net)
    z_expected, _, _ = _m3a_independent_z(split, 2)
    @test z_expected == [0.1, 0.5, 0.1, 0.8]
    @test live.result.domain.z == z_expected
    @test live.result.domain.z != sort(z_expected)
    @test live.result.domain.z != unique(z_expected)
    @test !isempty(live.samples)
    for sample in live.samples
        @test collect(sample.r_range) == z_expected
        @test collect(sample.r_range) != collect(range(0.05, 2.0; length = 80))
    end
end

@testset "T-B-COMPAT M3-C is internal; M2 surface stays untouched" begin
    @test isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)
    @test isdefined(BioDynaX, :assess_functional_identifiability)
    @test isdefined(BioDynaX, :FunctionalIdentifiabilityPair)
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test :assess_functional_identifiability ∉ names(BioDynaX)
    @test :FunctionalIdentifiabilityPair ∉ names(BioDynaX)
    @test :functional_identifiability ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :function_disagree ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    src = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    @test occursin("function assess_functional_identifiability", src)
    @test occursin("function assemble_functional_identifiability_diagnostic", src)
    rec = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    pipe = read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String)
    @test count("assess_functional_identifiability(", rec) == 0
    @test count("assess_functional_identifiability(", pipe) == 0
    train_body = let src = rec
        start = findfirst("function _train_unknown_edge", src)
        rest = src[first(start):end]
        nxt = findnext(r"\nfunction ", rest, 2)
        nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    end
    @test count("fit_unknown_destruction(", train_body) == 1
    @test occursin("split.train", train_body)
    @test !occursin("functional_identifiability", train_body)
end

function _m3c_length3_split()
    return _m3a_split(
        (_m3a_experiment(:T1, [0.1, 0.5]),),
        (_m3a_experiment(:H1, [0.8]),))
end

function _m3c_restart(seed, included)
    return FunctionalIdentifiabilityRestart(
        seed,
        included,
        included ? BioDynaX.Success : nothing,
        included ? :none : :fit_threw,
        included ? "" : "injected failure for seed $seed",
        UInt64(0),
        UInt64(0))
end

function _m3c_restarts(included_mask)
    return FunctionalIdentifiabilityRestart[
        _m3c_restart(seed, included)
        for (seed, included) in zip(FUNCTIONAL_ID_RESTART_SEEDS, included_mask)]
end

function _m3c_constant_pairs(included_seeds, d_scale, traj; d_raw = d_scale)
    seeds = sort(collect(Int, included_seeds))
    pairs = FunctionalIdentifiabilityPair[]
    for i in 1:(length(seeds) - 1)
        for j in (i + 1):length(seeds)
            push!(pairs, FunctionalIdentifiabilityPair(
                seeds[i], seeds[j], d_raw, d_scale, 1.0, 1.0, traj, traj))
        end
    end
    return pairs
end

function _m3c_run_assess(split, ude_net;
        throw_seeds = Int[],
        D_by_seed = nothing,
        X_value_by_seed = nothing,
        retcode = BioDynaX.Success,
        restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS,
        family::Symbol = :hill)
    entries = Any[]
    fit_results = Any[]
    samples = Any[]
    predicts = Any[]
    order = Symbol[]
    fit_calls = Ref(0)
    seed_attempts = Dict{Int,Int}()
    fp_to_seed = Dict{UInt64,Int}()
    result = with_fit_unknown_destruction_entry_observer(obs -> begin
            push!(order, :fit_entry)
            push!(entries, obs)
        end) do
        with_fit_unknown_destruction_observer(set -> begin
                fit_calls[] += 1
                k = fit_calls[]
                seed = k <= length(FUNCTIONAL_ID_RESTART_SEEDS) ?
                    FUNCTIONAL_ID_RESTART_SEEDS[k] : 1000 + k
                seed_attempts[seed] = get(seed_attempts, seed, 0) + 1
                if seed in throw_seeds
                    error("injected fit failure for seed $seed")
                end
                _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
                fit = _m3b_fake_fit(_m3b_shift_params(p0, seed), retcode)
                fp_to_seed[nn_parameter_fingerprint(fit.params.nn)] = seed
                push!(fit_results, fit)
                return fit
            end) do
            with_sample_unknown_destruction_result_observer(obs -> begin
                    push!(order, :sample)
                    if D_by_seed !== nothing
                        seed = fp_to_seed[obs.params_nn_fingerprint]
                        injected = D_by_seed[seed]
                        vec(obs.D) .= Float64.(injected)
                    end
                    push!(samples, obs)
                end) do
                with_predict_ude_observer(obs -> begin
                        push!(order, :predict)
                        if X_value_by_seed !== nothing
                            seed = fp_to_seed[nn_parameter_fingerprint(obs.params.nn)]
                            fill!(obs.X, Float64(X_value_by_seed[seed]))
                        end
                        push!(predicts, obs)
                    end) do
                    assess_functional_identifiability(
                        split, ude_net;
                        restart_seeds = restart_seeds,
                        family = family)
                end
            end
        end
    end
    return (;
        result, entries, fit_results, samples, predicts, order,
        fit_calls = fit_calls[], seed_attempts, fp_to_seed)
end

function _m3c_group_live_predictions(live, split)
    n_train = length(split.train.experiments)
    pred_train = Dict{Int,Vector}()
    pred_holdout = Dict{Int,Vector}()
    for obs in live.predicts
        seed = live.fp_to_seed[nn_parameter_fingerprint(obs.params.nn)]
        train_list = get!(Vector{Any}, pred_train, seed)
        hold_list = get!(Vector{Any}, pred_holdout, seed)
        if length(train_list) < n_train
            push!(train_list, obs.X)
        else
            push!(hold_list, obs.X)
        end
    end
    return pred_train, pred_holdout
end

function _m3c_independent_pair_metrics(D_i, D_j, pred_i_train, pred_j_train,
        pred_i_holdout, pred_j_holdout)
    dmet = _m3a_independent_ls(D_i, D_j)
    tmet = pairwise_trajectory_metrics(
        pred_i_train, pred_j_train, pred_i_holdout, pred_j_holdout)
    return merge(dmet, tmet)
end

function _m3c_independent_flags(diag)
    n_successful = count(restart -> restart.included, diag.restarts)
    complete = diag.n_attempted == 5 && n_successful >= 3
    scale_vals = [pair.d_rmse_scale_normalized for pair in diag.pairs]
    traj_train = [pair.traj_rmse_train for pair in diag.pairs]
    traj_hold = [pair.traj_rmse_holdout for pair in diag.pairs]
    med_scale = isempty(scale_vals) ? NaN : median(scale_vals)
    med_train = isempty(traj_train) ? NaN : median(traj_train)
    med_hold = isempty(traj_hold) ? NaN : median(traj_hold)
    function_disagree = complete && n_successful >= 2 && med_scale >= 0.20
    trajectory_agree = complete && med_train <= 0.05 && med_hold <= 0.05
    return (;
        complete,
        n_successful,
        function_disagree,
        trajectory_agree,
        trajectory_agree_function_disagree = trajectory_agree && function_disagree)
end

@testset "T-C-FIELDS locked surfaces stay closed" begin
    @test fieldnames(FunctionalIdentifiabilityRestart) === (
        :seed, :included, :training_retcode, :failure_reason, :message,
        :nn_init_fingerprint, :nn_final_fingerprint)
    @test fieldnames(FunctionalIdentifiabilityPair) === (
        :seed_i, :seed_j, :d_rmse_raw, :d_rmse_scale_normalized,
        :d_correlation, :scale_alpha, :traj_rmse_train, :traj_rmse_holdout)
    @test fieldnames(FunctionalIdentifiabilityDiagnostic) === (
        :family, :restart_seeds, :n_attempted, :n_successful, :n_failed,
        :complete, :domain, :restarts, :pairs, :median_d_rmse_raw,
        :median_d_rmse_scale_normalized, :median_d_correlation,
        :median_traj_rmse_train, :median_traj_rmse_holdout,
        :trajectory_agree, :function_disagree,
        :trajectory_agree_function_disagree, :status,
        :practical_not_structural)
    for name in (:success, :passed, :holdout, :payload, :misc, :extra,
                 :uncertainty, :hypothesis, :occupancy, :q4, :q7)
        @test name ∉ fieldnames(FunctionalIdentifiabilityRestart)
        @test name ∉ fieldnames(FunctionalIdentifiabilityPair)
        @test name ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
    end
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test :FunctionalIdentifiabilityPair ∉ names(BioDynaX)
    @test :assemble_functional_identifiability_diagnostic ∉ names(BioDynaX)
    @test :assess_functional_identifiability ∉ names(BioDynaX)
    @test :FUNCTIONAL_ID_REPORTING_CUTOFFS ∉ names(BioDynaX)
    @test FUNCTIONAL_ID_REPORTING_CUTOFFS === (
        min_successful_restarts = 3,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
    @test FUNCTIONAL_ID_STATUS_VOCABULARY === (
        :incomplete, :traj_disagree, :scale_ambiguity, :function_agree,
        :trajectory_agree_function_disagree)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
end

@testset "T-C-LEN wrong seed tuple raises" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    @test_throws ArgumentError assess_functional_identifiability(
        split, ude_net; restart_seeds = (201, 202, 203))
    @test_throws ArgumentError assess_functional_identifiability(
        split, ude_net; restart_seeds = (201, 202, 203, 204, 103))
    @test_throws ArgumentError assess_functional_identifiability(
        split, ude_net; restart_seeds = (205, 204, 203, 202, 201))
    @test_throws MethodError assess_functional_identifiability(
        split, ude_net; function_disagree = false)
    @test_throws MethodError assess_functional_identifiability(
        split, ude_net; trajectory_agree = true)
    @test_throws MethodError assess_functional_identifiability(
        split, ude_net; status = :function_agree)
    @test_throws MethodError assess_functional_identifiability(
        split, ude_net; complete = true)
    @test_throws MethodError assess_functional_identifiability(
        split, ude_net; truth_rate = x -> 0.0)
end

@testset "T-C-ACCT T-C-BIN T-C-IJ T-C-ZSAME T-C-ZIMM T-C-COMP live assess" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    z_expected, _, _ = _m3a_independent_z(split, 2)
    @test z_expected == [0.1, 0.5, 0.1, 0.8]
    cases = (
        (Int[203, 204, 205], 2, false),
        (Int[204, 205], 3, true),
        (Int[205], 4, true),
        (Int[], 5, true))
    domains = Vector{Float64}[]
    for (throw_seeds, n_successful, complete) in cases
        live = _m3c_run_assess(split, ude_net; throw_seeds = throw_seeds)
        diag = live.result
        @test diag isa FunctionalIdentifiabilityDiagnostic
        @test diag.restart_seeds === (201, 202, 203, 204, 205)
        @test diag.n_attempted == 5
        @test live.fit_calls == 5
        @test diag.n_successful == n_successful
        @test diag.n_successful == count(restart -> restart.included, diag.restarts)
        @test diag.n_failed == diag.n_attempted - diag.n_successful
        @test length(diag.restarts) == 5
        @test [restart.seed for restart in diag.restarts] == [201, 202, 203, 204, 205]
        @test diag.complete === complete
        @test diag.complete === (diag.n_attempted == 5 && diag.n_successful >= 3)
        @test length(diag.pairs) == binomial(n_successful, 2)
        included = [restart.seed for restart in diag.restarts if restart.included]
        expected_keys = Set{Tuple{Int,Int}}()
        for i in 1:(length(included) - 1)
            for j in (i + 1):length(included)
                seed_i, seed_j = included[i], included[j]
                seed_i < seed_j && push!(expected_keys, (seed_i, seed_j))
            end
        end
        keys = [(pair.seed_i, pair.seed_j) for pair in diag.pairs]
        @test Set(keys) == expected_keys
        @test all(pair -> pair.seed_i < pair.seed_j, diag.pairs)
        @test all(pair -> pair.seed_i != pair.seed_j, diag.pairs)
        @test length(unique(keys)) == length(keys)
        @test !any(pair -> (pair.seed_j, pair.seed_i) in keys, diag.pairs)
        @test diag.domain.z == z_expected
        @test diag.domain.z != sort(z_expected)
        @test diag.domain.z != unique(z_expected)
        push!(domains, copy(diag.domain.z))
        @test live.seed_attempts[201] == 1
    end
    @test all(z -> z == z_expected, domains)
    @test domains[1] == domains[end]
end

@testset "T-C-DBIND T-C-DSOURCE T-C-TBIND live pairs bind to sampled D and X" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    live = _m3c_run_assess(split, ude_net)
    diag = live.result
    @test diag.n_successful == 5
    @test length(diag.pairs) == 10
    D_by_seed = Dict{Int,Vector{Float64}}()
    independent_D = Dict{Int,Vector{Float64}}()
    z_expected = live.result.domain.z
    model_by_seed = Dict{Int,Any}()
    for obs in live.predicts
        seed = live.fp_to_seed[nn_parameter_fingerprint(obs.params.nn)]
        model_by_seed[seed] = obs.model
    end
    for sample in live.samples
        seed = live.fp_to_seed[sample.params_nn_fingerprint]
        D_by_seed[seed] = vec(Float64.(sample.D))
        raw_fit = live.fit_results[findfirst(==(seed),
            collect(FUNCTIONAL_ID_RESTART_SEEDS))]
        model = model_by_seed[seed]
        term = only_unknown_destruction(model)
        _, D_matrix, _ = sample_unknown_destruction_grid(
            model, raw_fit.params, term;
            r_range = z_expected, fill_value = 0.3)
        independent_D[seed] = vec(D_matrix)
        @test D_by_seed[seed] == independent_D[seed]
        @test length(D_by_seed[seed]) == length(z_expected)
    end
    pred_train, pred_holdout = _m3c_group_live_predictions(live, split)
    for pair in diag.pairs
        expected = _m3c_independent_pair_metrics(
            D_by_seed[pair.seed_i], D_by_seed[pair.seed_j],
            pred_train[pair.seed_i], pred_train[pair.seed_j],
            pred_holdout[pair.seed_i], pred_holdout[pair.seed_j])
        @test pair.scale_alpha == expected.alpha
        @test pair.d_rmse_raw == expected.d_rmse_raw
        @test pair.d_rmse_scale_normalized == expected.d_rmse_scale_normalized
        @test pair.d_correlation == expected.d_correlation
        @test pair.traj_rmse_train == expected.traj_rmse_train
        @test pair.traj_rmse_holdout == expected.traj_rmse_holdout
    end
    flags = _m3c_independent_flags(diag)
    @test diag.function_disagree === flags.function_disagree
    @test diag.trajectory_agree === flags.trajectory_agree
    @test diag.trajectory_agree_function_disagree ===
          flags.trajectory_agree_function_disagree
    @test diag.trajectory_agree !== diag.function_disagree ||
          diag.function_disagree === flags.function_disagree
end

@testset "T-C-LS-LIVE alignment is j -> i on the live pair" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3c_length3_split()
    D1 = [1.0, 2.0, 3.0]
    D2 = [2.0, 4.0, 9.0]
    D_by_seed = Dict(
        201 => D1,
        202 => D2,
        203 => D1,
        204 => D1,
        205 => D1)
    X_value_by_seed = Dict(seed => 1.0 for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    live = _m3c_run_assess(split, ude_net;
        D_by_seed = D_by_seed, X_value_by_seed = X_value_by_seed)
    pair = only(filter(p -> p.seed_i == 201 && p.seed_j == 202, live.result.pairs))
    expected = _m3a_independent_ls(D1, D2)
    @test expected.alpha == 37 / 101
    @test pair.scale_alpha == 37 / 101
    @test pair.scale_alpha != 37 / 14
    @test pair.scale_alpha != 3 / 9
    @test pair.d_rmse_scale_normalized == expected.d_rmse_scale_normalized
    @test live.result.domain.z == [0.1, 0.5, 0.8]
end

@testset "T-C-ZERO-LIVE zero D_j stays represented as NaN" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3c_length3_split()
    D1 = [1.0, 2.0, 3.0]
    D0 = zeros(3)
    D_by_seed = Dict(
        201 => D1,
        202 => D1,
        203 => D1,
        204 => D1,
        205 => D0)
    X_value_by_seed = Dict(seed => 1.0 for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    live = _m3c_run_assess(split, ude_net;
        D_by_seed = D_by_seed, X_value_by_seed = X_value_by_seed)
    diag = live.result
    @test length(diag.pairs) == 10
    zero_pairs = [pair for pair in diag.pairs if pair.seed_j == 205]
    @test length(zero_pairs) == 4
    for pair in zero_pairs
        @test pair.scale_alpha === NaN
        @test pair.d_rmse_scale_normalized === NaN
    end
    finite_pair = only(filter(p -> p.seed_i == 201 && p.seed_j == 203, diag.pairs))
    @test isfinite(finite_pair.scale_alpha)
    @test isfinite(finite_pair.d_rmse_scale_normalized)
end

@testset "T-C-DERIVE-LIVE A/B/C through assess" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    D_near = [1.0, 2.0, 1.0, 3.0]
    D_far = [1.0, 0.0, 1.0, 0.0]
    expected_far = _m3a_independent_ls(D_near, D_far)
    @test expected_far.d_rmse_scale_normalized >= 0.20
    D_A = Dict(seed => D_near for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    D_BC = Dict(
        201 => D_near, 202 => D_far, 203 => D_near, 204 => D_far, 205 => D_near)
    X_near = Dict(seed => 1.0 for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    X_far = Dict(201 => 1.0, 202 => 10.0, 203 => 1.0, 204 => 10.0, 205 => 1.0)
    live_a = _m3c_run_assess(split, ude_net;
        D_by_seed = D_A, X_value_by_seed = X_near)
    live_b = _m3c_run_assess(split, ude_net;
        D_by_seed = D_BC, X_value_by_seed = X_near)
    live_c = _m3c_run_assess(split, ude_net;
        D_by_seed = D_BC, X_value_by_seed = X_far)
    for live in (live_a, live_b, live_c)
        @test live.result.n_attempted == 5
        @test live.result.n_successful == 5
        @test live.result.complete
        @test length(live.result.pairs) == 10
        @test live.result.domain.z == [0.1, 0.5, 0.1, 0.8]
        @test live.result.practical_not_structural
        flags = _m3c_independent_flags(live.result)
        @test live.result.function_disagree === flags.function_disagree
        @test live.result.trajectory_agree === flags.trajectory_agree
        @test live.result.status in FUNCTIONAL_ID_STATUS_VOCABULARY
        @test live.result.status !== :structurally_identifiable
        @test live.result.status !== :functionally_identifiable
        @test live.result.status !== :passed
        @test live.result.status !== :success
    end
    a = live_a.result
    b = live_b.result
    c = live_c.result
    @test a.median_d_rmse_scale_normalized < 0.20
    @test b.median_d_rmse_scale_normalized >= 0.20
    @test c.median_d_rmse_scale_normalized >= 0.20
    @test a.median_traj_rmse_train <= 0.05
    @test a.median_traj_rmse_holdout <= 0.05
    @test b.median_traj_rmse_train <= 0.05
    @test b.median_traj_rmse_holdout <= 0.05
    @test c.median_traj_rmse_train > 0.05 || c.median_traj_rmse_holdout > 0.05
    @test a.function_disagree === false
    @test a.trajectory_agree === true
    @test a.trajectory_agree_function_disagree === false
    @test a.status === :function_agree
    @test b.function_disagree === true
    @test b.trajectory_agree === true
    @test b.trajectory_agree_function_disagree === true
    @test b.status === :trajectory_agree_function_disagree
    @test c.function_disagree === true
    @test c.trajectory_agree === false
    @test c.trajectory_agree_function_disagree === false
    @test c.status === :traj_disagree
    @test b.function_disagree !== b.trajectory_agree ||
          b.function_disagree === true
    @test c.function_disagree !== c.trajectory_agree
    @test a.function_disagree !== a.trajectory_agree
    high_error = maximum(pair.d_rmse_scale_normalized for pair in b.pairs)
    @test high_error >= 0.20
    @test length(b.pairs) == binomial(5, 2)
end

@testset "T-C-ABC T-C-STAT T-C-FLAG assemble path" begin
    domain = functional_identifiability_domain(_m3a_sentinel_split(), 2)
    included5 = _m3c_restarts((true, true, true, true, true))
    included2 = _m3c_restarts((true, true, false, false, false))
    pairs_a = _m3c_constant_pairs(201:205, 0.10, 0.01)
    pairs_b = _m3c_constant_pairs(201:205, 0.25, 0.01)
    pairs_c = _m3c_constant_pairs(201:205, 0.25, 0.20)
    pairs_scale = _m3c_constant_pairs(201:205, 0.05, 0.01; d_raw = 0.40)
    pairs_inc = _m3c_constant_pairs((201, 202), 0.25, 0.01)
    a = assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_a)
    b = assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_b)
    c = assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_c)
    scale = assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_scale)
    inc = assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included2, pairs_inc)
    @test a.function_disagree === false
    @test a.trajectory_agree === true
    @test a.status === :function_agree
    @test b.function_disagree === true
    @test b.trajectory_agree === true
    @test b.trajectory_agree_function_disagree === true
    @test b.status === :trajectory_agree_function_disagree
    @test c.trajectory_agree === false
    @test c.function_disagree === true
    @test c.status === :traj_disagree
    @test scale.status === :scale_ambiguity
    @test inc.complete === false
    @test inc.function_disagree === false
    @test inc.trajectory_agree === false
    @test inc.status === :incomplete
    @test Set((a.status, b.status, c.status, scale.status, inc.status)) ==
          Set(FUNCTIONAL_ID_STATUS_VOCABULARY)
    @test_throws ArgumentError assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_b;
        function_disagree = false)
    @test_throws ArgumentError assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_b;
        function_disagree = b.trajectory_agree && false)
    @test_throws ArgumentError FunctionalIdentifiabilityDiagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_a;
        function_disagree = a.trajectory_agree)
    @test_throws ArgumentError FunctionalIdentifiabilityDiagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_c;
        trajectory_agree = true)
    @test_throws ArgumentError FunctionalIdentifiabilityDiagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_b;
        status = :structurally_identifiable)
    @test_throws ArgumentError FunctionalIdentifiabilityDiagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included2, pairs_inc;
        complete = true)
    @test_throws ArgumentError FunctionalIdentifiabilityPair(201, 201, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1)
    @test_throws ArgumentError FunctionalIdentifiabilityPair(202, 201, 0.1, 0.1, 1.0, 1.0, 0.1, 0.1)
    consistent = FunctionalIdentifiabilityDiagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5, pairs_b;
        function_disagree = true)
    @test consistent.function_disagree === true
end

@testset "T-C-NC NotConverged remains included on the assess path" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    live = _m3c_run_assess(split, ude_net; retcode = BioDynaX.NotConverged)
    @test live.result.n_attempted == 5
    @test live.result.n_successful == 5
    @test live.result.n_failed == 0
    @test live.result.complete
    for restart in live.result.restarts
        @test restart.training_retcode === BioDynaX.NotConverged
        @test restart.included
    end
end

const _M3D_REQUIRED_PHRASES = (
    "practical functional diagnostic",
    "not a structural identifiability certificate",
    "not a unique-claim gate")

const _M3D_FORBIDDEN_PHRASES = (
    "functionally identifiable",
    "structurally identifiable",
    "Bayesian credible",
    "Q4 gate",
    "certified",
    "verified",
    "success gate",
    "passed",
    "credible interval",
    "credible level")

function _m3d_function_body(src, signature)
    start = findfirst(signature, src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function _m3d_q3_q4_sections(text)
    marker = "Q4 PRACTICAL FUNCTIONAL DIAGNOSTIC"
    idx = findfirst(marker, text)
    idx === nothing && return text, ""
    return text[1:(first(idx) - 1)], text[first(idx):end]
end

function _m3d_assert_required_and_forbidden(text)
    for phrase in _M3D_REQUIRED_PHRASES
        @test occursin(phrase, text)
    end
    for phrase in _M3D_FORBIDDEN_PHRASES
        @test !occursin(phrase, text)
    end
end

function _m3d_assert_diag_rows(text, diag)
    @test count("seed_i=", text) == length(diag.pairs)
    failed = [restart for restart in diag.restarts if !restart.included]
    @test count("included=false", text) == length(failed)
    for pair in diag.pairs
        @test occursin("seed_i=$(pair.seed_i)", text)
        @test occursin("seed_j=$(pair.seed_j)", text)
        @test occursin(
            "d_rmse_scale_normalized=$(pair.d_rmse_scale_normalized)", text)
        @test occursin("traj_rmse_train=$(pair.traj_rmse_train)", text)
        @test occursin("traj_rmse_holdout=$(pair.traj_rmse_holdout)", text)
        @test occursin("d_rmse_raw=$(pair.d_rmse_raw)", text)
        @test occursin("d_correlation=$(pair.d_correlation)", text)
        @test occursin("scale_alpha=$(pair.scale_alpha)", text)
    end
    for restart in diag.restarts
        @test occursin("seed=$(restart.seed)", text)
        @test occursin("included=$(restart.included)", text)
        @test occursin("failure_reason=$(restart.failure_reason)", text)
        if !restart.included
            @test restart.included === false
            @test !isempty(restart.message)
            @test occursin(
                "seed=$(restart.seed) included=false", text)
            @test occursin(restart.message, text)
        end
    end
    @test occursin("status: $(diag.status)", text)
    @test occursin("complete: $(diag.complete)", text)
    @test occursin(
        "median_d_rmse_scale_normalized: $(diag.median_d_rmse_scale_normalized)",
        text)
    @test occursin("median_traj_rmse_train: $(diag.median_traj_rmse_train)",
        text)
end

function _m3d_distinctive_pairs()
    seeds = collect(Int, FUNCTIONAL_ID_RESTART_SEEDS)
    pairs = FunctionalIdentifiabilityPair[]
    k = 0
    for i in 1:(length(seeds) - 1)
        for j in (i + 1):length(seeds)
            k += 1
            push!(pairs, FunctionalIdentifiabilityPair(
                seeds[i], seeds[j],
                0.02 * k, 0.01 * k + 0.001 * i, 0.5 + 0.01 * k,
                1.0 + 0.1 * k, 0.003 * k, 0.004 * k))
        end
    end
    return pairs
end

function _m3d_outlier_pairs()
    seeds = collect(Int, FUNCTIONAL_ID_RESTART_SEEDS)
    pairs = FunctionalIdentifiabilityPair[]
    k = 0
    for i in 1:(length(seeds) - 1)
        for j in (i + 1):length(seeds)
            k += 1
            d_scale = k == 1 ? 0.91 : 0.21
            push!(pairs, FunctionalIdentifiabilityPair(
                seeds[i], seeds[j], 0.21, d_scale, 1.0, 1.0, 0.01, 0.01))
        end
    end
    return pairs
end

function _m3d_status_diagnostics()
    domain = functional_identifiability_domain(_m3a_sentinel_split(), 2)
    included5 = _m3c_restarts((true, true, true, true, true))
    included2 = _m3c_restarts((true, true, false, false, false))
    return (
        incomplete = assemble_functional_identifiability_diagnostic(
            :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included2,
            _m3c_constant_pairs((201, 202), 0.25, 0.01)),
        traj_disagree = assemble_functional_identifiability_diagnostic(
            :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5,
            _m3c_constant_pairs(201:205, 0.25, 0.20)),
        scale_ambiguity = assemble_functional_identifiability_diagnostic(
            :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5,
            _m3c_constant_pairs(201:205, 0.05, 0.01; d_raw = 0.40)),
        function_agree = assemble_functional_identifiability_diagnostic(
            :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5,
            _m3c_constant_pairs(201:205, 0.10, 0.01)),
        trajectory_agree_function_disagree = assemble_functional_identifiability_diagnostic(
            :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5,
            _m3c_constant_pairs(201:205, 0.25, 0.01)))
end

@testset "T-D-SRC formatters stay internal and off the M2 surface" begin
    @test isdefined(BioDynaX, :format_functional_identifiability_diagnostic)
    @test isdefined(BioDynaX, :format_q3_q4_side_by_side)
    @test :format_functional_identifiability_diagnostic ∉ names(BioDynaX)
    @test :format_q3_q4_side_by_side ∉ names(BioDynaX)
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test :functional_identifiability ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test LOCKED_PUBLIC_EXPORTS === BioDynaX.LOCKED_PUBLIC_EXPORTS
    fi = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    rec = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    pipe = read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String)
    uc = read(joinpath(@__DIR__, "..", "src", "UniqueClaim.jl"), String)
    @test occursin("function format_functional_identifiability_diagnostic", fi)
    @test occursin("function format_q3_q4_side_by_side", fi)
    for src in (rec, pipe, uc)
        @test !occursin("format_functional_identifiability_diagnostic", src)
        @test !occursin("format_q3_q4_side_by_side", src)
        @test !occursin("assess_functional_identifiability", src)
    end
    fmt_body = _m3d_function_body(rec, "function format_protocol_result(ident;")
    rep_body = _m3d_function_body(pipe, "function report_recovery(evaled, ident;")
    hold_body = _m3d_function_body(rec, "function unique_claim_kpis_hold(kpis)")
    @test !isempty(fmt_body)
    @test !isempty(rep_body)
    @test !isempty(hold_body)
    @test !occursin("format_functional", fmt_body)
    @test !occursin("format_q3_q4", fmt_body)
    @test !occursin("assess_functional", fmt_body)
    @test !occursin("format_functional", rep_body)
    @test !occursin("format_q3_q4", rep_body)
    @test !occursin("assess_functional", rep_body)
    @test !occursin("function_disagree", hold_body)
    @test !occursin("assess_functional", hold_body)
    q4_body = _m3d_function_body(fi,
        "function format_functional_identifiability_diagnostic")
    pair_body = _m3d_function_body(fi, "function _format_functional_id_pair_row")
    @test occursin("diag.pairs", q4_body)
    @test occursin("diag.restarts", q4_body)
    @test !occursin("pairwise_destruction_metrics", q4_body)
    @test !occursin("pairwise_trajectory_metrics", q4_body)
    @test !occursin("assemble_functional_identifiability_diagnostic", q4_body)
    @test !occursin("rate_rel_rmse", q4_body)
    @test occursin("pair.d_rmse_scale_normalized", pair_body)
    @test !occursin("pairwise_destruction_metrics", pair_body)
    @test !occursin("unique_claim_kpis_hold", q4_body)
    @test !occursin("RECOVERY_THRESHOLDS", q4_body)
end

@testset "T-D-ALL T-D-FAIL live assess rows stay visible" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    live = _m3c_run_assess(split, ude_net; throw_seeds = Int[203, 204, 205])
    diag = live.result
    @test diag.n_attempted == 5
    @test diag.n_successful == 2
    @test diag.n_failed == 3
    @test length(diag.pairs) == 1
    @test diag.status === :incomplete
    text = format_functional_identifiability_diagnostic(diag)
    _m3d_assert_required_and_forbidden(text)
    _m3d_assert_diag_rows(text, diag)
    for seed in FUNCTIONAL_ID_RESTART_SEEDS
        @test occursin("seed=$seed", text)
    end
    failed = [restart for restart in diag.restarts if !restart.included]
    @test length(failed) == 3
    for restart in failed
        @test restart.failure_reason === :fit_threw
        @test occursin("included=false", text)
        @test occursin(restart.message, text)
        @test !isempty(restart.message)
    end
    pair = only(diag.pairs)
    @test occursin("seed_i=$(pair.seed_i)", text)
    @test occursin("seed_j=$(pair.seed_j)", text)
end

@testset "T-D-FIVE T-D-MUST T-D-BAN all five statuses" begin
    diags = _m3d_status_diagnostics()
    @test diags.incomplete.status === :incomplete
    @test diags.traj_disagree.status === :traj_disagree
    @test diags.scale_ambiguity.status === :scale_ambiguity
    @test diags.function_agree.status === :function_agree
    @test diags.trajectory_agree_function_disagree.status ===
          :trajectory_agree_function_disagree
    @test Set(diag.status for diag in values(diags)) ==
          Set(FUNCTIONAL_ID_STATUS_VOCABULARY)
    for (name, diag) in pairs(diags)
        text = format_functional_identifiability_diagnostic(diag)
        _m3d_assert_required_and_forbidden(text)
        _m3d_assert_diag_rows(text, diag)
        @test occursin("status: $(diag.status)", text)
        @test name === diag.status
    end
end

@testset "T-D-NOFILT formatter does not collapse pairs to the median" begin
    domain = functional_identifiability_domain(_m3a_sentinel_split(), 2)
    included5 = _m3c_restarts((true, true, true, true, true))
    distinctive = assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5,
        _m3d_distinctive_pairs())
    outlier = assemble_functional_identifiability_diagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, included5,
        _m3d_outlier_pairs())
    @test length(distinctive.pairs) == 10
    @test any(pair -> pair.d_rmse_scale_normalized !=
                      distinctive.median_d_rmse_scale_normalized,
        distinctive.pairs)
    text = format_functional_identifiability_diagnostic(distinctive)
    _m3d_assert_diag_rows(text, distinctive)
    @test count("seed_i=", text) == 10
    @test count("median_d_rmse_scale_normalized:", text) == 1
    outlier_text = format_functional_identifiability_diagnostic(outlier)
    high = only(filter(pair -> pair.d_rmse_scale_normalized == 0.91,
        outlier.pairs))
    @test high.d_rmse_scale_normalized != outlier.median_d_rmse_scale_normalized
    @test occursin(
        "d_rmse_scale_normalized=$(high.d_rmse_scale_normalized)",
        outlier_text)
    @test occursin(
        "median_d_rmse_scale_normalized: $(outlier.median_d_rmse_scale_normalized)",
        outlier_text)
    @test count("seed_i=", outlier_text) == 10
end

@testset "T-D-LIVE-METRIC formatter follows live pair metrics" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    D_near = [1.0, 2.0, 1.0, 3.0]
    D_far = [1.0, 0.0, 1.0, 0.0]
    expected_far = _m3a_independent_ls(D_near, D_far)
    @test expected_far.d_rmse_scale_normalized >= 0.20
    D_A = Dict(seed => D_near for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    D_B = Dict(
        201 => D_near, 202 => D_far, 203 => D_near, 204 => D_far, 205 => D_near)
    X_near = Dict(seed => 1.0 for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    live_a = _m3c_run_assess(split, ude_net;
        D_by_seed = D_A, X_value_by_seed = X_near)
    live_b = _m3c_run_assess(split, ude_net;
        D_by_seed = D_B, X_value_by_seed = X_near)
    diag_a = live_a.result
    diag_b = live_b.result
    @test diag_a isa FunctionalIdentifiabilityDiagnostic
    @test diag_b isa FunctionalIdentifiabilityDiagnostic
    @test length(diag_a.pairs) == 10
    @test length(diag_b.pairs) == 10
    text_a = format_functional_identifiability_diagnostic(diag_a)
    text_b = format_functional_identifiability_diagnostic(diag_b)
    _m3d_assert_diag_rows(text_a, diag_a)
    _m3d_assert_diag_rows(text_b, diag_b)
    @test text_a != text_b
    pair_a = only(filter(p -> p.seed_i == 201 && p.seed_j == 202, diag_a.pairs))
    pair_b = only(filter(p -> p.seed_i == 201 && p.seed_j == 202, diag_b.pairs))
    @test pair_a.d_rmse_scale_normalized != pair_b.d_rmse_scale_normalized
    @test pair_b.d_rmse_scale_normalized == expected_far.d_rmse_scale_normalized
    @test occursin(
        "d_rmse_scale_normalized=$(pair_a.d_rmse_scale_normalized)", text_a)
    @test occursin(
        "d_rmse_scale_normalized=$(pair_b.d_rmse_scale_normalized)", text_b)
    @test !occursin(
        "d_rmse_scale_normalized=$(pair_b.d_rmse_scale_normalized)", text_a)
    ident = (;
        unidentifiable_edge = true,
        production_param = :k_prod,
        collinearity = 0.99,
        condition_number = 1.0e7)
    side_a = format_q3_q4_side_by_side(ident, diag_a)
    side_b = format_q3_q4_side_by_side(ident, diag_b)
    @test side_a != side_b
    @test occursin(
        "d_rmse_scale_normalized=$(pair_b.d_rmse_scale_normalized)", side_b)
end

@testset "T-D-Q3Q4 Q3 and Q4 stay separate" begin
    diags = _m3d_status_diagnostics()
    ident_true = (;
        unidentifiable_edge = true,
        production_param = :k_prod,
        collinearity = 0.99,
        condition_number = 1.0e7)
    ident_false = (;
        unidentifiable_edge = false,
        production_param = :k_prod,
        collinearity = 0.11,
        condition_number = 10.0)
    @test diags.function_agree.function_disagree === false
    @test diags.trajectory_agree_function_disagree.function_disagree === true
    cross = format_q3_q4_side_by_side(ident_true, diags.function_agree)
    opposite = format_q3_q4_side_by_side(
        ident_false, diags.trajectory_agree_function_disagree)
    _m3d_assert_required_and_forbidden(cross)
    _m3d_assert_required_and_forbidden(opposite)
    q3_cross, q4_cross = _m3d_q3_q4_sections(cross)
    q3_opp, q4_opp = _m3d_q3_q4_sections(opposite)
    @test occursin("unidentifiable_edge: true", q3_cross)
    @test occursin("practical scale warning", q3_cross)
    @test occursin("asymptotic Fisher interval", q3_cross)
    @test !occursin("function_disagree", q3_cross)
    @test occursin("function_disagree: false", q4_cross)
    @test !occursin("unidentifiable_edge", q4_cross)
    @test occursin("unidentifiable_edge: false", q3_opp)
    @test !occursin("function_disagree", q3_opp)
    @test occursin("function_disagree: true", q4_opp)
    @test !occursin("unidentifiable_edge", q4_opp)
    for text in (cross, opposite)
        @test !occursin("unidentifiable_edge => function_disagree", text)
        @test !occursin("function_disagree => unidentifiable_edge", text)
    end
    minimal = format_q3_q4_side_by_side(
        (; unidentifiable_edge = true), diags.incomplete)
    q3_min, q4_min = _m3d_q3_q4_sections(minimal)
    @test occursin("unidentifiable_edge: true", q3_min)
    @test !occursin("function_disagree", q3_min)
    @test occursin("status: incomplete", q4_min)
end

@testset "T-D-GATE formatter does not modify Q3/Q1/Q5 hold" begin
    diags = _m3d_status_diagnostics()
    hold_kpis = (;
        unidentifiable_edge = true,
        data_residual = 0.003,
        support_recall = 0.99)
    fail_edge = (;
        unidentifiable_edge = false,
        data_residual = 0.003,
        support_recall = 0.99)
    @test unique_claim_kpis_hold(hold_kpis)
    @test !unique_claim_kpis_hold(fail_edge)
    @test UNIQUE_CLAIM_KPI_FIELDS ===
          (:unidentifiable_edge, :data_residual, :support_recall)
    format_functional_identifiability_diagnostic(
        diags.trajectory_agree_function_disagree)
    format_q3_q4_side_by_side(
        (; unidentifiable_edge = false, production_param = :k_prod,
            collinearity = 0.11),
        diags.function_agree)
    @test unique_claim_kpis_hold(hold_kpis)
    @test !unique_claim_kpis_hold(fail_edge)
    @test UNIQUE_CLAIM_KPI_FIELDS ===
          (:unidentifiable_edge, :data_residual, :support_recall)
    @test recovery_thresholds_hold()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
end

@testset "T-D-M2 M2 stdout and protocol fields stay unchanged" begin
    ident = (;
        unidentifiable_edge = true,
        collinearity = 0.99,
        production_param = :k_prod)
    proto = format_protocol_result(ident; residual = 0.003)
    @test startswith(proto, "IDENTIFIABILITY")
    @test occursin("\nFIT\n", proto)
    @test occursin("\nDISCOVERY\n", proto)
    @test occursin("\nREPRODUCTION\n", proto)
    @test format_protocol_result_field_order_holds(proto)
    @test UNIQUE_CLAIM_PRODUCT_BLOCKS ===
          (:IDENTIFIABILITY, :FIT, :DISCOVERY, :REPRODUCTION)
    @test PROTOCOL_RESULT_FIELDS === (
        :unknown_holes,
        :unidentifiable_edge,
        :coefficients_are_biological_constants,
        :data_residual,
        :support_recall,
        :support_f1,
        :extras,
        :canonical_hill_from_nn,
        :claim)
    @test !occursin("practical functional diagnostic", proto)
    @test !occursin("seed_i=", proto)
    @test !occursin("function_disagree", proto)
    @test !occursin("Q4 PRACTICAL FUNCTIONAL DIAGNOSTIC", proto)
    diags = _m3d_status_diagnostics()
    q4 = format_functional_identifiability_diagnostic(diags.function_agree)
    side = format_q3_q4_side_by_side(ident, diags.function_agree)
    proto2 = format_protocol_result(ident; residual = 0.003)
    @test proto == proto2
    @test q4 != proto
    @test side != proto
    @test !occursin("IDENTIFIABILITY", q4)
    @test occursin("IDENTIFIABILITY", proto)
end

# -- M3-E live assess-path binding --------------------------------------------

function _m3e_run_assess(split, ude_net;
        throw_seeds = Int[],
        predict_throw_seed = nothing,
        D_by_seed = nothing,
        X_value_by_seed = nothing,
        retcode = BioDynaX.Success,
        restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS,
        family::Symbol = :hill)
    assess_entries = Any[]
    entries = Any[]
    fit_results = Any[]
    samples = Any[]
    predicts = Any[]
    holdout_events = Any[]
    order = Symbol[]
    fit_calls = Ref(0)
    seed_attempts = Dict{Int,Int}()
    fp_to_seed = Dict{UInt64,Int}()
    predict_throw_fp = predict_throw_seed === nothing ? nothing :
        nn_parameter_fingerprint(_m3b_shift_params(
            last(_m3b_independent_p0_fingerprint(predict_throw_seed, ude_net)),
            predict_throw_seed).nn)
    result = with_assess_functional_identifiability_observer(obs -> begin
            push!(order, :assess)
            push!(assess_entries, obs)
        end) do
        with_fit_unknown_destruction_entry_observer(obs -> begin
                push!(order, :fit_entry)
                push!(entries, obs)
            end) do
            with_fit_unknown_destruction_observer(set -> begin
                    fit_calls[] += 1
                    k = fit_calls[]
                    seed = k <= length(FUNCTIONAL_ID_RESTART_SEEDS) ?
                        FUNCTIONAL_ID_RESTART_SEEDS[k] : 1000 + k
                    seed_attempts[seed] = get(seed_attempts, seed, 0) + 1
                    if seed in throw_seeds
                        error("injected fit failure for seed $seed")
                    end
                    _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
                    fit = _m3b_fake_fit(_m3b_shift_params(p0, seed), retcode)
                    fp_to_seed[nn_parameter_fingerprint(fit.params.nn)] = seed
                    push!(fit_results, fit)
                    return fit
                end) do
                with_sample_unknown_destruction_result_observer(obs -> begin
                        push!(order, :sample)
                        if D_by_seed !== nothing
                            seed = fp_to_seed[obs.params_nn_fingerprint]
                            injected = D_by_seed[seed]
                            vec(obs.D) .= Float64.(injected)
                        end
                        push!(samples, obs)
                    end) do
                    with_predict_ude_observer(obs -> begin
                            push!(order, :predict)
                            if predict_throw_fp !== nothing &&
                                    nn_parameter_fingerprint(obs.params.nn) ==
                                    predict_throw_fp
                                error("injected predict failure for seed $(predict_throw_seed)")
                            end
                            if X_value_by_seed !== nothing
                                seed = fp_to_seed[nn_parameter_fingerprint(obs.params.nn)]
                                fill!(obs.X, Float64(X_value_by_seed[seed]))
                            end
                            push!(predicts, obs)
                        end) do
                        with_evaluate_holdout_observer((args...) -> begin
                                push!(order, :holdout)
                                push!(holdout_events, args)
                                return nothing
                            end) do
                            assess_functional_identifiability(
                                split, ude_net;
                                restart_seeds = restart_seeds,
                                family = family)
                        end
                    end
                end
            end
        end
    end
    return (;
        result, assess_entries, entries, fit_results, samples, predicts,
        holdout_events, order, fit_calls = fit_calls[], seed_attempts,
        fp_to_seed)
end

function _m3e_approved_benchmark_entry(split, ude_net)
    return assess_functional_identifiability(split, ude_net)
end

function _m3e_constructor_only_benchmark(split, ude_net)
    domain = functional_identifiability_domain(
        split, _functional_id_regulator_index(ude_net))
    restarts = _m3c_restarts((true, true, true, true, true))
    pairs = _m3c_constant_pairs(201:205, 0.10, 0.01)
    return FunctionalIdentifiabilityDiagnostic(
        :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, restarts, pairs)
end

function _functional_id_regulator_index(ude_net)
    return only(neural_destruction_terms(ude_net)).regulator
end

function _m3e_normalize(src)
    return replace(src, "\r\n" => "\n")
end

function _m3e_function_body_from_src(src, name)
    needle = "function " * name * "("
    start = findfirst(needle, src)
    start === nothing && return nothing
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function _m3e_index_functions(src)
    src = _m3e_normalize(src)
    index = Dict{String,String}()
    for m in eachmatch(r"^function ([A-Za-z_][A-Za-z0-9_!]*)\("m, src)
        name = String(m.captures[1])
        body = _m3e_function_body_from_src(src, name)
        body !== nothing && (index[name] = body)
    end
    return index
end

function _m3e_callees(body)
    return [String(m.captures[1])
            for m in eachmatch(r"\b([A-Za-z_][A-Za-z0-9_!]*)\(", body)]
end

function _m3e_reachable(entry, index; stop = Set{String}())
    queue = String[entry]
    seen = Set{String}()
    bodies = Dict{String,String}()
    while !isempty(queue)
        name = popfirst!(queue)
        name in seen && continue
        push!(seen, name)
        name in stop && continue
        haskey(index, name) || continue
        body = index[name]
        bodies[name] = body
        for callee in _m3e_callees(body)
            callee in seen && continue
            callee in stop && continue
            haskey(index, callee) || continue
            push!(queue, callee)
        end
    end
    return bodies
end

const _M3E_WALK_STOP = Set((
    "build_ude_model",
    "fit_unknown_destruction",
    "sample_unknown_destruction_grid",
    "predict_ude",
    "only_unknown_destruction",
    "compile_mechanism",
    "rate_rel_rmse",
    "neural_destruction_terms"))

const _M3E_FORBIDDEN_TARGETS = (
    "discover_unknown_rate",
    "discover_equations",
    "discover_unknown_destruction",
    "run_recovery_suite",
    "_train_unknown_edge",
    "_evaluate_unknown_rate_recovery",
    "evaluate_holdout",
    "report_recovery",
    "_regulator_grid",
    "_unique_claim_external_regulator_band")

const _M3E_FORBIDDEN_TOKENS = (
    _M3E_FORBIDDEN_TARGETS...,
    "range(0.05, 2.0",
    "range(0.0, 1.0")

function _m3e_q4_source_index()
    root = joinpath(@__DIR__, "..", "src")
    src = _m3e_normalize(read(joinpath(root, "FunctionalIdentifiability.jl"), String)) *
          "\n" * _m3e_normalize(read(joinpath(root, "Recovery.jl"), String)) *
          "\n" * _m3e_normalize(read(joinpath(root, "RecoveryPipeline.jl"), String))
    return _m3e_index_functions(src)
end

function _m3e_has_token(bodies, token)
    return any(occursin(token, body) for body in values(bodies))
end

function _m3e_ast_has_call(src, name::Symbol)
    text = strip(src)
    isempty(text) && return false
    ex = try
        Meta.parse("begin\n" * text * "\nend")
    catch
        return false
    end
    found = Ref(false)
    walk = nothing
    walk = function (node)
        if node isa Expr
            if node.head === :call && !isempty(node.args)
                callee = node.args[1]
                if callee === name
                    found[] = true
                elseif callee isa Expr && callee.head === :. &&
                        length(callee.args) >= 2 &&
                        callee.args[2] === QuoteNode(name)
                    found[] = true
                end
            end
            for child in node.args
                walk(child)
            end
        end
        return nothing
    end
    walk(ex)
    return found[]
end

function _m3e_official_benchmark_path()
    return joinpath(@__DIR__, "..", "benchmark", "functional_identifiability.jl")
end

function _m3e_approved_benchmark_scripts()
    path = _m3e_official_benchmark_path()
    return isfile(path) ? [path] : String[]
end

function _m3e_is_official_benchmark_path(path)
    isfile(path) || return false
    basename(path) == "functional_identifiability.jl" || return false
    normalized = replace(abspath(path), '\\' => '/')
    return occursin("/benchmark/functional_identifiability.jl", normalized)
end

function _m3e_write_temp_benchmark(src; name = "functional_identifiability.jl")
    dir = mktempdir()
    path = joinpath(dir, name)
    write(path, src)
    return path
end

function _m3e_execute_benchmark_script(path)
    assess_n = Ref(0)
    fit_n = Ref(0)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    payload = nothing
    if isfile(path)
        payload = redirect_stdout(devnull) do
            with_assess_functional_identifiability_observer(_ ->
                    assess_n[] += 1) do
                with_fit_unknown_destruction_observer(_ -> begin
                        fit_n[] += 1
                        seed = FUNCTIONAL_ID_RESTART_SEEDS[min(fit_n[], 5)]
                        _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
                        return _m3b_fake_fit(_m3b_shift_params(p0, seed))
                    end) do
                    Base.include(Module(gensym("M3EBenchmarkExec")), path)
                end
            end
        end
    end
    result = nothing
    consumed = nothing
    if payload isa NamedTuple && haskey(payload, :diagnostic) &&
            haskey(payload, :report)
        result = payload.diagnostic
        consumed = payload.report
    elseif payload isa FunctionalIdentifiabilityDiagnostic
        result = payload
    end
    return (;
        assess_n = assess_n[],
        fit_n = fit_n[],
        result,
        consumed,
        payload)
end

function _m3e_script_live_bind(path)
    isfile(path) || return false
    src = read(path, String)
    _m3e_ast_has_call(src, :assess_functional_identifiability) || return false
    _m3e_ast_has_call(src, :format_functional_identifiability_diagnostic) ||
        return false
    live = _m3e_execute_benchmark_script(path)
    live.assess_n >= 1 || return false
    live.result isa FunctionalIdentifiabilityDiagnostic || return false
    live.consumed isa AbstractString || return false
    return live.consumed ==
           format_functional_identifiability_diagnostic(live.result)
end

function _m3e_official_discovery_binds(scripts)
    isempty(scripts) && return false
    length(scripts) == 1 || return false
    path = only(scripts)
    _m3e_is_official_benchmark_path(path) || return false
    return _m3e_script_live_bind(path)
end

@testset "M3-E seams stay unexported" begin
    for name in (
            :ASSESS_FUNCTIONAL_IDENTIFIABILITY_OBSERVER,
            :with_assess_functional_identifiability_observer,
            :assess_functional_identifiability,
            :FunctionalIdentifiabilityDiagnostic)
        @test isdefined(BioDynaX, name)
        @test name ∉ names(BioDynaX)
    end
    @test :functional_identifiability ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
end

@testset "T-E-LIVE-FIT live assess fit path" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3e_run_assess(split, ude_net)
    @test live.result isa FunctionalIdentifiabilityDiagnostic
    @test length(live.assess_entries) == 1
    @test live.assess_entries[1].split === split
    @test live.fit_calls == 5
    @test length(live.entries) == 5
    @test live.result.restart_seeds === (201, 202, 203, 204, 205)
    @test [restart.seed for restart in live.result.restarts] ==
          [201, 202, 203, 204, 205]
    @test all(==(1), values(live.seed_attempts))
    @test length(live.seed_attempts) == 5
    expected_config = (100, 50, Symbol[], nothing)
    wrong_config = (50, 50, Symbol[], nothing)
    for (k, seed) in enumerate(FUNCTIONAL_ID_RESTART_SEEDS)
        entry = live.entries[k]
        @test entry.fit_set === split.train
        @test entry.fit_set_length == 7
        @test length(entry.fit_set) == 7
        @test length(entry.fit_experiments_identity) == 7
        @test all(entry.fit_experiments_identity[i] ===
                  split.train.experiments[i] for i in 1:7)
        @test all(entry.fit_set[i] === split.train[i] for i in 1:7)
        holdout_ids = objectid.(split.holdout.experiments)
        @test all(objectid(entry.fit_set[i]) ∉ holdout_ids for i in 1:7)
        @test (entry.adam, entry.bfgs, entry.frozen_phys, entry.phys_init) ==
              expected_config
        @test (entry.adam, entry.bfgs, entry.frozen_phys, entry.phys_init) !=
              wrong_config
        @test entry.adam != _m3b_holdout_optimal_adam(split)
    end
    @test count(==(:fit_entry), live.order) == 5
    @test :assess in live.order
    @test findfirst(==(:assess), live.order) < findfirst(==(:fit_entry), live.order)
    @test :holdout ∉ live.order
    @test isempty(live.holdout_events)
    @test live.result isa FunctionalIdentifiabilityDiagnostic
    assess_from_helper = Ref(0)
    with_assess_functional_identifiability_observer(_ ->
            assess_from_helper[] += 1) do
        with_fit_unknown_destruction_observer(set -> begin
                seed = 201
                _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
                return _m3b_fake_fit(_m3b_shift_params(p0, seed))
            end) do
            train_functional_identifiability_restarts(split, ude_net)
        end
    end
    @test assess_from_helper[] == 0
end

@testset "T-E-LIVE-P0 live assess p0 fingerprints" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3e_run_assess(split, ude_net)
    @test length(live.entries) == 5
    @test length(live.assess_entries) == 1
    expected_p0 = Dict{Int,UInt64}()
    for seed in FUNCTIONAL_ID_RESTART_SEEDS
        expected_p0[seed] = nn_parameter_fingerprint(
            build_ude_model(MersenneTwister(seed), ude_net)[2].nn)
    end
    wrong_103 = nn_parameter_fingerprint(
        build_ude_model(MersenneTwister(103), ude_net)[2].nn)
    live_p0 = UInt64[]
    for (k, seed) in enumerate(FUNCTIONAL_ID_RESTART_SEEDS)
        live_fp = nn_parameter_fingerprint(live.entries[k].p0.nn)
        push!(live_p0, live_fp)
        @test live_fp == expected_p0[seed]
        @test live_fp != wrong_103
        @test live.result.restarts[k].nn_init_fingerprint == expected_p0[seed]
    end
    @test expected_p0[201] != expected_p0[202]
    @test length(unique(live_p0)) == 5
    shared = fill(expected_p0[201], 5)
    @test live_p0 != shared
end

@testset "T-E-LIVE-PARAMS live final TrainingResult.params bind D" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3e_run_assess(split, ude_net)
    @test live.result.n_successful == 5
    @test length(live.fit_results) == 5
    @test length(live.samples) == 5
    z_expected, _, _ = _m3a_independent_z(split, 2)
    live_final = UInt64[]
    live_D = Vector{Float64}[]
    independent_D = Vector{Float64}[]
    model_by_seed = Dict{Int,Any}()
    for obs in live.predicts
        seed = live.fp_to_seed[nn_parameter_fingerprint(obs.params.nn)]
        model_by_seed[seed] = obs.model
    end
    for (k, seed) in enumerate(FUNCTIONAL_ID_RESTART_SEEDS)
        fit = live.fit_results[k]
        sample = live.samples[k]
        @test live.result.restarts[k].seed == seed
        spy_fp = nn_parameter_fingerprint(fit.params.nn)
        sample_fp = nn_parameter_fingerprint(sample.params.nn)
        @test sample_fp == spy_fp
        @test sample.params_nn_fingerprint == spy_fp
        push!(live_final, spy_fp)
        model = model_by_seed[seed]
        term = only_unknown_destruction(model)
        _, D_matrix, _ = sample_unknown_destruction_grid(
            model, fit.params, term;
            r_range = collect(sample.r_range), fill_value = 0.3)
        push!(live_D, vec(sample.D))
        push!(independent_D, vec(D_matrix))
        @test vec(sample.D) == vec(D_matrix)
    end
    @test live_D == independent_D
    @test length(unique(live_final)) == 5
    shared = deepcopy(live.fit_results[1].params)
    attack_D = Vector{Float64}[]
    for seed in FUNCTIONAL_ID_RESTART_SEEDS
        model = model_by_seed[seed]
        term = only_unknown_destruction(model)
        _, D_matrix, _ = sample_unknown_destruction_grid(
            model, deepcopy(shared), term;
            r_range = z_expected, fill_value = 0.3)
        push!(attack_D, vec(D_matrix))
    end
    @test length(unique(nn_parameter_fingerprint.(attack_D))) == 1
    @test live_D != attack_D
    wrong_other = begin
        model = model_by_seed[202]
        term = only_unknown_destruction(model)
        _, D_matrix, _ = sample_unknown_destruction_grid(
            model, live.fit_results[1].params, term;
            r_range = z_expected, fill_value = 0.3)
        vec(D_matrix)
    end
    @test live_D[2] != wrong_other
end

@testset "T-E-LIVE-SAMPLE live r_range is the M3-A domain" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    live = _m3e_run_assess(split, ude_net)
    z_expected, r_train, r_holdout = _m3a_independent_z(split, 2)
    @test z_expected == [0.1, 0.5, 0.1, 0.8]
    @test r_train == [0.1, 0.5]
    @test r_holdout == [0.1, 0.8]
    @test live.result.domain.z == z_expected
    @test collect(live.result.domain.z) != sort(z_expected)
    @test collect(live.result.domain.z) != unique(z_expected)
    @test collect(live.result.domain.z) != collect(r_train)
    @test collect(live.result.domain.z) != collect(range(0.05, 2.0; length = 80))
    @test length(live.samples) == 5
    for sample in live.samples
        @test collect(sample.r_range) == z_expected
        @test collect(sample.r_range) == live.result.domain.z
        @test collect(sample.r_range) != sort(z_expected)
        @test collect(sample.r_range) != unique(z_expected)
        @test collect(sample.r_range) != collect(range(0.05, 2.0; length = 80))
    end
    live_fail = _m3e_run_assess(split, ude_net; throw_seeds = Int[203, 204, 205])
    @test live_fail.result.n_successful == 2
    @test live_fail.result.domain.z == z_expected
    @test live.result.domain.z == live_fail.result.domain.z
    for sample in live_fail.samples
        @test collect(sample.r_range) == z_expected
    end
end

@testset "T-E-LIVE-D live sample D binds pair metrics" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3c_length3_split()
    D1 = [1.0, 2.0, 3.0]
    D2 = [2.0, 4.0, 9.0]
    D0 = zeros(3)
    D_by_seed = Dict(
        201 => D1,
        202 => D2,
        203 => D1,
        204 => D1,
        205 => D0)
    X_value_by_seed = Dict(seed => 1.0 for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    live = _m3e_run_assess(split, ude_net;
        D_by_seed = D_by_seed, X_value_by_seed = X_value_by_seed)
    @test length(live.assess_entries) == 1
    @test length(live.samples) == 5
    live_D = Dict{Int,Vector{Float64}}()
    for sample in live.samples
        seed = live.fp_to_seed[sample.params_nn_fingerprint]
        live_D[seed] = vec(Float64.(sample.D))
    end
    @test live_D[201] == D1
    @test live_D[202] == D2
    @test live_D[205] == D0
    pair = only(filter(p -> p.seed_i == 201 && p.seed_j == 202, live.result.pairs))
    expected = _m3a_independent_ls(live_D[201], live_D[202])
    @test expected.alpha == 37 / 101
    @test pair.scale_alpha == 37 / 101
    @test pair.scale_alpha != 37 / 14
    @test pair.scale_alpha != 3 / 9
    @test pair.d_rmse_scale_normalized == expected.d_rmse_scale_normalized
    zero_pairs = [p for p in live.result.pairs if p.seed_j == 205]
    @test !isempty(zero_pairs)
    for zp in zero_pairs
        expected_z = _m3a_independent_ls(live_D[zp.seed_i], live_D[205])
        @test expected_z.alpha === NaN
        @test zp.scale_alpha === NaN
        @test zp.d_rmse_scale_normalized === NaN
        @test zp.d_rmse_scale_normalized === expected_z.d_rmse_scale_normalized
    end
end

@testset "T-E-LIVE-PRED live predict_ude X binds trajectory metrics" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    live = _m3e_run_assess(split, ude_net)
    @test live.result.n_successful == 5
    @test !isempty(live.predicts)
    @test all(obs -> obs.X isa AbstractArray, live.predicts)
    pred_train, pred_holdout = _m3c_group_live_predictions(live, split)
    D_by_seed = Dict{Int,Vector{Float64}}()
    for sample in live.samples
        seed = live.fp_to_seed[sample.params_nn_fingerprint]
        D_by_seed[seed] = vec(Float64.(sample.D))
    end
    manufactured = 0.123456
    for pair in live.result.pairs
        expected = _m3c_independent_pair_metrics(
            D_by_seed[pair.seed_i], D_by_seed[pair.seed_j],
            pred_train[pair.seed_i], pred_train[pair.seed_j],
            pred_holdout[pair.seed_i], pred_holdout[pair.seed_j])
        @test pair.traj_rmse_train == expected.traj_rmse_train
        @test pair.traj_rmse_holdout == expected.traj_rmse_holdout
        @test pair.traj_rmse_train != manufactured
        @test pair.traj_rmse_holdout != manufactured
    end
    swapped = pairwise_trajectory_metrics(
        pred_train[201], pred_train[205],
        pred_holdout[201], pred_holdout[205])
    pair_12 = only(filter(p -> p.seed_i == 201 && p.seed_j == 202,
        live.result.pairs))
    @test pair_12.traj_rmse_train == _m3c_independent_pair_metrics(
        D_by_seed[201], D_by_seed[202],
        pred_train[201], pred_train[202],
        pred_holdout[201], pred_holdout[202]).traj_rmse_train
    if swapped.traj_rmse_train != pair_12.traj_rmse_train
        @test pair_12.traj_rmse_train != swapped.traj_rmse_train
    end
end

@testset "T-E-LIVE-PRED203 predict throw is isolated on assess" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3b_protocol_split()
    live = _m3e_run_assess(split, ude_net; predict_throw_seed = 203)
    @test live.result isa FunctionalIdentifiabilityDiagnostic
    @test live.fit_calls == 5
    @test live.seed_attempts[203] == 1
    @test live.result.n_attempted == 5
    @test live.result.n_successful == 4
    @test live.result.n_failed == 1
    seeds = [restart.seed for restart in live.result.restarts]
    @test seeds == [201, 202, 203, 204, 205]
    @test count(==(203), seeds) == 1
    @test 206 ∉ seeds
    failed = live.result.restarts[3]
    @test failed.seed == 203
    @test !failed.included
    @test failed.failure_reason === :predict_threw
    @test !isempty(string(failed.failure_reason))
    @test !isempty(failed.message)
    for restart in live.result.restarts
        restart.seed == 203 && continue
        @test restart.included
        @test restart.failure_reason === :none
    end
    @test live.result.n_attempted == length(FUNCTIONAL_ID_RESTART_SEEDS)
    @test length(unique(seeds)) == 5
end

@testset "T-E-LIVE-FLAGS live pairs derive both disagree cells" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    D_near = [1.0, 2.0, 1.0, 3.0]
    D_far = [1.0, 0.0, 1.0, 0.0]
    @test _m3a_independent_ls(D_near, D_far).d_rmse_scale_normalized >= 0.20
    D_BC = Dict(
        201 => D_near, 202 => D_far, 203 => D_near, 204 => D_far, 205 => D_near)
    X_near = Dict(seed => 1.0 for seed in FUNCTIONAL_ID_RESTART_SEEDS)
    X_far = Dict(201 => 1.0, 202 => 10.0, 203 => 1.0, 204 => 10.0, 205 => 1.0)
    live_b = _m3e_run_assess(split, ude_net;
        D_by_seed = D_BC, X_value_by_seed = X_near)
    live_c = _m3e_run_assess(split, ude_net;
        D_by_seed = D_BC, X_value_by_seed = X_far)
    b = live_b.result
    c = live_c.result
    flags_b = _m3c_independent_flags(b)
    flags_c = _m3c_independent_flags(c)
    @test b.function_disagree === flags_b.function_disagree === true
    @test b.trajectory_agree === flags_b.trajectory_agree === true
    @test c.function_disagree === flags_c.function_disagree === true
    @test c.trajectory_agree === flags_c.trajectory_agree === false
    @test b.function_disagree !== b.trajectory_agree ||
          (b.function_disagree === true && b.trajectory_agree === true)
    @test c.function_disagree !== c.trajectory_agree
    @test b.function_disagree != b.trajectory_agree || b.trajectory_agree
    @test c.function_disagree != !c.trajectory_agree || c.function_disagree
    @test b.function_disagree !== c.trajectory_agree ||
          c.function_disagree === true
    @test !(b.function_disagree === b.trajectory_agree &&
            c.function_disagree === c.trajectory_agree)
    @test !(b.function_disagree === !b.trajectory_agree &&
            c.function_disagree === !c.trajectory_agree)
    med_b = median(pair.d_rmse_scale_normalized for pair in b.pairs)
    @test b.function_disagree === (
        b.complete && b.n_successful >= 2 && med_b >= 0.20)
    med_c = median(pair.d_rmse_scale_normalized for pair in c.pairs)
    @test c.function_disagree === (
        c.complete && c.n_successful >= 2 && med_c >= 0.20)
end

@testset "T-E-LIVE-HOLDOUT holdout does not choose restart settings" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split_a = _m3b_protocol_split()
    split_b = _m3b_protocol_split(; holdout_obs_scale = 1e6)
    @test split_a.holdout.experiments[1].observations !=
          split_b.holdout.experiments[1].observations
    live_a = _m3e_run_assess(split_a, ude_net)
    live_b = _m3e_run_assess(split_b, ude_net)
    @test live_a.result.restart_seeds === live_b.result.restart_seeds ===
          (201, 202, 203, 204, 205)
    p0_a = [nn_parameter_fingerprint(e.p0.nn) for e in live_a.entries]
    p0_b = [nn_parameter_fingerprint(e.p0.nn) for e in live_b.entries]
    @test p0_a == p0_b
    configs_a = [(e.adam, e.bfgs, e.frozen_phys, e.phys_init)
                 for e in live_a.entries]
    configs_b = [(e.adam, e.bfgs, e.frozen_phys, e.phys_init)
                 for e in live_b.entries]
    @test configs_a == configs_b
    @test all(cfg -> cfg == (100, 50, Symbol[], nothing), configs_a)
    @test all(restart -> restart.included, live_a.result.restarts)
    @test all(restart -> restart.included, live_b.result.restarts)
    @test :holdout ∉ live_a.order
    @test :holdout ∉ live_b.order
    @test isempty(live_a.holdout_events)
    @test isempty(live_b.holdout_events)
    holdout_errors = Float64[]
    pred_train, pred_holdout = _m3c_group_live_predictions(live_b, split_b)
    for seed in FUNCTIONAL_ID_RESTART_SEEDS
        errors = [sqrt(mean(abs2, vec(pred) .- vec(exp.observations)))
                  for (pred, exp) in zip(pred_holdout[seed],
            split_b.holdout.experiments)]
        push!(holdout_errors, mean(errors))
    end
    @test all(err -> err ≥ 1e3, holdout_errors)
    src = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    @test !occursin("evaluate_holdout", src)
end

@testset "T-E-LIVE-BENCHMARK approved entry calls assess live" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    assess_n = Ref(0)
    fit_n = Ref(0)
    live_diag = with_assess_functional_identifiability_observer(_ ->
            assess_n[] += 1) do
        with_fit_unknown_destruction_observer(set -> begin
                fit_n[] += 1
                seed = FUNCTIONAL_ID_RESTART_SEEDS[min(fit_n[], 5)]
                _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
                return _m3b_fake_fit(_m3b_shift_params(p0, seed))
            end) do
            _m3e_approved_benchmark_entry(split, ude_net)
        end
    end
    @test assess_n[] == 1
    @test fit_n[] == 5
    @test live_diag isa FunctionalIdentifiabilityDiagnostic
    @test live_diag.restart_seeds === (201, 202, 203, 204, 205)
    comment_only = "# assess_functional_identifiability(split, ude_net)\n"
    @test occursin("assess_functional_identifiability", comment_only)
    @test !_m3e_ast_has_call(comment_only, :assess_functional_identifiability)
    ctor_src = """
        FunctionalIdentifiabilityDiagnostic(
            :hill, (201, 202, 203, 204, 205), domain, restarts, pairs)
        """
    @test occursin("FunctionalIdentifiabilityDiagnostic", ctor_src)
    @test !_m3e_ast_has_call(ctor_src, :assess_functional_identifiability)
    @test _m3e_ast_has_call(ctor_src, :FunctionalIdentifiabilityDiagnostic)
    approved_src = "assess_functional_identifiability(split, ude_net)\n"
    @test _m3e_ast_has_call(approved_src, :assess_functional_identifiability)
    ctor_n = Ref(0)
    ctor_diag = with_assess_functional_identifiability_observer(_ ->
            ctor_n[] += 1) do
        _m3e_constructor_only_benchmark(split, ude_net)
    end
    @test ctor_n[] == 0
    @test ctor_diag isa FunctionalIdentifiabilityDiagnostic
    @test ctor_n[] != assess_n[]
    official = _m3e_official_benchmark_path()
    scripts = _m3e_approved_benchmark_scripts()
    @test isfile(official)
    @test !isempty(scripts)
    @test length(scripts) == 1
    @test only(scripts) == official
    @test _m3e_is_official_benchmark_path(only(scripts))
    @test basename(only(scripts)) == "functional_identifiability.jl"
    @test !isfile(joinpath(@__DIR__, "..", "benchmark",
        "m3_fake_functional_identifiability.jl"))
    @test !isfile(joinpath(@__DIR__, "..", "benchmark",
        "functional_identifiability_test.jl"))
    src = read(official, String)
    @test _m3e_ast_has_call(src, :assess_functional_identifiability)
    @test _m3e_ast_has_call(src, :format_functional_identifiability_diagnostic)
    @test _m3e_ast_has_call(src, :generate_recovery_experiments)
    @test _m3e_ast_has_call(src, :unique_claim_experiment_split)
    commented = join("# " .* Base.split(_m3e_normalize(src), '\n'), '\n')
    @test !_m3e_ast_has_call(commented, :assess_functional_identifiability)
    for token in _M3E_FORBIDDEN_TARGETS
        @test !occursin(token, src)
    end
    live = _m3e_execute_benchmark_script(official)
    @test live.assess_n >= 1
    @test live.fit_n == 5
    @test live.result isa FunctionalIdentifiabilityDiagnostic
    @test live.result.restart_seeds === (201, 202, 203, 204, 205)
    @test live.payload isa NamedTuple
    @test live.payload.diagnostic === live.result
    formatted = format_functional_identifiability_diagnostic(live.result)
    @test live.consumed == formatted
    @test occursin("PRACTICAL FUNCTIONAL DIAGNOSTIC", live.consumed)
    @test occursin("status: $(live.result.status)", live.consumed)
end

@testset "T-E-LIVE-BENCHMARK adversarial fakes stay red" begin
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    split = _m3a_sentinel_split()
    @test !_m3e_official_discovery_binds(String[])
    @test !_m3e_is_official_benchmark_path(joinpath(tempdir(),
        "functional_identifiability.jl"))
    @test !_m3e_script_live_bind(joinpath(tempdir(),
        "missing_functional_identifiability.jl"))
    renamed = _m3e_write_temp_benchmark(
        "assess_functional_identifiability(split, ude_net)\n";
        name = "m3_fake_functional_identifiability.jl")
    @test !_m3e_official_discovery_binds([renamed])
    comment_only = _m3e_write_temp_benchmark(
        "# assess_functional_identifiability(split, ude_net)\n" *
        "# format_functional_identifiability_diagnostic(diag)\n" *
        "println(\"comment only\")\n")
    comment_src = read(comment_only, String)
    @test occursin("assess_functional_identifiability", comment_src)
    @test !_m3e_ast_has_call(comment_src, :assess_functional_identifiability)
    comment_live = _m3e_execute_benchmark_script(comment_only)
    @test comment_live.assess_n == 0
    @test !_m3e_script_live_bind(comment_only)
    string_only = _m3e_write_temp_benchmark(
        "println(\"assess_functional_identifiability(split, ude_net)\")\n" *
        "println(\"PRACTICAL FUNCTIONAL DIAGNOSTIC\")\n")
    string_src = read(string_only, String)
    @test occursin("assess_functional_identifiability", string_src)
    @test !_m3e_ast_has_call(string_src, :assess_functional_identifiability)
    string_live = _m3e_execute_benchmark_script(string_only)
    @test string_live.assess_n == 0
    @test !_m3e_script_live_bind(string_only)
    ctor_src = """
        using BioDynaX
        using BioDynaX:
            FUNCTIONAL_ID_RESTART_SEEDS,
            FunctionalIdentifiabilityDiagnostic,
            FunctionalIdentifiabilityDomain,
            FunctionalIdentifiabilityPair,
            FunctionalIdentifiabilityRestart,
            format_functional_identifiability_diagnostic
        domain = FunctionalIdentifiabilityDomain(
            2, [0.1, 0.5, 0.1, 0.8], 2, 2, 0.3, :train_obs_union_holdout_obs)
        restarts = [
            FunctionalIdentifiabilityRestart(
                seed, true, BioDynaX.Success, :none, "", UInt64(0), UInt64(0))
            for seed in FUNCTIONAL_ID_RESTART_SEEDS]
        pairs = FunctionalIdentifiabilityPair[]
        seeds = collect(FUNCTIONAL_ID_RESTART_SEEDS)
        for i in 1:(length(seeds) - 1)
            for j in (i + 1):length(seeds)
                push!(pairs, FunctionalIdentifiabilityPair(
                    seeds[i], seeds[j], 0.01, 0.01, 1.0, 1.0, 0.01, 0.01))
            end
        end
        diag = FunctionalIdentifiabilityDiagnostic(
            :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, restarts, pairs)
        println(format_functional_identifiability_diagnostic(diag))
        diag
        """
    ctor_path = _m3e_write_temp_benchmark(ctor_src)
    @test occursin("FunctionalIdentifiabilityDiagnostic", ctor_src)
    @test _m3e_ast_has_call(ctor_src, :FunctionalIdentifiabilityDiagnostic)
    @test !_m3e_ast_has_call(ctor_src, :assess_functional_identifiability)
    ctor_live = _m3e_execute_benchmark_script(ctor_path)
    @test ctor_live.assess_n == 0
    @test ctor_live.result isa FunctionalIdentifiabilityDiagnostic
    @test !_m3e_script_live_bind(ctor_path)
    helper_n = Ref(0)
    helper_diag = with_assess_functional_identifiability_observer(_ ->
            helper_n[] += 1) do
        _m3e_constructor_only_benchmark(split, ude_net)
    end
    @test helper_n[] == 0
    @test helper_diag isa FunctionalIdentifiabilityDiagnostic
    no_assess = _m3e_write_temp_benchmark(
        "using BioDynaX\nprintln(\"executes without assess\")\n")
    no_assess_live = _m3e_execute_benchmark_script(no_assess)
    @test no_assess_live.assess_n == 0
    @test !_m3e_script_live_bind(no_assess)
end

@testset "T-E-WALK transitive assess graph has no forbidden path" begin
    dirty = """
function assess_functional_identifiability(split, ude_net)
    return _q4_helper(split, ude_net)
end
function _q4_helper(split, ude_net)
    return _forward(split)
end
function _forward(split)
    return evaluate_holdout(split, nothing, nothing, nothing, nothing, x -> 0.0)
end
"""
    dirty_bodies = _m3e_reachable(
        "assess_functional_identifiability", _m3e_index_functions(dirty))
    @test haskey(dirty_bodies, "_q4_helper")
    @test haskey(dirty_bodies, "_forward")
    @test _m3e_has_token(dirty_bodies, "evaluate_holdout(")
    assess_only = dirty_bodies["assess_functional_identifiability"]
    @test !occursin("evaluate_holdout", assess_only)
    index = _m3e_q4_source_index()
    bodies = _m3e_reachable(
        "assess_functional_identifiability", index; stop = _M3E_WALK_STOP)
    @test haskey(bodies, "assess_functional_identifiability")
    @test haskey(bodies, "train_functional_identifiability_restarts")
    @test haskey(bodies, "fit_functional_identifiability_restart")
    @test !haskey(bodies, "fit_unknown_destruction")
    @test !haskey(bodies, "sample_unknown_destruction_grid")
    @test !haskey(bodies, "evaluate_holdout")
    @test !haskey(bodies, "_train_unknown_edge")
    @test !haskey(bodies, "discover_unknown_rate")
    for token in _M3E_FORBIDDEN_TOKENS
        @test !_m3e_has_token(bodies, token)
    end
    fi = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    for token in _M3E_FORBIDDEN_TOKENS
        @test !occursin(token, fi)
    end
end

@testset "T-E-NOM4 T-E-COST T-E-THRESH T-E-TRUTH T-E-PROJ T-E-CUT" begin
    fi = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    for token in _M3E_FORBIDDEN_TOKENS
        @test !occursin(token, fi)
    end
    hard = read(joinpath(@__DIR__, "run_recovery_hard.jl"), String)
    runtests = read(joinpath(@__DIR__, "runtests.jl"), String)
    ci = read(joinpath(@__DIR__, "..", ".github", "workflows", "ci.yml"), String)
    @test !occursin("assess_functional_identifiability", hard)
    @test !occursin("assess_functional_identifiability", runtests)
    @test !occursin("functional_identifiability.jl", ci)
    @test occursin("include(\"test_functional_identifiability.jl\")", runtests)
    @test !occursin("RECOVERY_THRESHOLDS", fi)
    @test recovery_thresholds_hold()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test_throws MethodError assess_functional_identifiability(
        _m3a_sentinel_split(),
        build_hill_recovery_network(; known = false, hill_order = 2);
        truth_rate = x -> 0.0)
    @test_throws MethodError assess_functional_identifiability(
        _m3a_sentinel_split(),
        build_hill_recovery_network(; known = false, hill_order = 2);
        D_true = [1.0, 2.0])
    project = read(joinpath(@__DIR__, "..", "Project.toml"), String)
    @test !occursin("StructuralIdentifiability", project)
    @test !occursin("Turing", project)
    @test !occursin("MCMC", project)
    @test FUNCTIONAL_ID_REPORTING_CUTOFFS === (
        min_successful_restarts = 3,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
    @test FUNCTIONAL_ID_REPORTING_CUTOFFS != (
        min_successful_restarts = 2,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
end

@testset "T-E-HOLD unique_claim_kpis_hold stays Q3+Q1+Q5" begin
    hold = (;
        unidentifiable_edge = true,
        data_residual = 0.003,
        support_recall = 0.99)
    miss = (;
        unidentifiable_edge = false,
        data_residual = 0.003,
        support_recall = 0.99)
    @test UNIQUE_CLAIM_KPI_FIELDS ===
          (:unidentifiable_edge, :data_residual, :support_recall)
    @test :function_disagree ∉ UNIQUE_CLAIM_KPI_FIELDS
    @test unique_claim_kpis_hold(hold)
    @test !unique_claim_kpis_hold(miss)
    with_q4 = (;
        unidentifiable_edge = true,
        data_residual = 0.003,
        support_recall = 0.99,
        function_disagree = true)
    @test unique_claim_kpis_hold(with_q4)
    src = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    body = _m3e_function_body_from_src(src, "unique_claim_kpis_hold")
    @test occursin("unidentifiable_edge", body)
    @test occursin("data_residual", body)
    @test occursin("support_recall", body)
    @test !occursin("function_disagree", body)
    @test !occursin("assess_functional_identifiability", body)
end

@testset "T-E-M2HASH M2 lock IDs and Q4 surface stay" begin
    holdout = read(joinpath(@__DIR__, "test_holdout.jl"), String)
    recovery = read(joinpath(@__DIR__, "test_recovery_pipeline.jl"), String)
    hard = read(joinpath(@__DIR__, "test_recovery_hard.jl"), String)
    corpus = holdout * "\n" * recovery * "\n" * hard
    for id in (
            "L-SPLIT-ID", "L-SET-INTACT", "L-FIT-A", "L-FIT-B", "L-RNG",
            "L-DOM-A", "L-DOM-B", "L-D-OCC", "L-OVERFIT", "L-RES-HOLD",
            "L-RES-LEGACY", "L-DISC-A", "L-DISC-B-1", "L-DISC-B-2",
            "L-DISC-B-3", "L-EARLY", "L-GATE", "L-SITE", "L-API", "M2-G1",
            "M2-G2")
        @test occursin(id, corpus)
    end
    @test occursin("no M3 or M4 fields", holdout)
    @test occursin(":FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)",
        holdout)
    @test occursin(":FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)",
        recovery)
    @test !occursin(
        "!isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)", corpus)
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test :functional_identifiability ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :function_disagree ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
end

const _M3F_FORBIDDEN_PHRASES = (
    "Bayesian",
    "credible interval",
    "credible level",
    "posterior")

const _M3F_FORBIDDEN_FIELDS = (
    :posterior,
    :credible_interval,
    :credible_level,
    :structural_identifiability)

const _M3F_TRADEOFF_KEYS = (
    :production_param,
    :condition_number,
    :production_correlation,
    :collinearity,
    :unidentifiable_edge,
    :fisher)

const _M3F_PHASE1_FISHER = [
    1.948946020842544e13 -5.014202383474795e13 -1.1163107161215307e13
    -5.014202383474795e13 1.3015736872831088e14 2.9873326111898336e13
    -1.1163107161215307e13 2.9873326111898336e13 2.70365750557671e13]

const _M3F_PHASE1_CONDITION = 1086.662678334793
const _M3F_PHASE1_LOWER = [0.9999952004396508, 0.9999981216038665, 0.9999995560356232]
const _M3F_PHASE1_UPPER = [1.0000047995603496, 1.000001878396134, 1.0000004439643773]

function _m3f_phase1_fisher_fixture()
    rng = MersenneTwister(51)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    u0 = [0.2, 0.1]
    tspan = (0.0, 2.0)
    times, clean, _, _ = generate_data(
        rng; network = network, u0 = u0, tspan = tspan,
        n_points = 12, noise_σ = 0.0)
    return (; model, params, clean, times, u0, tspan)
end

function _m3f_independent_fisher(model, params, data, times, u0, tspan)
    jacobian, names = BioDynaX.trajectory_jacobian(
        model, params, u0, tspan, times)
    residual = data .- predict_ude(params, u0, tspan, times, model)
    σ² = max(eps(), sum(abs2, residual) / max(1, length(residual)))
    information = (jacobian' * jacobian) ./ σ²
    return (; jacobian, names, residual_variance = σ², information)
end

function _m3f_wald_interval(center, variance, z)
    half = z * sqrt(max(variance, zero(variance)))
    return (max(0.0, center - half), center + half)
end

function _m3f_pci_docstring(src)
    fn = findfirst("function parameter_credible_intervals", src)
    fn === nothing && return ""
    before = src[1:(first(fn) - 1)]
    closer = findlast("\"\"\"", before)
    closer === nothing && return ""
    opener = findprev("\"\"\"", before, first(closer) - 1)
    opener === nothing && return ""
    return before[(last(opener) + 1):(first(closer) - 1)]
end

@testset "T-F-SRC T-F-NOM T-F-PCI Fisher terminology" begin
    ident_src = read(joinpath(@__DIR__, "..", "src", "Identifiability.jl"),
        String)
    q4_src = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    for phrase in _M3F_FORBIDDEN_PHRASES
        @test !occursin(phrase, ident_src)
        @test !occursin(phrase, q4_src)
    end
    @test occursin("asymptotic Fisher interval", ident_src)
    @test occursin("nominal coverage", ident_src)
    @test occursin("This is not structural identifiability", ident_src)
    pci_doc = _m3f_pci_docstring(ident_src)
    @test !isempty(pci_doc)
    @test occursin("Asymptotic Fisher", pci_doc)
    @test occursin("nominal coverage", pci_doc)
    @test !occursin("Asymptotic normal credible intervals", pci_doc)
    @test !occursin("credible interval", pci_doc)
    @test !occursin("credible level", pci_doc)
    @test occursin("function parameter_credible_intervals", ident_src)
end

@testset "T-F-Z T-F-KEYS Fisher math, Q3 warning, Q4, exports" begin
    @test BioDynaX._z_score(0.90) == 1.6448536269514722
    @test BioDynaX._z_score(0.95) == 1.959963984540054
    @test BioDynaX._z_score(0.99) == 2.5758293035489004
    thrown = try
        BioDynaX._z_score(0.80)
        nothing
    catch err
        err
    end
    @test thrown isa ArgumentError
    @test occursin("nominal coverage", thrown.msg)
    @test !occursin("credible level", thrown.msg)
    @test !occursin("credible interval", thrown.msg)

    fixture = _m3f_phase1_fisher_fixture()
    rebuilt = _m3f_independent_fisher(
        fixture.model, fixture.params, fixture.clean, fixture.times,
        fixture.u0, fixture.tspan)
    report = assess_identifiability(
        fixture.model, fixture.params, fixture.clean, fixture.times,
        fixture.u0, fixture.tspan)
    report_again = assess_identifiability(
        fixture.model, fixture.params, fixture.clean, fixture.times,
        fixture.u0, fixture.tspan)
    @test report.parameter_names == rebuilt.names
    @test report.fisher_information == rebuilt.information
    @test report.residual_variance == rebuilt.residual_variance
    @test report.fisher_information == report_again.fisher_information
    @test report.condition_number == report_again.condition_number
    @test report.residual_variance == report_again.residual_variance
    @test report.parameter_names == [:k_ba, :k_a, :k_b]
    @test size(report.fisher_information) == (3, 3)
    @test report.fisher_information ≈ _M3F_PHASE1_FISHER rtol=1e-12 atol=0
    @test report.condition_number ≈ _M3F_PHASE1_CONDITION rtol=1e-12 atol=0
    @test report.residual_variance == eps(Float64)
    @test all(report.identifiable)

    z95 = 1.959963984540054
    variances = LinearAlgebra.diag(pinv(report.fisher_information))
    intervals = BioDynaX.parameter_credible_intervals(
        report, fixture.params; level = 0.95)
    uncertainty = estimate_parameter_uncertainty(
        fixture.model, fixture.params, fixture.clean, fixture.times,
        fixture.u0, fixture.tspan; level = 0.95)
    @test uncertainty.method === :fisher
    @test uncertainty.level == 0.95
    @test uncertainty.parameter_names == report.parameter_names
    @test uncertainty.lower ≈ _M3F_PHASE1_LOWER rtol=1e-12 atol=0
    @test uncertainty.upper ≈ _M3F_PHASE1_UPPER rtol=1e-12 atol=0
    for (i, name) in enumerate(report.parameter_names)
        center = positive_parameter(getproperty(fixture.params.phys, name))
        expected = _m3f_wald_interval(center, variances[i], z95)
        @test intervals[name] == expected
        @test uncertainty.estimates[i] == center
        @test uncertainty.lower[i] == expected[1]
        @test uncertainty.upper[i] == expected[2]
    end

    tradeoff = BioDynaX.production_destruction_tradeoff(
        fixture.model, fixture.params, fixture.clean, fixture.times,
        fixture.u0, fixture.tspan)
    @test keys(tradeoff) === _M3F_TRADEOFF_KEYS
    for name in _M3F_FORBIDDEN_FIELDS
        @test !hasproperty(tradeoff, name)
        @test name ∉ fieldnames(BioDynaX.IdentifiabilityReport)
        @test name ∉ fieldnames(BioDynaX.ParameterUncertainty)
        @test name ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
        @test name ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    end

    hit = BioDynaX.format_production_destruction_warning((;
        unidentifiable_edge = true,
        production_param = :k_prod,
        collinearity = 0.99))
    @test hit ==
          "Practical warning: production (k_prod) and unknown D(z) scale are collinear (cosine=0.99). Observed concentrations do not pin that scale. This is not structural identifiability."
    miss = BioDynaX.format_production_destruction_warning((;
        unidentifiable_edge = false,
        production_param = :k_prod,
        collinearity = 0.11))
    @test miss ==
          "Practical k_prod↔D collinearity cosine=0.11 (below threshold)."

    fid = _m3d_status_diagnostics().function_agree
    cross = format_q3_q4_side_by_side((;
            unidentifiable_edge = true,
            production_param = :k_prod,
            collinearity = 0.99,
            condition_number = 1.0e7),
        fid)
    q3, q4 = _m3d_q3_q4_sections(cross)
    @test occursin("practical scale warning", q3)
    @test occursin("asymptotic Fisher interval", q3)
    @test !occursin("function_disagree", q3)
    @test occursin("function_disagree: false", q4)
    @test !occursin("unidentifiable_edge", q4)
    @test UNIQUE_CLAIM_KPI_FIELDS ===
          (:unidentifiable_edge, :data_residual, :support_recall)
    @test :function_disagree ∉ UNIQUE_CLAIM_KPI_FIELDS
    @test unique_claim_kpis_hold((;
        unidentifiable_edge = true,
        data_residual = 0.003,
        support_recall = 0.99,
        function_disagree = true))
    @test !unique_claim_kpis_hold((;
        unidentifiable_edge = false,
        data_residual = 0.003,
        support_recall = 0.99))
    @test :parameter_credible_intervals ∉ names(BioDynaX)
    @test :parameter_credible_intervals ∉ LOCKED_PUBLIC_EXPORTS
    @test :assess_identifiability ∉ names(BioDynaX)
    @test :estimate_parameter_uncertainty ∉ names(BioDynaX)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
end

function _m3g_ast_call_count(src, name::Symbol)
    text = strip(src)
    isempty(text) && return 0
    ex = try
        Meta.parse("begin\n" * text * "\nend")
    catch
        return 0
    end
    n = Ref(0)
    walk = nothing
    walk = function (node)
        if node isa Expr
            if node.head === :call && !isempty(node.args)
                callee = node.args[1]
                if callee === name
                    n[] += 1
                elseif callee isa Expr && callee.head === :. &&
                        length(callee.args) >= 2 &&
                        callee.args[2] === QuoteNode(name)
                    n[] += 1
                end
            end
            for child in node.args
                walk(child)
            end
        end
        return nothing
    end
    walk(ex)
    return n[]
end

function _m3g_call_kwargs(src, name::Symbol)
    text = strip(src)
    isempty(text) && return Dict{Symbol,Any}[]
    ex = try
        Meta.parse("begin\n" * text * "\nend")
    catch
        return Dict{Symbol,Any}[]
    end
    found = Dict{Symbol,Any}[]
    walk = nothing
    walk = function (node)
        if node isa Expr
            if node.head === :call && !isempty(node.args)
                callee = node.args[1]
                is_target = callee === name ||
                    (callee isa Expr && callee.head === :. &&
                     length(callee.args) >= 2 &&
                     callee.args[2] === QuoteNode(name))
                if is_target
                    kwargs = Dict{Symbol,Any}()
                    for arg in node.args[2:end]
                        if arg isa Expr && arg.head === :kw &&
                                length(arg.args) >= 2 && arg.args[1] isa Symbol
                            kwargs[arg.args[1]] = arg.args[2]
                        elseif arg isa Expr && arg.head === :parameters
                            for p in arg.args
                                if p isa Expr && p.head === :kw &&
                                        length(p.args) >= 2 &&
                                        p.args[1] isa Symbol
                                    kwargs[p.args[1]] = p.args[2]
                                elseif p isa Symbol
                                    kwargs[p] = p
                                end
                            end
                        end
                    end
                    push!(found, kwargs)
                end
            end
            for child in node.args
                walk(child)
            end
        end
        return nothing
    end
    walk(ex)
    return found
end

function _m3g_execute_benchmark_script(path; predict_throw_seed = nothing)
    assess_n = Ref(0)
    assess_entries = Any[]
    fit_n = Ref(0)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    predict_throw_fp = predict_throw_seed === nothing ? nothing :
        nn_parameter_fingerprint(_m3b_shift_params(
            last(_m3b_independent_p0_fingerprint(predict_throw_seed, ude_net)),
            predict_throw_seed).nn)
    payload = nothing
    if isfile(path)
        payload = redirect_stdout(devnull) do
            with_assess_functional_identifiability_observer(entry -> begin
                    assess_n[] += 1
                    push!(assess_entries, entry)
                end) do
                with_fit_unknown_destruction_observer(_ -> begin
                        fit_n[] += 1
                        seed = FUNCTIONAL_ID_RESTART_SEEDS[min(fit_n[], 5)]
                        _, p0 = build_ude_model(MersenneTwister(seed), ude_net)
                        return _m3b_fake_fit(_m3b_shift_params(p0, seed))
                    end) do
                    with_predict_ude_observer(obs -> begin
                            if predict_throw_fp !== nothing &&
                                    nn_parameter_fingerprint(obs.params.nn) ==
                                    predict_throw_fp
                                error("injected predict failure for seed $(predict_throw_seed)")
                            end
                            return nothing
                        end) do
                        Base.include(Module(gensym("M3GBenchmarkExec")), path)
                    end
                end
            end
        end
    end
    result = nothing
    consumed = nothing
    if payload isa NamedTuple && haskey(payload, :diagnostic) &&
            haskey(payload, :report)
        result = payload.diagnostic
        consumed = payload.report
    elseif payload isa FunctionalIdentifiabilityDiagnostic
        result = payload
    end
    return (;
        assess_n = assess_n[],
        assess_entries,
        fit_n = fit_n[],
        result,
        consumed,
        payload)
end

const _M3G_M4_SEEDS = (103, 107, 111, 113, 127)
const _M3G_FORBIDDEN_PIPELINE = (
    _M3E_FORBIDDEN_TARGETS...,
    "consume_shared_suite_rng",
    "FunctionalIdentifiabilityDiagnostic(")

@testset "T-G-AST official script has a live assess Call" begin
    official = _m3e_official_benchmark_path()
    src = read(official, String)
    @test isfile(official)
    @test _m3e_ast_has_call(src, :assess_functional_identifiability)
    @test _m3e_ast_has_call(src, :format_functional_identifiability_diagnostic)
    @test _m3g_ast_call_count(src, :assess_functional_identifiability) == 1
    @test _m3g_ast_call_count(src, :format_functional_identifiability_diagnostic) ==
          1
    @test !_m3e_ast_has_call(src, :FunctionalIdentifiabilityDiagnostic)
    commented = join("# " .* Base.split(_m3e_normalize(src), '\n'), '\n')
    @test occursin("assess_functional_identifiability", commented)
    @test !_m3e_ast_has_call(commented, :assess_functional_identifiability)
    @test _m3g_ast_call_count(commented, :assess_functional_identifiability) == 0
end

@testset "T-G-CALL official script executes assess once" begin
    official = _m3e_official_benchmark_path()
    src = read(official, String)
    @test _m3e_ast_has_call(src, :assess_functional_identifiability)
    live = _m3e_execute_benchmark_script(official)
    again = _m3e_execute_benchmark_script(official)
    observed = _m3g_execute_benchmark_script(official)
    @test live.assess_n == 1
    @test again.assess_n == 1
    @test observed.assess_n == 1
    @test live.assess_n == _m3g_ast_call_count(src,
        :assess_functional_identifiability)
    @test live.fit_n == 5
    @test observed.fit_n == 5
    @test live.result isa FunctionalIdentifiabilityDiagnostic
    @test observed.result isa FunctionalIdentifiabilityDiagnostic
    @test live.payload isa NamedTuple
    @test live.payload.diagnostic === live.result
    @test length(observed.assess_entries) == 1
    @test observed.assess_entries[1].restart_seeds === FUNCTIONAL_ID_RESTART_SEEDS
    @test _m3e_script_live_bind(official)
    ctor_n = Ref(0)
    with_assess_functional_identifiability_observer(_ -> ctor_n[] += 1) do
        _m3e_constructor_only_benchmark(
            _m3a_sentinel_split(),
            build_hill_recovery_network(; known = false, hill_order = 2))
    end
    @test ctor_n[] == 0
    @test ctor_n[] != live.assess_n
end

@testset "T-G-FMT returned diagnostic is consumed" begin
    official = _m3e_official_benchmark_path()
    live = _m3e_execute_benchmark_script(official)
    @test live.result isa FunctionalIdentifiabilityDiagnostic
    @test live.consumed isa AbstractString
    formatted = format_functional_identifiability_diagnostic(live.result)
    @test live.consumed == formatted
    @test live.payload.report === live.consumed
    fake = _m3e_constructor_only_benchmark(
        _m3a_sentinel_split(),
        build_hill_recovery_network(; known = false, hill_order = 2))
    @test live.consumed != format_functional_identifiability_diagnostic(fake)
    @test occursin("PRACTICAL FUNCTIONAL DIAGNOSTIC", live.consumed)
    @test occursin("status: $(live.result.status)", live.consumed)
    @test occursin("complete: $(live.result.complete)", live.consumed)
    @test occursin("n_attempted: $(live.result.n_attempted)", live.consumed)
    @test occursin("n_successful: $(live.result.n_successful)", live.consumed)
    @test occursin("n_failed: $(live.result.n_failed)", live.consumed)
    @test occursin("RESTARTS", live.consumed)
    @test occursin("PAIRS", live.consumed)
    @test occursin("failure_reason=", live.consumed)
    @test occursin("MEDIANS", live.consumed)
    for seed in FUNCTIONAL_ID_RESTART_SEEDS
        @test occursin("seed=$seed", live.consumed)
    end
    @test live.result.n_attempted == 5
    @test live.result.n_failed ==
          live.result.n_attempted - live.result.n_successful
    @test length(live.result.restarts) == 5
end

@testset "T-G-KW generate uses protocol kwargs" begin
    official = _m3e_official_benchmark_path()
    src = read(official, String)
    @test _m3e_ast_has_call(src, :generate_recovery_experiments)
    @test _m3e_ast_has_call(src, :unique_claim_experiment_split)
    @test occursin("UNIQUE_CLAIM_PROTOCOL", src)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.seed", src)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.tspan", src)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.n_points", src)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.observation_noise", src)
    @test !occursin("protocol=", src)
    @test !occursin("protocol =", src)
    @test !occursin("consume_shared_suite_rng", src)
    calls = _m3g_call_kwargs(src, :generate_recovery_experiments)
    @test length(calls) == 1
    kwargs = only(calls)
    @test haskey(kwargs, :tspan)
    @test haskey(kwargs, :n_points)
    @test haskey(kwargs, :noise_σ)
    @test !haskey(kwargs, :protocol)
    @test UNIQUE_CLAIM_PROTOCOL.seed == 103
    @test UNIQUE_CLAIM_PROTOCOL.tspan === (0.0, 8.0)
    @test UNIQUE_CLAIM_PROTOCOL.n_points == 50
    @test UNIQUE_CLAIM_PROTOCOL.observation_noise == 0.0
    @test UNIQUE_CLAIM_PROTOCOL.n_ics == 9
end

@testset "T-G-SEEDS five locked restart seeds" begin
    official = _m3e_official_benchmark_path()
    src = read(official, String)
    live = _m3g_execute_benchmark_script(official)
    @test FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
    @test length(FUNCTIONAL_ID_RESTART_SEEDS) == 5
    @test 103 ∉ FUNCTIONAL_ID_RESTART_SEEDS
    @test UNIQUE_CLAIM_PROTOCOL.seed == 103
    @test isempty(intersect(FUNCTIONAL_ID_RESTART_SEEDS, _M3G_M4_SEEDS))
    @test occursin("FUNCTIONAL_ID_RESTART_SEEDS", src)
    @test !occursin("107, 111, 113, 127", src)
    assess_kwargs = _m3g_call_kwargs(src, :assess_functional_identifiability)
    @test length(assess_kwargs) == 1
    @test haskey(only(assess_kwargs), :restart_seeds)
    @test live.result.restart_seeds === FUNCTIONAL_ID_RESTART_SEEDS
    @test [restart.seed for restart in live.result.restarts] ==
          [201, 202, 203, 204, 205]
    @test live.assess_entries[1].restart_seeds === (201, 202, 203, 204, 205)
    @test live.fit_n == 5
    @test live.result.n_attempted == 5
end

@testset "T-G-FAIL failures stay in five-attempt accounting" begin
    official = _m3e_official_benchmark_path()
    src = read(official, String)
    live = _m3g_execute_benchmark_script(official; predict_throw_seed = 203)
    @test live.assess_n == 1
    @test live.result isa FunctionalIdentifiabilityDiagnostic
    @test live.result.n_attempted == 5
    @test length(live.result.restarts) == 5
    @test live.result.n_successful == count(r -> r.included, live.result.restarts)
    @test live.result.n_failed ==
          live.result.n_attempted - live.result.n_successful
    @test live.result.n_failed >= 1
    failed = only(filter(r -> r.seed == 203, live.result.restarts))
    @test failed.included == false
    @test failed.failure_reason === :predict_threw
    @test !isempty(failed.message)
    @test occursin("seed=203", live.consumed)
    @test occursin("predict_threw", live.consumed)
    @test occursin("n_failed: $(live.result.n_failed)", live.consumed)
    @test occursin("n_attempted: 5", live.consumed)
    @test !occursin("error(diagnostic", src)
    @test !occursin("exit(", src)
    @test !occursin("@test", src)
    for token in _M3G_FORBIDDEN_PIPELINE
        @test !occursin(token, src)
    end
end

@testset "T-G-CI benchmark is not a PR gate" begin
    official = _m3e_official_benchmark_path()
    runtests = read(joinpath(@__DIR__, "runtests.jl"), String)
    hard = read(joinpath(@__DIR__, "run_recovery_hard.jl"), String)
    workflow_dir = joinpath(@__DIR__, "..", ".github", "workflows")
    ci = read(joinpath(workflow_dir, "ci.yml"), String)
    @test occursin("include(\"test_functional_identifiability.jl\")", runtests)
    @test !occursin("assess_functional_identifiability", runtests)
    @test !occursin("benchmark/functional_identifiability.jl", runtests)
    @test !occursin("assess_functional_identifiability", hard)
    @test !occursin("functional_identifiability.jl", hard)
    @test occursin("test/run_recovery_hard.jl", ci)
    for name in readdir(workflow_dir)
        endswith(name, ".yml") || continue
        text = read(joinpath(workflow_dir, name), String)
        @test !occursin("functional_identifiability.jl", text)
        @test !occursin("assess_functional_identifiability", text)
    end
    src = read(official, String)
    @test occursin("not a PR gate", src)
    @test occursin("deferred to M7", src)
    contributing = read(joinpath(@__DIR__, "..", "CONTRIBUTING.md"), String)
    @test occursin("benchmark/functional_identifiability.jl", contributing)
    @test occursin("not a PR gate", contributing)
end

@testset "T-G-API M2 locks and public exports stay" begin
    @test :assess_functional_identifiability ∉ names(BioDynaX)
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test :FUNCTIONAL_ID_RESTART_SEEDS ∉ names(BioDynaX)
    @test :FUNCTIONAL_ID_REPORTING_CUTOFFS ∉ names(BioDynaX)
    @test :format_functional_identifiability_diagnostic ∉ names(BioDynaX)
    @test :functional_identifiability ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test public_export_list_holds()
    @test LOCKED_PUBLIC_EXPORTS === BioDynaX.LOCKED_PUBLIC_EXPORTS
    @test recovery_thresholds_hold()
    @test FUNCTIONAL_ID_REPORTING_CUTOFFS === (
        min_successful_restarts = 3,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
    holdout = read(joinpath(@__DIR__, "test_holdout.jl"), String)
    recovery = read(joinpath(@__DIR__, "test_recovery_pipeline.jl"), String)
    hard = read(joinpath(@__DIR__, "test_recovery_hard.jl"), String)
    corpus = holdout * "\n" * recovery * "\n" * hard
    @test occursin("M2-G1", corpus)
    @test occursin("M2-G2", corpus)
    @test occursin(":FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)",
        holdout)
    @test !occursin(
        "!isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)", corpus)
    official = _m3e_official_benchmark_path()
    src = read(official, String)
    for token in _M3G_FORBIDDEN_PIPELINE
        @test !occursin(token, src)
    end
end

@testset "T-G-FAKE constructor and comment scripts stay red" begin
    comment_only = _m3e_write_temp_benchmark(
        "# assess_functional_identifiability(split, ude_net)\n" *
        "# format_functional_identifiability_diagnostic(diag)\n")
    comment_live = _m3e_execute_benchmark_script(comment_only)
    @test comment_live.assess_n == 0
    @test !_m3e_script_live_bind(comment_only)
    string_only = _m3e_write_temp_benchmark(
        "println(\"assess_functional_identifiability(split, ude_net)\")\n")
    @test _m3e_execute_benchmark_script(string_only).assess_n == 0
    ctor_src = """
        using BioDynaX
        using BioDynaX:
            FUNCTIONAL_ID_RESTART_SEEDS,
            FunctionalIdentifiabilityDiagnostic,
            FunctionalIdentifiabilityDomain,
            FunctionalIdentifiabilityPair,
            FunctionalIdentifiabilityRestart,
            format_functional_identifiability_diagnostic
        domain = FunctionalIdentifiabilityDomain(
            2, [0.1, 0.5, 0.1, 0.8], 2, 2, 0.3, :train_obs_union_holdout_obs)
        restarts = [
            FunctionalIdentifiabilityRestart(
                seed, true, BioDynaX.Success, :none, "", UInt64(0), UInt64(0))
            for seed in FUNCTIONAL_ID_RESTART_SEEDS]
        pairs = FunctionalIdentifiabilityPair[]
        seeds = collect(FUNCTIONAL_ID_RESTART_SEEDS)
        for i in 1:(length(seeds) - 1)
            for j in (i + 1):length(seeds)
                push!(pairs, FunctionalIdentifiabilityPair(
                    seeds[i], seeds[j], 0.01, 0.01, 1.0, 1.0, 0.01, 0.01))
            end
        end
        diag = FunctionalIdentifiabilityDiagnostic(
            :hill, FUNCTIONAL_ID_RESTART_SEEDS, domain, restarts, pairs)
        report = format_functional_identifiability_diagnostic(diag)
        println(report)
        (; diagnostic = diag, report)
        """
    ctor_path = _m3e_write_temp_benchmark(ctor_src)
    ctor_live = _m3e_execute_benchmark_script(ctor_path)
    @test ctor_live.assess_n == 0
    @test ctor_live.result isa FunctionalIdentifiabilityDiagnostic
    @test !_m3e_script_live_bind(ctor_path)
    official = _m3e_official_benchmark_path()
    @test !_m3e_official_discovery_binds([ctor_path])
    @test _m3e_official_discovery_binds([official])
end
