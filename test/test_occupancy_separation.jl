# Live separation tests: occupancy sampling and the functional-identifiability domain.
# A1 T-A-DOMAIN-SEP means occupancy sampling uses occupancy.X.
# A2 T-A2-DOMAIN-SEP means functional-identifiability must NOT use occupancy.X.
# These IDs are intentionally opposite and must coexist.

using Test
using Random
using BioDynaX
if !@isdefined(_reference_protocol_rate_recovery)
    include(joinpath(@__DIR__, "internals.jl"))
end

const _A2_N_ICS = REFERENCE_PROTOCOL.n_ics
const _A2_N_POINTS = REFERENCE_PROTOCOL.n_points
const _A2_N_TRAIN = length(REFERENCE_PROTOCOL_TRAIN_INDICES)
const _A2_N_HOLD = length(REFERENCE_PROTOCOL_HOLDOUT_INDICES)
const _A2_TRAIN_COLS = _A2_N_TRAIN * _A2_N_POINTS
const _A2_HOLD_COLS = _A2_N_HOLD * _A2_N_POINTS
const _A2_DOMAIN_LEN = (_A2_N_TRAIN + _A2_N_HOLD) * _A2_N_POINTS
const _A2_FIXED_R = collect(range(0.05, 2.0; length = 80))
const _A2_DOMAIN_SEED = first(FUNCTIONAL_ID_RESTART_SEEDS)

function _a2_protocol_set()
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    proto = REFERENCE_PROTOCOL
    return generate_recovery_experiments(
        MersenneTwister(proto.seed), truth_net, truth;
        tspan = proto.tspan, n_points = proto.n_points,
        noise_σ = proto.observation_noise)
end

function _a2_probe_models()
    rng = MersenneTwister(1)
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    model, params = build_ude_model(rng, net)
    term = only_unknown_destruction(model)
    return model, params, term, net
end

function _a2_independent_z(split, regulator)
    r_train = reduce(
        vcat, (Float64.(exp.observations[regulator, :])
        for exp in split.train.experiments))
    r_hold = reduce(vcat,
        (Float64.(exp.observations[regulator, :])
        for exp in split.holdout.experiments))
    return vcat(r_train, r_hold)
end

function _a2_independent_holdout_r(split, term)
    return reduce(vcat,
        (Float64.(exp.observations[term.regulator, :])
        for exp in split.holdout.experiments))
end

function _a2_hill_truth()
    return r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2)
end

function _a2_matching_truth(model, params, term)
    return function (r)
        (_, D, _) = sample_unknown_destruction_grid(
            model, params, term; r_range = r, fill_value = 0.3)
        return vec(D)
    end
end

function _a2_dummy_discovery()
    return DiscoveryResult(false, "a2 dummy discovery", "",
        nothing, nothing, [], (;))
end

function _a2_fake_fit(params)
    return TrainingResult(
        params, Float64[], 1.0, 0.4,
        RunMetadata(seed = 0), (;), true, BioDynaX.Success)
end

function _a2_new_logs()
    return (;
        grid = Any[],
        sample = Any[],
        result = Any[],
        range = Any[],
        discover = Any[],
        holdout = Any[])
end

function _a2_occupancy_classified(X, train_occ, hold_occ)
    mixed = hcat(train_occ.X, hold_occ.X)
    return X === train_occ.X || X == train_occ.X ||
           X === hold_occ.X || X == hold_occ.X ||
           X == mixed
end

function _a2_occupancy_classified_sample_calls(logs, train_occ, hold_occ)
    return count(obs -> _a2_occupancy_classified(obs.X, train_occ, hold_occ),
        logs.sample)
end

function _a2_is_fill_grid(X, r_range, term, nstates; fill_value = 0.3)
    X isa Matrix || return false
    size(X, 1) == nstates || return false
    size(X, 2) == length(r_range) || return false
    collect(X[term.regulator, :]) == collect(r_range) || return false
    for i in 1:nstates
        if i != term.regulator
            all(==(fill_value), view(X, i, :)) || return false
        end
    end
    return true
end

function _a2_assert_fill_grid_not_occupancy(
        X, r_range, term, nstates, train_occ, hold_occ, domain;
        fill_value = 0.3)
    @test X isa Matrix
    @test size(X, 1) == nstates
    @test size(X, 2) == length(r_range)
    @test collect(X[term.regulator, :]) == collect(r_range)
    for i in 1:nstates
        if i != term.regulator
            @test all(==(fill_value), view(X, i, :))
        end
    end
    @test _a2_is_fill_grid(X, r_range, term, nstates; fill_value = fill_value)
    @test X !== train_occ.X
    @test X != train_occ.X
    @test X !== hold_occ.X
    @test X != hold_occ.X
    @test X != hcat(train_occ.X, hold_occ.X)
    @test X != reshape(domain.z, 1, :)
    @test !(X isa TrajectoryOccupancy)
    return nothing
end

function _a2_with_live_observers(f, logs)
    return with_evaluate_unknown_rate_recovery_range_observer(r_range -> begin
        push!(logs.range, r_range)
        nothing
    end) do
        with_sample_unknown_destruction_grid_observer(r_range -> begin
            push!(logs.grid, r_range)
            nothing
        end) do
            with_sample_unknown_destruction_observer(obs -> begin
                push!(logs.sample, obs)
                nothing
            end) do
                with_sample_unknown_destruction_result_observer(obs -> begin
                    push!(logs.result, obs)
                    nothing
                end) do
                    with_discover_unknown_rate_observer(
                        (R, times, D, config) -> begin
                        push!(logs.discover,
                            (;
                                R = copy(R),
                                times = copy(times),
                                D = copy(D),
                                config = config))
                        return _a2_dummy_discovery()
                    end) do
                        with_evaluate_holdout_observer((args...) -> begin
                            push!(logs.holdout, args)
                            nothing
                        end) do
                            f()
                        end
                    end
                end
            end
        end
    end
end

function _a2_live_m1(model, params, term, set, truth_rate)
    logs = _a2_new_logs()
    evaled = _a2_with_live_observers(logs) do
        _reference_protocol_rate_recovery(
            model, params, term, truth_rate, set;
            order = 2, family = :hill, noise_σ = 0.0,
            data_residual_fn = _ -> 0.0)
    end
    return evaled, logs
end

function _a2_live_domain(split, ude_net, domain; seed = _A2_DOMAIN_SEED)
    logs = _a2_new_logs()
    captured_p0 = Ref{Any}()
    restart = _a2_with_live_observers(logs) do
        with_fit_unknown_destruction_entry_observer(obs -> begin
            captured_p0[] = obs.p0
            nothing
        end) do
            with_fit_unknown_destruction_observer(_ -> _a2_fake_fit(captured_p0[])) do
                fit_functional_identifiability_restart(
                    split, ude_net, seed, domain)
            end
        end
    end
    return restart, logs
end

function _a2_live_holdout(split, model, params, term, truth_rate)
    logs = _a2_new_logs()
    ev = _a2_with_live_observers(logs) do
        evaluate_holdout(
            split, (; term = term), model, params, term, truth_rate)
    end
    return ev, logs
end

function _a2_apply_occupancy_sentinel!(train_occ, hold_occ, term)
    nstates = size(train_occ.X, 1)
    train_occ.X[term.regulator, :] .= 777.777
    hold_occ.X[term.regulator, :] .= 666.666
    for i in 1:nstates
        if i != term.regulator
            train_occ.X[i, :] .= 888.888
            hold_occ.X[i, :] .= 555.555
        end
    end
    return nothing
end

const _A2_SET = _a2_protocol_set()
const _A2_SPLIT = reference_protocol_experiment_split(_A2_SET)

@testset "T-A2-COMPOSER live composer uses _regulator_grid, not occupancy" begin
    @test _A2_N_ICS == 9
    @test _A2_N_POINTS == 50
    @test _A2_N_TRAIN == 7
    @test _A2_N_HOLD == 2
    @test _A2_DOMAIN_LEN == 450
    @test length(_A2_SET) == REFERENCE_PROTOCOL.n_ics == 9
    @test REFERENCE_PROTOCOL.seed == 103
    @test _A2_SPLIT.train_indices === REFERENCE_PROTOCOL_TRAIN_INDICES ===
          (1, 2, 3, 4, 5, 6, 7)
    @test _A2_SPLIT.holdout_indices === REFERENCE_PROTOCOL_HOLDOUT_INDICES === (8, 9)
    @test all(size(exp.observations, 2) == _A2_N_POINTS
    for exp in _A2_SET.experiments)
    model, params, term, _ = _a2_probe_models()
    expected_r = collect(_regulator_grid(_A2_SPLIT.train, term))
    domain = functional_identifiability_domain(_A2_SPLIT, term.regulator)
    z_expected = _a2_independent_z(_A2_SPLIT, term.regulator)
    train_occ = collect_observed_occupancy(
        _A2_SPLIT, :train_observed_states)
    hold_occ = collect_observed_occupancy(
        _A2_SPLIT, :holdout_observed_states)
    @test length(expected_r) == 80
    @test length(domain.z) == 450
    @test length(z_expected) == 450
    @test domain.z == z_expected
    @test size(train_occ.X, 2) == _A2_TRAIN_COLS == 350
    @test size(hold_occ.X, 2) == _A2_HOLD_COLS == 100
    evaled, logs = _a2_live_m1(model, params, term, _A2_SET, r -> zeros(length(r)))
    @test length(logs.grid) >= 1
    @test length(logs.range) >= 1
    @test !(logs.range[1] isa TrajectoryOccupancy)
    @test !(logs.grid[1] isa TrajectoryOccupancy)
    captured_grid_r = logs.grid[1]
    @test collect(captured_grid_r) == expected_r
    @test collect(captured_grid_r) != collect(train_occ.X[term.regulator, :])
    @test collect(captured_grid_r) != collect(hold_occ.X[term.regulator, :])
    @test collect(captured_grid_r) != collect(domain.z)
    @test collect(captured_grid_r) != z_expected
    @test collect(captured_grid_r) != _A2_FIXED_R
    @test _a2_occupancy_classified_sample_calls(logs, train_occ, hold_occ) == 0
    @test !isempty(logs.sample)
    captured_X = logs.sample[1].X
    nstates = model.compiled.nstates
    _a2_assert_fill_grid_not_occupancy(
        captured_X, expected_r, term, nstates, train_occ, hold_occ, domain)
    @test evaled.term === term
end

@testset "T-A2-COMPOSER-TIME live composer discovery still receives dummy time" begin
    model, params, term, _ = _a2_probe_models()
    expected_r = collect(_regulator_grid(_A2_SPLIT.train, term))
    dummy = collect(range(0.0, 1.0; length = length(expected_r)))
    train_occ = collect_observed_occupancy(
        _A2_SPLIT, :train_observed_states)
    hold_occ = collect_observed_occupancy(
        _A2_SPLIT, :holdout_observed_states)
    domain = functional_identifiability_domain(_A2_SPLIT, term.regulator)
    truth_rate = _a2_matching_truth(model, params, term)
    _, logs = _a2_live_m1(model, params, term, _A2_SET, truth_rate)
    @test length(logs.discover) >= 1
    @test logs.discover[1].times == dummy
    @test length(logs.discover[1].times) == 80
    @test logs.discover[1].times != train_occ.times
    @test logs.discover[1].times != hold_occ.times
    @test length(logs.grid) >= 1
    @test collect(logs.grid[1]) == expected_r
    @test !(logs.grid[1] isa TrajectoryOccupancy)
    @test _a2_occupancy_classified_sample_calls(logs, train_occ, hold_occ) == 0
    @test !isempty(logs.sample)
    _a2_assert_fill_grid_not_occupancy(
        logs.sample[1].X, expected_r, term, model.compiled.nstates,
        train_occ, hold_occ, domain)
end

@testset "T-A2-DOMAIN live functional-identifiability domain is train-then-holdout z, not occupancy" begin
    model, params, term, ude_net = _a2_probe_models()
    reg = term.regulator
    z_expected = _a2_independent_z(_A2_SPLIT, reg)
    @test length(z_expected) == 450
    domain = functional_identifiability_domain(_A2_SPLIT, reg)
    @test domain.z == z_expected
    @test length(domain.z) == 450
    @test domain.fill_value == 0.3
    @test domain.construction === :train_obs_union_holdout_obs
    train_occ = collect_observed_occupancy(
        _A2_SPLIT, :train_observed_states)
    hold_occ = collect_observed_occupancy(
        _A2_SPLIT, :holdout_observed_states)
    @test size(train_occ.X, 1) > 1
    @test size(hold_occ.X, 1) > 1
    @test train_occ.X != reshape(domain.z, 1, :)
    @test hold_occ.X != reshape(domain.z, 1, :)
    restart, logs = _a2_live_domain(_A2_SPLIT, ude_net, domain)
    @test length(logs.grid) >= 1
    captured_r = logs.grid[1]
    @test collect(captured_r) == z_expected
    @test collect(captured_r) == domain.z
    @test length(captured_r) == 450
    @test _a2_occupancy_classified_sample_calls(logs, train_occ, hold_occ) == 0
    @test !isempty(logs.sample)
    captured_X = logs.sample[1].X
    _a2_assert_fill_grid_not_occupancy(
        captured_X, z_expected, term, model.compiled.nstates,
        train_occ, hold_occ, domain)
    @test collect(captured_X[term.regulator, :]) == domain.z
    @test restart !== nothing
end

@testset "T-A2-DOMAIN-SEP functional-identifiability does not read occupancy after sentinel mutation" begin
    model, params, term, ude_net = _a2_probe_models()
    reg = term.regulator
    z_expected = _a2_independent_z(_A2_SPLIT, reg)
    domain = functional_identifiability_domain(_A2_SPLIT, reg)
    train_occ = collect_observed_occupancy(
        _A2_SPLIT, :train_observed_states)
    hold_occ = collect_observed_occupancy(
        _A2_SPLIT, :holdout_observed_states)
    @test domain.z == z_expected
    restart, logs = _a2_live_domain(_A2_SPLIT, ude_net, domain)
    @test collect(logs.grid[1]) == z_expected
    @test length(logs.grid[1]) == 450
    @test _a2_occupancy_classified_sample_calls(logs, train_occ, hold_occ) == 0
    _a2_assert_fill_grid_not_occupancy(
        logs.sample[1].X, z_expected, term, model.compiled.nstates,
        train_occ, hold_occ, domain)
    _a2_apply_occupancy_sentinel!(train_occ, hold_occ, term)
    @test all(==(777.777), train_occ.X[term.regulator, :])
    @test all(==(666.666), hold_occ.X[term.regulator, :])
    restart2, logs2 = _a2_live_domain(_A2_SPLIT, ude_net, domain)
    @test collect(logs2.grid[1]) == z_expected
    @test collect(logs2.grid[1]) == domain.z
    @test length(logs2.grid[1]) == 450
    @test collect(logs2.grid[1]) != collect(train_occ.X[term.regulator, :])
    @test collect(logs2.grid[1]) != collect(hold_occ.X[term.regulator, :])
    @test _a2_occupancy_classified_sample_calls(logs2, train_occ, hold_occ) == 0
    captured_X = logs2.sample[1].X
    @test collect(captured_X[term.regulator, :]) == z_expected
    @test collect(captured_X[term.regulator, :]) == domain.z
    for i in 1:size(captured_X, 1)
        if i != term.regulator
            @test all(==(0.3), view(captured_X, i, :))
        end
    end
    @test captured_X != train_occ.X
    @test captured_X != hold_occ.X
    @test captured_X != hcat(train_occ.X, hold_occ.X)
    @test captured_X != reshape(domain.z, 1, :)
    @test restart !== nothing
    @test restart2 !== nothing
end

@testset "T-A2-HOLDOUT T-A2-HOLDOUT-D live holdout grids are fill-grids; two D oracles" begin
    @test fieldnames(HoldoutEvidence) == (
        :data_residual_train,
        :data_residual_holdout,
        :d_rmse_holdout,
        :d_rmse_holdout_domain)
    @test :occupancy ∉ fieldnames(HoldoutEvidence)
    @test :occupancy ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
    model, params, term, _ = _a2_probe_models()
    truth_rate = _a2_hill_truth()
    r_holdout_expected = _a2_independent_holdout_r(_A2_SPLIT, term)
    r_band_expected = collect(
        _reference_protocol_external_regulator_band(_A2_SPLIT.train, term))
    domain = functional_identifiability_domain(_A2_SPLIT, term.regulator)
    train_occ = collect_observed_occupancy(
        _A2_SPLIT, :train_observed_states)
    hold_occ = collect_observed_occupancy(
        _A2_SPLIT, :holdout_observed_states)
    ev, logs = _a2_live_holdout(
        _A2_SPLIT, model, params, term, truth_rate)
    evaluate_holdout_call_count = length(logs.holdout)
    grid_call_count = length(logs.grid)
    occupancy_classified_sample_calls = _a2_occupancy_classified_sample_calls(
        logs, train_occ, hold_occ)
    @test ev isa HoldoutEvidence
    @test evaluate_holdout_call_count == 1
    @test grid_call_count == 2
    @test occupancy_classified_sample_calls == 0
    @test length(logs.sample) == 2
    @test length(logs.result) == 2
    @test collect(logs.grid[1]) == r_holdout_expected
    @test collect(logs.grid[2]) == r_band_expected
    nstates = model.compiled.nstates
    X1 = logs.sample[1].X
    X2 = logs.sample[2].X
    @test _a2_is_fill_grid(X1, r_holdout_expected, term, nstates;
        fill_value = 0.3)
    @test _a2_is_fill_grid(X2, r_band_expected, term, nstates;
        fill_value = 0.3)
    _a2_assert_fill_grid_not_occupancy(
        X1, r_holdout_expected, term, nstates, train_occ, hold_occ, domain)
    _a2_assert_fill_grid_not_occupancy(
        X2, r_band_expected, term, nstates, train_occ, hold_occ, domain)
    inferred_fill_1 = X1[term.regulator == 1 ? 2 : 1, 1]
    inferred_fill_2 = X2[term.regulator == 1 ? 2 : 1, 1]
    @test inferred_fill_1 == 0.3
    @test inferred_fill_2 == 0.3
    captured_holdout_R = logs.result[1].R
    captured_holdout_D = logs.result[1].D
    captured_band_R = logs.result[2].R
    captured_band_D = logs.result[2].D
    expected_holdout = _finite_rate_rel_rmse(
        captured_holdout_D, truth_rate(vec(captured_holdout_R)))
    expected_domain = _finite_rate_rel_rmse(
        captured_band_D, truth_rate(vec(captured_band_R)))
    @test ev.d_rmse_holdout == expected_holdout
    @test ev.d_rmse_holdout_domain == expected_domain
    production_grid_calls = length(logs.grid)
    production_sample_calls = length(logs.sample)
    replay_grid = Any[]
    replay_sample = Any[]
    with_sample_unknown_destruction_grid_observer(r -> begin
        push!(replay_grid, r)
        nothing
    end) do
        with_sample_unknown_destruction_observer(obs -> begin
            push!(replay_sample, obs)
            nothing
        end) do
            (replay_hold_R, replay_hold_D, _) = sample_unknown_destruction_grid(
                model, params, term;
                r_range = r_holdout_expected, fill_value = 0.3)
            (replay_band_R, replay_band_D, _) = sample_unknown_destruction_grid(
                model, params, term;
                r_range = r_band_expected, fill_value = 0.3)
            independent_holdout_rmse = _finite_rate_rel_rmse(
                replay_hold_D, truth_rate(vec(replay_hold_R)))
            independent_domain_rmse = _finite_rate_rel_rmse(
                replay_band_D, truth_rate(vec(replay_band_R)))
            @test ev.d_rmse_holdout == independent_holdout_rmse
            @test ev.d_rmse_holdout_domain == independent_domain_rmse
            @test independent_holdout_rmse == expected_holdout
            @test independent_domain_rmse == expected_domain
        end
    end
    @test length(logs.grid) == production_grid_calls == 2
    @test length(logs.sample) == production_sample_calls == 2
    @test length(replay_grid) == 2
    @test occupancy_classified_sample_calls == 0
    @test _a2_occupancy_classified_sample_calls(logs, train_occ, hold_occ) == 0
end
