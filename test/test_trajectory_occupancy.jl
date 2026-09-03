# M4-A1 occupancy is produced from observed Experiment.observations.
# Sentinel matrices distinguish full X from regulator-only X, keep
# duplicates and original order, and keep non-regulator states off 0.3.

const _M4A_FORBIDDEN_TOKENS = (
    "predict_ude",
    "_collect_trajectory_data",
    "_regulator_grid",
    "_unique_claim_external_regulator_band",
    "sample_unknown_destruction_grid",
    "sample_learned_function",
    "hill_rate_truth",
    "estimate_derivatives",
    "fill_value",
    "equation_to_function",
    "normalize_destruction_samples")

const _M4A_FORBIDDEN_MUTATORS = (
    "splice!", "deleteat!", "pop!", "push!", "insert!",
    "append!", "resize!", "setindex!", "replace!")

const _M4A_PROTOCOL_TRAIN_COLS = 7 * UNIQUE_CLAIM_PROTOCOL.n_points
const _M4A_PROTOCOL_HOLDOUT_COLS = 2 * UNIQUE_CLAIM_PROTOCOL.n_points
const _M4A_PROTOCOL_TOTAL_COLS =
    _M4A_PROTOCOL_TRAIN_COLS + _M4A_PROTOCOL_HOLDOUT_COLS

function _m4a_source()
    return read(joinpath(@__DIR__, "..", "src", "TrajectoryOccupancy.jl"),
        String)
end

function _m4a_function_body(name)
    src = _m4a_source()
    start = findfirst("function " * name, src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function _m4a_recovery_function_body(name)
    src = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    start = findfirst("function " * name, src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function _m4a_experiment(name::Symbol, times, observations)
    obs = Matrix{Float64}(observations)
    t = collect(Float64, times)
    return Experiment(name, t, obs, obs[:, 1])
end

# Independent oracle: same hcat contract, not the occupancy constructor.
function _m4a_independent_X(experiments)
    return reduce(hcat, (Float64.(exp.observations) for exp in experiments))
end

function _m4a_independent_times(experiments)
    return reduce(vcat, (Float64.(exp.times) for exp in experiments))
end

# Adversarial sentinel: regulator [0.1, 0.5, 0.1, 0.8] is unsorted and
# duplicated. Non-regulator rows are off 0.3 and cannot be rebuilt from
# the regulator row alone (the two 0.1 columns have different S and Q).
function _m4a_three_state_split()
    train = (
        _m4a_experiment(:T1, [10.0, 11.0, 12.0], [
            4.40 7.70 1.10;
            0.10 0.50 0.10;
            2.20 5.50 8.80]),
        _m4a_experiment(:T2, [1.0], reshape([9.90, 0.80, 3.30], 3, 1)))
    holdout = (
        _m4a_experiment(:H1, [30.0, 31.0], [
            6.60 0.44;
            1.75 1.75;
            9.91 0.22]),
        _m4a_experiment(:H2, [0.5], reshape([1.33, 0.40, 4.40], 3, 1)))
    return ExperimentSplit(
        UNIQUE_CLAIM_TRAIN_INDICES,
        UNIQUE_CLAIM_HOLDOUT_INDICES,
        ExperimentSet(collect(train), [:S, :R, :Q]),
        ExperimentSet(collect(holdout), [:S, :R, :Q]))
end

function _m4a_two_state_split()
    train = (
        _m4a_experiment(:T1, [10.0, 11.0, 12.0], [
            4.40 7.70 1.10;
            0.10 0.50 0.10]),
        _m4a_experiment(:T2, [1.0], reshape([9.90, 0.80], 2, 1)))
    holdout = (
        _m4a_experiment(:H1, [30.0, 31.0], [
            6.60 0.44;
            1.75 1.75]),
        _m4a_experiment(:H2, [0.5], reshape([1.33, 0.40], 2, 1)))
    return ExperimentSplit(
        UNIQUE_CLAIM_TRAIN_INDICES,
        UNIQUE_CLAIM_HOLDOUT_INDICES,
        ExperimentSet(collect(train), [:S, :R]),
        ExperimentSet(collect(holdout), [:S, :R]))
end

function _m4a_nine_ic_set()
    experiments = [_m4a_experiment(Symbol("E$i"), [Float64(i)],
        reshape(Float64[0.71, 0.10 + 0.01 * i, 4.4], 3, 1)) for i in 1:9]
    return ExperimentSet(experiments, [:S, :R, :Q])
end

function _m4a_protocol_set()
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    proto = UNIQUE_CLAIM_PROTOCOL
    return generate_recovery_experiments(
        MersenneTwister(proto.seed), truth_net, truth;
        tspan = proto.tspan, n_points = proto.n_points,
        noise_σ = proto.observation_noise)
end

function _m4a_composer_set()
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    return generate_recovery_experiments(
        MersenneTwister(23), truth_net, truth;
        tspan = (0.0, 1.0), n_points = 5, noise_σ = 0.0)
end

function _m4a_probe_models()
    rng = MersenneTwister(1)
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    model, params = build_ude_model(rng, net)
    term = only_unknown_destruction(model)
    return model, params, term
end

function _m4a_three_state_models()
    rng = MersenneTwister(1)
    net = BioDynaX.build_three_state_unknown_network(; with_distractor = false)
    model, params = build_ude_model(rng, net)
    term = only_unknown_destruction(model)
    return model, params, term
end

function _m4a_dummy_evaled(term)
    return (;
        nn_correlation = 0.0,
        nn_rate_rmse = Inf,
        success = false,
        retcode = DiscoveryFailed,
        message = "m4a composer probe",
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

function _m4a_dummy_discovery()
    return DiscoveryResult(false, "m4a probe", "", nothing, nothing, [], (;))
end

function _m4a_matching_truth(model, params, term)
    return function (r)
        (_, D, _) = sample_unknown_destruction_grid(
            model, params, term; r_range = r, fill_value = 0.3)
        return vec(D)
    end
end

function _m4a_snapshot_set(set::ExperimentSet)
    return (;
        vec = set.experiments,
        experiments = [set.experiments[i] for i in eachindex(set.experiments)],
        observations = [set.experiments[i].observations
                        for i in eachindex(set.experiments)],
        times = [set.experiments[i].times for i in eachindex(set.experiments)],
        u0 = [set.experiments[i].u0 for i in eachindex(set.experiments)],
        mask = [set.experiments[i].mask for i in eachindex(set.experiments)],
        name = [set.experiments[i].name for i in eachindex(set.experiments)],
        exp_metadata = [deepcopy(set.experiments[i].metadata)
                        for i in eachindex(set.experiments)],
        state_names = copy(set.state_names),
        units = copy(set.units),
        metadata = deepcopy(set.metadata),
        fingerprint = experiment_fingerprint(set),
        fields = fieldnames(typeof(set)),
        n = length(set))
end

function _m4a_assert_intact(set::ExperimentSet, snap)
    @test set.experiments === snap.vec
    @test length(set.experiments) == snap.n
    @test length(set) == snap.n
    @test all(set.experiments[i] === snap.experiments[i]
              for i in eachindex(snap.experiments))
    @test set.state_names == snap.state_names
    @test set.units == snap.units
    @test set.metadata == snap.metadata
    @test experiment_fingerprint(set) == snap.fingerprint
    @test fieldnames(typeof(set)) == snap.fields
    @test fieldnames(typeof(set)) ==
          (:experiments, :state_names, :units, :metadata)
    @test !hasfield(ExperimentSet, :train)
    @test !hasfield(ExperimentSet, :holdout)
    @test !haskey(set.metadata, :train)
    @test !haskey(set.metadata, :holdout)
    for i in eachindex(snap.experiments)
        @test set.experiments[i].observations === snap.observations[i]
        @test set.experiments[i].times === snap.times[i]
        @test set.experiments[i].u0 === snap.u0[i]
        @test set.experiments[i].mask === snap.mask[i]
        @test set.experiments[i].name === snap.name[i]
        @test set.experiments[i].metadata == snap.exp_metadata[i]
    end
    return nothing
end

# Full call log for sample_unknown_destruction. Isolated M4-A1 sampling
# tests must empty the log, push every invocation, assert
# length(captured_calls) == 1, and inspect captured_calls[1]. Do not
# keep only the last log entry.
function _m4a_with_sample_call_log(f::Function, captured_calls::Vector)
    empty!(captured_calls)
    return with_sample_unknown_destruction_observer(f, obs -> begin
            push!(captured_calls, obs)
            nothing
        end)
end

function _m4a_protocol_fake_X(occupancy, term)
    fake = fill(0.3, size(occupancy.X))
    fake[term.regulator, :] .= occupancy.X[term.regulator, :]
    return fake
end

mutable struct _M4AMutationLog
    calls::Vector{String}
end

struct _M4ALoggedExperiments{E} <: AbstractVector{E}
    data::Vector{E}
    log::_M4AMutationLog
end

Base.size(v::_M4ALoggedExperiments) = size(v.data)
Base.getindex(v::_M4ALoggedExperiments, i::Int) = v.data[i]
function Base.setindex!(v::_M4ALoggedExperiments, value, i)
    push!(v.log.calls, "setindex!")
    return setindex!(v.data, value, i)
end
function Base.splice!(v::_M4ALoggedExperiments, args...)
    push!(v.log.calls, "splice!")
    return splice!(v.data, args...)
end
function Base.deleteat!(v::_M4ALoggedExperiments, args...)
    push!(v.log.calls, "deleteat!")
    return deleteat!(v.data, args...)
end
function Base.pop!(v::_M4ALoggedExperiments)
    push!(v.log.calls, "pop!")
    return pop!(v.data)
end
function Base.push!(v::_M4ALoggedExperiments, args...)
    push!(v.log.calls, "push!")
    return push!(v.data, args...)
end
function Base.insert!(v::_M4ALoggedExperiments, args...)
    push!(v.log.calls, "insert!")
    return insert!(v.data, args...)
end
function Base.append!(v::_M4ALoggedExperiments, args...)
    push!(v.log.calls, "append!")
    return append!(v.data, args...)
end
function Base.resize!(v::_M4ALoggedExperiments, n)
    push!(v.log.calls, "resize!")
    return resize!(v.data, n)
end
function Base.replace!(v::_M4ALoggedExperiments, args...; kwargs...)
    push!(v.log.calls, "replace!")
    return replace!(v.data, args...; kwargs...)
end

@testset "T-A-API M4-A1 occupancy helpers stay unexported" begin
    @test :TrajectoryOccupancy ∉ names(BioDynaX)
    @test :collect_observed_occupancy ∉ names(BioDynaX)
    @test :sample_destruction_occupancy ∉ names(BioDynaX)
    @test :TRAJECTORY_OCCUPANCY_PROVENANCES ∉ names(BioDynaX)
    @test :with_sample_unknown_destruction_observer ∉ names(BioDynaX)
    @test :SAMPLE_UNKNOWN_DESTRUCTION_OBSERVER ∉ names(BioDynaX)
    @test :TrajectoryOccupancy ∉ LOCKED_PUBLIC_EXPORTS
    @test :collect_observed_occupancy ∉ LOCKED_PUBLIC_EXPORTS
    @test :sample_destruction_occupancy ∉ LOCKED_PUBLIC_EXPORTS
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test FUNCTIONAL_ID_REPORTING_CUTOFFS === (
        min_successful_restarts = 3,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
    @test FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
end

@testset "M4-A1 field contract and closed Q4 construction" begin
    @test fieldnames(TrajectoryOccupancy) === (
        :X, :experiment_index, :sample_index_in_exp, :times,
        :provenance, :split_indices, :n_points, :construction)
    @test TRAJECTORY_OCCUPANCY_PROVENANCES === (
        :train_observed_states, :holdout_observed_states)
    split = _m4a_three_state_split()
    X = _m4a_independent_X(split.train.experiments)
    n = size(X, 2)
    idx = [1, 1, 1, 2]
    samp = [1, 2, 3, 1]
    times = _m4a_independent_times(split.train.experiments)
    @test_throws ArgumentError TrajectoryOccupancy(
        X, idx, samp, times, :train_obs_union_holdout_obs, (1, 2), n,
        :train_obs_union_holdout_obs)
    @test_throws ArgumentError TrajectoryOccupancy(
        X, idx, samp, times, :train_observed_states, (1, 2), n,
        :train_obs_union_holdout_obs)
    @test_throws ArgumentError collect_observed_occupancy(
        split, :train_obs_union_holdout_obs)
end

@testset "T-A-SRC producer uses observed hcat, not a hand-built constructor" begin
    split = _m4a_three_state_split()
    expected_X = _m4a_independent_X(split.train.experiments)
    @test expected_X == [
        4.40 7.70 1.10 9.90;
        0.10 0.50 0.10 0.80;
        2.20 5.50 8.80 3.30]
    occupancy = collect_observed_occupancy(split, :train_observed_states)
    @test occupancy isa TrajectoryOccupancy
    @test occupancy.X == expected_X
    @test occupancy.X !== expected_X
    predict_hits = Ref(0)
    with_predict_ude_observer(_ -> (predict_hits[] += 1; nothing)) do
        again = collect_observed_occupancy(split, :train_observed_states)
        @test again.X == expected_X
    end
    @test predict_hits[] == 0
end

@testset "T-A-XNEQ full X is not regulator-only, not fill 0.3, and not Q4 z" begin
    split = _m4a_three_state_split()
    occupancy = collect_observed_occupancy(split, :train_observed_states)
    hold = collect_observed_occupancy(split, :holdout_observed_states)
    regulator_only = occupancy.X[2:2, :]
    q4_like_z = vcat(occupancy.X[2, :], hold.X[2, :])
    reconstructed = fill(0.3, size(occupancy.X))
    reconstructed[2, :] .= occupancy.X[2, :]
    @test size(occupancy.X) == (3, 4)
    @test size(regulator_only) == (1, 4)
    @test occupancy.X != regulator_only
    @test occupancy.X[2, :] == vec(regulator_only)
    @test occupancy.X[2, :] != q4_like_z
    @test vec(occupancy.X) != q4_like_z
    @test occupancy.X != q4_like_z
    @test all(occupancy.X[1, :] .!= 0.3)
    @test all(occupancy.X[3, :] .!= 0.3)
    @test occupancy.X[1, :] != fill(0.3, occupancy.n_points)
    @test occupancy.X[3, :] != fill(0.3, occupancy.n_points)
    @test occupancy.X != reconstructed
    @test occupancy.X[1, 1] != occupancy.X[1, 3]
    @test occupancy.X[3, 1] != occupancy.X[3, 3]
    @test occupancy.X[2, 1] == occupancy.X[2, 3] == 0.10
end

@testset "T-A-SRC preserves exact order and duplicates" begin
    split = _m4a_three_state_split()
    occupancy = collect_observed_occupancy(split, :train_observed_states)
    @test occupancy.X[2, :] == [0.1, 0.5, 0.1, 0.8]
    @test occupancy.X[2, :] != sort([0.1, 0.5, 0.1, 0.8])
    @test occupancy.X[2, :] != unique([0.1, 0.5, 0.1, 0.8])
    @test occupancy.X[1, :] == [4.40, 7.70, 1.10, 9.90]
    @test occupancy.X[3, :] == [2.20, 5.50, 8.80, 3.30]
    @test length(occupancy.X[2, :]) == 4
    @test length(unique(occupancy.X[2, :])) == 3
    @test occupancy.X != occupancy.X[:, sortperm(occupancy.X[2, :])]
    @test occupancy.times == [10.0, 11.0, 12.0, 1.0]
    @test occupancy.times != sort(occupancy.times)
    @test occupancy.times != collect(range(0.0, 1.0; length = occupancy.n_points))
    @test occupancy.experiment_index == [1, 1, 1, 2]
    @test occupancy.sample_index_in_exp == [1, 2, 3, 1]
    @test occupancy.split_indices == (1, 2)
    @test occupancy.n_points == 4
    @test occupancy.n_points == size(occupancy.X, 2)
end

@testset "T-A-PROV T-A-SPLIT train and holdout occupancy stay separate" begin
    split = _m4a_three_state_split()
    train_occ = collect_observed_occupancy(split, :train_observed_states)
    hold_occ = collect_observed_occupancy(split, :holdout_observed_states)
    hold_X = _m4a_independent_X(split.holdout.experiments)
    mixed_X = hcat(train_occ.X, hold_X)
    @test train_occ.provenance === :train_observed_states
    @test train_occ.construction === :train_observed_states
    @test train_occ.construction !== :train_obs_union_holdout_obs
    @test hold_occ.provenance === :holdout_observed_states
    @test hold_occ.construction === :holdout_observed_states
    @test hold_occ.construction !== :train_obs_union_holdout_obs
    @test hold_occ.X == hold_X
    @test hold_occ.X == [
        6.60 0.44 1.33;
        1.75 1.75 0.40;
        9.91 0.22 4.40]
    @test hold_occ.experiment_index == [8, 8, 9]
    @test hold_occ.sample_index_in_exp == [1, 2, 1]
    @test hold_occ.split_indices == (8, 9)
    @test hold_occ.n_points == 3
    @test train_occ.X != mixed_X
    @test hold_occ.X != mixed_X
    @test size(train_occ.X, 2) != size(mixed_X, 2)
    @test all(c -> train_occ.X[:, c] != hold_X[:, 1], 1:train_occ.n_points)
    @test_throws ArgumentError collect_observed_occupancy(
        _m4a_nine_ic_set(), :train_observed_states)
    @test_throws ArgumentError collect_observed_occupancy(
        _m4a_nine_ic_set(), :holdout_observed_states)
end

@testset "T-A-LEN T-A-PROV T-A-SPLIT T-A-R T-A-Q4SEP protocol 9x50 occupancy" begin
    set = _m4a_protocol_set()
    @test length(set) == UNIQUE_CLAIM_PROTOCOL.n_ics == 9
    @test all(size(exp.observations, 2) == UNIQUE_CLAIM_PROTOCOL.n_points
              for exp in set.experiments)
    model, params, term = _m4a_probe_models()
    split = unique_claim_experiment_split(set)
    @test split.train_indices === UNIQUE_CLAIM_TRAIN_INDICES ===
          (1, 2, 3, 4, 5, 6, 7)
    @test split.holdout_indices === UNIQUE_CLAIM_HOLDOUT_INDICES === (8, 9)
    @test length(split.train) == 7
    @test length(split.holdout) == 2
    train_occ = collect_observed_occupancy(split, :train_observed_states)
    hold_occ = collect_observed_occupancy(split, :holdout_observed_states)
    expected_train = _m4a_independent_X(split.train.experiments)
    expected_hold = _m4a_independent_X(split.holdout.experiments)
    @test train_occ.n_points == _M4A_PROTOCOL_TRAIN_COLS == 350
    @test hold_occ.n_points == _M4A_PROTOCOL_HOLDOUT_COLS == 100
    @test size(train_occ.X, 2) == 350
    @test size(hold_occ.X, 2) == 100
    @test size(train_occ.X, 2) + size(hold_occ.X, 2) ==
          _M4A_PROTOCOL_TOTAL_COLS == 450
    @test train_occ.X == expected_train
    @test hold_occ.X == expected_hold
    @test train_occ.split_indices == (1, 2, 3, 4, 5, 6, 7)
    @test hold_occ.split_indices == (8, 9)
    @test train_occ.provenance === :train_observed_states
    @test hold_occ.provenance === :holdout_observed_states
    @test train_occ.construction !== :train_obs_union_holdout_obs
    @test hold_occ.construction !== :train_obs_union_holdout_obs
    mixed = hcat(train_occ.X, hold_occ.X)
    @test train_occ.X != mixed
    @test hold_occ.X != mixed
    domain = functional_identifiability_domain(split, term.regulator)
    @test length(domain.z) == 450
    @test domain.n_train_points == 350
    @test domain.n_holdout_points == 100
    @test domain.construction === :train_obs_union_holdout_obs
    @test train_occ.X[term.regulator, :] == domain.z[1:350]
    @test hold_occ.X[term.regulator, :] == domain.z[351:450]
    @test size(train_occ.X, 1) > 1
    nonreg = filter(!=(term.regulator), 1:size(train_occ.X, 1))
    @test !isempty(nonreg)
    @test any(train_occ.X[i, :] != train_occ.X[term.regulator, :]
              for i in nonreg)
    @test train_occ.X != domain.z
    @test hold_occ.X != domain.z
    @test size(train_occ.X) != size(reshape(domain.z, 1, :))
end

@testset "T-A-SAMP captured sample_unknown_destruction X === occupancy.X" begin
    split = _m4a_three_state_split()
    occupancy = collect_observed_occupancy(split, :train_observed_states)
    @test size(occupancy.X) == (3, 4)
    @test occupancy.X[2, :] == [0.1, 0.5, 0.1, 0.8]
    @test all(occupancy.X[1, :] .!= 0.3)
    @test all(occupancy.X[3, :] .!= 0.3)
    reconstructed = fill(0.3, size(occupancy.X))
    reconstructed[2, :] .= occupancy.X[2, :]
    @test occupancy.X != reconstructed
    model, params, term = _m4a_three_state_models()
    @test term.regulator == 2
    captured_calls = Any[]
    grid_hits = Ref(0)
    result_hits = Ref(0)
    predict_hits = Ref(0)
    sampled = _m4a_with_sample_call_log(captured_calls) do
        with_sample_unknown_destruction_grid_observer(
            _ -> (grid_hits[] += 1; nothing)) do
            with_sample_unknown_destruction_result_observer(
                _ -> (result_hits[] += 1; nothing)) do
                with_predict_ude_observer(
                    _ -> (predict_hits[] += 1; nothing)) do
                    return sample_destruction_occupancy(
                        model, params, term, occupancy)
                end
            end
        end
    end
    @test length(captured_calls) == 1
    @test captured_calls[1].model === model
    @test captured_calls[1].params === params
    @test captured_calls[1].term === term
    @test captured_calls[1].X === occupancy.X
    @test size(captured_calls[1].X) == size(occupancy.X)
    @test captured_calls[1].X == occupancy.X
    @test captured_calls[1].X != reconstructed
    @test captured_calls[1].X !== reconstructed
    expected = sample_unknown_destruction(model, params, occupancy.X; term)
    @test sampled == expected
    @test sampled[1] == occupancy.X[term.regulators, :]
    @test grid_hits[] == 0
    @test result_hits[] == 0
    @test predict_hits[] == 0
    body = _m4a_function_body("sample_destruction_occupancy")
    @test occursin("sample_unknown_destruction(model, params, occupancy.X; term)",
        body)
    @test !occursin("occupancy.times", body)
    @test !occursin("sample_unknown_destruction_grid", body)
end

@testset "T-A-SAMP T-A-Q4SEP real train 350-column live production call" begin
    set = _m4a_protocol_set()
    @test length(set) == UNIQUE_CLAIM_PROTOCOL.n_ics == 9
    @test all(size(exp.observations, 2) == UNIQUE_CLAIM_PROTOCOL.n_points
              for exp in set.experiments)
    model, params, term = _m4a_probe_models()
    split = unique_claim_experiment_split(set)
    @test split.train_indices === UNIQUE_CLAIM_TRAIN_INDICES ===
          (1, 2, 3, 4, 5, 6, 7)
    train_occupancy = collect_observed_occupancy(split, :train_observed_states)
    @test train_occupancy.provenance === :train_observed_states
    @test size(train_occupancy.X, 2) == 350
    @test size(train_occupancy.X, 1) > 1
    domain = functional_identifiability_domain(split, term.regulator)
    @test train_occupancy.X[term.regulator, :] == domain.z[1:350]
    fake = _m4a_protocol_fake_X(train_occupancy, term)
    q4_X = reshape(domain.z, 1, :)
    q4_train_X = reshape(domain.z[1:350], 1, :)
    @test train_occupancy.X != fake
    @test train_occupancy.X != q4_X
    @test train_occupancy.X != q4_train_X
    captured_calls = Any[]
    sampled = _m4a_with_sample_call_log(captured_calls) do
        return sample_destruction_occupancy(
            model, params, term, train_occupancy)
    end
    @test length(captured_calls) == 1
    @test captured_calls[1].model === model
    @test captured_calls[1].params === params
    @test captured_calls[1].term === term
    @test captured_calls[1].X === train_occupancy.X
    @test size(captured_calls[1].X) == size(train_occupancy.X)
    @test captured_calls[1].X == train_occupancy.X
    @test captured_calls[1].X != fake
    @test captured_calls[1].X !== fake
    @test captured_calls[1].X != q4_X
    @test captured_calls[1].X != q4_train_X
    @test size(captured_calls[1].X, 1) > 1
    @test size(captured_calls[1].X) != size(q4_X)
    @test size(captured_calls[1].X) != size(q4_train_X)
    expected = sample_unknown_destruction(
        model, params, train_occupancy.X; term)
    @test sampled == expected
end

@testset "T-A-SAMP T-A-Q4SEP real holdout 100-column live production call" begin
    set = _m4a_protocol_set()
    @test length(set) == UNIQUE_CLAIM_PROTOCOL.n_ics == 9
    @test all(size(exp.observations, 2) == UNIQUE_CLAIM_PROTOCOL.n_points
              for exp in set.experiments)
    model, params, term = _m4a_probe_models()
    split = unique_claim_experiment_split(set)
    @test split.holdout_indices === UNIQUE_CLAIM_HOLDOUT_INDICES === (8, 9)
    holdout_occupancy = collect_observed_occupancy(
        split, :holdout_observed_states)
    @test holdout_occupancy.provenance === :holdout_observed_states
    @test size(holdout_occupancy.X, 2) == 100
    @test size(holdout_occupancy.X, 1) > 1
    domain = functional_identifiability_domain(split, term.regulator)
    @test holdout_occupancy.X[term.regulator, :] == domain.z[351:450]
    fake = _m4a_protocol_fake_X(holdout_occupancy, term)
    q4_X = reshape(domain.z, 1, :)
    q4_hold_X = reshape(domain.z[351:450], 1, :)
    @test holdout_occupancy.X != fake
    @test holdout_occupancy.X != q4_X
    @test holdout_occupancy.X != q4_hold_X
    captured_calls = Any[]
    sampled = _m4a_with_sample_call_log(captured_calls) do
        return sample_destruction_occupancy(
            model, params, term, holdout_occupancy)
    end
    @test length(captured_calls) == 1
    @test captured_calls[1].model === model
    @test captured_calls[1].params === params
    @test captured_calls[1].term === term
    @test captured_calls[1].X === holdout_occupancy.X
    @test size(captured_calls[1].X) == size(holdout_occupancy.X)
    @test captured_calls[1].X == holdout_occupancy.X
    @test captured_calls[1].X != fake
    @test captured_calls[1].X !== fake
    @test captured_calls[1].X != q4_X
    @test captured_calls[1].X != q4_hold_X
    @test size(captured_calls[1].X, 1) > 1
    @test size(captured_calls[1].X) != size(q4_X)
    @test size(captured_calls[1].X) != size(q4_hold_X)
    expected = sample_unknown_destruction(
        model, params, holdout_occupancy.X; term)
    @test sampled == expected
end

@testset "T-A-DTRUTH occupancy D uses the learned-model path" begin
    split = _m4a_three_state_split()
    occupancy = collect_observed_occupancy(split, :train_observed_states)
    model, params, term = _m4a_three_state_models()
    captured_calls = Any[]
    sampled = _m4a_with_sample_call_log(captured_calls) do
        with_discover_unknown_rate_observer(
            (_...) -> error("discover_unknown_rate entered occupancy sampling")) do
            with_discover_equations_observer(
                (_...) -> error("discover_equations entered occupancy sampling")) do
                return sample_destruction_occupancy(
                    model, params, term, occupancy)
            end
        end
    end
    @test length(captured_calls) == 1
    @test captured_calls[1].X === occupancy.X
    replayed = sample_unknown_destruction(
        captured_calls[1].model, captured_calls[1].params, captured_calls[1].X;
        term = captured_calls[1].term)
    @test sampled == replayed
    r = vec(occupancy.X[term.regulator, :])
    truth_D = hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2)
    @test vec(sampled[2]) != truth_D
    normalized, _ = BioDynaX.normalize_destruction_samples(sampled[2])
    @test sampled[2] != normalized
    sample_body = _m4a_function_body("sample_destruction_occupancy")
    occ_src = _m4a_source()
    for token in ("hill_rate_truth", "equation_to_function",
                  "normalize_destruction_samples")
        @test !occursin(token, sample_body)
        @test !occursin(token, occ_src)
    end
end

@testset "T-A-M1 composer still uses _regulator_grid(split.train), not occupancy" begin
    model, params, term = _m4a_probe_models()
    set = _m4a_composer_set()
    split = unique_claim_experiment_split(set)
    expected = collect(_regulator_grid(split.train, term))
    captured = Ref{Any}()
    occupancy_hits = Ref(0)
    evaled = with_evaluate_unknown_rate_recovery_range_observer(r_range -> begin
            captured[] = r_range
            occupancy_hits[] += r_range isa TrajectoryOccupancy
            _m4a_dummy_evaled(term)
        end) do
        _unique_claim_rate_recovery(
            model, params, term, _ -> 0.0, set;
            order = 2, family = :hill, noise_σ = 0.0,
            data_residual_fn = _ -> 0.0)
    end
    @test captured[] !== nothing
    @test !(captured[] isa TrajectoryOccupancy)
    @test occupancy_hits[] == 0
    @test collect(captured[]) == expected
    @test collect(captured[]) != collect(_regulator_grid(split.holdout, term))
    @test collect(captured[]) != collect(_regulator_grid(set, term))
    helper = _m4a_recovery_function_body("_unique_claim_rate_recovery")
    @test occursin("r_range = _regulator_grid(split.train, term)", helper)
    @test !occursin("TrajectoryOccupancy", helper)
    @test !occursin("collect_observed_occupancy", helper)
    @test !occursin("sample_destruction_occupancy", helper)
    @test evaled.term === term
end

@testset "T-A-TIME M1 discovery still receives dummy time" begin
    model, params, term = _m4a_probe_models()
    set = _m4a_composer_set()
    occupancy = collect_observed_occupancy(
        unique_claim_experiment_split(set), :train_observed_states)
    captured = Any[]
    truth_rate = _m4a_matching_truth(model, params, term)
    with_discover_unknown_rate_observer((R, times, D, config) -> begin
            push!(captured, (;
                R = copy(R), times = copy(times), D = copy(D), config))
            return _m4a_dummy_discovery()
        end) do
        _unique_claim_rate_recovery(
            model, params, term, truth_rate, set;
            order = 2, family = :hill, noise_σ = 0.0,
            data_residual_fn = _ -> 0.0)
    end
    @test !isempty(captured)
    dummy = collect(range(0.0, 1.0; length = size(captured[1].R, 2)))
    @test captured[1].times == dummy
    @test captured[1].times != occupancy.times
    @test captured[1].times != collect(occupancy.times)
    composer = _m4a_recovery_function_body("_evaluate_unknown_rate_recovery")
    @test occursin("times = collect(range(0.0, 1.0; length = length(r)))",
        composer)
    sample_hits = Ref(0)
    with_discover_unknown_rate_observer((_...) ->
        error("occupancy sampling must not enter discover_unknown_rate")) do
        with_sample_unknown_destruction_observer(_ ->
            (sample_hits[] += 1; nothing)) do
            sample_destruction_occupancy(model, params, term, occupancy)
        end
    end
    @test sample_hits[] == 1
end

@testset "T-A-M2 T-A-RES occupancy is not attached to M1/M2/M3 objects" begin
    @test :occupancy ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test fieldnames(HoldoutEvidence) == (
        :data_residual_train, :data_residual_holdout,
        :d_rmse_holdout, :d_rmse_holdout_domain)
    @test length(fieldnames(HoldoutEvidence)) == 4
    @test :occupancy ∉ fieldnames(HoldoutEvidence)
    @test :X ∉ fieldnames(HoldoutEvidence)
    @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
    @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDomain)
    src = _m4a_source()
    for token in _M4A_FORBIDDEN_TOKENS
        @test !occursin(token, src)
    end
    collect_body = _m4a_function_body("collect_observed_occupancy")
    @test occursin("exp.observations", collect_body)
    @test occursin("reduce(hcat", collect_body)
    q4 = read(joinpath(@__DIR__, "..", "src", "FunctionalIdentifiability.jl"),
        String)
    @test !occursin("TrajectoryOccupancy", q4)
    @test !occursin("collect_observed_occupancy", q4)
    @test !occursin("sample_destruction_occupancy", q4)
    hold = read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String)
    @test !occursin("occupancy::", hold)
    @test occursin("struct HoldoutEvidence", hold)
end

@testset "T-A-INTACT ExperimentSet identity/value unchanged after occupancy" begin
    set = _m4a_protocol_set()
    snap = _m4a_snapshot_set(set)
    split = unique_claim_experiment_split(set)
    _m4a_assert_intact(set, snap)
    train_occ = collect_observed_occupancy(split, :train_observed_states)
    hold_occ = collect_observed_occupancy(split, :holdout_observed_states)
    _m4a_assert_intact(set, snap)
    @test train_occ.n_points == 350
    @test hold_occ.n_points == 100
    @test_throws ArgumentError collect_observed_occupancy(
        set, :train_observed_states)
    @test_throws ArgumentError collect_observed_occupancy(
        set, :holdout_observed_states)
    _m4a_assert_intact(set, snap)
    for i in 1:7
        @test split.train[i] === set.experiments[i]
        @test split.train[i].observations === set.experiments[i].observations
    end
    for i in 1:2
        @test split.holdout[i] === set.experiments[7 + i]
        @test split.holdout[i].observations === set.experiments[7 + i].observations
    end
    occ_src = _m4a_source()
    for mutator in _M4A_FORBIDDEN_MUTATORS
        @test !occursin(mutator * "(", occ_src)
    end
    @test !occursin(r"\.experiments\s*\[[^\]]+\]\s*=", occ_src)
end

@testset "T-A-VECTOR ambiguous 9-experiment vector input is rejected" begin
    set = _m4a_nine_ic_set()
    logged = _M4ALoggedExperiments(collect(set.experiments),
        _M4AMutationLog(String[]))
    @test_throws ArgumentError collect_observed_occupancy(
        logged, :train_observed_states)
    @test_throws ArgumentError collect_observed_occupancy(
        logged, :holdout_observed_states)
    @test_throws ArgumentError collect_observed_occupancy(
        collect(set.experiments), :train_observed_states)
    @test_throws ArgumentError collect_observed_occupancy(
        collect(set.experiments), :holdout_observed_states)
    @test_throws DimensionMismatch collect_observed_occupancy(
        collect(set.experiments), :train_observed_states;
        split_indices = UNIQUE_CLAIM_TRAIN_INDICES)
    @test_throws DimensionMismatch collect_observed_occupancy(
        collect(set.experiments), :holdout_observed_states;
        split_indices = UNIQUE_CLAIM_HOLDOUT_INDICES)
    @test isempty(logged.log.calls)
    proto = _m4a_protocol_set()
    proto_logged = _M4ALoggedExperiments(collect(proto.experiments),
        _M4AMutationLog(String[]))
    @test_throws ArgumentError collect_observed_occupancy(
        proto_logged, :train_observed_states)
    @test_throws ArgumentError collect_observed_occupancy(
        proto_logged, :holdout_observed_states)
    @test isempty(proto_logged.log.calls)
    split = unique_claim_experiment_split(proto)
    train_logged = _M4ALoggedExperiments(collect(split.train.experiments),
        _M4AMutationLog(String[]))
    train_occ = collect_observed_occupancy(
        train_logged, :train_observed_states;
        split_indices = UNIQUE_CLAIM_TRAIN_INDICES)
    @test train_occ.n_points == 350
    @test train_occ.split_indices == (1, 2, 3, 4, 5, 6, 7)
    @test isempty(train_logged.log.calls)
    hold_logged = _M4ALoggedExperiments(collect(split.holdout.experiments),
        _M4AMutationLog(String[]))
    hold_occ = collect_observed_occupancy(
        hold_logged, :holdout_observed_states;
        split_indices = UNIQUE_CLAIM_HOLDOUT_INDICES)
    @test hold_occ.n_points == 100
    @test hold_occ.split_indices == (8, 9)
    @test isempty(hold_logged.log.calls)
end

@testset "T-A-SAMP occupancy spy is a full call log" begin
    src = read(@__FILE__, String)
    last_entry = string("captured_calls", "[", "end", "]")
    @test occursin("push!(captured_calls", src)
    @test occursin("length(captured_calls) == 1", src)
    @test !occursin(last_entry, src)
    @test !occursin(r"captured\[\]\s*=\s*obs", src)
end
