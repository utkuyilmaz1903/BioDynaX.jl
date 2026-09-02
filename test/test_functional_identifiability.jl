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
        "assess_functional_identifiability",
        "FunctionalIdentifiabilityDiagnostic",
        "build_ude_model",
        "fit_unknown_destruction",
        "predict_ude",
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
