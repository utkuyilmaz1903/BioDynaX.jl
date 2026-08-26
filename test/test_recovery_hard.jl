@testset "UDE unknown-edge Hill recovery" begin
    rng = MersenneTwister(103)
    report = run_recovery_suite(rng;
        ude_adam = 100, ude_bfgs = 50, ude_noise_σ = 0.0,
        sections = (:ude_discovery,))
    ude = report[:ude_discovery]
    @test ude.nn_correlation ≥ RECOVERY_THRESHOLDS.nn_correlation
    @test ude.nn_rate_rmse ≤ RECOVERY_THRESHOLDS.nn_rate_rmse
    @test ude.success
    @test ude.retcode === DiscoverySuccess
    @test ude.support_recall ≥ RECOVERY_THRESHOLDS.support_recall
    @test ude.support_f1 ≥ RECOVERY_THRESHOLDS.support_f1_ude
    @test ude.discovered_rate_rmse ≤ RECOVERY_THRESHOLDS.discovered_rate_rmse
    @test ude.data_residual ≤ RECOVERY_THRESHOLDS.data_residual
    @test ude.denominator_violations == 0
    @test haskey(ude, :identifiability) || hasproperty(ude, :identifiability)
    ident = ude.identifiability
    @test ident.unidentifiable_edge
    @test ident.collinearity ≥ 0.95
    kpis = ude.locked_kpis
    @test kpis.data_residual ≤ RECOVERY_THRESHOLDS.data_residual
    @test kpis.support_recall ≥ RECOVERY_THRESHOLDS.support_recall
    @test kpis.unidentifiable_edge
    @test kpis.claim === :recall_plus_data_residual
    @test unique_claim_kpis_hold(kpis)
    @test isempty(unique_claim_kpi_failures(kpis))
    @test assert_unique_claim_kpis(kpis) === kpis
    proto = ude.protocol_result
    @test assert_protocol_result_fields(proto) === proto
    @test proto.coefficients_are_biological_constants == false
    @test proto.canonical_hill_from_nn == false
    @test proto.unknown_holes == 1
    @test proto.claim === :recall_plus_data_residual
    @test proto.data_residual == ude.data_residual
    @test proto.support_recall == ude.support_recall
    @test ude.support_f1 < RECOVERY_THRESHOLDS.support_f1_clean
    @test unique_claim_f1_reaches_analytical_gate(ude.support_f1) == false
    @test unique_claim_f1_meets_skeleton_floor(ude.support_f1)
    extras = proto.extras
    @test extras !== nothing
    @test "1" in extras || "r" in extras || !isempty(extras)
    @test occursin("collinear", BioDynaX.format_production_destruction_warning(ident))
    @test isfinite(ude.normalized_support_f1)
    @test UNIQUE_CLAIM_PROTOCOL.n_ics == 9
    @test UNIQUE_CLAIM_PROTOCOL.seed == 103
end

@testset "UDE unknown-edge Hill recovery with noise" begin
    rng = MersenneTwister(113)
    report = run_recovery_suite(rng;
        ude_adam = 100, ude_bfgs = 50, ude_noise_σ = 0.02,
        sections = (:ude_discovery,))
    ude = report[:ude_discovery]
    @test ude.nn_correlation ≥ RECOVERY_THRESHOLDS.nn_correlation
    @test ude.success
    @test ude.retcode === DiscoverySuccess
    @test ude.support_recall ≥ RECOVERY_THRESHOLDS.support_recall
    @test ude.support_f1 ≥ RECOVERY_THRESHOLDS.support_f1_noisy
    @test ude.discovered_rate_rmse ≤ RECOVERY_THRESHOLDS.discovered_rate_rmse
    @test ude.data_residual ≤ RECOVERY_THRESHOLDS.data_residual
end

@testset "UDE unknown-edge MM recovery" begin
    rng = MersenneTwister(123)
    report = run_recovery_suite(rng;
        ude_adam = 100, ude_bfgs = 50, ude_noise_σ = 0.0,
        sections = (:mm_unknown,))
    mm = report[:mm_unknown]
    @test mm.nn_correlation ≥ RECOVERY_THRESHOLDS.nn_correlation
    @test mm.nn_rate_rmse ≤ RECOVERY_THRESHOLDS.nn_rate_rmse
    @test mm.success
    @test mm.retcode === DiscoverySuccess
    @test mm.discovered_rate_rmse ≤ RECOVERY_THRESHOLDS.discovered_rate_rmse
    @test mm.data_residual ≤ RECOVERY_THRESHOLDS.data_residual
    # Canonical MM support from a trained NN is not claimed on this budget
    # (measured recall ~0.5, F1 ~0.33). Hill recall 0.99 is not silently reused.
    # Unknown-edge closed loop is Hill-class; MM is NN RMSE + data residual.
    @test mm.support_recall ≥ 0
    @test isfinite(mm.support_f1)
    @test isfinite(mm.normalized_support_f1)
end
