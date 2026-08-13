@testset "UDE unknown-edge Hill recovery" begin
    rng = MersenneTwister(103)
    report = run_recovery_suite(rng;
        ude_adam = 80, ude_bfgs = 40, ude_noise_σ = 0.0,
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
end

@testset "UDE unknown-edge Hill recovery with noise" begin
    rng = MersenneTwister(113)
    report = run_recovery_suite(rng;
        ude_adam = 80, ude_bfgs = 40, ude_noise_σ = 0.02,
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
        ude_adam = 80, ude_bfgs = 40, ude_noise_σ = 0.0,
        sections = (:mm_unknown,))
    mm = report[:mm_unknown]
    @test mm.nn_correlation ≥ RECOVERY_THRESHOLDS.nn_correlation
    @test mm.nn_rate_rmse ≤ RECOVERY_THRESHOLDS.nn_rate_rmse
    @test mm.success
    @test mm.retcode === DiscoverySuccess
    @test mm.support_recall ≥ RECOVERY_THRESHOLDS.support_recall
    @test mm.support_f1 ≥ RECOVERY_THRESHOLDS.support_f1_ude
    @test mm.discovered_rate_rmse ≤ RECOVERY_THRESHOLDS.discovered_rate_rmse
    @test mm.data_residual ≤ RECOVERY_THRESHOLDS.data_residual
end
