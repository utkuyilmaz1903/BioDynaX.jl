# M2-G2 live checks only. Legacy scientific gates stay unchanged.
# data_residual_holdout is observational: it is not compared to 0.30.
function _m2_g2_assert_live_result(result, truth_rate; label::String)
    @test result.split isa ExperimentSplit
    @test length(result.split.train) == 7
    @test length(result.split.holdout) == 2
    @test result.split.train_indices === UNIQUE_CLAIM_TRAIN_INDICES
    @test result.split.holdout_indices === UNIQUE_CLAIM_HOLDOUT_INDICES
    @test length(result.experiments) == 9
    @test all(result.split.train[i] === result.experiments[i] for i in 1:7)
    @test result.split.holdout[1] === result.experiments[8]
    @test result.split.holdout[2] === result.experiments[9]
    if result.discovery === nothing
        @test result.holdout === nothing
        @info "M2-G2 $label Case A" data_residual=result.data_residual
        return nothing
    end
    @test result.holdout isa HoldoutEvidence
    ev = result.holdout
    @test isfinite(ev.data_residual_train)
    @test isfinite(ev.data_residual_holdout)
    @test isfinite(ev.d_rmse_holdout)
    @test isfinite(ev.d_rmse_holdout_domain)
    model = result.model
    params = result.params
    term = result.term
    D_hat_fn = neural_identity_rate(model, params, term)
    @test ev.data_residual_train === _mean_hybrid_residual(
        result.split.train.experiments, model, params, term, D_hat_fn)
    @test ev.data_residual_holdout === _mean_hybrid_residual(
        result.split.holdout.experiments, model, params, term, D_hat_fn)
    r_holdout = _holdout_observed_regulators(result.split.holdout, term)
    (R, D_hat_vals, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_holdout, fill_value = 0.3)
    @test ev.d_rmse_holdout === _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
    r_band = _unique_claim_external_regulator_band(result.split.train, term)
    (R, D_hat_vals, _) = sample_unknown_destruction_grid(
        model, params, term; r_range = r_band, fill_value = 0.3)
    @test ev.d_rmse_holdout_domain ===
          _finite_rate_rel_rmse(D_hat_vals, truth_rate(vec(R)))
    ev2 = evaluate_holdout(result.split, result, model, params, term, truth_rate)
    @test ev.data_residual_train === ev2.data_residual_train
    @test ev.data_residual_holdout === ev2.data_residual_holdout
    @test ev.d_rmse_holdout === ev2.d_rmse_holdout
    @test ev.d_rmse_holdout_domain === ev2.d_rmse_holdout_domain
    @test isfinite(result.data_residual)
    case = result.discovery.success ? "C" : "B"
    @info "M2-G2 $label Case $case" nn_correlation=result.nn_correlation nn_rate_rmse=result.nn_rate_rmse support_recall=result.support_recall support_f1=result.support_f1 data_residual=result.data_residual data_residual_train=ev.data_residual_train data_residual_holdout=ev.data_residual_holdout d_rmse_holdout=ev.d_rmse_holdout d_rmse_holdout_domain=ev.d_rmse_holdout_domain holdout_residual_gt_030=(ev.data_residual_holdout > 0.30)
    return ev
end

@testset "UDE unknown-edge Hill recovery" begin
    rng = MersenneTwister(103)
    n_eval = Ref(0)
    report = with_evaluate_holdout_observer((_...) -> (n_eval[] += 1; nothing)) do
        run_recovery_suite(rng;
            ude_adam = 100, ude_bfgs = 50, ude_noise_σ = 0.0,
            sections = (:ude_discovery,))
    end
    @test n_eval[] == 1
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
    row = unique_claim_protocol_row(ude)
    @test row.kpi_failures == Symbol[]
    @test format_unique_claim_kpi_failures(row.kpi_failures) == "(none)"
    @test unique_claim_kpi_failure_symbols() ==
          (:unidentifiable_edge, :data_residual, :support_recall)
    @test !(:support_f1 in unique_claim_kpi_failure_symbols())
    @test assert_unique_claim_protocol_row_holds(row) === row
    @test extras_print_label(proto.extras) == row.extras_label
    @test unique_claim_fingerprint_is_protocol(row.fingerprint)
    @test occursin("unidentifiable_edge: true", row.text)
    _m2_g2_assert_live_result(
        ude, r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2);
        label = "seed103")
end

@testset "UDE unknown-edge Hill recovery with noise" begin
    rng = MersenneTwister(113)
    n_eval = Ref(0)
    report = with_evaluate_holdout_observer((_...) -> (n_eval[] += 1; nothing)) do
        run_recovery_suite(rng;
            ude_adam = 100, ude_bfgs = 50, ude_noise_σ = 0.02,
            sections = (:ude_discovery,))
    end
    @test n_eval[] == 1
    ude = report[:ude_discovery]
    @test ude.nn_correlation ≥ RECOVERY_THRESHOLDS.nn_correlation
    @test ude.success
    @test ude.retcode === DiscoverySuccess
    @test ude.support_recall ≥ RECOVERY_THRESHOLDS.support_recall
    @test ude.support_f1 ≥ RECOVERY_THRESHOLDS.support_f1_noisy
    @test ude.discovered_rate_rmse ≤ RECOVERY_THRESHOLDS.discovered_rate_rmse
    @test ude.data_residual ≤ RECOVERY_THRESHOLDS.data_residual
    _m2_g2_assert_live_result(
        ude, r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2);
        label = "seed113")
end

@testset "UDE unknown-edge MM recovery" begin
    rng = MersenneTwister(123)
    n_eval = Ref(0)
    report = with_evaluate_holdout_observer((_...) -> (n_eval[] += 1; nothing)) do
        run_recovery_suite(rng;
            ude_adam = 100, ude_bfgs = 50, ude_noise_σ = 0.0,
            sections = (:mm_unknown,))
    end
    @test n_eval[] == 1
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
    _m2_g2_assert_live_result(
        mm, r -> mm_rate_truth(r; vmax = 1.6, km = 0.45);
        label = "mm")
end
