@testset "claim-metric honesty helpers are not exported" begin
    @test !(:ude_f1_attempt_live_row in names(BioDynaX))
    @test !(:unidentifiable_edge_from_fisher in names(BioDynaX))
    @test !(:mm_unknown_claim_holds in names(BioDynaX))
    @test !(:mm_unknown_claim_gates in names(BioDynaX))
    @test !(:claim_metric_honesty_contract_holds in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "same-library UDE extras do not reach the analytical F1 gate" begin
    row = BioDynaX.ude_f1_attempt_live_row()
    @test row.success
    @test row.extras_one
    @test row.extras_r
    @test row.reaches_clean == false
    @test row.extracted_hill == false
    @test row.meets_skeleton
    @test row.f1 < RECOVERY_THRESHOLDS.support_f1_clean
    @test unique_claim_f1_reaches_analytical_gate(row.f1) == false
    @test row.verdict === :extras_remain_claim_stays_recall_plus_residual
    @test row.floor == 0.50
    @test row.clean_gate == 0.99
    @test row.holds
    @test unique_claim_f1_attempt_holds()
    @test BioDynaX.extracted_hill_docs_hold()
    @test isempty(BioDynaX.extracted_hill_forbidden_hits())
end

@testset "unidentifiable_edge is Fisher cosine or condition number" begin
    formula = BioDynaX.unidentifiable_edge_formula_row()
    @test formula.cond_only
    @test formula.cosine_only
    @test formula.neither == false
    @test formula.nan_pair == false
    @test formula.coeff_when_edge == false
    @test formula.holds
    @test BioDynaX.printed_protocol_not_structural_holds()
    @test BioDynaX.identifiability_docs_not_structural_hold()
    ident = (; unidentifiable_edge = true, production_param = :k_prod)
    @test coefficients_are_biological_constants(ident) == false
    @test coefficients_are_biological_constants((;
        unidentifiable_edge = false))
    text = format_protocol_result(ident; residual = 0.003)
    @test occursin("not StructuralIdentifiability.jl", text)
    @test occursin("practical Fisher/Jacobian", text)
end

@testset "MM unknown does not reuse the Hill recall gate" begin
    gates = BioDynaX.mm_unknown_claim_gates()
    @test gates.applies_hill_recall == false
    @test gates.applies_hill_f1_clean == false
    @test gates.family === :mm
    @test gates.nn_rate_rmse == RECOVERY_THRESHOLDS.nn_rate_rmse
    @test gates.data_residual == RECOVERY_THRESHOLDS.data_residual
    @test gates.hill_recall == 0.99
    @test gates.measured_recall == 0.5
    @test gates.measured_f1 == 0.33
    mm_ok = (;
        nn_rate_rmse = 0.04,
        data_residual = 0.01,
        support_recall = 0.5,
        support_f1 = 0.33)
    @test BioDynaX.mm_unknown_claim_holds(mm_ok)
    hill_kpis = locked_ude_kpis((;
        data_residual = 0.01,
        support_recall = 0.5,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpis_hold(hill_kpis) == false
    @test unique_claim_kpi_failures(hill_kpis) == [:support_recall]
    @test BioDynaX.mm_unknown_hard_job_source_holds()
    @test BioDynaX.mm_unknown_recovery_family_source_holds()
    @test BioDynaX.mm_unknown_docs_hold()
    mm_bad = (;
        nn_rate_rmse = 0.20,
        data_residual = 0.01,
        support_recall = 0.99,
        support_f1 = 0.99)
    @test BioDynaX.mm_unknown_claim_holds(mm_bad) == false
end

@testset "claim-metric honesty contract" begin
    @test isfile(BioDynaX.claim_metric_honesty_docs_path())
    @test BioDynaX.claim_metric_honesty_contract_holds()
end
