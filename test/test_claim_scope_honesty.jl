@testset "claim-scope honesty helpers are not exported" begin
    @test !(:howto_csv_fixture_row in names(BioDynaX))
    @test !(:assert_partial_obs_does_not_claim_ude_mask_train in names(BioDynaX))
    @test !(:graph_prior_boolean_lock in names(BioDynaX))
    @test !(:recovery_seeds_ude_is_report_row in names(BioDynaX))
    @test !(:unknown_topology_out_of_scope_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "howto CSV is a synthetic fixture; licensed series are absent" begin
    row = BioDynaX.howto_csv_fixture_row()
    @test row.csv_exists
    @test row.header == "t,S,R"
    @test row.synthetic
    @test row.not_licensed
    @test row.sentence
    @test row.absence
    @test row.result
    @test row.no_wetlab
    @test row.no_experimental_claim
    @test row.holds
end

@testset "partial observation does not claim masked-state UDE training" begin
    ok = (; ude_mask_train_claimed = false, closed_loop_vs_data = true)
    @test BioDynaX.assert_partial_obs_does_not_claim_ude_mask_train(ok) === ok
    @test_throws ErrorException BioDynaX.assert_partial_obs_does_not_claim_ude_mask_train((;
        ude_mask_train_claimed = true))
    @test_throws ErrorException BioDynaX.assert_partial_obs_does_not_claim_ude_mask_train((;
        mask_used = true))
    @test BioDynaX.partial_obs_ude_fit_claim_source_holds()
    @test BioDynaX.partial_obs_does_not_train_unknown_edge_source()
end

@testset "graph priors lock parent booleans, not a DataDrivenSparse F1 win" begin
    lock = BioDynaX.graph_prior_boolean_lock()
    @test lock.kpi_is_f1 == false
    @test lock.beats_datadriven == false
    @test :local_has_true_parent in lock.three_state
    @test :local_false_parent in lock.three_state
    @test :Z_in_local_library in lock.six_state
    @test BioDynaX.graph_prior_docs_hold()
    three = BioDynaX.three_state_library_row()
    @test three.holds
    six = BioDynaX.six_state_library_row()
    @test six.holds
end

@testset "multi-seed UDE is a report; red gate stays 103/104" begin
    row = BioDynaX.recovery_seeds_ude_is_report_row()
    @test row.script_exists
    @test row.disclaimer
    @test row.ude_flag
    @test row.red_gate_script
    @test row.contributing_gate
    @test row.ci_runs_ude == false
    @test row.sentence
    @test row.holds
end

@testset "unknown topology and general CRN stay out of scope" begin
    row = BioDynaX.unknown_topology_out_of_scope_row()
    @test row.page_exists
    @test row.sentence
    @test row.unique_not_crn
    @test isempty(row.missing)
    @test row.holds
    @test isfile(BioDynaX.claim_scope_honesty_docs_path())
    @test BioDynaX.claim_scope_honesty_contract_holds()
end
