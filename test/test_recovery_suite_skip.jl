@testset "recovery suite skip helpers are not exported" begin
    @test !(:RecoverySuiteSectionSpec in names(BioDynaX))
    @test !(:recovery_suite_plan in names(BioDynaX))
    @test !(:with_train_unknown_edge_counter in names(BioDynaX))
    @test !(:recovery_suite_section_body in names(BioDynaX))
    @test !(:skipped_unique_claim_does_not_train in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "section spec matrix names unique-claim trainers" begin
    matrix = BioDynaX.recovery_suite_spec_matrix()
    @test matrix.holds
    @test issetequal(matrix.trainers, (:ude_discovery, :mm_unknown))
    @test :linear in matrix.open_known
    @test :ude_discovery in matrix.unique_claim
    spec = BioDynaX.recovery_suite_section_spec(:ablation)
    @test spec.trains_unknown_edge == false
    @test spec.compiles == false
    @test spec.discovers
end

@testset "default plan still includes both unique-claim trainers" begin
    default = BioDynaX.default_suite_plan_includes_trainers()
    @test default.holds
    empty = BioDynaX.skip_empty_unique_claim_plan()
    @test empty.holds
    @test empty.would_train == false
    @test BioDynaX.recovery_suite_would_train_unknown_edge()
    @test !BioDynaX.recovery_suite_would_train_unknown_edge((:linear, :mm))
end

@testset "every suite section is gated by if :name in wanted" begin
    @test BioDynaX.recovery_suite_all_sections_gated()
    source = BioDynaX.recovery_suite_section_source_matrix()
    @test source.holds
    @test source.gated
    @test issetequal(source.trainer_sections, (:ude_discovery, :mm_unknown))
    @test BioDynaX.train_unknown_edge_only_in_unique_claim_source()
    @test BioDynaX.recovery_suite_default_sections_source()
end

@testset "ident_interventions and partial_obs do not call _train_unknown_edge" begin
    @test BioDynaX.ident_interventions_does_not_train_unknown_edge_source()
    @test BioDynaX.partial_obs_does_not_train_unknown_edge_source()
    @test BioDynaX.unique_claim_non_trainers_source_hold()
end

@testset "default suite minus unique-claim trainers does not train them" begin
    mm = BioDynaX.skip_mm_only_report()
    @test mm.holds
    @test mm.counter == 0
    @test mm.keys == (:mm,)
    minus = BioDynaX.skip_default_minus_trainers_report()
    @test minus.holds
    @test minus.counter == 0
    @test :ude_discovery in minus.skipped_trainers
    @test :mm_unknown in minus.skipped_trainers
    @test !(:ude_discovery in minus.keys)
    @test !(:mm_unknown in minus.keys)
end

@testset "skipped linear section does not train a unique-claim UDE" begin
    report = BioDynaX.skip_linear_only_report()
    @test report.holds
    @test report.counter == 0
    @test report.keys == (:linear,)
    @test issetequal(report.skipped_trainers, (:ude_discovery, :mm_unknown))
end

@testset "skipped linear section still compiles the linear model only" begin
    report = BioDynaX.skip_linear_compile_report()
    @test report.holds
    @test report.train == 0
    @test report.compile ≥ 1
end

@testset "ablation, identifiability, and literature skip unique-claim train" begin
    ablation = BioDynaX.skip_ablation_only_report()
    @test ablation.holds
    @test ablation.counter == 0
    ident = BioDynaX.skip_identifiability_only_report()
    @test ident.holds
    @test ident.counter == 0
    literature = BioDynaX.skip_literature_only_report()
    @test literature.holds
    @test literature.counter == 0
end

@testset "cost catalog matches the hole-policy matrix" begin
    cost = BioDynaX.recovery_suite_cost_matrix()
    @test cost.holds
    @test cost.n == length(recovery_suite_sections())
    ude = BioDynaX.recovery_suite_section_cost_row(:ude_discovery)
    @test ude.trains_unknown_edge
    @test ude.uses_admit
    @test ude.hole_policy === :exactly_one
    linear = BioDynaX.recovery_suite_section_cost_row(:linear)
    @test linear.trains_ude
    @test linear.trains_unknown_edge == false
    @test linear.hole_policy === :open
end

@testset "skipped reports keep the catalog keys and zero train count" begin
    keys = BioDynaX.recovery_suite_report_key_matrix()
    @test keys.holds
    @test :protocol_result in keys.ude
    literature = BioDynaX.skip_literature_only_report()
    @test BioDynaX.skip_report_has_expected_keys(literature, :literature)
    @test literature.report[:literature].experimental_csv == false
    @test literature.report[:literature].unique_claim_protocol == false
end

@testset "graph-prior sections skip unique-claim train" begin
    report = BioDynaX.skip_graph_prior_report()
    @test report.holds
    @test report.three.counter == 0
    @test report.wrong.counter == 0
end

@testset "competitive_unknown skip stays analytical" begin
    report = BioDynaX.skip_competitive_unknown_key_report()
    @test report.holds
    @test report.report.report[:competitive_unknown].canonical_f1_claimed == false
end

@testset "skip index joins spec, needles, and report keys" begin
    index = BioDynaX.recovery_suite_skip_index()
    @test index.holds
    @test index.n == length(recovery_suite_sections())
    @test issetequal(index.trainers, (:ude_discovery, :mm_unknown))
    rng = BioDynaX.recovery_suite_shared_rng_honesty()
    @test rng.holds
    @test rng.skip_linear_changes_ude_rng
    text = BioDynaX.format_recovery_suite_skip_index()
    @test occursin("ude_discovery", text)
    @test occursin("exactly_one", text)
    @test BioDynaX.recovery_suite_skip_markdown_holds()
end

@testset "every gated section body keeps its catalog needles" begin
    matrix = BioDynaX.recovery_suite_needles_matrix()
    @test matrix.holds
    @test matrix.n == length(recovery_suite_sections())
    @test BioDynaX.unique_claim_trainer_keeps_protocol_source()
    @test BioDynaX.recovery_suite_section_needles_hold(:ude_discovery)
    @test BioDynaX.recovery_suite_section_needles_hold(:literature)
end

@testset "skip contract and docs hold" begin
    @test BioDynaX.recovery_suite_benchmark_fast_skips_trainers()
    @test BioDynaX.recovery_suite_seeds_uses_ude_only()
    @test BioDynaX.recovery_suite_sindy_baseline_uses_ablation_only()
    @test BioDynaX.recovery_suite_benchmark_skip_source_holds()
    @test BioDynaX.recovery_suite_skip_source_holds()
    @test BioDynaX.unique_claim_skip_source_holds()
    @test BioDynaX.recovery_suite_skip_docs_hold()
    @test BioDynaX.recovery_suite_skip_landing_docs_hold()
    @test BioDynaX.recovery_suite_skip_contract_holds()
    violations = BioDynaX.recovery_suite_skip_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
end

@testset "skip path does not loosen locked claim numbers" begin
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test recovery_thresholds_lock() == RECOVERY_THRESHOLDS
    @test issetequal(names(BioDynaX), collect(locked_public_names()))
    zero = build_zero_unknown_linear_network()
    @test validate_network(zero) === zero
end
