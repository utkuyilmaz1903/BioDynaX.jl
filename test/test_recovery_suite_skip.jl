@testset "recovery suite skip helpers are not exported" begin
    @test !(:RecoverySuiteSectionSpec in names(HybridKinetics))
    @test !(:recovery_suite_plan in names(HybridKinetics))
    @test !(:with_train_unknown_edge_counter in names(HybridKinetics))
    @test !(:recovery_suite_section_body in names(HybridKinetics))
    @test !(:skipped_reference_protocol_does_not_train in names(HybridKinetics))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "section spec matrix names reference protocol trainers" begin
    matrix = HybridKinetics.recovery_suite_spec_matrix()
    @test matrix.holds
    @test issetequal(matrix.trainers, (:ude_discovery, :mm_unknown))
    @test :linear in matrix.open_known
    @test :ude_discovery in matrix.reference_protocol
    spec = HybridKinetics.recovery_suite_section_spec(:ablation)
    @test spec.trains_unknown_edge == false
    @test spec.compiles == false
    @test spec.discovers
end

@testset "default plan still includes both reference protocol trainers" begin
    default = HybridKinetics.default_suite_plan_includes_trainers()
    @test default.holds
    empty = HybridKinetics.skip_empty_reference_protocol_plan()
    @test empty.holds
    @test empty.would_train == false
    @test HybridKinetics.recovery_suite_would_train_unknown_edge()
    @test !HybridKinetics.recovery_suite_would_train_unknown_edge((:linear, :mm))
end

@testset "every suite section is checked by if :name in wanted" begin
    @test HybridKinetics.recovery_suite_all_sections_checked()
    source = HybridKinetics.recovery_suite_section_source_matrix()
    @test source.holds
    @test source.checked
    @test issetequal(source.trainer_sections, (:ude_discovery, :mm_unknown))
    @test HybridKinetics.train_unknown_edge_only_in_reference_protocol_source()
    @test HybridKinetics.recovery_suite_default_sections_source()
end

@testset "ident_interventions and partial_obs do not call _train_unknown_edge" begin
    @test HybridKinetics.ident_interventions_does_not_train_unknown_edge_source()
    @test HybridKinetics.partial_obs_does_not_train_unknown_edge_source()
    @test reference_protocol_non_trainers_source_hold()
end

@testset "default suite minus reference protocol trainers does not train them" begin
    mm = HybridKinetics.skip_mm_only_report()
    @test mm.holds
    @test mm.counter == 0
    @test mm.keys == (:mm,)
    minus = HybridKinetics.skip_default_minus_trainers_report()
    @test minus.holds
    @test minus.counter == 0
    @test :ude_discovery in minus.skipped_trainers
    @test :mm_unknown in minus.skipped_trainers
    @test !(:ude_discovery in minus.keys)
    @test !(:mm_unknown in minus.keys)
end

@testset "skipped linear section does not train a reference protocol UDE" begin
    report = HybridKinetics.skip_linear_only_report()
    @test report.holds
    @test report.counter == 0
    @test report.keys == (:linear,)
    @test issetequal(report.skipped_trainers, (:ude_discovery, :mm_unknown))
end

@testset "skipped linear section still compiles the linear model only" begin
    report = HybridKinetics.skip_linear_compile_report()
    @test report.holds
    @test report.train == 0
    @test report.compile ≥ 1
end

@testset "ablation, identifiability, and literature skip reference protocol train" begin
    ablation = HybridKinetics.skip_ablation_only_report()
    @test ablation.holds
    @test ablation.counter == 0
    ident = HybridKinetics.skip_identifiability_only_report()
    @test ident.holds
    @test ident.counter == 0
    literature = HybridKinetics.skip_literature_only_report()
    @test literature.holds
    @test literature.counter == 0
end

@testset "cost catalog matches the hole-policy matrix" begin
    cost = HybridKinetics.recovery_suite_cost_matrix()
    @test cost.holds
    @test cost.n == length(recovery_suite_sections())
    ude = HybridKinetics.recovery_suite_section_cost_row(:ude_discovery)
    @test ude.trains_unknown_edge
    @test ude.uses_admit
    @test ude.hole_policy === :exactly_one
    linear = HybridKinetics.recovery_suite_section_cost_row(:linear)
    @test linear.trains_ude
    @test linear.trains_unknown_edge == false
    @test linear.hole_policy === :open
end

@testset "skipped reports keep the catalog keys and zero train count" begin
    keys = HybridKinetics.recovery_suite_report_key_matrix()
    @test keys.holds
    @test :protocol_result in keys.ude
    literature = HybridKinetics.skip_literature_only_report()
    @test HybridKinetics.skip_report_has_expected_keys(literature, :literature)
    @test literature.report[:literature].experimental_csv == false
    @test literature.report[:literature].reference_protocol_protocol == false
end

@testset "graph-prior sections skip reference protocol train" begin
    report = HybridKinetics.skip_graph_prior_report()
    @test report.holds
    @test report.three.counter == 0
    @test report.wrong.counter == 0
end

@testset "competitive_unknown skip stays analytical" begin
    report = HybridKinetics.skip_competitive_unknown_key_report()
    @test report.holds
    @test report.report.report[:competitive_unknown].canonical_f1_claimed == false
end

@testset "skip index joins spec, needles, and report keys" begin
    index = HybridKinetics.recovery_suite_skip_index()
    @test index.holds
    @test index.n == length(recovery_suite_sections())
    @test issetequal(index.trainers, (:ude_discovery, :mm_unknown))
    rng = HybridKinetics.recovery_suite_shared_rng_consistency()
    @test rng.holds
    @test rng.skip_linear_changes_ude_rng
    text = HybridKinetics.format_recovery_suite_skip_index()
    @test occursin("ude_discovery", text)
    @test occursin("exactly_one", text)
    @test HybridKinetics.recovery_suite_skip_markdown_holds()
end

@testset "every checked section body keeps its catalog needles" begin
    matrix = HybridKinetics.recovery_suite_needles_matrix()
    @test matrix.holds
    @test matrix.n == length(recovery_suite_sections())
    @test HybridKinetics.reference_protocol_trainer_keeps_protocol_source()
    @test HybridKinetics.recovery_suite_section_needles_hold(:ude_discovery)
    @test HybridKinetics.recovery_suite_section_needles_hold(:literature)
end

@testset "skip rules and docs hold" begin
    @test HybridKinetics.recovery_suite_benchmark_fast_skips_trainers()
    @test HybridKinetics.recovery_suite_seeds_uses_ude_only()
    @test HybridKinetics.recovery_suite_sindy_baseline_uses_ablation_only()
    @test HybridKinetics.recovery_suite_benchmark_skip_source_holds()
    @test reference_protocol_skip_source_holds()
end

@testset "skip path does not loosen locked claim numbers" begin
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test recovery_thresholds_lock() == RECOVERY_THRESHOLDS
    @test issetequal(names(HybridKinetics), collect(locked_public_names()))
    zero = build_zero_unknown_linear_network()
    @test validate_network(zero) === zero
end
