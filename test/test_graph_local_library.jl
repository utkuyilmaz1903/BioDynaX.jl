@testset "graph-local library helpers are not exported" begin
    @test !(:graph_vs_global_library_row in names(HybridKinetics))
    @test !(:local_has_true_parent_check in names(HybridKinetics))
    @test !(:GraphLocalLibraryRow in names(HybridKinetics))
    @test !(:wrong_graph_parent_row in names(HybridKinetics))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing checks stay locked" begin
    @test local_basis_scope_source_holds()
    @test local_has_true_parent_check_source_holds()
    @test recovery_suite_uses_parent_checks_source_holds()
    @test candidate_parents_source_holds()
    @test graph_local_library_index_holds()
end

@testset "graph library keeps true parents and drops wrong ones" begin
    three = HybridKinetics.three_state_library_row()
    @test three.holds
    @test 2 in three.parents
    wrong = HybridKinetics.wrong_graph_library_row()
    @test wrong.holds
    @test wrong.true_missing
    six = HybridKinetics.six_state_library_row()
    @test six.holds
    six_wrong = HybridKinetics.six_state_wrong_graph_library_row()
    @test six_wrong.holds
    ablation = HybridKinetics.ablation_library_row()
    @test ablation.holds
    @test ablation.global_has_z
    @test !ablation.graph_has_z
end

@testset "fixture libraries stay compile-free topology checks" begin
    hill = HybridKinetics.hill_unknown_library_row()
    @test hill.holds
    mm = HybridKinetics.mm_unknown_library_row()
    @test mm.holds
    two = HybridKinetics.two_regulator_library_row()
    @test two.holds
    default = HybridKinetics.default_example_library_row()
    @test default.holds
    linear = HybridKinetics.linear_zero_library_row()
    @test linear.holds
    dual = HybridKinetics.dual_library_row()
    @test dual.holds
    @test dual.admits == false
    remap = HybridKinetics.remapped_library_row()
    @test remap.holds
    repress = HybridKinetics.repressilator_library_row()
    @test repress.holds
    hill_k = HybridKinetics.hill_known_library_row()
    @test hill_k.holds
    mm_k = HybridKinetics.mm_known_library_row()
    @test mm_k.holds
end

@testset "parent checks and discovery consistency" begin
    none = HybridKinetics.nothing_candidate_check_row()
    @test none.holds
    scope = HybridKinetics.default_scope_is_graph_row()
    @test scope.holds
    invalid = HybridKinetics.invalid_scope_throws_row()
    @test invalid.holds
    oor = HybridKinetics.target_out_of_range_row()
    @test oor.holds
    degree = HybridKinetics.degree_widens_global_row()
    @test degree.holds
    inter = HybridKinetics.interactions_widen_library_row()
    @test inter.holds
    suite = HybridKinetics.suite_gate_symbols_row()
    @test suite.holds
    smoke = HybridKinetics.smoke_vs_protocol_discovery_row()
    @test smoke.holds
    @test smoke.protocol_ics == 9
    abl = HybridKinetics.ablation_discovery_check_row()
    @test abl.holds
    @test abl.n_ics == 1
    three = HybridKinetics.three_state_discovery_check_row()
    @test three.holds
    wrong = HybridKinetics.wrong_graph_discovery_check_row()
    @test wrong.holds
    @test wrong.local_has_true == false
    typed = HybridKinetics.graph_local_library_typed_matrix()
    @test typed.holds
    middle = HybridKinetics.skipped_middle_library_row()
    @test middle.holds
    kinetic = HybridKinetics.kinetic_library_row()
    @test kinetic.holds
    nodist = HybridKinetics.three_state_no_distractor_library_row()
    @test nodist.holds
    skipped = HybridKinetics.skipped_duplicate_library_row()
    @test skipped.holds
    comp = HybridKinetics.competitive_library_row()
    @test comp.holds
    suite_lib = HybridKinetics.suite_section_library_matrix()
    @test suite_lib.holds
    graph_secs = HybridKinetics.graph_prior_suite_sections_row()
    @test graph_secs.holds
    six_targets = HybridKinetics.six_state_per_target_library_row()
    @test six_targets.holds
    three_targets = HybridKinetics.three_state_per_target_library_row()
    @test three_targets.holds
    @test suite_library_index_holds()
    screen = HybridKinetics.screen_variables_bound_row()
    @test screen.holds
    ev = HybridKinetics.evaluate_graph_library_finite_row()
    @test ev.holds
    @test HybridKinetics.recovery_thresholds_untouched_library_row().holds
    @test HybridKinetics.suite_parent_catalog_holds()
    @test HybridKinetics.remapped_per_target_library_row().holds
    @test HybridKinetics.dual_per_target_library_row().holds
    @test HybridKinetics.default_per_target_library_row().holds
    extra = HybridKinetics.extra_candidates_do_not_shrink_graph_row()
    @test extra.holds
    @test HybridKinetics.public_export_list_untouched_library_row().holds
    ics = HybridKinetics.reference_protocol_not_faster_by_dropping_ics_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test ics.n_table == 9
    catalog = HybridKinetics.suite_parent_set_catalog()
    @test catalog.holds
    @test catalog.n ≥ length(HybridKinetics.recovery_suite_sections())
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"GraphLocalLibrary.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "GraphLocalLibrary.jl"))
    recovery = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    @test occursin("function local_has_true_parent_check", recovery)
    @test occursin("local_has_true_parent = local_has_true_parent_check(", recovery)
end
