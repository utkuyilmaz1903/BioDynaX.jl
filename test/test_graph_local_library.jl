@testset "graph-local library helpers are not exported" begin
    @test !(:graph_vs_global_library_row in names(BioDynaX))
    @test !(:local_has_true_parent_gate in names(BioDynaX))
    @test !(:GraphLocalLibraryRow in names(BioDynaX))
    @test !(:wrong_graph_parent_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.local_basis_scope_source_holds()
    @test BioDynaX.local_has_true_parent_gate_source_holds()
    @test BioDynaX.recovery_suite_uses_parent_gates_source_holds()
    @test BioDynaX.candidate_parents_source_holds()
    @test BioDynaX.graph_local_library_source_holds()
    @test BioDynaX.graph_local_library_docs_hold()
    @test BioDynaX.graph_local_library_landing_docs_hold()
    @test BioDynaX.graph_local_library_docs_mention_helpers()
    @test BioDynaX.graph_local_library_example_source_holds()
    @test BioDynaX.graph_local_library_index_holds()
    @test BioDynaX.graph_local_library_contract() ==
          BioDynaX.graph_local_library_locked_sentences().prior
    violations = BioDynaX.graph_local_library_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    @test BioDynaX.graph_local_library_contract_holds()
end

@testset "graph library keeps true parents and drops wrong ones" begin
    three = BioDynaX.three_state_library_row()
    @test three.holds
    @test 2 in three.parents
    wrong = BioDynaX.wrong_graph_library_row()
    @test wrong.holds
    @test wrong.true_missing
    six = BioDynaX.six_state_library_row()
    @test six.holds
    six_wrong = BioDynaX.six_state_wrong_graph_library_row()
    @test six_wrong.holds
    ablation = BioDynaX.ablation_library_row()
    @test ablation.holds
    @test ablation.global_has_z
    @test !ablation.graph_has_z
end

@testset "fixture libraries stay compile-free topology checks" begin
    hill = BioDynaX.hill_unknown_library_row()
    @test hill.holds
    mm = BioDynaX.mm_unknown_library_row()
    @test mm.holds
    two = BioDynaX.two_regulator_library_row()
    @test two.holds
    default = BioDynaX.default_example_library_row()
    @test default.holds
    linear = BioDynaX.linear_zero_library_row()
    @test linear.holds
    dual = BioDynaX.dual_library_row()
    @test dual.holds
    @test dual.admits == false
    remap = BioDynaX.remapped_library_row()
    @test remap.holds
    repress = BioDynaX.repressilator_library_row()
    @test repress.holds
    hill_k = BioDynaX.hill_known_library_row()
    @test hill_k.holds
    mm_k = BioDynaX.mm_known_library_row()
    @test mm_k.holds
end

@testset "parent gates and discovery honesty" begin
    none = BioDynaX.nothing_candidate_gate_row()
    @test none.holds
    scope = BioDynaX.default_scope_is_graph_row()
    @test scope.holds
    invalid = BioDynaX.invalid_scope_throws_row()
    @test invalid.holds
    oor = BioDynaX.target_out_of_range_row()
    @test oor.holds
    degree = BioDynaX.degree_widens_global_row()
    @test degree.holds
    inter = BioDynaX.interactions_widen_library_row()
    @test inter.holds
    suite = BioDynaX.suite_gate_symbols_row()
    @test suite.holds
    smoke = BioDynaX.smoke_vs_protocol_discovery_row()
    @test smoke.holds
    @test smoke.protocol_ics == 9
    abl = BioDynaX.ablation_discovery_gate_row()
    @test abl.holds
    @test abl.n_ics == 1
    three = BioDynaX.three_state_discovery_gate_row()
    @test three.holds
    wrong = BioDynaX.wrong_graph_discovery_gate_row()
    @test wrong.holds
    @test wrong.local_has_true == false
    typed = BioDynaX.graph_local_library_typed_matrix()
    @test typed.holds
    middle = BioDynaX.skipped_middle_library_row()
    @test middle.holds
    kinetic = BioDynaX.kinetic_library_row()
    @test kinetic.holds
    nodist = BioDynaX.three_state_no_distractor_library_row()
    @test nodist.holds
    skipped = BioDynaX.skipped_duplicate_library_row()
    @test skipped.holds
    comp = BioDynaX.competitive_library_row()
    @test comp.holds
    suite_lib = BioDynaX.suite_section_library_matrix()
    @test suite_lib.holds
    graph_secs = BioDynaX.graph_prior_suite_sections_row()
    @test graph_secs.holds
    six_targets = BioDynaX.six_state_per_target_library_row()
    @test six_targets.holds
    three_targets = BioDynaX.three_state_per_target_library_row()
    @test three_targets.holds
    @test BioDynaX.suite_library_index_holds()
    screen = BioDynaX.screen_variables_bound_row()
    @test screen.holds
    ev = BioDynaX.evaluate_graph_library_finite_row()
    @test ev.holds
    @test BioDynaX.recovery_thresholds_untouched_library_row().holds
    @test BioDynaX.suite_parent_catalog_holds()
    @test BioDynaX.remapped_per_target_library_row().holds
    @test BioDynaX.dual_per_target_library_row().holds
    @test BioDynaX.default_per_target_library_row().holds
    extra = BioDynaX.extra_candidates_do_not_shrink_graph_row()
    @test extra.holds
    @test BioDynaX.public_export_list_untouched_library_row().holds
    ics = BioDynaX.unique_claim_not_faster_by_dropping_ics_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test ics.n_table == 9
    catalog = BioDynaX.suite_parent_set_catalog()
    @test catalog.holds
    @test catalog.n ≥ length(BioDynaX.recovery_suite_sections())
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"GraphLocalLibrary.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "docs", "src", "graph-local-library.md"))
    @test isfile(joinpath(@__DIR__, "..", "src", "GraphLocalLibrary.jl"))
    recovery = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    @test occursin("function local_has_true_parent_gate", recovery)
    @test occursin("local_has_true_parent = local_has_true_parent_gate(", recovery)
    make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    @test occursin("graph-local-library.md", make)
    howto = read(joinpath(@__DIR__, "..", "docs", "src", "howto.md"), String)
    @test occursin("graph-local-library", howto)
    @test occursin("local_has_true_parent_gate", howto)
    sciml = read(joinpath(@__DIR__, "..", "docs", "src", "sciml.md"), String)
    @test occursin(BioDynaX.graph_local_library_contract(), sciml)
    @test occursin("graph-local-library",
        join(BioDynaX.unique_claim_user_doc_paths(), " "))
end
