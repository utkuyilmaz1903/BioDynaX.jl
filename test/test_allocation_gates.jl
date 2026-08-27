@testset "allocation gate helpers are not exported" begin
    @test !(:allocation_hot in names(BioDynaX))
    @test !(:AllocationGateRow in names(BioDynaX))
    @test !(:pack_parameters_allocation_row in names(BioDynaX))
    @test !(:stlsq_workspace_reuse_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test :pack_parameters in names(BioDynaX)
    @test :parameter_schema in names(BioDynaX)
    @test :allocate_cache in names(BioDynaX)
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.allocation_gates_source_holds()
    @test BioDynaX.allocation_hot_source_holds()
    @test BioDynaX.stlsq_reuse_source_holds()
    @test BioDynaX.discovery_report_source_holds()
    @test BioDynaX.quality_gates_test_still_holds()
    @test BioDynaX.quality_gates_extended_source_holds()
    @test BioDynaX.quality_gates_ude_rhs_source_holds()
    @test BioDynaX.allocation_gates_docs_hold()
    @test BioDynaX.allocation_gates_landing_docs_hold()
    @test BioDynaX.allocation_gates_docs_mention_helpers()
    @test BioDynaX.allocation_gates_example_source_holds()
    @test BioDynaX.allocation_gates_index_holds()
    @test BioDynaX.allocation_gates_unique_claim_path_holds()
    @test BioDynaX.allocation_gates_make_listed_holds()
    @test BioDynaX.allocation_gates_changelog_holds()
    @test BioDynaX.allocation_gates_news_holds()
    @test BioDynaX.allocation_gates_architecture_holds()
    @test BioDynaX.allocation_gates_tutorial_holds()
    @test BioDynaX.allocation_gates_internals_hold()
    @test BioDynaX.allocation_workspace_catalog_holds()
    @test BioDynaX.allocation_gates_contract() ==
          BioDynaX.allocation_gates_locked_sentences().measured
    violations = BioDynaX.allocation_gates_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    @test BioDynaX.allocation_gates_contract_holds()
end

@testset "measured pack, schema, and ude_rhs! ceilings can fail" begin
    pack = BioDynaX.pack_parameters_allocation_row()
    @test pack.holds
    @test pack.typed.hot ≤ BioDynaX.ALLOCATION_GATE_LIMITS.pack_parameters
    @test BioDynaX.assert_allocation_under(
        pack.typed.hot, pack.typed.limit, :pack)
    unpack = BioDynaX.unpack_parameters_allocation_row()
    @test unpack.holds
    schema = BioDynaX.parameter_schema_allocation_row()
    @test schema.holds
    pos = BioDynaX.positive_parameter_allocation_row()
    @test pos.holds
    @test pos.typed.hot == 0
    inv = BioDynaX.inverse_softplus_allocation_row()
    @test inv.holds
    @test inv.typed.hot == 0
    stype = BioDynaX.schema_type_row()
    @test stype.holds
    @test stype.schema_type <: BioDynaX.ParameterSchema
    remap = BioDynaX.remapped_pack_allocation_row()
    @test remap.holds
    @test remap.nn_heads == 2
    kinetic = BioDynaX.kinetic_schema_allocation_row()
    @test kinetic.holds
    @test kinetic.has_custom
    rhs = BioDynaX.quality_gates_ude_rhs_live_row()
    @test rhs.holds
    @test rhs.linear_hot ≤ 512
    @test rhs.p53_hot ≤ 4096
    typed = BioDynaX.allocation_gates_typed_matrix()
    @test typed.holds
    @test typed.pos.hot == 0
end

@testset "workspace reuse must not increment same-shape resize_count" begin
    stlsq = BioDynaX.stlsq_workspace_reuse_row()
    @test stlsq.holds
    @test stlsq.same_shape
    @test stlsq.after_grow
    implicit = BioDynaX.implicit_workspace_reuse_row()
    @test implicit.holds
    @test implicit.same
    streaming = BioDynaX.streaming_implicit_reuse_row()
    @test streaming.holds
    @test streaming.same
    chunk = BioDynaX.library_chunk_reuse_row()
    @test chunk.holds
    @test chunk.same
    report = BioDynaX.discovery_stlsq_reuse_row()
    @test report.holds
    @test report.reuse_smaller
    @test report.reused_bytes < report.naive_bytes
    active = BioDynaX.collect_active_allocation_row()
    @test active.holds
    @test active.n == 16
    reuse = BioDynaX.allocation_gates_live_reuse_row()
    @test reuse.holds
end

@testset "type rows stay concrete on unexported workspaces" begin
    @test BioDynaX.pack_type_row().holds
    @test BioDynaX.allocate_cache_type_row().holds
    @test BioDynaX.unpack_type_row().holds
    @test BioDynaX.positive_parameter_type_row().holds
    @test BioDynaX.training_solver_lock_type_row().holds
    hybrid = BioDynaX.hybrid_identity_type_row()
    @test hybrid.holds
    @test hybrid.type <: BioDynaX.NeuralDestructionTerm
    @test BioDynaX.candidate_type_row().holds
    @test BioDynaX.fingerprint_type_row().holds
    @test BioDynaX.schema_head_type_row().holds
    @test BioDynaX.contract_string_type_row().holds
    @test BioDynaX.allocation_gates_live_type_row().holds
    @test BioDynaX.default_example_cache_type_row().holds
    remap = BioDynaX.remapped_cache_type_row()
    @test remap.holds
    @test remap.heads == 2
    kinetic = BioDynaX.kinetic_cache_type_row()
    @test kinetic.holds
    dummy = BioDynaX.zero_hole_dummy_head_not_schema_row()
    @test dummy.holds
    @test dummy.schema_heads == 0
end

@testset "denominator, library, and H-L surfaces keep measured gates" begin
    @test BioDynaX.denominator_violation_allocation_row().holds
    @test BioDynaX.denominator_split_allocation_row().holds
    @test BioDynaX.local_basis_allocation_row().holds
    @test BioDynaX.graph_vs_global_allocation_row().holds
    @test BioDynaX.extras_denominator_allocation_row().holds
    @test BioDynaX.library_contains_allocation_row().holds
    @test BioDynaX.graph_parent_set_allocation_row().holds
    @test BioDynaX.default_phys_allocation_row().holds
    @test BioDynaX.frozen_zero_allocation_row().holds
    parent = BioDynaX.local_parent_gate_allocation_row()
    @test parent.holds
    @test parent.closed == false
    @test BioDynaX.hybrid_residual_type_row().holds
    @test BioDynaX.identifiability_product_type_row().holds
    @test BioDynaX.graph_local_type_row().holds
    @test BioDynaX.denominator_domain_type_row().holds
    @test BioDynaX.schema_pack_type_row().holds
    @test BioDynaX.docs_executable_type_row().holds
    @test BioDynaX.docs_executable_join_type_row().holds
    leftover = BioDynaX.leftover_empty_type_row()
    @test leftover.holds
    @test leftover.n == 0
    @test BioDynaX.allocation_gates_hl_contract_strings_row().holds
    @test BioDynaX.allocation_gates_ag_helpers_exist_row().holds
    @test BioDynaX.workspace_not_exported_row().holds
    @test BioDynaX.pack_does_not_compile_row().holds
    @test BioDynaX.allocation_gates_live_surface_row().holds
end

@testset "protocol, exports, and combined F1 stay honest" begin
    smoke = BioDynaX.smoke_vs_protocol_allocation_row()
    @test smoke.holds
    @test smoke.proto_ics == 9
    @test smoke.smoke_ics == 1
    ics = BioDynaX.unique_claim_not_faster_allocation_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test BioDynaX.combined_f1_not_allocation_kpi_row().holds
    @test BioDynaX.hill_from_nn_closed_allocation_row().holds
    @test BioDynaX.validate_open_allocation_row().holds
    @test BioDynaX.recovery_thresholds_untouched_allocation_row().holds
    @test BioDynaX.public_export_untouched_allocation_row().holds
    matrix = BioDynaX.allocation_gates_fixture_matrix()
    @test matrix.holds
    @test BioDynaX.allocation_gates_report_holds()
    names = BioDynaX.allocation_gates_fixture_names()
    @test length(unique(names)) == length(names)
    @test length(names) ≥ 50
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"AllocationGates.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "docs", "src", "allocation-gates.md"))
    @test isfile(joinpath(@__DIR__, "..", "src", "AllocationGates.jl"))
    make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    @test occursin("allocation-gates.md", make)
    howto = read(joinpath(@__DIR__, "..", "docs", "src", "howto.md"), String)
    @test occursin("allocation-gates", howto)
    @test occursin("allocation_hot", howto)
    @test occursin("STLSQWorkspace", howto)
    sciml = read(joinpath(@__DIR__, "..", "docs", "src", "sciml.md"), String)
    @test occursin(BioDynaX.allocation_gates_contract(), sciml)
    tutorial = read(joinpath(@__DIR__, "..", "docs", "src", "tutorial.md"), String)
    @test occursin("allocation-gates", tutorial)
    @test occursin("allocation_hot", tutorial)
    @test occursin("allocation-gates.md",
        join(BioDynaX.unique_claim_user_doc_paths(), " "))
    quality = read(joinpath(@__DIR__, "test_quality_gates.jl"), String)
    @test occursin("allocation_gates_contract_holds", quality)
    @test occursin("pack_parameters_allocation_row", quality)
    @test occursin("stlsq_workspace_reuse_row", quality)
end
