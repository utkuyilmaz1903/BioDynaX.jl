@testset "hybrid compose helpers are not exported" begin
    @test !(:neural_identity_rate in names(BioDynaX))
    @test !(:neural_identity_rhs_row in names(BioDynaX))
    @test !(:hybrid_identity_residual_row in names(BioDynaX))
    @test !(:remapped_compose_row in names(BioDynaX))
    @test !(:discover_then_compose_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.compose_hybrid_rhs_source_holds()
    @test BioDynaX.hybrid_data_residual_source_holds()
    @test BioDynaX.export_rhs_rejects_failure_source_holds()
    @test BioDynaX.sample_unknown_destruction_source_holds()
    @test BioDynaX.hybrid_compose_source_holds()
    @test BioDynaX.hybrid_compose_docs_hold()
    @test BioDynaX.hybrid_compose_landing_docs_hold()
    @test BioDynaX.hybrid_compose_docs_mention_helpers()
    @test BioDynaX.hybrid_compose_example_source_holds()
    @test BioDynaX.hybrid_compose_index_holds()
    @test BioDynaX.hybrid_compose_contract() ==
          BioDynaX.hybrid_compose_locked_sentences().identity
    violations = BioDynaX.hybrid_compose_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    @test BioDynaX.hybrid_compose_contract_holds()
end

@testset "neural identity recovers ude_system" begin
    hill = BioDynaX.hill_ude_identity_path()
    @test hill.holds
    mm = BioDynaX.mm_unknown_identity_path()
    @test mm.holds
    two = BioDynaX.two_regulator_identity_path()
    @test two.holds
    six = BioDynaX.six_state_identity_path()
    @test six.holds
    default = BioDynaX.default_example_identity_path()
    @test default.holds
    three = BioDynaX.three_state_identity_path()
    @test three.holds
end

@testset "zero-hole and multi-head honesty" begin
    zero = BioDynaX.linear_zero_hole_compose_row()
    @test zero.holds
    @test zero.n_terms == 0
    dual = BioDynaX.dual_only_throws_row()
    @test dual.holds
    remap = BioDynaX.remapped_compose_row()
    @test remap.holds
    @test remap.compiles == 0
    skipped = BioDynaX.skipped_duplicate_compose_row()
    @test skipped.holds
    mm_known = BioDynaX.mm_known_no_compose_row()
    @test mm_known.holds
    repress = BioDynaX.repressilator_no_compose_row()
    @test repress.holds
    middle = BioDynaX.skipped_middle_compose_row()
    @test middle.holds
end

@testset "failed discovery does not export a hybrid RHS" begin
    failed = BioDynaX.failed_export_rhs_row()
    @test failed.holds
    empty = BioDynaX.empty_export_rhs_row()
    @test empty.holds
    discover = BioDynaX.discover_then_compose_row()
    @test discover.holds
    @test discover.n_ics == 1
end

@testset "residuals stay compile-free data residuals" begin
    known = BioDynaX.hill_known_generate_unknown_identity_row()
    @test known.holds
    multi = BioDynaX.multi_ic_identity_residual_row()
    @test multi.holds
    @test multi.compiles == 0
    constant = BioDynaX.constant_rate_changes_residual_row()
    @test constant.holds
    sample = BioDynaX.sample_destruction_matches_identity_row()
    @test sample.holds
    irregular = BioDynaX.irregular_times_residual_row()
    @test irregular.holds
    exploding = BioDynaX.failed_solve_residual_is_inf_row()
    @test exploding.holds
    competitive = BioDynaX.competitive_unknown_identity_path()
    @test competitive.holds
    dual_terms = BioDynaX.dual_per_term_compose_row()
    @test dual_terms.holds
    session = BioDynaX.session_predict_hybrid_row()
    @test session.holds
    @test session.compiles == 0
    normalized = BioDynaX.normalize_destruction_honesty_row()
    @test normalized.holds
    explicit_fn = BioDynaX.equation_to_function_explicit_row()
    @test explicit_fn.holds
    no_compile = BioDynaX.compose_does_not_compile_row()
    @test no_compile.holds
    shape = BioDynaX.residual_shape_guard_row()
    @test shape.holds
    kinetic = BioDynaX.kinetic_known_no_compose_row()
    @test kinetic.holds
    typed = BioDynaX.hybrid_compose_typed_matrix()
    @test typed.holds
    smoke = BioDynaX.unique_claim_smoke_identity_row()
    @test smoke.holds
    @test smoke.n_ics == 1
    @test smoke.n_points == 8
    vs = BioDynaX.hybrid_compose_smoke_vs_protocol_row()
    @test vs.holds
    @test vs.protocol_ics == 9
    @test vs.protocol_points == 50
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"HybridCompose.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "docs", "src", "hybrid-compose.md"))
    @test isfile(joinpath(@__DIR__, "..", "src", "HybridCompose.jl"))
    make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    @test occursin("hybrid-compose.md", make)
    howto = read(joinpath(@__DIR__, "..", "docs", "src", "howto.md"), String)
    @test occursin("hybrid-compose", howto)
    @test occursin("compose_hybrid_rhs", howto)
    sciml = read(joinpath(@__DIR__, "..", "docs", "src", "sciml.md"), String)
    @test occursin(BioDynaX.hybrid_compose_contract(), sciml)
    @test occursin("hybrid-compose", join(BioDynaX.unique_claim_user_doc_paths(), " "))
end
