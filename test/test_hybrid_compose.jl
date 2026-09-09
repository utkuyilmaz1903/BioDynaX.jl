@testset "hybrid compose helpers are not exported" begin
    @test !(:neural_identity_rate in names(HybridKinetics))
    @test !(:neural_identity_rhs_row in names(HybridKinetics))
    @test !(:hybrid_identity_residual_row in names(HybridKinetics))
    @test !(:remapped_compose_row in names(HybridKinetics))
    @test !(:discover_then_compose_row in names(HybridKinetics))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "source and landing checks stay locked" begin
    @test compose_hybrid_rhs_source_holds()
    @test hybrid_data_residual_source_holds()
    @test export_rhs_rejects_failure_source_holds()
    @test sample_unknown_destruction_source_holds()
    @test hybrid_compose_index_holds()
end

@testset "neural identity recovers ude_system" begin
    hill = HybridKinetics.hill_ude_identity_path()
    @test hill.holds
    mm = HybridKinetics.mm_unknown_identity_path()
    @test mm.holds
    two = HybridKinetics.two_regulator_identity_path()
    @test two.holds
    six = HybridKinetics.six_state_identity_path()
    @test six.holds
    default = HybridKinetics.default_example_identity_path()
    @test default.holds
    three = HybridKinetics.three_state_identity_path()
    @test three.holds
end

@testset "zero-hole and multi-head consistency" begin
    zero = HybridKinetics.linear_zero_hole_compose_row()
    @test zero.holds
    @test zero.n_terms == 0
    dual = HybridKinetics.dual_only_throws_row()
    @test dual.holds
    remap = HybridKinetics.remapped_compose_row()
    @test remap.holds
    @test remap.compiles == 0
    skipped = HybridKinetics.skipped_duplicate_compose_row()
    @test skipped.holds
    mm_known = HybridKinetics.mm_known_no_compose_row()
    @test mm_known.holds
    repress = HybridKinetics.repressilator_no_compose_row()
    @test repress.holds
    middle = HybridKinetics.skipped_middle_compose_row()
    @test middle.holds
end

@testset "failed discovery does not export a hybrid RHS" begin
    failed = HybridKinetics.failed_export_rhs_row()
    @test failed.holds
    empty = HybridKinetics.empty_export_rhs_row()
    @test empty.holds
    discover = HybridKinetics.discover_then_compose_row()
    @test discover.holds
    @test discover.n_ics == 1
end

@testset "residuals stay compile-free data residuals" begin
    known = HybridKinetics.hill_known_generate_unknown_identity_row()
    @test known.holds
    multi = HybridKinetics.multi_ic_identity_residual_row()
    @test multi.holds
    @test multi.compiles == 0
    constant = HybridKinetics.constant_rate_changes_residual_row()
    @test constant.holds
    sample = HybridKinetics.sample_destruction_matches_identity_row()
    @test sample.holds
    irregular = HybridKinetics.irregular_times_residual_row()
    @test irregular.holds
    exploding = HybridKinetics.failed_solve_residual_is_inf_row()
    @test exploding.holds
    competitive = HybridKinetics.competitive_unknown_identity_path()
    @test competitive.holds
    dual_terms = HybridKinetics.dual_per_term_compose_row()
    @test dual_terms.holds
    session = HybridKinetics.session_predict_hybrid_row()
    @test session.holds
    @test session.compiles == 0
    normalized = normalize_destruction_honesty_row()
    @test normalized.holds
    explicit_fn = HybridKinetics.equation_to_function_explicit_row()
    @test explicit_fn.holds
    no_compile = HybridKinetics.compose_does_not_compile_row()
    @test no_compile.holds
    shape = HybridKinetics.residual_shape_guard_row()
    @test shape.holds
    kinetic = HybridKinetics.kinetic_known_no_compose_row()
    @test kinetic.holds
    typed = HybridKinetics.hybrid_compose_typed_matrix()
    @test typed.holds
    smoke = HybridKinetics.reference_protocol_smoke_identity_row()
    @test smoke.holds
    @test smoke.n_ics == 1
    @test smoke.n_points == 8
    vs = HybridKinetics.hybrid_compose_smoke_vs_protocol_row()
    @test vs.holds
    @test vs.protocol_ics == 9
    @test vs.protocol_points == 50
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"HybridCompose.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "HybridCompose.jl"))
end
