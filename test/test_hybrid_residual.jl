@testset "hybrid residual helpers are not exported" begin
    @test !(:hybrid_residual_sciml_solve in names(HybridKinetics))
    @test !(:residual_solver_agreement_row in names(HybridKinetics))
    @test !(:noise0_vs_noisy_residual_row in names(HybridKinetics))
    @test !(:smoke_vs_protocol_residual_row in names(HybridKinetics))
    @test !(:HybridResidualRow in names(HybridKinetics))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing checks stay locked" begin
    @test hybrid_data_residual_uses_sciml_solve_source_holds()
    @test hybrid_residual_sciml_solve_source_holds()
    @test hybrid_residual_model_solve_source_holds()
    @test predict_ude_uses_odeproblem_source_holds()
    @test hybrid_residual_index_holds()
end

@testset "identity residual agrees with SciMLBase.solve and predict_ude" begin
    hill = HybridKinetics.hill_residual_solver_path()
    @test hill.holds
    @test hill.agree.compiles == 0
    mm = HybridKinetics.mm_residual_solver_path()
    @test mm.holds
    two = HybridKinetics.two_regulator_residual_solver_path()
    @test two.holds
    six = HybridKinetics.six_state_residual_solver_path()
    @test six.holds
    default = HybridKinetics.default_example_residual_solver_path()
    @test default.holds
    three = HybridKinetics.three_state_residual_solver_path()
    @test three.holds
    competitive = HybridKinetics.competitive_residual_solver_path()
    @test competitive.holds
end

@testset "failed compose paths stay failed" begin
    linear = HybridKinetics.failed_compose_linear_term_row()
    @test linear.holds
    empty = HybridKinetics.failed_compose_empty_terms_row()
    @test empty.holds
    dual = HybridKinetics.failed_compose_dual_only_row()
    @test dual.holds
    @test dual.admits == false
    failed = HybridKinetics.failed_compose_export_row()
    @test failed.holds
    empty_export = HybridKinetics.failed_compose_empty_export_row()
    @test empty_export.holds
    exploding = HybridKinetics.hybrid_residual_failed_solve_row()
    @test exploding.holds
    shape = HybridKinetics.hybrid_residual_shape_guard_row()
    @test shape.holds
    wrong = HybridKinetics.failed_compose_wrong_rate_row()
    @test wrong.holds
end

@testset "noise-0 residual is not the noisy residual" begin
    noise = HybridKinetics.noise_does_not_paint_f1_row()
    @test noise.holds
    @test noise.vs_clean < 1e-6
    @test noise.vs_noisy > noise.vs_clean
    built = HybridKinetics.hybrid_linear_unknown_model(401)
    grid = HybridKinetics.noise_grid_residual_row(
        built.model, built.packed, [0.30, 0.25])
    @test grid.holds
    @test grid.rows[1].noise_σ == 0.0
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
end

@testset "smoke residual is not the protocol residual" begin
    smoke = HybridKinetics.smoke_vs_protocol_residual_row()
    @test smoke.holds
    @test smoke.smoke_ics == 1
    @test smoke.protocol_ics == 9
    @test smoke.smoke_points == 8
    @test smoke.protocol_points == 50
    self = HybridKinetics.smoke_identity_on_self_row()
    @test self.holds
    protocol = HybridKinetics.protocol_fingerprint_not_dropped_row()
    @test protocol.holds
    @test protocol.n_ics == 9
    @test protocol.n_points == 50
    @test protocol.seed == 103
end

@testset "multi-head and multi-IC residuals stay compile-free" begin
    remap = HybridKinetics.remapped_residual_solver_row()
    @test remap.holds
    @test remap.compiles == 0
    skipped = HybridKinetics.skipped_duplicate_residual_solver_row()
    @test skipped.holds
    middle = HybridKinetics.skipped_middle_residual_solver_row()
    @test middle.holds
    multi = HybridKinetics.multi_ic_residual_solver_row()
    @test multi.holds
    @test multi.compiles == 0
    known = HybridKinetics.hill_known_generate_unknown_solver_row()
    @test known.holds
    session = HybridKinetics.session_residual_solver_path()
    @test session.holds
    typed = HybridKinetics.hybrid_residual_typed_matrix()
    @test typed.holds
    zero = HybridKinetics.linear_zero_hole_residual_row()
    @test zero.holds
    mm_known = HybridKinetics.mm_known_no_residual_row()
    @test mm_known.holds
    repress = HybridKinetics.repressilator_no_residual_row()
    @test repress.holds
    kinetic = HybridKinetics.kinetic_known_no_residual_row()
    @test kinetic.holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"HybridResidual.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "HybridResidual.jl"))
end
