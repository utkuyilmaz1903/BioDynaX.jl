@testset "hybrid residual helpers are not exported" begin
    @test !(:hybrid_residual_sciml_solve in names(BioDynaX))
    @test !(:residual_solver_agreement_row in names(BioDynaX))
    @test !(:noise0_vs_noisy_residual_row in names(BioDynaX))
    @test !(:smoke_vs_protocol_residual_row in names(BioDynaX))
    @test !(:HybridResidualRow in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.hybrid_data_residual_uses_sciml_solve_source_holds()
    @test BioDynaX.hybrid_residual_sciml_solve_source_holds()
    @test BioDynaX.hybrid_residual_model_solve_source_holds()
    @test BioDynaX.predict_ude_uses_odeproblem_source_holds()
    @test BioDynaX.hybrid_residual_source_holds()
    @test BioDynaX.hybrid_residual_docs_hold()
    @test BioDynaX.hybrid_residual_landing_docs_hold()
    @test BioDynaX.hybrid_residual_docs_mention_helpers()
    @test BioDynaX.hybrid_residual_example_source_holds()
    @test BioDynaX.hybrid_residual_index_holds()
    @test BioDynaX.hybrid_residual_contract() ==
          BioDynaX.hybrid_residual_locked_sentences().solve
    violations = BioDynaX.hybrid_residual_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    @test BioDynaX.hybrid_residual_contract_holds()
end

@testset "identity residual agrees with SciMLBase.solve and predict_ude" begin
    hill = BioDynaX.hill_residual_solver_path()
    @test hill.holds
    @test hill.agree.compiles == 0
    mm = BioDynaX.mm_residual_solver_path()
    @test mm.holds
    two = BioDynaX.two_regulator_residual_solver_path()
    @test two.holds
    six = BioDynaX.six_state_residual_solver_path()
    @test six.holds
    default = BioDynaX.default_example_residual_solver_path()
    @test default.holds
    three = BioDynaX.three_state_residual_solver_path()
    @test three.holds
    competitive = BioDynaX.competitive_residual_solver_path()
    @test competitive.holds
end

@testset "failed compose paths stay failed" begin
    linear = BioDynaX.failed_compose_linear_term_row()
    @test linear.holds
    empty = BioDynaX.failed_compose_empty_terms_row()
    @test empty.holds
    dual = BioDynaX.failed_compose_dual_only_row()
    @test dual.holds
    @test dual.admits == false
    failed = BioDynaX.failed_compose_export_row()
    @test failed.holds
    empty_export = BioDynaX.failed_compose_empty_export_row()
    @test empty_export.holds
    exploding = BioDynaX.hybrid_residual_failed_solve_row()
    @test exploding.holds
    shape = BioDynaX.hybrid_residual_shape_guard_row()
    @test shape.holds
    wrong = BioDynaX.failed_compose_wrong_rate_row()
    @test wrong.holds
end

@testset "noise-0 residual is not the noisy residual" begin
    noise = BioDynaX.noise_does_not_paint_f1_row()
    @test noise.holds
    @test noise.vs_clean < 1e-6
    @test noise.vs_noisy > noise.vs_clean
    built = BioDynaX.hybrid_linear_unknown_model(401)
    grid = BioDynaX.noise_grid_residual_row(
        built.model, built.packed, [0.30, 0.25])
    @test grid.holds
    @test grid.rows[1].noise_σ == 0.0
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
end

@testset "smoke residual is not the protocol residual" begin
    smoke = BioDynaX.smoke_vs_protocol_residual_row()
    @test smoke.holds
    @test smoke.smoke_ics == 1
    @test smoke.protocol_ics == 9
    @test smoke.smoke_points == 8
    @test smoke.protocol_points == 50
    self = BioDynaX.smoke_identity_on_self_row()
    @test self.holds
    protocol = BioDynaX.protocol_fingerprint_not_dropped_row()
    @test protocol.holds
    @test protocol.n_ics == 9
    @test protocol.n_points == 50
    @test protocol.seed == 103
end

@testset "multi-head and multi-IC residuals stay compile-free" begin
    remap = BioDynaX.remapped_residual_solver_row()
    @test remap.holds
    @test remap.compiles == 0
    skipped = BioDynaX.skipped_duplicate_residual_solver_row()
    @test skipped.holds
    middle = BioDynaX.skipped_middle_residual_solver_row()
    @test middle.holds
    multi = BioDynaX.multi_ic_residual_solver_row()
    @test multi.holds
    @test multi.compiles == 0
    known = BioDynaX.hill_known_generate_unknown_solver_row()
    @test known.holds
    session = BioDynaX.session_residual_solver_path()
    @test session.holds
    typed = BioDynaX.hybrid_residual_typed_matrix()
    @test typed.holds
    zero = BioDynaX.linear_zero_hole_residual_row()
    @test zero.holds
    mm_known = BioDynaX.mm_known_no_residual_row()
    @test mm_known.holds
    repress = BioDynaX.repressilator_no_residual_row()
    @test repress.holds
    kinetic = BioDynaX.kinetic_known_no_residual_row()
    @test kinetic.holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"HybridResidual.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "docs", "src", "hybrid-residual.md"))
    @test isfile(joinpath(@__DIR__, "..", "src", "HybridResidual.jl"))
    make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    @test occursin("hybrid-residual.md", make)
    howto = read(joinpath(@__DIR__, "..", "docs", "src", "howto.md"), String)
    @test occursin("hybrid-residual", howto)
    @test occursin("hybrid_data_residual", howto)
    sciml = read(joinpath(@__DIR__, "..", "docs", "src", "sciml.md"), String)
    @test occursin(BioDynaX.hybrid_residual_contract(), sciml)
    @test occursin("hybrid-residual", join(BioDynaX.unique_claim_user_doc_paths(), " "))
end
