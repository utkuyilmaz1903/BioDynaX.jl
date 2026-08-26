@testset "solve surface helpers are not exported" begin
    @test !(:SolveSurfaceRow in names(BioDynaX))
    @test !(:solve_surface_row in names(BioDynaX))
    @test !(:sensealg_boundary_row in names(BioDynaX))
    @test !(:cache_reuse_row in names(BioDynaX))
    @test !(:remake_field_row in names(BioDynaX))
    @test !(:odefunction_rhs_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "ODEFunction matches ude_system and inplace RHS" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    row = BioDynaX.odefunction_rhs_row(model, params, [0.22, 0.14])
    @test row.holds
    @test row.matches_function
    @test row.matches_inplace
    @test row.no_compile
    @test row.oop_inplace == false
    @test row.ip_inplace
end

@testset "remake of p, u0, tspan matches a fresh ODEProblem" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    row = BioDynaX.remake_field_row(model, packed, [0.22, 0.14])
    @test row.holds
    @test row.matches_p
    @test row.matches_u0
    @test row.matches_tspan
    @test row.no_compile
end

@testset "inplace cache pointer is stable across remakes" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    row = BioDynaX.cache_reuse_row(model, params, [0.22, 0.14])
    @test row.holds
    @test row.same_du
    @test row.no_compile
end

@testset "linear solve surface agrees every SciML entry" begin
    path = BioDynaX.linear_solve_surface_path()
    @test path.holds
    @test path.row.holds
    @test path.row.matches_predict
    @test path.row.matches_session
    @test path.row.matches_generate
    @test path.row.no_compile
    @test path.boundary.crossing
    @test path.cfg.algorithm_is_tsit5
    @test path.failed.nan_threw
    @test path.saveat.holds
end

@testset "sensealg 64/65 boundary is mechanistic-only" begin
    matrix = BioDynaX.sensealg_boundary_matrix()
    @test matrix.holds
    @test matrix.linear.n64.zygote_name === :backsolve_mechanistic
    @test matrix.linear.n65.zygote_name === :interpolating_default
    @test matrix.hill.n64.zygote_name === :interpolating_neural
    @test matrix.hill.n65.zygote_name === :interpolating_neural
    @test matrix.competitive.crossing
    @test matrix.six.neural
end

@testset "neural and remapped fixtures stay on the solve surface" begin
    hill = BioDynaX.hill_ude_solve_surface_path()
    @test hill.holds
    remap = BioDynaX.remapped_solve_surface_path()
    @test remap.holds
    @test remap.row.n_heads == 2
    two = BioDynaX.two_regulator_solve_surface_path()
    @test two.holds
    six = BioDynaX.six_state_solve_surface_path()
    @test six.holds
    @test six.row.nstates == 6
end

@testset "zero-hole, dual, MM, and skipped heads stay compile-open" begin
    zero = BioDynaX.zero_hole_solve_surface_path()
    @test zero.holds
    @test zero.validate_open
    dual = BioDynaX.dual_unknown_solve_surface_path()
    @test dual.holds
    @test dual.row.n_heads == 2
    mm = BioDynaX.mm_unknown_solve_surface_path()
    @test mm.holds
    skipped = BioDynaX.skipped_duplicate_solve_surface_path()
    @test skipped.holds
    middle = BioDynaX.skipped_middle_solve_surface_path()
    @test middle.holds
    @test middle.row.dense
end

@testset "competitive known kinetics stay mechanistic on the surface" begin
    path = BioDynaX.competitive_solve_surface_path()
    @test path.holds
    @test path.row.n_heads == 0
    @test path.boundary.mechanistic
end

@testset "unique-claim smoke experiment set stays on the solve surface" begin
    path = BioDynaX.unique_claim_solve_surface_path(; smoke = true)
    @test path.holds
    @test path.compiled_once
    @test path.row.n_heads == 1
end

@testset "multi-IC remake does not compile" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    row = BioDynaX.multi_ic_remake_row(
        model, packed, [[0.22, 0.14], [0.30, 0.18]])
    @test row.holds
    @test row.no_compile
    @test row.n_ics == 2
end

@testset "solver-config agreement does not introduce a new algorithm" begin
    rng = MersenneTwister(7)
    model, _ = build_ude_model(rng, build_linear_test_network())
    row = BioDynaX.solver_config_agreement_row(model)
    @test row.holds
    @test row.algorithm_is_tsit5
    @test row.prod_nothing_unlocked
end

@testset "ProductionAD inplace matches Zygote out-of-place" begin
    matrix = BioDynaX.production_inplace_matrix()
    @test matrix.holds
    @test matrix.linear.inplace
    @test matrix.linear.oop_inplace == false
end

@testset "SciMLBase.solve disables dense output" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    row = BioDynaX.dense_disabled_row(model, packed, [0.22, 0.14])
    @test row.holds
    @test row.dense == false
    @test row.source_disables_dense
end

@testset "irregular saveat keeps the requested times" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    row = BioDynaX.irregular_saveat_row(model, packed, [0.22, 0.14])
    @test row.holds
    @test row.times_match
end

@testset "Zygote gradient through predict_ude stays finite and compile-free" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    row = BioDynaX.zygote_gradient_finite_row(model, packed, [0.22, 0.14])
    @test row.holds
    @test row.finite
    @test row.no_compile
end

@testset "recommend_sensealg rationale strings stay locked" begin
    matrix = BioDynaX.recommend_sensealg_rationale_matrix()
    @test matrix.holds
    @test matrix.linear20.name === :backsolve_mechanistic
    @test matrix.linear100.name === :interpolating_default
    @test matrix.hill.name === :interpolating_neural
end

@testset "known MM and repressilator stay on the mechanistic surface" begin
    mm = BioDynaX.mm_known_solve_surface_path()
    @test mm.holds
    @test mm.row.n_heads == 0
    rep = BioDynaX.repressilator_solve_surface_path()
    @test rep.holds
    @test rep.nstates == 3
end

@testset "default example stays on the solve surface after remapping" begin
    path = BioDynaX.default_example_solve_surface_path()
    @test path.holds
    @test path.duplicate
    @test path.row.dense
end

@testset "solve-surface contract and docs hold" begin
    @test BioDynaX.sciml_solve_surface_source_holds()
    @test BioDynaX.sciml_recommend_sensealg_source_holds()
    @test BioDynaX.sciml_odeproblem_uses_build_ude_function()
    @test BioDynaX.sciml_solve_uses_odeproblem()
    @test BioDynaX.sciml_interface_adds_no_solver()
    @test BioDynaX.sciml_solve_surface_docs_hold()
    @test BioDynaX.sciml_solve_surface_landing_docs_hold()
    @test BioDynaX.sciml_solve_surface_contract_holds()
    violations = BioDynaX.sciml_solve_surface_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
end

@testset "solve surface does not loosen locked claim numbers" begin
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test recovery_thresholds_lock() == RECOVERY_THRESHOLDS
    @test issetequal(names(BioDynaX), collect(locked_public_names()))
    zero = build_zero_unknown_linear_network()
    @test validate_network(zero) === zero
end
