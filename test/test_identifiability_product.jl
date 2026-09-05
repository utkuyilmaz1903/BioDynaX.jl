@testset "identifiability product helpers are not exported" begin
    @test !(:live_production_destruction_tradeoff in names(BioDynaX))
    @test !(:join_tradeoff_protocol_row in names(BioDynaX))
    @test !(:IdentifiabilityProductRow in names(BioDynaX))
    @test !(:format_protocol_collinearity_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.production_destruction_tradeoff_source_holds()
    @test BioDynaX.format_production_destruction_warning_source_holds()
    @test BioDynaX.format_protocol_result_collinearity_source_holds()
    @test BioDynaX.coefficients_are_biological_constants_source_holds()
    @test BioDynaX.identifiability_product_index_holds()
end

@testset "coefficients follow unidentifiable_edge" begin
    coeff = BioDynaX.coefficients_are_biological_constants_row()
    @test coeff.holds
    @test coeff.true_coeff == false
    @test coeff.false_coeff == true
    hill_closed = BioDynaX.protocol_row_rejects_hill_from_nn_row()
    @test hill_closed.holds
    extras = BioDynaX.extras_not_invented_on_join_row()
    @test extras.holds
    kpi = BioDynaX.kpi_f1_not_a_failure_on_join_row()
    @test kpi.holds
    @test :support_f1 ∉ kpi.failures
end

@testset "collinearity print and warning stay honest" begin
    col = BioDynaX.format_protocol_collinearity_row()
    @test col.holds
    @test col.prints_finite
    @test col.silent_nan
    warning = BioDynaX.collinearity_warning_row()
    @test warning.holds
    sections = BioDynaX.format_protocol_sections_row()
    @test sections.holds
    smoke = BioDynaX.smoke_vs_protocol_print_row()
    @test smoke.holds
    @test smoke.protocol_ics
    @test smoke.smoke_ics
end

@testset "live tradeoff joins UniqueClaimProtocolRow" begin
    known = BioDynaX.hill_known_tradeoff_path()
    @test known.holds
    @test known.collinearity_nan
    unknown = BioDynaX.hill_unknown_tradeoff_path()
    @test unknown.holds
    @test unknown.collinearity_finite
    mm_u = BioDynaX.mm_unknown_tradeoff_path()
    @test mm_u.holds
    mm_k = BioDynaX.mm_known_tradeoff_path()
    @test mm_k.holds
    linear = BioDynaX.linear_zero_hole_tradeoff_path()
    @test linear.holds
    two = BioDynaX.two_regulator_tradeoff_path()
    @test two.holds
    default = BioDynaX.default_example_tradeoff_path()
    @test default.holds
end

@testset "multi-head and honesty tradeoff rows" begin
    remap = BioDynaX.remapped_tradeoff_path()
    @test remap.holds
    @test remap.admits == false
    dual = BioDynaX.dual_tradeoff_path()
    @test dual.holds
    six = BioDynaX.six_state_tradeoff_path()
    @test six.holds
    three = BioDynaX.three_state_tradeoff_path()
    @test three.holds
    skipped = BioDynaX.skipped_duplicate_tradeoff_path()
    @test skipped.holds
    repress = BioDynaX.repressilator_tradeoff_path()
    @test repress.holds
    comp = BioDynaX.competitive_unknown_tradeoff_path()
    @test comp.holds
    missing = BioDynaX.missing_production_param_row()
    @test missing.holds
    frozen = BioDynaX.frozen_k_prod_raw_unchanged_row()
    @test frozen.holds
    compile_free = BioDynaX.compile_free_tradeoff_row()
    @test compile_free.holds
    verbose = BioDynaX.report_verbose_tradeoff_row()
    @test verbose.holds
    typed = BioDynaX.identifiability_product_typed_matrix()
    @test typed.holds
    middle = BioDynaX.skipped_middle_tradeoff_path()
    @test middle.holds
    kinetic = BioDynaX.kinetic_known_tradeoff_path()
    @test kinetic.holds
    cond = BioDynaX.condition_threshold_row()
    @test cond.holds
    matched = BioDynaX.format_matches_joined_protocol_row()
    @test matched.holds
    @test matched.n_ics == 9
    blocks = BioDynaX.unique_claim_product_blocks_hold_on_join()
    @test blocks.holds
    untouched = BioDynaX.recovery_thresholds_untouched_row()
    @test untouched.holds
    @test untouched.ude == 0.50
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"IdentifiabilityProduct.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "IdentifiabilityProduct.jl"))
end
