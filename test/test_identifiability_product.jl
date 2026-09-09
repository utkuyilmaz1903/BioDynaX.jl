@testset "identifiability product helpers are not exported" begin
    @test !(:live_production_destruction_tradeoff in names(HybridKinetics))
    @test !(:join_tradeoff_protocol_row in names(HybridKinetics))
    @test !(:IdentifiabilityProductRow in names(HybridKinetics))
    @test !(:format_protocol_collinearity_row in names(HybridKinetics))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing checks stay locked" begin
    @test production_destruction_tradeoff_source_holds()
    @test format_production_destruction_warning_source_holds()
    @test format_protocol_result_collinearity_source_holds()
    @test coefficients_are_biological_constants_source_holds()
    @test identifiability_product_index_holds()
end

@testset "coefficients follow unidentifiable_edge" begin
    coeff = HybridKinetics.coefficients_are_biological_constants_row()
    @test coeff.holds
    @test coeff.true_coeff == false
    @test coeff.false_coeff == true
    hill_not_attempted = HybridKinetics.protocol_row_rejects_hill_from_nn_row()
    @test hill_not_attempted.holds
    extras = HybridKinetics.extras_not_invented_on_join_row()
    @test extras.holds
    kpi = HybridKinetics.kpi_f1_not_a_failure_on_join_row()
    @test kpi.holds
    @test :support_f1 ∉ kpi.failures
end

@testset "collinearity print and warning stay consistent" begin
    col = HybridKinetics.format_protocol_collinearity_row()
    @test col.holds
    @test col.prints_finite
    @test col.silent_nan
    warning = HybridKinetics.collinearity_warning_row()
    @test warning.holds
    sections = HybridKinetics.format_protocol_sections_row()
    @test sections.holds
    smoke = HybridKinetics.smoke_vs_protocol_print_row()
    @test smoke.holds
    @test smoke.protocol_ics
    @test smoke.smoke_ics
end

@testset "live tradeoff joins ReferenceProtocolRow" begin
    known = HybridKinetics.hill_known_tradeoff_path()
    @test known.holds
    @test known.collinearity_nan
    unknown = HybridKinetics.hill_unknown_tradeoff_path()
    @test unknown.holds
    @test unknown.collinearity_finite
    mm_u = HybridKinetics.mm_unknown_tradeoff_path()
    @test mm_u.holds
    mm_k = HybridKinetics.mm_known_tradeoff_path()
    @test mm_k.holds
    linear = HybridKinetics.linear_zero_hole_tradeoff_path()
    @test linear.holds
    two = HybridKinetics.two_regulator_tradeoff_path()
    @test two.holds
    default = HybridKinetics.default_example_tradeoff_path()
    @test default.holds
end

@testset "multi-head and consistency tradeoff rows" begin
    remap = HybridKinetics.remapped_tradeoff_path()
    @test remap.holds
    @test remap.admits == false
    dual = HybridKinetics.dual_tradeoff_path()
    @test dual.holds
    six = HybridKinetics.six_state_tradeoff_path()
    @test six.holds
    three = HybridKinetics.three_state_tradeoff_path()
    @test three.holds
    skipped = HybridKinetics.skipped_duplicate_tradeoff_path()
    @test skipped.holds
    repress = HybridKinetics.repressilator_tradeoff_path()
    @test repress.holds
    comp = HybridKinetics.competitive_unknown_tradeoff_path()
    @test comp.holds
    missing = HybridKinetics.missing_production_param_row()
    @test missing.holds
    frozen = HybridKinetics.frozen_k_prod_raw_unchanged_row()
    @test frozen.holds
    compile_free = HybridKinetics.compile_free_tradeoff_row()
    @test compile_free.holds
    verbose = HybridKinetics.report_verbose_tradeoff_row()
    @test verbose.holds
    typed = HybridKinetics.identifiability_product_typed_matrix()
    @test typed.holds
    middle = HybridKinetics.skipped_middle_tradeoff_path()
    @test middle.holds
    kinetic = HybridKinetics.kinetic_known_tradeoff_path()
    @test kinetic.holds
    cond = HybridKinetics.condition_threshold_row()
    @test cond.holds
    matched = HybridKinetics.format_matches_joined_protocol_row()
    @test matched.holds
    @test matched.n_ics == 9
    blocks = HybridKinetics.reference_protocol_product_blocks_hold_on_join()
    @test blocks.holds
    untouched = HybridKinetics.recovery_thresholds_untouched_row()
    @test untouched.holds
    @test untouched.ude == 0.50
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"IdentifiabilityProduct.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "IdentifiabilityProduct.jl"))
end
