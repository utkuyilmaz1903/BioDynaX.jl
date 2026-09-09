@testset "parameter schema pack helpers are not exported" begin
    @test !(:unpack_parameters in names(HybridKinetics))
    @test !(:ParameterSchemaPackRow in names(HybridKinetics))
    @test !(:remapped_pack_unpack_row in names(HybridKinetics))
    @test !(:kinetic_custom_in_schema_row in names(HybridKinetics))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test :pack_parameters in names(HybridKinetics)
    @test :parameter_schema in names(HybridKinetics)
    @test :positive_parameter in names(HybridKinetics)
end

@testset "source and landing checks stay locked" begin
    @test custom_kinetic_schema_source_holds()
    @test unpack_parameters_source_holds()
    @test pack_parameters_source_holds()
    @test frozen_phys_source_holds()
    @test default_phys_includes_custom_source_holds()
    @test parameter_schema_pack_index_holds()
end

@testset "pack/unpack and :k_custom schema" begin
    rt = HybridKinetics.positive_parameter_roundtrip_row()
    @test rt.holds
    linear = HybridKinetics.pack_unpack_linear_row()
    @test linear.holds
    remap = HybridKinetics.remapped_pack_unpack_row()
    @test remap.holds
    @test remap.nn_heads == 2
    two = HybridKinetics.two_regulator_pack_unpack_row()
    @test two.holds
    kinetic = HybridKinetics.kinetic_custom_in_schema_row()
    @test kinetic.holds
    @test kinetic.has_custom
    missing = HybridKinetics.missing_custom_validate_throws_row()
    @test missing.holds
    defaults = HybridKinetics.default_parameters_include_custom_row()
    @test defaults.holds
    repack = HybridKinetics.unpack_then_repack_row()
    @test repack.holds
end

@testset "schema vs compiled NN tree and frozen_phys" begin
    tree = HybridKinetics.schema_vs_compiled_nn_tree_row()
    @test tree.holds
    dummy = HybridKinetics.dummy_head_on_zero_hole_row()
    @test dummy.holds
    @test dummy.schema_heads == 0
    remap = HybridKinetics.remapped_schema_matches_heads_row()
    @test remap.holds
    dual = HybridKinetics.dual_schema_matches_heads_row()
    @test dual.holds
    middle = HybridKinetics.skipped_middle_schema_heads_row()
    @test middle.holds
    z = HybridKinetics.frozen_phys_zero_gradient_row()
    @test z.holds
    r = HybridKinetics.frozen_phys_restore_row()
    @test r.holds
    c = HybridKinetics.frozen_phys_config_copy_row()
    @test c.holds
    rf = HybridKinetics.remapped_frozen_phys_row()
    @test rf.holds
end

@testset "fixture schemas stay honest" begin
    @test HybridKinetics.hill_unknown_schema_row().holds
    @test HybridKinetics.hill_known_schema_row().holds
    @test HybridKinetics.mm_unknown_schema_row().holds
    @test HybridKinetics.default_example_schema_row().holds
    @test HybridKinetics.three_state_schema_row().holds
    @test HybridKinetics.repressilator_schema_row().holds
    typed = HybridKinetics.parameter_schema_pack_typed_matrix()
    @test typed.holds
    @test typed.kinetic.has_custom
    smoke = HybridKinetics.smoke_vs_protocol_schema_row()
    @test smoke.holds
    @test smoke.proto_ics == 9
    @test HybridKinetics.suite_schema_catalog_holds()
    @test HybridKinetics.kinetic_known_tradeoff_now_predicts_row().holds
    ics = HybridKinetics.reference_protocol_not_faster_by_dropping_ics_schema_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test HybridKinetics.bounded_parameter_row().holds
    @test HybridKinetics.two_regulator_input_dim_row().holds
    @test HybridKinetics.remapped_input_dims_row().holds
    @test HybridKinetics.schema_name_catalog_row().holds
    @test HybridKinetics.format_pack_markdown_holds()
    @test HybridKinetics.linear_schema_names_are_mass_action_row().holds
    @test HybridKinetics.pack_rejects_nonpositive_phys_row().holds
    @test HybridKinetics.validate_rejects_nonpositive_row().holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"ParameterSchemaPack.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "ParameterSchemaPack.jl"))
    schema = read(joinpath(@__DIR__, "..", "src", "ParameterSchema.jl"), String)
    @test occursin("CUSTOM_KINETIC", schema)
    @test occursin(":k_custom", schema)
    ude = read(joinpath(@__DIR__, "..", "src", "UDE.jl"), String)
    @test occursin("function unpack_parameters(p)", ude)
end
