@testset "parameter schema pack helpers are not exported" begin
    @test !(:unpack_parameters in names(BioDynaX))
    @test !(:ParameterSchemaPackRow in names(BioDynaX))
    @test !(:remapped_pack_unpack_row in names(BioDynaX))
    @test !(:kinetic_custom_in_schema_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test :pack_parameters in names(BioDynaX)
    @test :parameter_schema in names(BioDynaX)
    @test :positive_parameter in names(BioDynaX)
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.custom_kinetic_schema_source_holds()
    @test BioDynaX.unpack_parameters_source_holds()
    @test BioDynaX.pack_parameters_source_holds()
    @test BioDynaX.frozen_phys_source_holds()
    @test BioDynaX.default_phys_includes_custom_source_holds()
    @test BioDynaX.parameter_schema_pack_index_holds()
end

@testset "pack/unpack and :k_custom schema" begin
    rt = BioDynaX.positive_parameter_roundtrip_row()
    @test rt.holds
    linear = BioDynaX.pack_unpack_linear_row()
    @test linear.holds
    remap = BioDynaX.remapped_pack_unpack_row()
    @test remap.holds
    @test remap.nn_heads == 2
    two = BioDynaX.two_regulator_pack_unpack_row()
    @test two.holds
    kinetic = BioDynaX.kinetic_custom_in_schema_row()
    @test kinetic.holds
    @test kinetic.has_custom
    missing = BioDynaX.missing_custom_validate_throws_row()
    @test missing.holds
    defaults = BioDynaX.default_parameters_include_custom_row()
    @test defaults.holds
    repack = BioDynaX.unpack_then_repack_row()
    @test repack.holds
end

@testset "schema vs compiled NN tree and frozen_phys" begin
    tree = BioDynaX.schema_vs_compiled_nn_tree_row()
    @test tree.holds
    dummy = BioDynaX.dummy_head_on_zero_hole_row()
    @test dummy.holds
    @test dummy.schema_heads == 0
    remap = BioDynaX.remapped_schema_matches_heads_row()
    @test remap.holds
    dual = BioDynaX.dual_schema_matches_heads_row()
    @test dual.holds
    middle = BioDynaX.skipped_middle_schema_heads_row()
    @test middle.holds
    z = BioDynaX.frozen_phys_zero_gradient_row()
    @test z.holds
    r = BioDynaX.frozen_phys_restore_row()
    @test r.holds
    c = BioDynaX.frozen_phys_config_copy_row()
    @test c.holds
    rf = BioDynaX.remapped_frozen_phys_row()
    @test rf.holds
end

@testset "fixture schemas stay honest" begin
    @test BioDynaX.hill_unknown_schema_row().holds
    @test BioDynaX.hill_known_schema_row().holds
    @test BioDynaX.mm_unknown_schema_row().holds
    @test BioDynaX.default_example_schema_row().holds
    @test BioDynaX.three_state_schema_row().holds
    @test BioDynaX.repressilator_schema_row().holds
    typed = BioDynaX.parameter_schema_pack_typed_matrix()
    @test typed.holds
    @test typed.kinetic.has_custom
    smoke = BioDynaX.smoke_vs_protocol_schema_row()
    @test smoke.holds
    @test smoke.proto_ics == 9
    @test BioDynaX.suite_schema_catalog_holds()
    @test BioDynaX.kinetic_known_tradeoff_now_predicts_row().holds
    ics = BioDynaX.unique_claim_not_faster_by_dropping_ics_schema_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test BioDynaX.bounded_parameter_row().holds
    @test BioDynaX.two_regulator_input_dim_row().holds
    @test BioDynaX.remapped_input_dims_row().holds
    @test BioDynaX.schema_name_catalog_row().holds
    @test BioDynaX.format_pack_markdown_holds()
    @test BioDynaX.linear_schema_names_are_mass_action_row().holds
    @test BioDynaX.pack_rejects_nonpositive_phys_row().holds
    @test BioDynaX.validate_rejects_nonpositive_row().holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"ParameterSchemaPack.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "ParameterSchemaPack.jl"))
    schema = read(joinpath(@__DIR__, "..", "src", "ParameterSchema.jl"), String)
    @test occursin("CUSTOM_KINETIC", schema)
    @test occursin(":k_custom", schema)
    ude = read(joinpath(@__DIR__, "..", "src", "UDE.jl"), String)
    @test occursin("function unpack_parameters(p)", ude)
end
