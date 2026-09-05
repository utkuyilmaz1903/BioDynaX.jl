@testset "denominator domain helpers are not exported" begin
    @test !(:denominator_split_counts in names(BioDynaX))
    @test !(:ude_extras_denominator_row in names(BioDynaX))
    @test !(:DenominatorDomainRow in names(BioDynaX))
    @test !(:synthetic_safe_implicit_candidate in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.denominator_violation_count_source_holds()
    @test BioDynaX.denominator_split_counts_source_holds()
    @test BioDynaX.ude_extras_denominator_source_holds()
    @test BioDynaX.extras_path_calls_split_source_holds()
    @test BioDynaX.implicit_discovery_uses_domain_grid_source_holds()
    @test BioDynaX.explicit_path_skips_domain_grid_source_holds()
    @test BioDynaX.domain_grid_clips_source_holds()
    @test BioDynaX.denominator_domain_index_holds()
end

@testset "split counts distinguish safe, unsafe, and explicit" begin
    safe = BioDynaX.safe_split_row()
    @test safe.holds
    @test safe.raw == 0
    unsafe = BioDynaX.unsafe_split_row()
    @test unsafe.holds
    @test unsafe.raw > 0
    explicit = BioDynaX.explicit_candidate_zero_violations_row()
    @test explicit.holds
    missing = BioDynaX.missing_candidate_typemax_row()
    @test missing.holds
    near = BioDynaX.near_zero_split_row()
    @test near.holds
    two = BioDynaX.two_state_unsafe_split_row()
    @test two.holds
end

@testset "domain grid stays in the positive orthant" begin
    grid = BioDynaX.domain_grid_nonneg_row()
    @test grid.holds
    disabled = BioDynaX.domain_grid_disabled_row()
    @test disabled.holds
    clip = BioDynaX.domain_grid_clips_negative_pad_row()
    @test clip.holds
    span = BioDynaX.domain_grid_spans_observed_row()
    @test span.holds
    seed = BioDynaX.domain_grid_seed_reproducible_row()
    @test seed.holds
    reject = BioDynaX.implicit_safety_rejects_unsafe_row()
    @test reject.holds
    accept = BioDynaX.implicit_safety_accepts_safe_row()
    @test accept.holds
end

@testset "UDE extras still walk the domain grid" begin
    live = BioDynaX.extras_denominator_live_row()
    @test live.holds
    @test live.extras_live
    empty = BioDynaX.extras_empty_still_splits_row()
    @test empty.holds
    na = BioDynaX.extras_nothing_is_na_row()
    @test na.holds
    hard = BioDynaX.extras_hardcoded_attempt_rejected_row()
    @test hard.holds
    hill = BioDynaX.extras_on_hill_truth_discovery_row()
    @test hill.holds
    mm = BioDynaX.extras_on_mm_truth_discovery_row()
    @test mm.holds
    smoke = BioDynaX.smoke_vs_protocol_denominator_row()
    @test smoke.holds
    @test smoke.proto_ics == 9
    empty_domain = BioDynaX.empty_domain_split_is_train_val_only_row()
    @test empty_domain.holds
end

@testset "fixture libraries keep a zero-coefficient denominator safe" begin
    hill = BioDynaX.hill_unknown_denominator_row()
    @test hill.holds
    mm = BioDynaX.mm_unknown_denominator_row()
    @test mm.holds
    two = BioDynaX.two_regulator_denominator_row()
    @test two.holds
    three = BioDynaX.three_state_denominator_row()
    @test three.holds
    wrong = BioDynaX.wrong_graph_denominator_row()
    @test wrong.holds
    default = BioDynaX.default_example_denominator_row()
    @test default.holds
    remap = BioDynaX.remapped_denominator_row()
    @test remap.holds
    dual = BioDynaX.dual_denominator_row()
    @test dual.holds
    linear = BioDynaX.linear_zero_denominator_row()
    @test linear.holds
    typed = BioDynaX.denominator_domain_typed_matrix()
    @test typed.holds
    catalog = BioDynaX.suite_section_denominator_catalog()
    @test catalog.holds
    @test BioDynaX.suite_denominator_catalog_holds()
    ics = BioDynaX.unique_claim_not_faster_by_dropping_ics_denominator_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test BioDynaX.discovery_config_domain_samples_row().holds
    @test BioDynaX.default_backend_domain_samples_row().holds
    @test BioDynaX.floor_sensitivity_row().holds
    @test BioDynaX.split_matches_sum_row().holds
    @test BioDynaX.extra_candidates_keep_denominator_row().holds
    @test BioDynaX.six_state_per_target_denominator_row().holds
    @test BioDynaX.default_per_target_denominator_row().holds
    @test BioDynaX.remapped_per_target_denominator_row().holds
    @test BioDynaX.format_split_markdown_holds()
    @test BioDynaX.combined_f1_not_a_denominator_kpi_row().holds
    @test BioDynaX.single_sample_split_row().holds
    @test BioDynaX.all_zero_state_grid_row().holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"DenominatorDomain.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "DenominatorDomain.jl"))
    recovery = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    @test occursin("function ude_extras_denominator_row", recovery)
    pipeline = read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String)
    @test occursin("extras_denominator = ude_extras_denominator_row(", pipeline)
end
