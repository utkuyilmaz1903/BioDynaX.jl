@testset "denominator domain helpers are not exported" begin
    @test !(:denominator_split_counts in names(HybridKinetics))
    @test !(:ude_extras_denominator_row in names(HybridKinetics))
    @test !(:DenominatorDomainRow in names(HybridKinetics))
    @test !(:synthetic_safe_implicit_candidate in names(HybridKinetics))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing checks stay locked" begin
    @test denominator_violation_count_source_holds()
    @test denominator_split_counts_source_holds()
    @test ude_extras_denominator_source_holds()
    @test extras_path_calls_split_source_holds()
    @test implicit_discovery_uses_domain_grid_source_holds()
    @test explicit_path_skips_domain_grid_source_holds()
    @test domain_grid_clips_source_holds()
    @test denominator_domain_index_holds()
end

@testset "split counts distinguish safe, unsafe, and explicit" begin
    safe = HybridKinetics.safe_split_row()
    @test safe.holds
    @test safe.raw == 0
    unsafe = HybridKinetics.unsafe_split_row()
    @test unsafe.holds
    @test unsafe.raw > 0
    explicit = HybridKinetics.explicit_candidate_zero_violations_row()
    @test explicit.holds
    missing = HybridKinetics.missing_candidate_typemax_row()
    @test missing.holds
    near = HybridKinetics.near_zero_split_row()
    @test near.holds
    two = HybridKinetics.two_state_unsafe_split_row()
    @test two.holds
end

@testset "domain grid stays in the positive orthant" begin
    grid = HybridKinetics.domain_grid_nonneg_row()
    @test grid.holds
    disabled = HybridKinetics.domain_grid_disabled_row()
    @test disabled.holds
    clip = HybridKinetics.domain_grid_clips_negative_pad_row()
    @test clip.holds
    span = HybridKinetics.domain_grid_spans_observed_row()
    @test span.holds
    seed = HybridKinetics.domain_grid_seed_reproducible_row()
    @test seed.holds
    reject = HybridKinetics.implicit_safety_rejects_unsafe_row()
    @test reject.holds
    accept = HybridKinetics.implicit_safety_accepts_safe_row()
    @test accept.holds
end

@testset "UDE extras still walk the domain grid" begin
    live = HybridKinetics.extras_denominator_live_row()
    @test live.holds
    @test live.extras_live
    empty = HybridKinetics.extras_empty_still_splits_row()
    @test empty.holds
    na = HybridKinetics.extras_nothing_is_na_row()
    @test na.holds
    hard = HybridKinetics.extras_hardcoded_attempt_rejected_row()
    @test hard.holds
    hill = HybridKinetics.extras_on_hill_truth_discovery_row()
    @test hill.holds
    mm = HybridKinetics.extras_on_mm_truth_discovery_row()
    @test mm.holds
    smoke = HybridKinetics.smoke_vs_protocol_denominator_row()
    @test smoke.holds
    @test smoke.proto_ics == 9
    empty_domain = HybridKinetics.empty_domain_split_is_train_val_only_row()
    @test empty_domain.holds
end

@testset "fixture libraries keep a zero-coefficient denominator safe" begin
    hill = HybridKinetics.hill_unknown_denominator_row()
    @test hill.holds
    mm = HybridKinetics.mm_unknown_denominator_row()
    @test mm.holds
    two = HybridKinetics.two_regulator_denominator_row()
    @test two.holds
    three = HybridKinetics.three_state_denominator_row()
    @test three.holds
    wrong = HybridKinetics.wrong_graph_denominator_row()
    @test wrong.holds
    default = HybridKinetics.default_example_denominator_row()
    @test default.holds
    remap = HybridKinetics.remapped_denominator_row()
    @test remap.holds
    dual = HybridKinetics.dual_denominator_row()
    @test dual.holds
    linear = HybridKinetics.linear_zero_denominator_row()
    @test linear.holds
    typed = HybridKinetics.denominator_domain_typed_matrix()
    @test typed.holds
    catalog = HybridKinetics.suite_section_denominator_catalog()
    @test catalog.holds
    @test HybridKinetics.suite_denominator_catalog_holds()
    ics = HybridKinetics.reference_protocol_not_faster_by_dropping_ics_denominator_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test HybridKinetics.discovery_config_domain_samples_row().holds
    @test HybridKinetics.default_backend_domain_samples_row().holds
    @test HybridKinetics.floor_sensitivity_row().holds
    @test HybridKinetics.split_matches_sum_row().holds
    @test HybridKinetics.extra_candidates_keep_denominator_row().holds
    @test HybridKinetics.six_state_per_target_denominator_row().holds
    @test HybridKinetics.default_per_target_denominator_row().holds
    @test HybridKinetics.remapped_per_target_denominator_row().holds
    @test HybridKinetics.format_split_markdown_holds()
    @test HybridKinetics.combined_f1_not_a_denominator_kpi_row().holds
    @test HybridKinetics.single_sample_split_row().holds
    @test HybridKinetics.all_zero_state_grid_row().holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "HybridKinetics.jl"), String)
    @test occursin("include(\"DenominatorDomain.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "DenominatorDomain.jl"))
    recovery = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    @test occursin("function ude_extras_denominator_row", recovery)
    pipeline = read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String)
    @test occursin("extras_denominator = ude_extras_denominator_row(", pipeline)
end
