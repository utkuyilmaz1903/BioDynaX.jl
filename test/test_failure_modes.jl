@testset "failure-mode helpers are not exported" begin
    @test !(:discovery_retcode_catalog in names(BioDynaX))
    @test !(:insufficient_samples_row in names(BioDynaX))
    @test !(:hole_validate_matrix in names(BioDynaX))
    @test !(:kpi_failure_grid in names(BioDynaX))
    @test !(:extras_print_catalog_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "source and landing checks stay locked" begin
    @test discovery_retcode_mapper_source_holds()
    @test discovery_sample_floor_source_holds()
    @test discovery_n_samples_entry_source_holds()
    @test extras_source_holds()
    @test failure_mode_index_holds()
    @test BioDynaX.failure_mode_formatter_lock_holds()
end

@testset "DiscoveryRetcode catalog and mapper" begin
    catalog = BioDynaX.discovery_retcode_catalog()
    @test catalog.holds
    @test catalog.n == 6
    mapper = BioDynaX.discovery_retcode_mapper_row()
    @test mapper.holds
    @test mapper.insufficient === InsufficientSamples
    @test mapper.empty === EmptySupport
    @test mapper.den === DenominatorUnsafe
    @test mapper.singular === SingularLibrary
    @test mapper.other === DiscoveryFailed
end

@testset "live insufficient and empty-support retcodes" begin
    insufficient = BioDynaX.insufficient_samples_row()
    @test insufficient.holds
    @test insufficient.retcode === InsufficientSamples
    boundary = BioDynaX.insufficient_samples_boundary_row()
    @test boundary.holds
    entry = BioDynaX.n_samples_entry_throws_row()
    @test entry.holds
    empty = BioDynaX.empty_support_row()
    @test empty.holds
    @test empty.retcode === EmptySupport
    implicit = BioDynaX.implicit_insufficient_row()
    @test implicit.holds
    rate = BioDynaX.discover_unknown_rate_insufficient_row()
    @test rate.holds
    failed = BioDynaX.failed_discovery_result_row()
    @test failed.holds
    @test failed.retcode === DiscoveryFailed
    hill_short = BioDynaX.hill_insufficient_samples_row()
    @test hill_short.holds
    mm_short = BioDynaX.mm_insufficient_samples_row()
    @test mm_short.holds
    select_fail = BioDynaX.select_discovery_all_fail_row()
    @test select_fail.holds
    @test BioDynaX.discovery_retcode_exports_hold()
end

@testset "explicit success is a consistent DiscoverySuccess" begin
    success = BioDynaX.explicit_success_row()
    @test success.holds
    @test success.retcode === DiscoverySuccess
    hill = BioDynaX.hill_known_explicit_success_row()
    @test hill.holds
    @test hill.retcode === DiscoverySuccess
end

@testset "0/2-hole networks still validate" begin
    matrix = BioDynaX.hole_validate_matrix()
    @test matrix.holds
    @test matrix.all_validate_open
    @test matrix.n_zero ≥ 4
    @test matrix.n_one ≥ 4
    @test matrix.n_multi ≥ 2
    pair = BioDynaX.hole_validate_zero_and_dual_row()
    @test pair.holds
    @test pair.zero.admits == false
    @test pair.dual.admits == false
    @test pair.single.admits
    zero = BioDynaX.discovery_on_zero_hole_row()
    @test zero.holds
    dual = BioDynaX.discovery_on_dual_hole_row()
    @test dual.holds
end

@testset "KPI failures omit combined F1" begin
    grid = BioDynaX.kpi_failure_grid()
    @test grid.holds
    @test grid.n == 72
    @test grid.painted == 0
    examples = BioDynaX.kpi_failure_named_examples()
    @test examples.holds
    f1 = BioDynaX.kpi_f1_never_failure_symbol_row()
    @test f1.holds
    @test f1.high_kpis_hold
    @test f1.low_kpis_hold
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    boundary = BioDynaX.kpi_threshold_boundary_row()
    @test boundary.holds
end

@testset "extras NA, empty, live, and hardcoded attempt" begin
    catalog = BioDynaX.extras_print_catalog_row()
    @test catalog.holds
    @test catalog.has_na
    @test catalog.has_none
    @test catalog.has_live
    @test catalog.has_hardcoded
    empty_na = BioDynaX.extras_empty_vs_na_row()
    @test empty_na.holds
    hardcoded = BioDynaX.extras_hardcoded_attempt_row()
    @test hardcoded.holds
    print_row = BioDynaX.failed_protocol_print_row()
    @test print_row.holds
    two = BioDynaX.failed_protocol_two_hole_print_row()
    @test two.holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"FailureModes.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "src", "FailureModes.jl"))
end
