@testset "run_discovery.jl is a debug runner, not the product" begin
    script = joinpath(pkgdir(BioDynaX), "scripts", "run_discovery.jl")
    src = read(script, String)
    contributing = read(joinpath(pkgdir(BioDynaX), "CONTRIBUTING.md"), String)
    readme = read(joinpath(pkgdir(BioDynaX), "README.md"), String)
    workspaces = read(
        joinpath(pkgdir(BioDynaX), "docs", "src",
            "internal-workspaces.md"), String)
    @test occursin("Debug runner, not the product", src)
    @test occursin("examples/unknown_inhibition.jl", src)
    @test occursin("scripts/run_discovery.jl", contributing)
    @test occursin("debug runner, not the product", contributing)
    @test occursin("examples/unknown_inhibition.jl", contributing)
    @test occursin("The golden path is `examples/unknown_inhibition.jl`",
        workspaces)
    @test occursin("train_experiments_with_warmup", workspaces)
    @test occursin("denominator_split_counts", workspaces)
    @test occursin("Not a general network solver", readme)
    @test occursin("Not v1.0", readme)
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
end

@testset "format job lists stacked and science Julia files" begin
    ci = read(joinpath(pkgdir(BioDynaX), ".github", "workflows", "ci.yml"),
        String)
    needed = (
        "src/TrainingReuse.jl",
        "src/AllocationGates.jl",
        "src/ClaimMetricHonesty.jl",
        "src/ClaimScopeHonesty.jl",
        "test/test_training_reuse.jl",
        "test/test_allocation_gates.jl",
        "test/test_claim_metric_honesty.jl",
        "test/test_claim_scope_honesty.jl",
        "test/test_software_hygiene.jl")
    for file in needed
        @test occursin(file, ci)
    end
end
