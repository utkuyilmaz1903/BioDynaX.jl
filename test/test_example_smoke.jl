@testset "golden-path example smoke" begin
    fixture = joinpath(@__DIR__, "..", "examples", "data", "unknown_inhibition.csv")
    fixture_mtime = mtime(fixture)
    example = joinpath(@__DIR__, "..", "examples", "unknown_inhibition.jl")
    # Same process: a child `julia` restart spends minutes compiling, not training.
    include(example)
    _, residual, ident = main(; adam_iters = 2, bfgs_iters = 0, smoke = true)
    @test residual isa Real
    @test ident.unidentifiable_edge isa Bool
    @test mtime(fixture) == fixture_mtime
end
