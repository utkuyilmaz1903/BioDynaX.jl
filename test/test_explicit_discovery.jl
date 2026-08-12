@testset "explicit STLSQ discovery backend" begin
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    x = collect(range(0.1, 2.0; length = 200))
    X = reshape(x, 1, :)
    derivative = 1.5 .* x .+ 0.3

    spec = local_basis(
        network, 1; degree = 1, include_interactions = false,
        X, derivative, max_variables = 1)
    library = evaluate_library(spec.numerator, X)
    coefficients = BioDynaX._stlsq(library, derivative, 1e-7)
    prediction = library * coefficients
    @test mean(abs2, prediction .- derivative) < 1e-8

    rng = MersenneTwister(0)
    model, params = build_ude_model(rng, build_linear_test_network())
    result = discover_equations(
        params, model;
        u0 = [0.2, 0.1],
        tspan = (0.0, 2.0),
        n_samples = 80,
        config = DiscoveryConfig(backend = ExplicitSTLSQ(threshold = 1e-2)),
        verbose = false)
    @test result.success
    @test result.candidates isa Vector{ExplicitCandidate}
    @test !isempty(result.equations)
end
