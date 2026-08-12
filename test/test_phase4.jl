@testset "phase 4 chunked STLSQ parity" begin
    rng = MersenneTwister(4)
    A = randn(rng, 120, 6)
    y = A * [1.0, 0.0, 0.5, 0.0, -0.25, 0.0] .+ 1e-3 .* randn(rng, 120)
    dense = BioDynaX._stlsq(A, y, 1e-2)
    blocked = BioDynaX._stlsq_blocked(A, y, 1e-2; chunk_size = 32)
    @test dense ≈ blocked atol = 5e-3
    X = rand(rng, 2, 80)
    terms = local_basis(build_linear_test_network(), 1).numerator
    full = evaluate_library(terms, X)
    chunked = zeros(eltype(X), size(X, 2), length(terms))
    for (buffer, sample_range) in each_library_chunk(terms, X; chunk_size = 17)
        chunked[sample_range, :] .= buffer
    end
    @test chunked ≈ full
    range_buf = zeros(eltype(X), 10, length(terms))
    evaluate_library_range!(range_buf, terms, X, 21:30)
    @test range_buf ≈ full[21:30, :]
end

@testset "phase 4 denominator domain safety" begin
    network = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
    x = collect(range(0.05, 2.0; length = 80))
    X = reshape(x, 1, :)
    derivative = 2.0 .* x ./ (0.5 .+ x)
    config = DiscoveryConfig(
        backend = ImplicitSINDyPI(
            threshold = 1e-6, max_degree = 1, max_hill_degree = 1,
            bootstrap_samples = 0, validation_fraction = 0.2,
            denominator_floor = 1e-8, domain_samples = 64, chunk_size = 16),
        include_interactions = false)
    result = discover_equations(
        X, collect(range(0.0, 1.0; length = length(x))), network;
        derivatives = reshape(derivative, 1, :), config = config,
        verbose = false)
    @test result.success
    @test only(result.candidates).denominator_minimum ≥ 1e-8

    # Force a singular denominator report via an artificial candidate check.
    spec = local_basis(network, 1; degree = 1, include_interactions = false)
    @test_throws DomainError BioDynaX._check_denominator_safety(
        spec, [0.0, 1.0], [-2.0], X, X[:, 1:10], X, 1e-3)
end

@testset "phase 4 raw-data MM recovery" begin
    network = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
    times = collect(range(0.0, 4.0; length = 200))
    # Exact MM trajectory from separable ODE dx/dt = -vmax x/(km+x) is hard;
    # instead feed exact states + exact derivatives through the raw-data API.
    x = collect(range(0.1, 2.5; length = length(times)))
    X = reshape(x, 1, :)
    vmax, km = 1.7, 0.55
    dX = reshape(vmax .* x ./ (km .+ x), 1, :)
    config = DiscoveryConfig(
        backend = ImplicitSINDyPI(
            threshold = 1e-7, max_degree = 1, max_hill_degree = 1,
            bootstrap_samples = 4, validation_fraction = 0.2,
            domain_samples = 32, chunk_size = 32),
        include_interactions = false, seed = 7)
    result = discover_equations(
        X, times, network; derivatives = dX, config = config, verbose = false)
    @test result.success
    cand = only(result.candidates)
    @test cand.numerator_coefficients[2] ≈ vmax / km atol = 5e-2
    @test cand.denominator_coefficients[1] ≈ inv(km) atol = 5e-2
    dX_est = estimate_derivatives(X, times)
    @test size(dX_est) == size(X)
    @test all(isfinite, dX_est)
end

@testset "phase 4 export and model selection" begin
    network = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
    x = collect(range(0.2, 2.0; length = 120))
    X = reshape(x, 1, :)
    times = collect(range(0.0, 1.0; length = length(x)))
    vmax, km = 2.0, 0.8
    dX = reshape(vmax .* x ./ (km .+ x), 1, :)
    config = DiscoveryConfig(
        backend = ImplicitSINDyPI(
            threshold = 1e-6, max_degree = 1, max_hill_degree = 1,
            bootstrap_samples = 0, domain_samples = 16),
        include_interactions = false)
    result = discover_equations(
        X, times, network; derivatives = dX, config = config, verbose = false)
    @test result.success
    cand = only(result.candidates)
    latex = equation_to_latex(cand)
    @test occursin(r"\\dot\{x\}_1", latex) || occursin("\\dot{x}_{1}", latex)
    f = equation_to_function(cand)
    pred, _ = BioDynaX._evaluate_candidate(
        cand.specification, cand.numerator_coefficients,
        cand.denominator_coefficients, X[:, 1:5])
    for j in 1:5
        @test f(X[:, j]) ≈ pred[j] atol = 1e-8
    end
    rhs = export_rhs(result)
    @test rhs([x[1]])[1] ≈ f([x[1]]) atol = 1e-10
    @test information_criterion(100, 1.0, 3; criterion = :aic) <
          information_criterion(100, 1.0, 3; criterion = :bic)
    @test isfinite(score_candidate(cand, X, vec(dX); criterion = :aic))

    rng = MersenneTwister(21)
    model, params = build_ude_model(rng, build_linear_test_network())
    selected = select_discovery_config(
        params, model;
        thresholds = (1e-1, 1e-2),
        criterion = :aic,
        n_samples = 80,
        u0 = [0.2, 0.1],
        tspan = (0.0, 1.0),
        config = DiscoveryConfig(
            backend = ExplicitSTLSQ(threshold = 1e-2),
            include_interactions = false),
        verbose = false)
    @test selected.success
    @test haskey(selected.metadata.config, :selected_threshold)
    @test haskey(selected.metadata.config, :selection)
end
