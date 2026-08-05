@testset "implicit rational recovery" begin
    network = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
    x = collect(range(0.05, 3.0; length = 240))
    X = reshape(x, 1, :)
    vmax, km = 2.4, 0.7
    derivative = vmax .* x ./ (km .+ x)
    spec = local_basis(
        network, 1; degree = 1, include_interactions = false,
        X, derivative, max_variables = 1)
    numerator, denominator = BioDynaX._fit_implicit(
        spec, X, derivative, collect(eachindex(x)), 1e-7)
    prediction, denominator_values = BioDynaX._evaluate_candidate(
        spec, numerator, denominator, X)

    @test mean(abs2, prediction .- derivative) < 1e-8
    @test minimum(abs, denominator_values) > 0
    @test numerator[2] ≈ vmax / km atol = 1e-3
    @test denominator[1] ≈ inv(km) atol = 1e-3
end

@testset "Hill and competitive inhibition recovery" begin
    single = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    x = collect(range(0.1, 2.0; length = 300))
    X = reshape(x, 1, :)
    vmax, k = 1.8, 0.6
    hill = vmax .* x .^ 4 ./ (k^4 .+ x .^ 4)
    hill_spec = local_basis(
        single, 1; degree = 4, include_interactions = false,
        X, derivative = hill)
    numerator, denominator = BioDynaX._fit_implicit(
        hill_spec, X, hill, collect(eachindex(x)), 1e-7)
    prediction, denominator_values = BioDynaX._evaluate_candidate(
        hill_spec, numerator, denominator, X)
    @test mean(abs2, prediction .- hill) < 1e-7
    @test minimum(abs, denominator_values) > 0

    nodes = [NodeSpec(name = :substrate), NodeSpec(name = :inhibitor)]
    network = BiologicalNetwork(
        nodes, [EdgeSpec(source = 2, target = 1, kind = INHIBITION,
                         family = COMPETITIVE)])
    substrate = repeat(collect(range(0.1, 2.0; length = 30)), 30)
    inhibitor = repeat(collect(range(0.0, 1.5; length = 30)); inner = 30)
    states = permutedims(hcat(substrate, inhibitor))
    km, ki = 0.4, 0.8
    rate = 2.0 .* substrate ./
           (km .* (1 .+ inhibitor ./ ki) .+ substrate)
    spec = local_basis(
        network, 1; degree = 1, include_interactions = false,
        X = states, derivative = rate)
    num, den = BioDynaX._fit_implicit(
        spec, states, rate, collect(eachindex(rate)), 1e-7)
    fitted, denominator_values = BioDynaX._evaluate_candidate(
        spec, num, den, states)
    @test mean(abs2, fitted .- rate) < 1e-7
    @test minimum(abs, denominator_values) > 0
end

@testset "graph-local basis scales with indegree" begin
    nodes = [NodeSpec(name = Symbol("x$i")) for i in 1:50]
    edges = [EdgeSpec(source = i, target = i + 1, kind = ACTIVATION)
             for i in 1:49]
    network = BiologicalNetwork(nodes, edges)
    counts = [candidate_count(local_basis(network, target; degree = 2))
              for target in 1:50]
    @test maximum(counts) ≤ 11
    @test sum(counts) < 50 * 12
end
