# Opt-in JET analysis of a small public surface. Typo-mode stays in quality.jl.
# This is not a package-wide type-stability claim.

function standards_jet_reports(f, types)
    result = JET.report_call(f, types; target_modules = (BioDynaX,))
    return JET.get_reports(result)
end

function standards_jet_opt_reports(f, types)
    result = JET.report_opt(f, types; target_modules = (BioDynaX,))
    return JET.get_reports(result)
end

@testset "JET report_call on train_ude discover_unknown_rate compose_hybrid_rhs" begin
    rng = MersenneTwister(41)
    model, params = build_ude_model(
        rng, build_hill_recovery_network(; known = false, hill_order = 2))
    term = only(filter(t -> t isa NeuralDestructionTerm,
        model.compiled.destruction_terms))
    rate_fn = r -> 0.25 * r[1] / (0.4 + r[1])
    times = collect(range(0.0, 0.4; length = 6))
    data = rand(rng, 2, length(times))
    u0 = [0.3, 0.2]
    tspan = (0.0, 0.4)
    R = rand(rng, 1, 16)
    D = rand(rng, 1, 16)
    t_disc = collect(range(0.0, 1.0; length = 16))

    compose_types = typeof((model, params, term, rate_fn))
    discover_types = typeof((R, t_disc, D))
    train_types = Tuple{
        typeof(params), typeof(data), typeof(times),
        typeof(u0), typeof(tspan), UDEModel}

    compose_reports = standards_jet_reports(compose_hybrid_rhs, compose_types)
    discover_reports = standards_jet_reports(discover_unknown_rate, discover_types)
    train_reports = standards_jet_reports(train_ude, train_types)

    @test isempty(compose_reports)
    @test isempty(discover_reports)
    @test isempty(train_reports)
end

@testset "JET report_opt on the same public verbs" begin
    rng = MersenneTwister(43)
    model, params = build_ude_model(
        rng, build_hill_recovery_network(; known = false, hill_order = 2))
    term = only(filter(t -> t isa NeuralDestructionTerm,
        model.compiled.destruction_terms))
    rate_fn = r -> 0.25 * r[1] / (0.4 + r[1])
    times = collect(range(0.0, 0.4; length = 6))
    data = rand(rng, 2, length(times))
    u0 = [0.3, 0.2]
    tspan = (0.0, 0.4)
    R = rand(rng, 1, 16)
    D = rand(rng, 1, 16)
    t_disc = collect(range(0.0, 1.0; length = 16))

    compose_types = typeof((model, params, term, rate_fn))
    discover_types = typeof((R, t_disc, D))
    train_types = Tuple{
        typeof(params), typeof(data), typeof(times),
        typeof(u0), typeof(tspan), UDEModel}

    @test isempty(standards_jet_opt_reports(compose_hybrid_rhs, compose_types))
    @test isempty(standards_jet_opt_reports(discover_unknown_rate, discover_types))
    @test isempty(standards_jet_opt_reports(train_ude, train_types))
end
