@testset "compiler kinetic families" begin
    rng = MersenneTwister(31)
    nn, nn_ps, st = build_ude_nn(rng)

    hill_nodes = [
        NodeSpec(name = :S),
        NodeSpec(name = :T)
    ]
    hill_reactions = [
        ReactionSpec(name = :drive,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = Dict(:rate_param => :k_prod)),
        ReactionSpec(name = :hill_decay,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = true, family = HILL,
            metadata = Dict(:vmax_param => :vmax, :k_param => :K,
                :hill_order => 4)),
        ReactionSpec(name = :linear,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = Dict(:rate_param => :gamma))
    ]
    hill_network = BiologicalNetwork(hill_nodes, EdgeSpec[];
        reactions = hill_reactions)
    hill_model = compile_network(hill_network, nn, st)
    @test any(t -> t isa BioDynaX.HillDestructionTerm,
        hill_model.compiled.destruction_terms)

    comp_nodes = [
        NodeSpec(name = :E),
        NodeSpec(name = :S),
        NodeSpec(name = :I)
    ]
    comp_reactions = [
        ReactionSpec(name = :source,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = Dict(:rate_param => :k_in)),
        ReactionSpec(name = :competitive,
            stoichiometry = Dict(1 => -1.0),
            regulators = [2, 3], known = true, family = COMPETITIVE,
            metadata = Dict(:vmax_param => :vmax, :km_param => :km,
                :ki_param => :ki)),
        ReactionSpec(name = :substrate_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = Dict(:rate_param => :k_s)),
        ReactionSpec(name = :inhibitor_decay,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = Dict(:rate_param => :k_i))
    ]
    comp_network = BiologicalNetwork(comp_nodes, EdgeSpec[];
        reactions = comp_reactions)
    comp_model = compile_network(comp_network, nn, st)
    @test any(t -> t isa BioDynaX.CompetitiveDestructionTerm,
        comp_model.compiled.destruction_terms)

    hill_params = pack_parameters(
        (k_prod = 1.0, vmax = 2.0, K = 0.5, gamma = 0.7), nn_ps)
    x = [0.3, 0.4]
    dx = ude_system(x, hill_params, 0.0, hill_model)
    @test all(isfinite, dx)

    comp_params = pack_parameters(
        (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6, k_s = 0.8, k_i = 0.5),
        nn_ps)
    x2 = [0.2, 0.5, 0.1]
    dx2 = ude_system(x2, comp_params, 0.0, comp_model)
    @test all(isfinite, dx2)

    @test_throws ArgumentError compile_mechanism(BiologicalNetwork(
        [NodeSpec(name = :only)], EdgeSpec[];
        reactions = [ReactionSpec(name = :bad,
            stoichiometry = Dict(1 => -1.0),
            regulators = [1], known = true,
            family = HILL)]))
end

@testset "general pipeline second topology" begin
    rng = MersenneTwister(99)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    tspan = (0.0, 1.0)
    times = collect(range(tspan...; length = 8))
    prediction = predict_ude(
        params, [0.2, 0.1], tspan, times, model;
        solver_config = SolverConfig())
    @test all(isfinite, prediction)
    _, _, noisy, _ = generate_data(
        rng; u0 = [0.2, 0.1], network = network, n_points = 8, noise_σ = 0.01)
    result = train_ude(
        params, noisy, times, [0.2, 0.1], tspan, model;
        adam_iters = 4, bfgs_iters = 0, log_every = 100, verbose = false)
    @test result.final_loss ≤ result.initial_loss
end
