@testset "typed kinetic metadata" begin
    meta = InputDriveMetadata(
        rate_param = :α, input_param = :signal, input_node = 1)
    @test BioDynaX._meta_symbol(meta, :rate_param, :default) == :α
    @test BioDynaX._meta_symbol(meta, :input_param, :default) == :signal
    @test BioDynaX._meta_haskey(meta, :drive)

    hill = HillMetadata(vmax_param = :vmax, k_param = :K, hill_order = 3)
    @test BioDynaX._meta_int(hill, :hill_order, 4) == 3
    @test BioDynaX._meta_symbol(hill, :k_param, :default) == :K

    dict = Dict(:rate_param => :k1, :order => 2)
    @test BioDynaX._meta_symbol(dict, :rate_param, :default) == :k1
    @test BioDynaX._meta_int(dict, :order, 1) == 2
end

@testset "typed metadata compiles p53 network" begin
    network = build_network()
    for reaction in network.reactions
        @test reaction.metadata isa KineticMetadata
    end
    rng = MersenneTwister(7)
    nn, nn_ps, st = build_ude_nn(rng)
    model = compile_network(network, nn, st)
    @test model.compiled.nstates == 2
    params = pack_parameters(
        (α_p53 = 0.9, β_mdm2 = 1.1, γ_mdm2 = 1.5, signal = 1.0), nn_ps)
    dx = ude_system([0.2, 0.1], params, 0.0, model)
    @test all(isfinite, dx)
end

@testset "dict metadata backward compatibility" begin
    rng = MersenneTwister(11)
    nn, nn_ps, st = build_ude_nn(rng)
    network = BiologicalNetwork(
        [NodeSpec(name = :x), NodeSpec(name = :y)],
        EdgeSpec[];
        reactions = [
            ReactionSpec(name = :drive,
                         stoichiometry = Dict(1 => 1.0), regulators = [2],
                         metadata = Dict(:rate_param => :k_xy)),
            ReactionSpec(name = :decay_x,
                         stoichiometry = Dict(1 => -1.0), regulators = Int[],
                         metadata = Dict(:rate_param => :k_x)),
            ReactionSpec(name = :decay_y,
                         stoichiometry = Dict(2 => -1.0), regulators = Int[],
                         metadata = Dict(:rate_param => :k_y)),
        ])
    model = compile_network(network, nn, st)
    params = pack_parameters((k_xy = 1.0, k_x = 0.5, k_y = 0.4), nn_ps)
    @test all(isfinite, ude_system([0.3, 0.2], params, 0.0, model))
end
