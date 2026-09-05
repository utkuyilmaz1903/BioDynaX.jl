@testset "typed biological network" begin
    network = build_network()
    @test length(network.nodes) == 3
    @test state_nodes(network) == [2, 3]
    @test candidate_parents(network, 2) == [1, 3]
    @test network.interactions[(3, 2)].family == HILL
    @test !network.interactions[(3, 2)].known
    @test validate_network(network) === network

    duplicate = [
        EdgeSpec(source = 1, target = 2, kind = ACTIVATION),
        EdgeSpec(source = 1, target = 2, kind = ACTIVATION)
    ]
    @test_throws ArgumentError BiologicalNetwork(
        [NodeSpec(name = :a), NodeSpec(name = :b)], duplicate)

    mechanism = compile_mechanism(network)
    @test mechanism.nstates == 2
    @test length(mechanism.production_terms) == 2
    @test length(mechanism.destruction_terms) == 2
    @test mechanism.production_terms[1] isa BioDynaX.InputProductionTerm
    @test mechanism.destruction_terms[1] isa BioDynaX.NeuralDestructionTerm
end
