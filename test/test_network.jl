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

@testset "graph edges derived from unknown reactions" begin
    # An unknown term declared as a reaction alone gives the same graph parents
    # as one that also declares the edge; known reactions add no edges.
    reaction_only = BioDynaX.build_hill_recovery_network(; known = false)
    @test candidate_parents(reaction_only, 1) == [2]
    @test candidate_parents(reaction_only, 2) == Int[]
    @test isempty(reaction_only.interactions)
    @test BioDynaX.Graphs.ne(reaction_only.graph) == 1
    graph_spec = local_basis(reaction_only, 1; degree = 2, include_interactions = false,
        scope = :graph)
    @test graph_spec.variables == [1, 2]
    with_edge = BioDynaX._library_study_two_state_graph_network(; parent = 2)
    @test candidate_parents(with_edge, 1) == candidate_parents(reaction_only, 1)
    @test BioDynaX.Graphs.ne(with_edge.graph) == 1
    @test local_basis(with_edge, 1; degree = 2, include_interactions = false,
        scope = :graph).variables == graph_spec.variables
    # Known kinetics: the same reactions with known = true derive nothing.
    known = BioDynaX.build_hill_recovery_network(; known = true)
    @test candidate_parents(known, 1) == Int[]
    @test BioDynaX.Graphs.ne(known.graph) == 0
    # A regulator that is the changed species itself gives a self-loop, as the
    # explicitly declared S -> S edge of the study's wrong two-state graph does.
    self_regulated = BioDynaX.build_hill_recovery_network(; known = false, parent = 1)
    @test candidate_parents(self_regulated, 1) == [1]
    @test BioDynaX.Graphs.ne(self_regulated.graph) == 1
    @test candidate_parents(self_regulated, 1) == candidate_parents(
        BioDynaX._library_study_two_state_graph_network(; parent = 1), 1)
    # Two regulators of one unknown reaction both become parents.
    two = BioDynaX.build_two_regulator_unknown_network()
    @test candidate_parents(two, 1) == [1, 2]
    # Fixtures that declare their edges explicitly are unchanged.
    four = BioDynaX.build_three_state_unknown_network()
    @test candidate_parents(four, 1) == [2]
    @test all(isempty(candidate_parents(four, i)) for i in 2:4)
    wrong = BioDynaX.build_wrong_graph_unknown_network()
    @test candidate_parents(wrong, 1) == [3]
end
