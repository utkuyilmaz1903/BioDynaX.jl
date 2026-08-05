@setup_workload begin
    network = build_network()
    @compile_workload begin
        validate_network(network)
        state_nodes(network)
        compile_mechanism(network)
        local_basis(network, 1; degree = 2)
    end
end
