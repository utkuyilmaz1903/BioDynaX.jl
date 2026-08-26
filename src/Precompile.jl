@setup_workload begin
    network = build_linear_test_network()
    rng = MersenneTwister(0)
    @compile_workload begin
        validate_network(network)
        state_nodes(network)
        compile_mechanism(network)
        local_basis(network, 1; degree = 2)
        model, params = build_ude_model(rng, network)
        u = [0.2, 0.1]
        ude_system(u, params, 0.0, model)
        cache = allocate_cache(model, Float64)
        ude_rhs!(cache.du, u, params, 0.0, model, cache)
        SciMLBase.ODEProblem(model, u, (0.0, 1.0), params)
        unique_claim_fingerprint()
        unique_claim_fingerprint(; smoke = true)
        neural_index_is_dense(compile_mechanism(network))
        extras_print_label(nothing)
    end
end
