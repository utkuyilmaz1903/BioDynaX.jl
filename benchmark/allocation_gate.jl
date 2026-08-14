using BioDynaX
using Random

function allocation_gate()
    rng = MersenneTwister(0)
    network = BioDynaX.build_linear_test_network()
    model, params = build_ude_model(rng, network)
    parameters = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    u = [0.2, 0.1]
    cache = allocate_cache(model, Float64)
    for _ in 1:20
        ude_rhs!(cache.du, u, parameters, 0.0, model, cache)
    end
    bytes = @allocated ude_rhs!(cache.du, u, parameters, 0.0, model, cache)
    println((kernel = :ude_rhs!, network = :linear_test, allocated_bytes = bytes))
    bytes == 0 || bytes ≤ 512 || error("allocation gate failed with $bytes bytes")
end

allocation_gate()
