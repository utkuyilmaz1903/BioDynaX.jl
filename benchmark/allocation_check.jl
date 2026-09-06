#!/usr/bin/env julia
# Allocation check for the in-place right-hand side `ude_rhs!` on a two-state
# linear network. Warms up, then prints the bytes allocated by one call.
# Runs in CI on every push (job "allocation-check"); the result must stay at
# zero bytes. Runtime: under a minute after precompilation.
# Run:  julia --project=. benchmark/allocation_check.jl

using BioDynaX
using Random

function allocation_check()
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
    bytes == 0 || bytes ≤ 512 || error("allocation check failed with $bytes bytes")
end

allocation_check()
