#!/usr/bin/env julia
# Library size versus network size. Builds sparse networks with bounded
# in-degree and prints the number of graph-local library terms per target
# for increasing node counts. Not run in CI. Runtime: under a minute.
# Run:  julia --project=. benchmark/scale_basis.jl

using BioDynaX
using BioDynaX: candidate_count, each_library_chunk, evaluate_library
using Random

function sparse_network(node_count; indegree = 3)
    nodes = [NodeSpec(name = Symbol("x$i")) for i in 1:node_count]
    interactions = EdgeSpec[]
    for target in 2:node_count
        for source in max(1, target - indegree):(target - 1)
            push!(interactions,
                EdgeSpec(source = source, target = target, kind = ACTIVATION,
                    known = true, family = MASS_ACTION, max_order = 2))
        end
    end
    return BiologicalNetwork(nodes, interactions)
end

function benchmark_basis(sizes = [10, 50, 100, 250]; n_samples = 400,
        chunk_size = 64)
    rng = MersenneTwister(0)
    for node_count in sizes
        network = sparse_network(node_count)
        elapsed_spec = @elapsed specifications = [local_basis(network, target; degree = 3,
                                                      max_variables = 8)
                                                  for target in 1:node_count]
        columns = sum(candidate_count, specifications)
        X = rand(rng, node_count, n_samples)
        y = randn(rng, n_samples)
        spec = specifications[1]
        elapsed_eval = @elapsed begin
            total = 0.0
            for (buffer, _) in each_library_chunk(
                spec.numerator, X; chunk_size = chunk_size)
                total += sum(buffer)
            end
            total
        end
        design_terms = spec.numerator
        elapsed_fit = @elapsed begin
            library = evaluate_library(design_terms, X)
            BioDynaX._stlsq_blocked(library, y, 1e-2; chunk_size = chunk_size)
        end
        bytes_chunk = chunk_size * length(design_terms) * sizeof(Float64)
        println((
            nodes = node_count,
            columns,
            seconds_spec = elapsed_spec,
            seconds_chunk_eval = elapsed_eval,
            seconds_blocked_fit = elapsed_fit,
            approx_chunk_bytes = bytes_chunk))
    end
end

benchmark_basis()
