using BioDynaX
using Graphs
using Random

function sparse_network(node_count; indegree = 3)
    nodes = [NodeSpec(name = Symbol("x$i")) for i in 1:node_count]
    interactions = EdgeSpec[]
    for target in 2:node_count
        for source in max(1, target - indegree):(target - 1)
            push!(interactions,
                  EdgeSpec(source, target, ACTIVATION, true, MASS_ACTION, 2,
                           Dict{Symbol,Any}()))
        end
    end
    return BiologicalNetwork(nodes, interactions)
end

function benchmark_basis(sizes = [10, 50, 100, 250])
    for node_count in sizes
        network = sparse_network(node_count)
        elapsed = @elapsed specifications = [
            local_basis(network, target; degree = 3, max_variables = 8)
            for target in 1:node_count
        ]
        columns = sum(candidate_count, specifications)
        println((nodes = node_count, columns, seconds = elapsed))
    end
end

benchmark_basis()
