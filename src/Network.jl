###############################################################################
# Network.jl — biological graph topology and the BiologicalNetwork struct.
#
# `include`d into the top-level `BioDynaX` module; all names defined here live
# in the BioDynaX namespace.
###############################################################################

"""
    EdgeKind

Conceptual semantics of an edge in a biological regulatory network.

* `ACTIVATION` — source promotes production of the target.
* `INHIBITION` — source linearly suppresses the target.
* `UNKNOWN_NN` — edge kinetics are unknown a priori and will be modeled by
                 a neural network embedded in the UDE right-hand side.
"""
@enum EdgeKind ACTIVATION INHIBITION UNKNOWN_NN

"""
    BiologicalNetwork

Immutable container describing a directed regulatory network.

# Fields
- `graph::SimpleDiGraph{Int}`             — topology (1-indexed nodes).
- `node_names::Dict{Int,String}`          — node-id ↦ species name.
- `edge_kinds::Dict{Tuple{Int,Int},EdgeKind}` — (src, dst) ↦ biological meaning.
"""
struct BiologicalNetwork
    graph::SimpleDiGraph{Int}
    node_names::Dict{Int,String}
    edge_kinds::Dict{Tuple{Int,Int},EdgeKind}
end

"""
    build_network() -> BiologicalNetwork

Canonical 3-node p53 / Mdm2 tumor-suppressor feedback loop:

        DNA_Damage(1) ──[ACT]──▶ p53(2) ──[ACT]──▶ Mdm2(3)
                                  ▲                  │
                                  └──[UNKNOWN_NN]────┘

The `Mdm2 → p53` edge has unknown kinetics — this is what the neural
network embedded inside the UDE will learn, and what the symbolic
regression layer (`Discovery.jl`) will later try to recover in closed form.
"""
function build_network()::BiologicalNetwork
    g = SimpleDiGraph(3)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 2)

    node_names = Dict(1 => "DNA_Damage", 2 => "p53", 3 => "Mdm2")
    edge_kinds = Dict(
        (1, 2) => ACTIVATION,
        (2, 3) => ACTIVATION,
        (3, 2) => UNKNOWN_NN,
    )
    return BiologicalNetwork(g, node_names, edge_kinds)
end

"""
    describe_network(net) -> Nothing

Pretty-print the network topology and the semantic edge labels to stdout.
The name is `describe_network` (not `describe`) to avoid clashing with
`Base`/`InteractiveUtils` exports.
"""
function describe_network(net::BiologicalNetwork)
    println("BiologicalNetwork: ", nv(net.graph), " nodes, ",
            ne(net.graph), " edges")
    for e in edges(net.graph)
        s, d = src(e), dst(e)
        kind = get(net.edge_kinds, (s, d), :unknown)
        println("    [", lpad(s, 2), "] ", net.node_names[s],
                "  →  ", net.node_names[d], "   (", kind, ")")
    end
    return nothing
end
