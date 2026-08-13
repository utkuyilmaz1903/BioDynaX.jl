@enum EdgeKind ACTIVATION INHIBITION UNKNOWN_NN
@enum NodeKind STATE INPUT LATENT
@enum KineticFamily MASS_ACTION SATURATION HILL COMPETITIVE CUSTOM_KINETIC

Base.@kwdef struct NodeSpec
    name::Symbol
    kind::NodeKind = STATE
    unit::Symbol = :dimensionless
    lower::Float64 = 0.0
    upper::Float64 = Inf
    observed::Bool = true
end

Base.@kwdef struct EdgeSpec
    source::Int
    target::Int
    kind::EdgeKind
    known::Bool = true
    family::KineticFamily = MASS_ACTION
    max_order::Int = 2
    metadata::MetadataLike = EmptyMetadata()
end

"""
Stoichiometric reaction. `stoichiometry` maps state-node ids to signed
coefficients; regulators affect the rate but are not necessarily consumed.
"""
Base.@kwdef struct ReactionSpec
    name::Symbol
    stoichiometry::Dict{Int,Float64}
    regulators::Vector{Int} = Int[]
    known::Bool = true
    family::KineticFamily = MASS_ACTION
    metadata::MetadataLike = EmptyMetadata()
end

"""
    BiologicalNetwork

Species graph (`NodeSpec`), directed interactions (`EdgeSpec`), and
stoichiometric reactions (`ReactionSpec`). Unknown edges and reactions compile
to neural destruction terms; known families become mechanistic IR.
"""
struct BiologicalNetwork
    graph::SimpleDiGraph{Int}
    nodes::Vector{NodeSpec}
    interactions::Dict{Tuple{Int,Int},EdgeSpec}
    reactions::Vector{ReactionSpec}
    # Compatibility views retained through the 0.x migration.
    node_names::Dict{Int,String}
    edge_kinds::Dict{Tuple{Int,Int},EdgeKind}
end

function BiologicalNetwork(nodes::Vector{NodeSpec}, edges::Vector{EdgeSpec};
                           reactions::Vector{ReactionSpec} = ReactionSpec[])
    g = SimpleDiGraph(length(nodes))
    interactions = Dict{Tuple{Int,Int},EdgeSpec}()
    for edge in edges
        1 ≤ edge.source ≤ length(nodes) ||
            throw(ArgumentError("invalid source node $(edge.source)"))
        1 ≤ edge.target ≤ length(nodes) ||
            throw(ArgumentError("invalid target node $(edge.target)"))
        key = (edge.source, edge.target)
        haskey(interactions, key) &&
            throw(ArgumentError("duplicate interaction $key"))
        add_edge!(g, edge.source, edge.target)
        interactions[key] = edge
    end
    names = Dict(i => String(node.name) for (i, node) in pairs(nodes))
    kinds = Dict(key => edge.kind for (key, edge) in interactions)
    network = BiologicalNetwork(g, nodes, interactions, reactions, names, kinds)
    validate_network(network)
    return network
end

function _validate_reaction_metadata!(network::BiologicalNetwork, reaction::ReactionSpec)
    meta = reaction.metadata
    if reaction.family == SATURATION
        _meta_haskey(meta, :vmax_param) ||
            meta isa SaturationMetadata ||
            throw(ArgumentError(
                "reaction $(reaction.name): SATURATION requires vmax_param metadata"))
        _meta_haskey(meta, :km_param) ||
            meta isa SaturationMetadata ||
            throw(ArgumentError(
                "reaction $(reaction.name): SATURATION requires km_param metadata"))
        length(reaction.regulators) == 1 ||
            throw(ArgumentError(
                "reaction $(reaction.name): SATURATION requires exactly one regulator"))
    end
    if reaction.family == CUSTOM_KINETIC
        has_eval = meta isa CustomKineticMetadata && meta.evaluator !== nothing
        has_preset = meta isa CustomKineticMetadata && meta.preset != :none
        dict_eval = meta isa AbstractDict{Symbol} && meta[:evaluator] isa Function
        dict_preset = meta isa AbstractDict{Symbol} && get(meta, :preset, :none) != :none
        (has_eval || has_preset || dict_eval || dict_preset) ||
            throw(ArgumentError(
                "reaction $(reaction.name): CUSTOM_KINETIC requires evaluator or preset"))
        isempty(reaction.regulators) &&
            throw(ArgumentError(
                "reaction $(reaction.name): CUSTOM_KINETIC requires regulators"))
    end
    if reaction.family == HILL && reaction.known
        length(reaction.regulators) == 1 ||
            throw(ArgumentError(
                "reaction $(reaction.name): HILL requires exactly one regulator"))
    end
    if reaction.family == COMPETITIVE && reaction.known
        length(reaction.regulators) == 2 ||
            throw(ArgumentError(
                "reaction $(reaction.name): COMPETITIVE requires two regulators"))
    end
    return nothing
end

function _validate_edge_metadata!(network::BiologicalNetwork, edge::EdgeSpec)
    if edge.family == SATURATION
        meta = edge.metadata
        _meta_haskey(meta, :vmax_param) || meta isa SaturationMetadata ||
            throw(ArgumentError(
                "edge $(edge.source)→$(edge.target): SATURATION requires vmax_param"))
    end
    if edge.kind == UNKNOWN_NN && edge.known
        throw(ArgumentError(
            "edge $(edge.source)→$(edge.target): UNKNOWN_NN edges must set known=false"))
    end
    return nothing
end

function validate_network(network::BiologicalNetwork)
    isempty(network.nodes) && throw(ArgumentError("network cannot be empty"))
    names = getfield.(network.nodes, :name)
    allunique(names) || throw(ArgumentError("node names must be unique"))
    for node in network.nodes
        node.lower ≤ node.upper ||
            throw(ArgumentError("invalid bounds for node $(node.name)"))
        node.kind == INPUT && node.observed == false &&
            @warn "Input node $(node.name) is not observed; identifiability may fail."
    end
    for reaction in network.reactions
        isempty(reaction.stoichiometry) &&
            throw(ArgumentError("reaction $(reaction.name) has no stoichiometry"))
        all(1 ≤ i ≤ length(network.nodes) for i in keys(reaction.stoichiometry)) ||
            throw(ArgumentError("reaction $(reaction.name) references invalid node"))
        all(1 ≤ i ≤ length(network.nodes) for i in reaction.regulators) ||
            throw(ArgumentError("reaction $(reaction.name) references invalid regulator"))
        for coefficient in values(reaction.stoichiometry)
            coefficient == zero(coefficient) &&
                throw(ArgumentError(
                    "reaction $(reaction.name) has zero stoichiometry"))
        end
        _validate_reaction_metadata!(network, reaction)
    end
    for edge in values(network.interactions)
        _validate_edge_metadata!(network, edge)
    end
    return network
end

state_nodes(network::BiologicalNetwork) =
    findall(node -> node.kind != INPUT, network.nodes)

candidate_parents(network::BiologicalNetwork, target::Integer) =
    collect(inneighbors(network.graph, target))

function build_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :DNA_Damage, kind = INPUT, observed = true),
        NodeSpec(name = :p53),
        NodeSpec(name = :Mdm2),
    ]
    edges = [
        EdgeSpec(source = 1, target = 2, kind = ACTIVATION,
                 family = MASS_ACTION),
        EdgeSpec(source = 2, target = 3, kind = ACTIVATION,
                 family = MASS_ACTION),
        EdgeSpec(source = 3, target = 2, kind = UNKNOWN_NN, known = false,
                 family = HILL, max_order = 4),
    ]
    reactions = [
        ReactionSpec(name = :input_drives_p53,
                     stoichiometry = Dict(2 => 1.0),
                     regulators = Int[],
                     metadata = InputDriveMetadata(
                         input_node = 1,
                         rate_param = :α_p53,
                         input_param = :signal)),
        ReactionSpec(name = :p53_to_Mdm2,
                     stoichiometry = Dict(3 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :β_mdm2)),
        ReactionSpec(name = :Mdm2_degrades_p53,
                     stoichiometry = Dict(2 => -1.0), regulators = [3],
                     known = false, family = HILL,
                     metadata = HillMetadata()),
        ReactionSpec(name = :Mdm2_linear_decay,
                     stoichiometry = Dict(3 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :γ_mdm2)),
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

"""Frozen p53/Mdm2 fixture used by legacy five-argument `ude_system` calls."""
const DEFAULT_EXAMPLE_NETWORK = build_network()

"""Small fully-known network for allocation and compiler parity gates."""
function build_linear_test_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :A),
        NodeSpec(name = :B),
    ]
    reactions = [
        ReactionSpec(name = :b_drives_a,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :k_ba)),
        ReactionSpec(name = :a_linear_decay,
                     stoichiometry = Dict(1 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_a)),
        ReactionSpec(name = :b_linear_decay,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_b)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

function describe_network(network::BiologicalNetwork)
    println("BiologicalNetwork: ", nv(network.graph), " nodes, ",
            ne(network.graph), " edges, ", length(network.reactions),
            " reactions")
    for edge in values(network.interactions)
        source = network.nodes[edge.source].name
        target = network.nodes[edge.target].name
        status = edge.known ? "known" : "unknown"
        println("    ", source, " → ", target, " (", edge.kind, ", ",
                edge.family, ", ", status, ")")
    end
    return nothing
end
