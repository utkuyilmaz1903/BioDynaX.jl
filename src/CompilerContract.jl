###############################################################################
# Compiler remapping contract (not exported).
#
# Duplicate unknown reaction+edge pairs skip the edge after incrementing
# nn_index. compile_mechanism reindexes kept NeuralDestructionTerm heads to
# 1:n so ude_system / ude_rhs! / allocate_cache stay in bounds.
# validate_network does not own this contract. generate_data /
# default_parameters build the compiled NN tree (multi-head and
# multi-regulator); they are not a 1-input dummy hole.
###############################################################################

"""Source strings that prove compile_mechanism remaps kept heads."""
const COMPILER_REINDEX_MUST_CONTAIN = (
    "function compile_mechanism",
    "_reindex_neural_destruction!",
    "Renumber kept heads to 1:n")

"""Phrases that would put a unique-claim hole gate into the compiler."""
const COMPILER_REINDEX_MUST_NOT_CONTAIN = (
    "assert_single_unknown_destruction",
    "unique-claim protocol requires exactly one")

function neural_destruction_terms(compiled::CompiledMechanism)
    [term for term in compiled.destruction_terms if term isa NeuralDestructionTerm]
end

function neural_destruction_terms(network::BiologicalNetwork)
    neural_destruction_terms(compile_mechanism(network))
end

function neural_nn_indices(compiled::CompiledMechanism)
    Int[term.nn_index for term in neural_destruction_terms(compiled)]
end

neural_nn_indices(model::UDEModel) = neural_nn_indices(model.compiled)

function neural_nn_indices(network::BiologicalNetwork)
    neural_nn_indices(compile_mechanism(network))
end

function neural_regulator_arities(compiled::CompiledMechanism)
    return Int[length(term.regulators) for term in neural_destruction_terms(compiled)]
end

neural_regulator_arities(model::UDEModel) = neural_regulator_arities(model.compiled)

neural_head_count(compiled::CompiledMechanism) = length(neural_destruction_terms(compiled))

neural_head_count(model::UDEModel) = model.n_neural

function neural_head_count(network::BiologicalNetwork)
    neural_head_count(compile_mechanism(network))
end

"""True when kept `nn_index` values are exactly `1:n` in some order."""
function neural_index_is_dense(compiled::CompiledMechanism)
    idx = sort(neural_nn_indices(compiled))
    return idx == collect(1:length(idx))
end

neural_index_is_dense(model::UDEModel) = neural_index_is_dense(model.compiled)

function neural_index_is_dense(network::BiologicalNetwork)
    neural_index_is_dense(compile_mechanism(network))
end

function assert_dense_neural_index(compiled::CompiledMechanism)
    neural_index_is_dense(compiled) || throw(ErrorException(
        "NeuralDestructionTerm nn_index must be dense 1:n; got $(neural_nn_indices(compiled))"))
    return compiled
end

function assert_dense_neural_index(model::UDEModel)
    assert_dense_neural_index(model.compiled)
    return model
end

"""
    neural_cache_matches_heads(model, cache) -> Bool

`allocate_cache` columns are the kept-head count. A gapped `nn_index`
would write past that matrix.
"""
function neural_cache_matches_heads(model::UDEModel, cache)
    terms = neural_destruction_terms(model)
    n = length(terms)
    size(cache.nn_inputs, 2) == n || return false
    n == 0 && return size(cache.nn_inputs, 1) == 0
    maximum(term.nn_index for term in terms) == n || return false
    max_in = maximum(length(term.regulators) for term in terms)
    return size(cache.nn_inputs, 1) >= max_in
end

function neural_multihead_matches(model::UDEModel)
    n = neural_head_count(model)
    n <= 1 && return !(model.nn isa MultiHeadNetwork)
    model.nn isa MultiHeadNetwork || return false
    return length(model.nn.heads) == n
end

function compile_mechanism_source_path()
    joinpath(pkgdir(BioDynaX), "src", "MechanismCompiler.jl")
end

function compile_mechanism_reindexes_source()
    src = read(compile_mechanism_source_path(), String)
    return all(occursin(needle, src) for needle in COMPILER_REINDEX_MUST_CONTAIN) &&
           !any(occursin(needle, src) for needle in COMPILER_REINDEX_MUST_NOT_CONTAIN)
end

function compile_mechanism_source_violations()
    src = read(compile_mechanism_source_path(), String)
    missing = [s for s in COMPILER_REINDEX_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in COMPILER_REINDEX_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

"""
    evaluate_compiled_rhs(model, params, x) -> NamedTuple

Finite `ude_system` / `ude_rhs!` pair used to lock remapping. A gapped
slot throws BoundsError here.
"""
function evaluate_compiled_rhs(model::UDEModel, params, x)
    dx = ude_system(x, params, 0.0, model)
    cache = allocate_cache(model, Float64)
    ude_rhs!(cache.du, x, params, 0.0, model, cache)
    return (;
        dx,
        cache_dx = copy(cache.du),
        finite = all(isfinite, dx) && all(isfinite, cache.du),
        parity = Vector(cache.du) ≈ dx,
        dense = neural_index_is_dense(model),
        cache_matches = neural_cache_matches_heads(model, cache),
        multihead_matches = neural_multihead_matches(model))
end

"""
    build_skipped_duplicate_unknown_network()

Unknown reaction plus a matching UNKNOWN_NN edge (the DEFAULT_EXAMPLE
dual-declaration) and a later kept unknown. Before remapping the second
head received slot 3 on a 2-head network.
"""
function build_skipped_duplicate_unknown_network()::BiologicalNetwork
    nodes = [NodeSpec(name = :A), NodeSpec(name = :B), NodeSpec(name = :C)]
    reactions = [
        ReactionSpec(name = :drive_a,
            stoichiometry = Dict(1 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_ca)),
        ReactionSpec(name = :unknown_ab,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = false, family = HILL, metadata = HillMetadata()),
        ReactionSpec(name = :b_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_b)),
        ReactionSpec(name = :c_decay,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_c))
    ]
    edges = [
        EdgeSpec(source = 2, target = 1, kind = UNKNOWN_NN, known = false,
            family = HILL),
        EdgeSpec(source = 3, target = 1, kind = UNKNOWN_NN, known = false,
            family = HILL)
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

"""
    build_skipped_middle_unknown_network()

Three unknown destructions where the middle one is a duplicate
reaction+edge pair. Remapping must keep slots 1:2, not 1 and 3.
"""
function build_skipped_middle_unknown_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :A),
        NodeSpec(name = :B),
        NodeSpec(name = :C),
        NodeSpec(name = :D)
    ]
    reactions = [
        ReactionSpec(name = :drive_a,
            stoichiometry = Dict(1 => 1.0), regulators = [4],
            metadata = MassActionMetadata(rate_param = :k_da)),
        ReactionSpec(name = :unknown_ab,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = false, family = HILL, metadata = HillMetadata()),
        ReactionSpec(name = :unknown_bc,
            stoichiometry = Dict(2 => -1.0), regulators = [3],
            known = false, family = HILL, metadata = HillMetadata()),
        ReactionSpec(name = :unknown_cd,
            stoichiometry = Dict(3 => -1.0), regulators = [4],
            known = false, family = HILL, metadata = HillMetadata()),
        ReactionSpec(name = :d_decay,
            stoichiometry = Dict(4 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_d))
    ]
    edges = [
        EdgeSpec(source = 3, target = 2, kind = UNKNOWN_NN, known = false,
        family = HILL)
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

"""
    build_two_regulator_unknown_network()

Unknown destruction with two regulators `D(S, I)`. `build_ude_model`
must size the NN input to 2. This fixture is not the unique-claim path.
"""
function build_two_regulator_unknown_network()::BiologicalNetwork
    nodes = [NodeSpec(name = :S), NodeSpec(name = :I), NodeSpec(name = :E)]
    reactions = [
        ReactionSpec(name = :drive_s,
            stoichiometry = Dict(1 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_es)),
        ReactionSpec(name = :unknown_si,
            stoichiometry = Dict(1 => -1.0), regulators = [1, 2],
            known = false, family = COMPETITIVE, metadata = HillMetadata()),
        ReactionSpec(name = :i_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_i)),
        ReactionSpec(name = :e_decay,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_e))
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""
    build_zero_unknown_linear_network()

Alias of the fully-known linear fixture. Recovery must reject it;
`validate_network` and compile must not.
"""
build_zero_unknown_linear_network() = build_linear_test_network()

function default_example_has_duplicate_unknown_declaration()
    net = DEFAULT_EXAMPLE_NETWORK
    unknown_reactions = count(r -> !r.known, net.reactions)
    unknown_edges = count(e -> !e.known && e.kind == UNKNOWN_NN, values(net.interactions))
    return unknown_reactions == 1 && unknown_edges == 1
end

"""
    compile_unknown_topology(network) -> NamedTuple

Compile + `build_ude_model` snapshot used by remapping tests. Does not
train a UDE and does not call `generate_data` with custom `truth_params`.
"""
function compile_unknown_topology(network::BiologicalNetwork;
        rng::AbstractRNG = MersenneTwister(13),
        x = nothing)
    compiled = compile_mechanism(network)
    assert_dense_neural_index(compiled)
    model, params = build_ude_model(rng, network)
    assert_dense_neural_index(model)
    n = compiled.nstates
    state = x === nothing ? fill(0.25, n) : Float64.(x)
    rhs = evaluate_compiled_rhs(model, params, state)
    schema = parameter_schema(model)
    return (;
        network,
        compiled,
        model,
        params,
        schema,
        n_heads = neural_head_count(compiled),
        indices = sort(neural_nn_indices(compiled)),
        arities = neural_regulator_arities(compiled),
        rhs,
        validate_open = validate_network(network) === network)
end
