using StaticArrays: SVector, MVector

abstract type MechanismTerm end

"""Scale factor from reaction stoichiometry (|coefficient|)."""
@inline _scaled(value, scale::Float64) = scale * value

struct InputProductionTerm <: MechanismTerm
    target::Int
    rate_param::Symbol
    input_param::Symbol
    scale::Float64
end

struct MassActionProductionTerm <: MechanismTerm
    target::Int
    regulator::Int
    param::Symbol
    order::Int
    scale::Float64
end

struct LinearDestructionTerm <: MechanismTerm
    target::Int
    param::Symbol
    scale::Float64
end

struct HillDestructionTerm <: MechanismTerm
    target::Int
    regulator::Int
    vmax_param::Symbol
    k_param::Symbol
    hill_order::Int
    scale::Float64
end

struct SaturationDestructionTerm <: MechanismTerm
    target::Int
    regulator::Int
    vmax_param::Symbol
    km_param::Symbol
    scale::Float64
end

struct SaturationProductionTerm <: MechanismTerm
    target::Int
    regulator::Int
    vmax_param::Symbol
    km_param::Symbol
    scale::Float64
end

struct CompetitiveDestructionTerm <: MechanismTerm
    target::Int
    substrate::Int
    inhibitor::Int
    vmax_param::Symbol
    km_param::Symbol
    ki_param::Symbol
    scale::Float64
end

struct CustomDestructionTerm{F} <: MechanismTerm
    target::Int
    regulators::Vector{Int}
    scale::Float64
    evaluator::F
end

"""Unknown destruction rate compiled to a positivity-preserving neural head."""
struct NeuralDestructionTerm <: MechanismTerm
    target::Int
    regulator::Int
    nn_index::Int
    scale::Float64
end

"""Small networks (`n ≤ STATIC_STATE_THRESHOLD`) dispatch `ude_system` through StaticArrays when the state is already an `SVector`."""
const STATIC_STATE_THRESHOLD = 4

struct CompiledMechanism{P,D}
    nstates::Int
    state_ids::Vector{Int}
    node_to_state::Dict{Int,Int}
    production_terms::P
    destruction_terms::D
end

@inline _phys_param(p, name::Symbol) =
    positive_parameter(getproperty(p.phys, name))

@inline _nonneg_state(x, index::Int) = _nonneg(x[index])

@inline _term_production_value(term::InputProductionTerm, p) =
    _phys_param(p, term.rate_param) * _phys_param(p, term.input_param)

@inline function _term_production_value(term::MassActionProductionTerm, x, p)
    value = _nonneg_state(x, term.regulator)
    for _ in 2:term.order
        value *= _nonneg_state(x, term.regulator)
    end
    return _phys_param(p, term.param) * value
end

@inline _term_destruction_value(term::LinearDestructionTerm, p) =
    _phys_param(p, term.param)

@inline function _term_destruction_value(term::HillDestructionTerm, x, p)
    reg = _nonneg_state(x, term.regulator)
    kn = _phys_param(p, term.k_param)^term.hill_order
    regn = reg^term.hill_order
    return _phys_param(p, term.vmax_param) * regn / (kn + regn + eps(typeof(reg)))
end

@inline function _term_destruction_value(term::CompetitiveDestructionTerm, x, p)
    sub = _nonneg_state(x, term.substrate)
    inh = _nonneg_state(x, term.inhibitor)
    km = _phys_param(p, term.km_param)
    ki = _phys_param(p, term.ki_param)
    return _phys_param(p, term.vmax_param) * sub / (km * (1 + inh / ki) + sub + eps(typeof(sub)))
end

@inline function _term_production_value(term::SaturationProductionTerm, x, p)
    reg = _nonneg_state(x, term.regulator)
    km = _phys_param(p, term.km_param)
    return _phys_param(p, term.vmax_param) * reg / (km + reg + eps(typeof(reg)))
end

@inline function _term_destruction_value(term::SaturationDestructionTerm, x, p)
    reg = _nonneg_state(x, term.regulator)
    km = _phys_param(p, term.km_param)
    return _phys_param(p, term.vmax_param) * reg / (km + reg + eps(typeof(reg)))
end

@inline function _term_destruction_value(term::CustomDestructionTerm, x, p)
    rate = term.evaluator(x, p, term.regulators)
    return max(zero(eltype(x)), rate)
end

@inline function _nn_scalar_output!(nn, p_nn, st, value, slot::Int, cache)
    cache.nn_inputs[1, slot] = value
    if nn isa MultiHeadNetwork
        head = nn.heads[slot]
        p_h, s_h = _nn_head_params(p_nn, st, slot)
        output, _ = head(@view(cache.nn_inputs[:, slot:slot]), p_h, s_h)
        return output[1]
    end
    output, _ = nn(@view(cache.nn_inputs[:, slot:slot]), p_nn, st)
    return output[1]
end

@inline function _term_destruction_value(
        term::NeuralDestructionTerm, x, p, nn, st, cache)
    return _nn_scalar_output!(
        nn, p.nn, st, _nonneg_state(x, term.regulator), term.nn_index, cache)
end

@inline function _accumulate_production!(P, term::InputProductionTerm, p)
    P[term.target] += _scaled(_term_production_value(term, p), term.scale)
    return nothing
end

@inline function _accumulate_production!(P, term::MassActionProductionTerm, x, p)
    P[term.target] += _scaled(_term_production_value(term, x, p), term.scale)
    return nothing
end

@inline function _accumulate_production!(P, term::SaturationProductionTerm, x, p)
    P[term.target] += _scaled(_term_production_value(term, x, p), term.scale)
    return nothing
end

@inline function _accumulate_destruction!(D, term::LinearDestructionTerm, p)
    D[term.target] += _scaled(_term_destruction_value(term, p), term.scale)
    return nothing
end

@inline function _accumulate_destruction!(D, term::HillDestructionTerm, x, p)
    D[term.target] += _scaled(_term_destruction_value(term, x, p), term.scale)
    return nothing
end

@inline function _accumulate_destruction!(D, term::SaturationDestructionTerm, x, p)
    D[term.target] += _scaled(_term_destruction_value(term, x, p), term.scale)
    return nothing
end

@inline function _accumulate_destruction!(D, term::CompetitiveDestructionTerm, x, p)
    D[term.target] += _scaled(_term_destruction_value(term, x, p), term.scale)
    return nothing
end

@inline function _accumulate_destruction!(D, term::CustomDestructionTerm, x, p)
    D[term.target] += _scaled(_term_destruction_value(term, x, p), term.scale)
    return nothing
end

@inline function _accumulate_destruction!(
        D, term::NeuralDestructionTerm, x, p, nn, st, cache)
    D[term.target] += _scaled(
        _term_destruction_value(term, x, p, nn, st, cache), term.scale)
    return nothing
end

function ude_rhs!(du, x, p, _t, model::UDEModel, cache::UDEModelCache)
    cm = model.compiled
    fill!(cache.production, zero(eltype(x)))
    fill!(cache.destruction, zero(eltype(x)))
    for term in cm.production_terms
        if term isa InputProductionTerm
            _accumulate_production!(cache.production, term, p)
        elseif term isa SaturationProductionTerm
            _accumulate_production!(cache.production, term, x, p)
        else
            _accumulate_production!(cache.production, term, x, p)
        end
    end
    for term in cm.destruction_terms
        if term isa LinearDestructionTerm
            _accumulate_destruction!(cache.destruction, term, p)
        elseif term isa NeuralDestructionTerm
            _accumulate_destruction!(
                cache.destruction, term, x, p, model.nn, model.st, cache)
        elseif term isa CustomDestructionTerm
            _accumulate_destruction!(cache.destruction, term, x, p)
        else
            _accumulate_destruction!(cache.destruction, term, x, p)
        end
    end
    @inbounds for i in 1:cm.nstates
        du[i] = cache.production[i] - cache.destruction[i] * x[i]
    end
    return du
end

@inline function _production_contribution(term::InputProductionTerm, target, x, p)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_production_value(term, p), term.scale)
end

@inline function _production_contribution(term::MassActionProductionTerm, target, x, p)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_production_value(term, x, p), term.scale)
end

@inline function _production_contribution(term::SaturationProductionTerm, target, x, p)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_production_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(term::LinearDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, p), term.scale)
end

@inline function _destruction_contribution(term::HillDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(term::SaturationDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(term::CompetitiveDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(term::CustomDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(term::NeuralDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    reg = _nonneg_state(x, term.regulator)
    if nn isa MultiHeadNetwork
        head = nn.heads[term.nn_index]
        p_h, s_h = _nn_head_params(p.nn, st, term.nn_index)
        nn_out, _ = head([reg], p_h, s_h)
        return _scaled(nn_out[1], term.scale)
    end
    nn_out, _ = nn([reg], p.nn, st)
    return _scaled(nn_out[1], term.scale)
end

function _state_production(target, x, p, terms)
    value = zero(eltype(x))
    for term in terms
        value = value + _production_contribution(term, target, x, p)
    end
    return value
end

function _state_destruction(target, x, p, terms, nn, st)
    value = zero(eltype(x))
    for term in terms
        value = value + _destruction_contribution(term, target, x, p, nn, st)
    end
    return value
end

function _ude_system_out_of_place(x, p, _t, model::UDEModel)
    cm = model.compiled
    n = cm.nstates
    T = eltype(x)
    return Vector{T}(map(1:n) do i
        _state_production(i, x, p, cm.production_terms) -
        _state_destruction(i, x, p, cm.destruction_terms, model.nn, model.st) * x[i]
    end)
end

@inline function _ude_system_static(x::SVector{N,T}, p, _t, model::UDEModel) where {N,T}
    cm = model.compiled
    du = MVector{N,T}(undef)
    @inbounds for i in 1:N
        prod = _state_production(i, x, p, cm.production_terms)
        dest = _state_destruction(i, x, p, cm.destruction_terms, model.nn, model.st)
        du[i] = prod - dest * x[i]
    end
    return SVector(du)
end

"""
    ude_system(x, p, t, model::UDEModel) -> dx

Evaluate the compiled production–destruction RHS. `SVector` states with
`n ≤ STATIC_STATE_THRESHOLD` use the StaticArrays kernel.
"""
function ude_system(x, p, t, model::UDEModel; cache = nothing)
    if cache !== nothing
        ude_rhs!(cache.du, x, p, t, model, cache)
        return copy(cache.du)
    end
    return _ude_system_out_of_place(x, p, t, model)
end

function ude_system(x::SVector{N,T}, p, t, model::UDEModel;
                    cache = nothing) where {N,T}
    if cache !== nothing
        return SVector{N,T}(ude_system(Vector(x), p, t, model; cache = cache))
    end
    if N ≤ STATIC_STATE_THRESHOLD && model.compiled.nstates == N
        return _ude_system_static(x, p, t, model)
    end
    return SVector{N,T}(_ude_system_out_of_place(x, p, t, model))
end

function ude_system(x, p, t, nn, st)
    return ude_system(x, p, t, nn, st, DEFAULT_EXAMPLE_NETWORK)
end

function ude_system(x, p, t, nn, st, network::BiologicalNetwork)
    model = Zygote.@ignore compile_network(network, nn, st)
    return ude_system(x, p, t, model)
end

function _custom_kinetic_evaluator(meta::MetadataLike, reaction_name::Symbol)
    if meta isa CustomKineticMetadata
        meta.evaluator !== nothing && return meta.evaluator
        meta.preset == :power_law &&
            return (x, p, regulators) -> begin
                reg = _nonneg_state(x, only(regulators))
                return _phys_param(p, meta.rate_param) * reg
            end
        throw(ArgumentError(
            "reaction $reaction_name: CUSTOM_KINETIC requires evaluator or supported preset"))
    end
    evaluator = _meta_get(meta, :evaluator)
    evaluator isa Function && return evaluator
    preset = _meta_get(meta, :preset)
    preset == :power_law &&
        return (x, p, regulators) -> begin
            rate_param = _meta_symbol(meta, :rate_param, reaction_name)
            reg = _nonneg_state(x, only(regulators))
            return _phys_param(p, rate_param) * reg
        end
    throw(ArgumentError(
        "reaction $reaction_name: CUSTOM_KINETIC metadata missing evaluator or preset"))
end

function _reaction_production_term(
        reaction::ReactionSpec, target::Int,
        node_to_state::Dict{Int,Int}, scale::Float64)
    meta = reaction.metadata
    if isempty(reaction.regulators)
        _meta_haskey(meta, :input_param) ||
            meta isa InputDriveMetadata ||
            throw(ArgumentError(
                "reaction $(reaction.name) requires input metadata for basal production"))
        return InputProductionTerm(
            target,
            _meta_symbol(meta, :rate_param, reaction.name),
            _meta_symbol(meta, :input_param, :signal),
            scale)
    end
    if _meta_haskey(meta, :drive) &&
       _meta_symbol(meta, :drive, :none) == :input
        return InputProductionTerm(
            target,
            _meta_symbol(meta, :rate_param, reaction.name),
            _meta_symbol(meta, :input_param, :signal),
            scale)
    end
    if reaction.family == SATURATION && length(reaction.regulators) == 1
        return SaturationProductionTerm(
            target, node_to_state[only(reaction.regulators)],
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :km_param, :km), scale)
    end
    length(reaction.regulators) == 1 ||
        throw(ArgumentError(
            "reaction $(reaction.name) must declare exactly one regulator"))
    regulator = node_to_state[only(reaction.regulators)]
    order = _meta_int(meta, :order, 1)
    return MassActionProductionTerm(
        target, regulator,
        _meta_symbol(meta, :rate_param, reaction.name), order, scale)
end

function _reaction_destruction_term(
        reaction::ReactionSpec, target::Int,
        node_to_state::Dict{Int,Int}, nn_index::Ref{Int}, scale::Float64)
    meta = reaction.metadata
    if !reaction.known
        length(reaction.regulators) == 1 ||
            throw(ArgumentError(
                "unknown reaction $(reaction.name) requires one regulator"))
        nn_index[] += 1
        return NeuralDestructionTerm(
            target, node_to_state[only(reaction.regulators)], nn_index[], scale)
    end
    if reaction.family == HILL && length(reaction.regulators) == 1
        return HillDestructionTerm(
            target, node_to_state[only(reaction.regulators)],
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :k_param, :K),
            _meta_int(meta, :hill_order, 4), scale)
    end
    if reaction.family == SATURATION && length(reaction.regulators) == 1
        return SaturationDestructionTerm(
            target, node_to_state[only(reaction.regulators)],
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :km_param, :km), scale)
    end
    if reaction.family == COMPETITIVE && length(reaction.regulators) == 2
        return CompetitiveDestructionTerm(
            target,
            node_to_state[reaction.regulators[1]],
            node_to_state[reaction.regulators[2]],
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :km_param, :km),
            _meta_symbol(meta, :ki_param, :ki), scale)
    end
    if reaction.family == CUSTOM_KINETIC
        regulators = [node_to_state[i] for i in reaction.regulators]
        evaluator = _custom_kinetic_evaluator(meta, reaction.name)
        return CustomDestructionTerm(target, regulators, scale, evaluator)
    end
    if isempty(reaction.regulators)
        return LinearDestructionTerm(
            target, _meta_symbol(meta, :rate_param, reaction.name), scale)
    end
    throw(ArgumentError(
        "reaction $(reaction.name) has unsupported known destruction form"))
end

function _edge_production_term(edge::EdgeSpec, node_to_state::Dict{Int,Int}, scale::Float64)
    edge.kind == ACTIVATION || return nothing
    haskey(node_to_state, edge.target) || return nothing
    meta = edge.metadata
    _meta_haskey(meta, :rate_param) || return nothing
    rate_param = _meta_symbol(meta, :rate_param, Symbol("k_", edge.source))
    target = node_to_state[edge.target]
    if haskey(node_to_state, edge.source)
        return MassActionProductionTerm(
            target, node_to_state[edge.source],
            rate_param, _meta_int(meta, :order, 1), scale)
    end
    _meta_haskey(meta, :input_param) || return nothing
    return InputProductionTerm(
        target, rate_param, _meta_symbol(meta, :input_param, :signal), scale)
end

function _edge_destruction_term(
        edge::EdgeSpec, node_to_state::Dict{Int,Int}, nn_index::Ref{Int},
        scale::Float64 = 1.0)
    haskey(node_to_state, edge.target) || return nothing
    target = node_to_state[edge.target]
    meta = edge.metadata
    if edge.kind == UNKNOWN_NN && !edge.known
        haskey(node_to_state, edge.source) || return nothing
        nn_index[] += 1
        return NeuralDestructionTerm(
            target, node_to_state[edge.source], nn_index[], scale)
    end
    if edge.kind == INHIBITION && edge.family == COMPETITIVE
        regulators = _meta_get(meta, :regulators)
        regulators isa Vector && length(regulators) == 2 ||
            (_meta_haskey(meta, :substrate) && _meta_haskey(meta, :inhibitor)) ||
            return nothing
        substrate_id = something(_meta_get(meta, :substrate), edge.source)
        inhibitor_id = something(_meta_get(meta, :inhibitor), edge.source)
        substrate = node_to_state[substrate_id]
        inhibitor = node_to_state[inhibitor_id]
        return CompetitiveDestructionTerm(
            target, substrate, inhibitor,
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :km_param, :km),
            _meta_symbol(meta, :ki_param, :ki), scale)
    end
    if edge.family == SATURATION && haskey(node_to_state, edge.source)
        return SaturationDestructionTerm(
            target, node_to_state[edge.source],
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :km_param, :km), scale)
    end
    return nothing
end

"""
    compile_mechanism(network) -> CompiledMechanism

Lower reactions and edges to production–destruction IR
(`duᵢ = Pᵢ − Dᵢ·uᵢ`). Unknown mechanisms become `NeuralDestructionTerm`.
"""
function compile_mechanism(network::BiologicalNetwork)
    state_ids = state_nodes(network)
    isempty(state_ids) &&
        throw(ArgumentError("network contains no dynamic states"))
    node_to_state = Dict(node => row for (row, node) in pairs(state_ids))
    production_terms = MechanismTerm[]
    destruction_terms = MechanismTerm[]
    nn_index = Ref(0)
    neural_keys = Set{Tuple{Int,Int}}()

    for reaction in network.reactions
        for (node, coefficient) in reaction.stoichiometry
            haskey(node_to_state, node) || continue
            target = node_to_state[node]
            scale = abs(coefficient)
            if coefficient > 0
                push!(production_terms, _reaction_production_term(
                    reaction, target, node_to_state, scale))
            elseif coefficient < 0
                term = _reaction_destruction_term(
                    reaction, target, node_to_state, nn_index, scale)
                if term isa NeuralDestructionTerm
                    push!(neural_keys, (term.target, term.regulator))
                end
                push!(destruction_terms, term)
            end
        end
    end

    for edge in values(network.interactions)
        prod = _edge_production_term(edge, node_to_state, 1.0)
        prod === nothing || push!(production_terms, prod)
        dest = _edge_destruction_term(edge, node_to_state, nn_index, 1.0)
        if dest !== nothing
            if dest isa NeuralDestructionTerm
                key = (dest.target, dest.regulator)
                key in neural_keys && continue
                push!(neural_keys, key)
            end
            push!(destruction_terms, dest)
        end
    end

    isempty(production_terms) &&
        throw(ArgumentError("compiled mechanism has no production terms"))
    isempty(destruction_terms) &&
        throw(ArgumentError("compiled mechanism has no destruction terms"))

    return CompiledMechanism(
        length(state_ids),
        copy(state_ids),
        node_to_state,
        Tuple(production_terms),
        Tuple(destruction_terms))
end

function compile_network(network::BiologicalNetwork, nn, st)
    compiled = compile_mechanism(network)
    unknown = [edge for edge in values(network.interactions) if !edge.known]
    isempty(unknown) &&
        !any(reaction -> !reaction.known, network.reactions) &&
        @debug "Compiled network has no unknown mechanisms."
    return UDEModel(network, nn, st, compiled, compiled.state_ids)
end

"""
    build_ude_model(rng, network=DEFAULT_EXAMPLE_NETWORK) -> (model, params)

Compile `network` into a `UDEModel` and default `ComponentVector` parameters
(`phys` + `nn`).
"""
function build_ude_model(rng::AbstractRNG,
                         network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK)
    compiled = compile_mechanism(network)
    n_heads = count(term -> term isa NeuralDestructionTerm,
                    compiled.destruction_terms)
    nn, nn_ps, st = build_ude_nn(rng; n_heads = max(n_heads, 1))
    model = compile_network(network, nn, st)
    schema = parameter_schema(model)
    params = pack_parameters(default_phys_parameters(schema), nn_ps)
    return model, params
end

function build_ude_rhs(model::UDEModel, cache::UDEModelCache)
    return SciMLBase.ODEFunction{true}((du, u, p, t) -> begin
        ude_rhs!(du, u, p, t, model, cache)
        return nothing
    end)
end
