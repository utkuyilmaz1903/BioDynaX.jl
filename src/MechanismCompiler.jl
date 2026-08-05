using StaticArrays: SMatrix

abstract type MechanismTerm end

struct InputProductionTerm <: MechanismTerm
    target::Int
    rate_param::Symbol
    input_param::Symbol
end

struct MassActionProductionTerm <: MechanismTerm
    target::Int
    regulator::Int
    param::Symbol
    order::Int
end

struct LinearDestructionTerm <: MechanismTerm
    target::Int
    param::Symbol
end

struct HillDestructionTerm <: MechanismTerm
    target::Int
    regulator::Int
    vmax_param::Symbol
    k_param::Symbol
    hill_order::Int
end

struct CompetitiveDestructionTerm <: MechanismTerm
    target::Int
    substrate::Int
    inhibitor::Int
    vmax_param::Symbol
    km_param::Symbol
    ki_param::Symbol
end

struct NeuralDestructionTerm <: MechanismTerm
    target::Int
    regulator::Int
    nn_index::Int
end

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

@inline function _nn_scalar_output!(nn, p_nn, st, value, slot::Int, cache)
    cache.nn_inputs[1, slot] = value
    output, _ = nn(@view(cache.nn_inputs[:, slot:slot]), p_nn, st)
    return output[1]
end

@inline function _term_destruction_value(
        term::NeuralDestructionTerm, x, p, nn, st, cache)
    return _nn_scalar_output!(
        nn, p.nn, st, _nonneg_state(x, term.regulator), term.nn_index, cache)
end

@inline function _accumulate_production!(P, term::InputProductionTerm, p)
    P[term.target] += _term_production_value(term, p)
    return nothing
end

@inline function _accumulate_production!(P, term::MassActionProductionTerm, x, p)
    P[term.target] += _term_production_value(term, x, p)
    return nothing
end

@inline function _accumulate_destruction!(D, term::LinearDestructionTerm, p)
    D[term.target] += _term_destruction_value(term, p)
    return nothing
end

@inline function _accumulate_destruction!(D, term::HillDestructionTerm, x, p)
    D[term.target] += _term_destruction_value(term, x, p)
    return nothing
end

@inline function _accumulate_destruction!(D, term::CompetitiveDestructionTerm, x, p)
    D[term.target] += _term_destruction_value(term, x, p)
    return nothing
end

@inline function _accumulate_destruction!(
        D, term::NeuralDestructionTerm, x, p, nn, st, cache)
    D[term.target] += _term_destruction_value(term, x, p, nn, st, cache)
    return nothing
end

function ude_rhs!(du, x, p, _t, model::UDEModel, cache::UDEModelCache)
    cm = model.compiled
    fill!(cache.production, zero(eltype(x)))
    fill!(cache.destruction, zero(eltype(x)))
    for term in cm.production_terms
        if term isa InputProductionTerm
            _accumulate_production!(cache.production, term, p)
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
    return _term_production_value(term, p)
end

@inline function _production_contribution(term::MassActionProductionTerm, target, x, p)
    term.target == target || return zero(eltype(x))
    return _term_production_value(term, x, p)
end

@inline function _destruction_contribution(term::LinearDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _term_destruction_value(term, p)
end

@inline function _destruction_contribution(term::HillDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _term_destruction_value(term, x, p)
end

@inline function _destruction_contribution(term::CompetitiveDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _term_destruction_value(term, x, p)
end

@inline function _destruction_contribution(term::NeuralDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    reg = _nonneg_state(x, term.regulator)
    nn_out, _ = nn([reg], p.nn, st)
    return nn_out[1]
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
    T = eltype(x)
    return Vector{T}(map(1:cm.nstates) do i
        _state_production(i, x, p, cm.production_terms) -
        _state_destruction(i, x, p, cm.destruction_terms, model.nn, model.st) * x[i]
    end)
end

function ude_system(x, p, t, model::UDEModel; cache = nothing)
    if cache !== nothing
        ude_rhs!(cache.du, x, p, t, model, cache)
        return copy(cache.du)
    end
    return _ude_system_out_of_place(x, p, t, model)
end

function ude_system(x, p, t, nn, st)
    return ude_system(x, p, t, nn, st, DEFAULT_EXAMPLE_NETWORK)
end

function ude_system(x, p, t, nn, st, network::BiologicalNetwork)
    model = Zygote.@ignore compile_network(network, nn, st)
    return ude_system(x, p, t, model)
end

function _meta_symbol(meta::Dict{Symbol,Any}, key::Symbol, default::Symbol)
    return get(meta, key, default)
end

function _meta_int(meta::Dict{Symbol,Any}, key::Symbol, default::Int)
    value = get(meta, key, default)
    return value isa Integer ? Int(value) : default
end

function _reaction_production_term(
        reaction::ReactionSpec, target::Int,
        node_to_state::Dict{Int,Int})
    meta = reaction.metadata
    if haskey(meta, :drive) && meta[:drive] == :input
        return InputProductionTerm(
            target,
            _meta_symbol(meta, :rate_param, reaction.name),
            _meta_symbol(meta, :input_param, :signal))
    end
    length(reaction.regulators) == 1 ||
        throw(ArgumentError(
            "reaction $(reaction.name) must declare exactly one regulator"))
    regulator = node_to_state[only(reaction.regulators)]
    order = _meta_int(meta, :order, 1)
    return MassActionProductionTerm(
        target, regulator,
        _meta_symbol(meta, :rate_param, reaction.name), order)
end

function _reaction_destruction_term(
        reaction::ReactionSpec, target::Int,
        node_to_state::Dict{Int,Int}, nn_index::Ref{Int})
    meta = reaction.metadata
    if !reaction.known
        length(reaction.regulators) == 1 ||
            throw(ArgumentError(
                "unknown reaction $(reaction.name) requires one regulator"))
        nn_index[] += 1
        return NeuralDestructionTerm(
            target, node_to_state[only(reaction.regulators)], nn_index[])
    end
    if reaction.family == HILL && length(reaction.regulators) == 1
        return HillDestructionTerm(
            target, node_to_state[only(reaction.regulators)],
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :k_param, :K),
            _meta_int(meta, :hill_order, 4))
    end
    if reaction.family == COMPETITIVE && length(reaction.regulators) == 2
        return CompetitiveDestructionTerm(
            target,
            node_to_state[reaction.regulators[1]],
            node_to_state[reaction.regulators[2]],
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :km_param, :km),
            _meta_symbol(meta, :ki_param, :ki))
    end
    if isempty(reaction.regulators)
        return LinearDestructionTerm(
            target, _meta_symbol(meta, :rate_param, reaction.name))
    end
    throw(ArgumentError(
        "reaction $(reaction.name) has unsupported known destruction form"))
end

function _edge_production_term(edge::EdgeSpec, node_to_state::Dict{Int,Int})
    edge.kind == ACTIVATION || return nothing
    haskey(node_to_state, edge.target) || return nothing
    meta = edge.metadata
    haskey(meta, :rate_param) || return nothing
    target = node_to_state[edge.target]
    if haskey(node_to_state, edge.source)
        return MassActionProductionTerm(
            target, node_to_state[edge.source],
            meta[:rate_param], _meta_int(meta, :order, 1))
    end
    haskey(meta, :input_param) || return nothing
    return InputProductionTerm(target, meta[:rate_param], meta[:input_param])
end

function _edge_destruction_term(
        edge::EdgeSpec, node_to_state::Dict{Int,Int}, nn_index::Ref{Int})
    haskey(node_to_state, edge.target) || return nothing
    target = node_to_state[edge.target]
    meta = edge.metadata
    if edge.kind == UNKNOWN_NN && !edge.known
        haskey(node_to_state, edge.source) || return nothing
        nn_index[] += 1
        return NeuralDestructionTerm(
            target, node_to_state[edge.source], nn_index[])
    end
    if edge.kind == INHIBITION && edge.family == COMPETITIVE
        length(edge.metadata[:regulators]) == 2 ||
            haskey(meta, :substrate) && haskey(meta, :inhibitor) ||
            return nothing
        substrate = node_to_state[get(meta, :substrate, edge.source)]
        inhibitor = node_to_state[get(meta, :inhibitor, edge.source)]
        return CompetitiveDestructionTerm(
            target, substrate, inhibitor,
            _meta_symbol(meta, :vmax_param, :vmax),
            _meta_symbol(meta, :km_param, :km),
            _meta_symbol(meta, :ki_param, :ki))
    end
    return nothing
end

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
            if coefficient > 0
                push!(production_terms, _reaction_production_term(
                    reaction, target, node_to_state))
            elseif coefficient < 0
                term = _reaction_destruction_term(
                    reaction, target, node_to_state, nn_index)
                if term isa NeuralDestructionTerm
                    push!(neural_keys, (term.target, term.regulator))
                end
                push!(destruction_terms, term)
            end
        end
    end

    for edge in values(network.interactions)
        prod = _edge_production_term(edge, node_to_state)
        prod === nothing || push!(production_terms, prod)
        dest = _edge_destruction_term(edge, node_to_state, nn_index)
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
        @warn "Compiled network has no unknown mechanisms."
    return UDEModel(network, nn, st, compiled, compiled.state_ids)
end

function build_ude_model(rng::AbstractRNG,
                         network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK)
    nn, nn_ps, st = build_ude_nn(rng)
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
