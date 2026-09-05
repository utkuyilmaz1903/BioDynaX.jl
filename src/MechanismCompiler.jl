using StaticArrays: SVector, MVector

abstract type MechanismTerm end

"""Scale factor from reaction stoichiometry (|coefficient|)."""
@inline _scaled(value, scale::Float64) = scale * value

struct InputProductionTerm{R, I} <: MechanismTerm
    target::Int
    rate_param::Symbol
    input_param::Symbol
    scale::Float64
    function InputProductionTerm(target::Int, rate_param::Symbol,
            input_param::Symbol, scale::Float64)
        return new{rate_param, input_param}(target, rate_param, input_param, scale)
    end
end

struct MassActionProductionTerm{P} <: MechanismTerm
    target::Int
    regulator::Int
    param::Symbol
    order::Int
    scale::Float64
    function MassActionProductionTerm(target::Int, regulator::Int,
            param::Symbol, order::Int, scale::Float64)
        return new{param}(target, regulator, param, order, scale)
    end
end

struct LinearDestructionTerm{P} <: MechanismTerm
    target::Int
    param::Symbol
    scale::Float64
    function LinearDestructionTerm(target::Int, param::Symbol, scale::Float64)
        return new{param}(target, param, scale)
    end
end

struct HillDestructionTerm{V, K} <: MechanismTerm
    target::Int
    regulator::Int
    vmax_param::Symbol
    k_param::Symbol
    hill_order::Int
    scale::Float64
    function HillDestructionTerm(target::Int, regulator::Int,
            vmax_param::Symbol, k_param::Symbol,
            hill_order::Int, scale::Float64)
        return new{vmax_param, k_param}(
            target, regulator, vmax_param, k_param, hill_order, scale)
    end
end

struct SaturationDestructionTerm{V, K} <: MechanismTerm
    target::Int
    regulator::Int
    vmax_param::Symbol
    km_param::Symbol
    scale::Float64
    function SaturationDestructionTerm(target::Int, regulator::Int,
            vmax_param::Symbol, km_param::Symbol,
            scale::Float64)
        return new{vmax_param, km_param}(
            target, regulator, vmax_param, km_param, scale)
    end
end

struct SaturationProductionTerm{V, K} <: MechanismTerm
    target::Int
    regulator::Int
    vmax_param::Symbol
    km_param::Symbol
    scale::Float64
    function SaturationProductionTerm(target::Int, regulator::Int,
            vmax_param::Symbol, km_param::Symbol,
            scale::Float64)
        return new{vmax_param, km_param}(
            target, regulator, vmax_param, km_param, scale)
    end
end

struct CompetitiveDestructionTerm{V, M, I} <: MechanismTerm
    target::Int
    substrate::Int
    inhibitor::Int
    vmax_param::Symbol
    km_param::Symbol
    ki_param::Symbol
    scale::Float64
    function CompetitiveDestructionTerm(target::Int, substrate::Int,
            inhibitor::Int, vmax_param::Symbol,
            km_param::Symbol, ki_param::Symbol,
            scale::Float64)
        return new{vmax_param, km_param, ki_param}(
            target, substrate, inhibitor, vmax_param, km_param, ki_param, scale)
    end
end

struct CustomDestructionTerm{F} <: MechanismTerm
    target::Int
    regulators::Vector{Int}
    scale::Float64
    evaluator::F
end

"""
    NeuralDestructionTerm

Unknown destruction rate compiled to a positivity-preserving neural head.
`regulators` are the inputs; `regulator` is the first (backward compatible).
"""
struct NeuralDestructionTerm <: MechanismTerm
    target::Int
    regulator::Int
    nn_index::Int
    scale::Float64
    regulators::Vector{Int}
end

function NeuralDestructionTerm(target::Int, regulator::Int, nn_index::Int, scale::Float64)
    NeuralDestructionTerm(target, regulator, nn_index, scale, Int[regulator])
end

"""Small networks (`n ≤ STATIC_STATE_THRESHOLD`) dispatch `ude_system` through StaticArrays when the state is already an `SVector`."""
const STATIC_STATE_THRESHOLD = 4

struct CompiledMechanism{P, D}
    nstates::Int
    state_ids::Vector{Int}
    node_to_state::Dict{Int, Int}
    production_terms::P
    destruction_terms::D
end

@inline _phys_param(p, name::Symbol) = positive_parameter(getproperty(p.phys, name))

@inline _phys_param(p, ::Val{name}) where {name} = positive_parameter(getproperty(
    p.phys, name))

@inline function _component_axis_type(::ComponentVector{T, V, <:Tuple{Ax}}) where {T, V, Ax}
    return Ax
end

function _phys_index_namedtuple(p::ComponentVector)
    names = Tuple(propertynames(p.phys))
    isempty(names) && return (;)
    Ax = _component_axis_type(p)
    return NamedTuple{names}(ntuple(i -> _phys_name_index(Ax, names[i]),
        length(names)))
end

function _phys_name_index(::Type{Ax}, name::Symbol) where {Ax}
    idxmap = ComponentArrays.indexmap(Ax)
    haskey(idxmap, :phys) || throw(ArgumentError("packed parameters need a phys block"))
    phys = getfield(idxmap, :phys)
    inner = ComponentArrays.indexmap(phys)
    haskey(inner, name) || throw(ArgumentError("missing physical parameter :$name"))
    rel = getfield(inner, name)
    rel isa Integer || throw(ArgumentError("physical parameter :$name is not a scalar"))
    start = first(ComponentArrays.viewindex(phys))
    return start + Int(rel) - 1
end

@generated function _phys_param(
        p::ComponentVector{T, <:AbstractVector, <:Tuple{Ax}},
        ::Val{name}) where {T, Ax, name}
    idx = _phys_name_index(Ax, name)
    return quote
        positive_parameter(@inbounds parent(p)[$idx])
    end
end

function ChainRulesCore.rrule(::typeof(_phys_param), p::ComponentVector,
        ::Val{name}) where {name}
    idx = _phys_name_index(_component_axis_type(p), name)
    raw = @inbounds parent(p)[idx]
    y = positive_parameter(raw)
    function phys_param_pullback(Δ)
        Δu = ChainRulesCore.unthunk(Δ)
        Δu isa ChainRulesCore.AbstractZero &&
            return (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent(),
                ChainRulesCore.NoTangent())
        flat = zero(parent(p))
        @inbounds flat[idx] = sigmoid(raw) * Δu
        tan = ComponentVector(flat, getfield(p, :axes))
        return (ChainRulesCore.NoTangent(), tan, ChainRulesCore.NoTangent())
    end
    return y, phys_param_pullback
end

@noinline _typed_cons(xs::Tuple, x) = (xs..., x)

@inline _nonneg_state(x, index::Int) = _nonneg(x[index])

@inline _term_production_value(term::InputProductionTerm{R, I}, p) where {R, I} = _phys_param(
    p, Val{R}()) * _phys_param(
    p, Val{I}())

@inline function _term_production_value(term::MassActionProductionTerm{P}, x, p) where {P}
    value = _nonneg_state(x, term.regulator)
    for _ in 2:(term.order)
        value *= _nonneg_state(x, term.regulator)
    end
    return _phys_param(p, Val{P}()) * value
end

@inline _term_destruction_value(term::LinearDestructionTerm{P}, p) where {P} = _phys_param(
    p, Val{P}())

@inline function _term_destruction_value(term::HillDestructionTerm{V, K}, x, p) where {V, K}
    reg = _nonneg_state(x, term.regulator)
    kn = _phys_param(p, Val{K}())^term.hill_order
    regn = reg^term.hill_order
    return _phys_param(p, Val{V}()) * regn / (kn + regn + eps(typeof(reg)))
end

@inline function _term_destruction_value(
        term::CompetitiveDestructionTerm{V, M, I}, x, p) where {V, M, I}
    sub = _nonneg_state(x, term.substrate)
    inh = _nonneg_state(x, term.inhibitor)
    km = _phys_param(p, Val{M}())
    ki = _phys_param(p, Val{I}())
    return _phys_param(p, Val{V}()) * sub / (km * (1 + inh / ki) + sub + eps(typeof(sub)))
end

@inline function _term_production_value(
        term::SaturationProductionTerm{V, K}, x, p) where {V, K}
    reg = _nonneg_state(x, term.regulator)
    km = _phys_param(p, Val{K}())
    return _phys_param(p, Val{V}()) * reg / (km + reg + eps(typeof(reg)))
end

@inline function _term_destruction_value(
        term::SaturationDestructionTerm{V, K}, x, p) where {V, K}
    reg = _nonneg_state(x, term.regulator)
    km = _phys_param(p, Val{K}())
    return _phys_param(p, Val{V}()) * reg / (km + reg + eps(typeof(reg)))
end

@inline function _term_destruction_value(term::CustomDestructionTerm, x, p)
    rate = term.evaluator(x, p, term.regulators)
    return max(zero(eltype(x)), rate)
end

@inline function _nn_scalar_output!(nn, p_nn, st, value, slot::Int, cache)
    cache.nn_inputs[1, slot] = value
    input = @view cache.nn_inputs[1:1, slot:slot]
    if nn isa MultiHeadNetwork
        head = nn.heads[slot]
        p_h, s_h = _nn_head_params(p_nn, st, slot)
        output, _ = head(input, p_h, s_h)
        return output[1]
    end
    output, _ = nn(input, p_nn, st)
    return output[1]
end

@inline function _nn_vector_output!(nn, p_nn, st, x, term::NeuralDestructionTerm, cache)
    nin = length(term.regulators)
    slot = term.nn_index
    @inbounds for (i, r) in pairs(term.regulators)
        cache.nn_inputs[i, slot] = _nonneg_state(x, r)
    end
    input = @view cache.nn_inputs[1:nin, slot:slot]
    if nn isa MultiHeadNetwork
        head = nn.heads[slot]
        p_h, s_h = _nn_head_params(p_nn, st, slot)
        output, _ = head(input, p_h, s_h)
        return output[1]
    end
    output, _ = nn(input, p_nn, st)
    return output[1]
end

@inline function _term_destruction_value(
        term::NeuralDestructionTerm, x, p, nn, st, cache)
    if length(term.regulators) == 1
        return _nn_scalar_output!(
            nn, p.nn, st, _nonneg_state(x, term.regulator), term.nn_index, cache)
    end
    return _nn_vector_output!(nn, p.nn, st, x, term, cache)
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

@inline function _require_matching_state_length(u, nstates::Int)
    length(u) == nstates || throw(DimensionMismatch(
        "state length $(length(u)) does not match compiled nstates $nstates"))
    return nothing
end

@inline function _extract_flat(p::ComponentVector)
    return getfield(p, :data)::Vector{Float64}
end

function ChainRulesCore.rrule(::typeof(_extract_flat), p::ComponentVector)
    flat = getfield(p, :data)::Vector{Float64}
    axes = getfield(p, :axes)
    function extract_flat_pullback(Δ)
        Δu = ChainRulesCore.unthunk(Δ)
        Δu isa ChainRulesCore.AbstractZero &&
            return (ChainRulesCore.NoTangent(), ChainRulesCore.ZeroTangent())
        tan = ComponentVector(convert(Vector{Float64}, Δu), axes)
        return (ChainRulesCore.NoTangent(), tan)
    end
    return flat, extract_flat_pullback
end

@inline function _linear_ab_from_flat(x, flat::Vector{Float64}, model::UDEModel)
    k_ba = positive_parameter(@inbounds flat[model.k_ba_idx])
    k_a = positive_parameter(@inbounds flat[model.k_a_idx])
    k_b = positive_parameter(@inbounds flat[model.k_b_idx])
    a = k_ba * _nonneg(x[2]) - k_a * x[1]
    b = -k_b * x[2]
    return a, b
end

"""
    ude_rhs!(du, x, p, t, model, cache)

In-place compiled production–destruction RHS using a preallocated cache.
"""
function ude_rhs!(du, x, p, t, model::UDEModel, cache::UDEModelCache)
    if model.is_linear_ab
        _require_matching_state_length(x, model.nstates)
        a, b = _linear_ab_from_flat(x, _extract_flat(p), model)
        @inbounds begin
            du[1] = a
            du[2] = b
        end
        return du
    end
    # Do not forward the isbits `t` through abstract `impl` dispatch.
    return _ude_rhs_impl!(du, x, p, model.impl, cache)
end

function ude_rhs!(du, x, p, _t, model::UDEModelImpl, cache::UDEModelCache)
    return _ude_rhs_impl!(du, x, p, model, cache)
end

function _ude_rhs_impl!(du, x, p, model::UDEModelImpl{N, NN, ST, C},
        cache::UDEModelCache) where {N, NN, ST, C}
    cm = model.compiled
    _require_matching_state_length(x, cm.nstates)
    fill!(cache.production, zero(eltype(x)))
    fill!(cache.destruction, zero(eltype(x)))
    _accumulate_production_terms!(cache.production, x, p, cm.production_terms)
    _accumulate_destruction_terms!(
        cache.destruction, x, p, model.nn, model.st, cache, cm.destruction_terms)
    @inbounds for i in 1:(cm.nstates)
        du[i] = cache.production[i] - cache.destruction[i] * x[i]
    end
    return du
end

@generated function _accumulate_production_terms!(P, x, p, terms::T) where {T <: Tuple}
    exprs = [:(_dispatch_production!(P, terms[$i], x, p)) for i in 1:fieldcount(T)]
    return quote
        $(exprs...)
        return nothing
    end
end

@generated function _accumulate_destruction_terms!(
        D, x, p, nn, st, cache, terms::T) where {T <: Tuple}
    exprs = [:(_dispatch_destruction!(D, terms[$i], x, p, nn, st, cache))
             for i in 1:fieldcount(T)]
    return quote
        $(exprs...)
        return nothing
    end
end

@inline function _accumulate_production_terms!(P, x, p, terms)
    for term in terms
        _dispatch_production!(P, term, x, p)
    end
    return nothing
end

@inline function _accumulate_destruction_terms!(D, x, p, nn, st, cache, terms)
    for term in terms
        _dispatch_destruction!(D, term, x, p, nn, st, cache)
    end
    return nothing
end

@inline _dispatch_production!(P, term::InputProductionTerm, x, p) = _accumulate_production!(
    P, term, p)
@inline _dispatch_production!(P, term, x, p) = _accumulate_production!(P, term, x, p)

@inline _dispatch_destruction!(D, term::LinearDestructionTerm, x, p, nn, st, cache) = _accumulate_destruction!(
    D, term, p)
@inline _dispatch_destruction!(D, term::NeuralDestructionTerm, x, p, nn, st, cache) = _accumulate_destruction!(
    D, term, x, p, nn, st, cache)
@inline _dispatch_destruction!(D, term, x, p, nn, st, cache) = _accumulate_destruction!(
    D, term, x, p)

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

@inline function _destruction_contribution(
        term::LinearDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, p), term.scale)
end

@inline function _destruction_contribution(term::HillDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(
        term::SaturationDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(
        term::CompetitiveDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _destruction_contribution(
        term::CustomDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    return _scaled(_term_destruction_value(term, x, p), term.scale)
end

@inline function _neural_regulators(x::SVector, term::NeuralDestructionTerm)
    regs = term.regulators
    n = length(regs)
    T = eltype(x)
    n == 1 && return SVector{1, T}(_nonneg_state(x, term.regulator))
    n == 2 && return SVector{2, T}(
        _nonneg_state(x, regs[1]), _nonneg_state(x, regs[2]))
    return T[_nonneg_state(x, r) for r in regs]
end

@inline function _neural_regulators(x, term::NeuralDestructionTerm)
    regs = term.regulators
    n = length(regs)
    T = eltype(x)
    # Heap states (training / Zygote): Vector inputs, not SVector.
    n == 1 && return T[_nonneg_state(x, term.regulator)]
    n == 2 && return T[_nonneg_state(x, regs[1]), _nonneg_state(x, regs[2])]
    return T[_nonneg_state(x, r) for r in regs]
end

@inline function _destruction_contribution(
        term::NeuralDestructionTerm, target, x, p, nn, st)
    term.target == target || return zero(eltype(x))
    input = _neural_regulators(x, term)
    if nn isa MultiHeadNetwork
        head = nn.heads[term.nn_index]
        p_h, s_h = _nn_head_params(p.nn, st, term.nn_index)
        nn_out, _ = head(input, p_h, s_h)
        return _scaled(nn_out[1], term.scale)
    end
    nn_out, _ = nn(input, p.nn, st)
    return _scaled(nn_out[1], term.scale)
end

@inline _state_production(target, x, p, terms::Tuple{}) = zero(eltype(x))
@inline function _state_production(target, x, p, terms::Tuple{A}) where {A}
    return _production_contribution(terms[1], target, x, p)
end
@inline function _state_production(target, x, p, terms::Tuple{A, B}) where {A, B}
    return _production_contribution(terms[1], target, x, p) +
           _production_contribution(terms[2], target, x, p)
end

@generated function _state_production(target, x, p, terms::T) where {T <: Tuple}
    body = [:(value += _production_contribution(terms[$i], target, x, p))
            for i in 1:fieldcount(T)]
    return quote
        value = zero(eltype(x))
        $(body...)
        return value
    end
end

function _state_production(target, x, p, terms)
    value = zero(eltype(x))
    for term in terms
        value = value + _production_contribution(term, target, x, p)
    end
    return value
end

@inline _state_destruction(target, x, p, terms::Tuple{}, nn, st) = zero(eltype(x))
@inline function _state_destruction(target, x, p, terms::Tuple{A}, nn, st) where {A}
    return _destruction_contribution(terms[1], target, x, p, nn, st)
end
@inline function _state_destruction(target, x, p, terms::Tuple{A, B}, nn, st) where {A, B}
    return _destruction_contribution(terms[1], target, x, p, nn, st) +
           _destruction_contribution(terms[2], target, x, p, nn, st)
end

@generated function _state_destruction(target, x, p, terms::T, nn, st) where {T <: Tuple}
    body = [:(value += _destruction_contribution(terms[$i], target, x, p, nn, st))
            for i in 1:fieldcount(T)]
    return quote
        value = zero(eltype(x))
        $(body...)
        return value
    end
end

function _state_destruction(target, x, p, terms, nn, st)
    value = zero(eltype(x))
    for term in terms
        value = value + _destruction_contribution(term, target, x, p, nn, st)
    end
    return value
end

function _ude_system_out_of_place(x, p, _t, model)
    cm = model.compiled
    n = cm.nstates
    T = eltype(x)
    # Build the Vector without `setindex!` so Zygote can differentiate.
    if n == 2
        p1 = _state_production(1, x, p, cm.production_terms)
        d1 = _state_destruction(1, x, p, cm.destruction_terms, model.nn, model.st)
        p2 = _state_production(2, x, p, cm.production_terms)
        d2 = _state_destruction(2, x, p, cm.destruction_terms, model.nn, model.st)
        return T[p1 - d1 * x[1], p2 - d2 * x[2]]
    end
    return map(1:n) do i
        prod = _state_production(i, x, p, cm.production_terms)
        dest = _state_destruction(i, x, p, cm.destruction_terms, model.nn, model.st)
        T(prod - dest * x[i])
    end
end

@inline function _s2_kernel(x::SVector{2, T}, p, prod, dest, nn, st) where {T}
    a = _state_production(1, x, p, prod) -
        _state_destruction(1, x, p, dest, nn, st) * x[1]
    b = _state_production(2, x, p, prod) -
        _state_destruction(2, x, p, dest, nn, st) * x[2]
    return SVector{2, T}(a, b)
end

@inline function _ude_system_static(x::SVector{2, T}, p, _t,
        cm::CompiledMechanism{P, D}, nn, st) where {T, P, D}
    return _s2_kernel(x, p, cm.production_terms, cm.destruction_terms, nn, st)
end

@generated function _ude_system_static(x::SVector{N, T}, p, _t,
        cm::CompiledMechanism{P, D}, nn, st) where {N, T, P, D}
    elems = [:(
                 _state_production($i, x, p, cm.production_terms) -
                 _state_destruction($i, x, p, cm.destruction_terms, nn, st) * x[$i]
             ) for i in 1:N]
    return :(SVector{$N, $T}($(elems...)))
end

@inline function _ude_system_static(x::SVector{N, T}, p, _t,
        model::UDEModel) where {N, T}
    return _ude_system_static_impl(x, p, model.impl)
end

@inline function _ude_system_static(x::SVector{N, T}, p, _t,
        model::UDEModelImpl) where {N, T}
    return _ude_system_static_impl(x, p, model)
end

"""
    ude_system(x, p, t, model::UDEModel) -> dx

Evaluate the compiled production–destruction RHS. `SVector` states with
`n ≤ STATIC_STATE_THRESHOLD` use the StaticArrays kernel.
"""
function ude_system(x, p, t, model::UDEModel)
    # Heap-state path: bounce to the specialized impl without forwarding
    # isbits `t`. The linear A/B fixture uses the Vector{Float64} method.
    return _ude_system_impl(x, p, model.impl, nothing)
end

function ude_system(x, p, t, model::UDEModelImpl)
    return _ude_system_impl(x, p, model, nothing)
end

function _ude_system_impl(
        x, p, model::UDEModelImpl{N, NN, ST, C}, cache) where {N, NN, ST, C}
    _require_matching_state_length(x, model.compiled.nstates)
    if cache !== nothing
        _ude_rhs_impl!(cache.du, x, p, model, cache)
        return copy(cache.du)
    end
    return _ude_system_out_of_place(x, p, 0.0, model)
end

@noinline function _generic_s2(x::SVector{2, Float64}, p, model::UDEModel)
    return _ude_system_static_impl(x, p, model.impl)::SVector{2, Float64}
end

@inline function ude_system(x::SVector{2, Float64}, p, t,
        model::UDEModel)::SVector{2, Float64}
    _require_matching_state_length(x, model.nstates)
    if model.is_linear_ab
        a, b = _linear_ab_from_flat(x, _extract_flat(p), model)
        return SVector{2, Float64}(a, b)
    end
    return _generic_s2(x, p, model)
end

@noinline function _generic_vec64(x::Vector{Float64}, p, model::UDEModel)
    return _ude_system_impl(x, p, model.impl, nothing)::Vector{Float64}
end

@inline function _linear_vec64(x, p, model::UDEModel)
    a, b = _linear_ab_from_flat(x, _extract_flat(p), model)
    return Float64[a, b]
end

@inline function _ude_system_vec64_primal(x, p, model::UDEModel)
    _require_matching_state_length(x, model.nstates)
    if model.is_linear_ab
        return _linear_vec64(x, p, model)
    end
    return _generic_vec64(x, p, model)
end

function ude_system(x::Vector{Float64}, p, t, model::UDEModel)
    return _ude_system_vec64_primal(x, p, model)
end

# Zygote traces both sides of `if model.is_linear_ab`. A custom rrule
# evaluates only the taken kernel so training stays on one AD graph.
function ChainRulesCore.rrule(::typeof(ude_system), x::Vector{Float64}, p, t,
        model::UDEModel)
    y = _ude_system_vec64_primal(x, p, model)
    function ude_system_vec64_pullback(Δ)
        Δu = ChainRulesCore.unthunk(Δ)
        back = if model.is_linear_ab
            last(Zygote.pullback(_linear_vec64, x, p, model))
        else
            last(Zygote.pullback(_generic_vec64, x, p, model))
        end
        gx, gp, gm = back(Δu)
        return (ChainRulesCore.NoTangent(), gx, gp,
            ChainRulesCore.NoTangent(), gm)
    end
    return y, ude_system_vec64_pullback
end

@inline function ude_system(x::SVector{N, T}, p, t, model::UDEModel) where {N, T}
    N == 2 && T === Float64 && return ude_system(SVector{2, Float64}(x), p, t, model)
    return _ude_system_static_impl(x, p, model.impl)
end

@inline function ude_system(x::SVector{N, T}, p, t, model::UDEModelImpl) where {N, T}
    return _ude_system_static_impl(x, p, model)
end

@inline function _ude_system_static_impl(x::SVector{N, T}, p,
        model::UDEModelImpl{Nnet, NN, ST, C}) where {N, T, Nnet, NN, ST, C}
    _require_matching_state_length(x, model.compiled.nstates)
    return _ude_system_static(x, p, 0.0, model.compiled, model.nn, model.st)
end

function ude_system(x::SVector{N, T}, p, t, model::UDEModel,
        cache::UDEModelCache) where {N, T}
    return ude_system(x, p, 0.0, model.impl, cache)
end

function ude_system(x::SVector{N, T}, p, t, model::UDEModelImpl,
        cache::UDEModelCache) where {N, T}
    _require_matching_state_length(x, model.compiled.nstates)
    ude_rhs!(cache.du, x, p, t, model, cache)
    return SVector{N, T}(ntuple(i -> cache.du[i], Val{N}()))
end

function ude_system(x, p, t, nn, st)
    return ude_system(x, p, t, nn, st, DEFAULT_EXAMPLE_NETWORK)
end

function ude_system(x, p, t, nn, st, network::BiologicalNetwork)
    model = ignore_derivatives(() -> compile_network(network, nn, st))
    return ude_system(x, p, t, model)
end

function _custom_kinetic_evaluator(meta::CustomKineticMetadata, reaction_name::Symbol)
    meta.evaluator !== nothing && return meta.evaluator
    meta.preset === :power_law &&
        return (x, p, regulators) -> begin
            reg = _nonneg_state(x, only(regulators))
            return _phys_param(p, meta.rate_param) * reg
        end
    throw(ArgumentError(
        "reaction $reaction_name: CUSTOM_KINETIC requires evaluator or supported preset"))
end

function _custom_kinetic_evaluator(meta::AbstractDict{Symbol}, reaction_name::Symbol)
    evaluator = _meta_get(meta, :evaluator)
    evaluator isa Function && return evaluator
    preset = _meta_get(meta, :preset)
    preset === :power_law &&
        return (x, p, regulators) -> begin
            rate_param = _meta_symbol(meta, :rate_param, reaction_name)
            reg = _nonneg_state(x, only(regulators))
            return _phys_param(p, rate_param) * reg
        end
    throw(ArgumentError(
        "reaction $reaction_name: CUSTOM_KINETIC metadata missing evaluator or preset"))
end

function _custom_kinetic_evaluator(::KineticMetadata, reaction_name::Symbol)
    throw(ArgumentError(
        "reaction $reaction_name: CUSTOM_KINETIC metadata missing evaluator or preset"))
end

function _reaction_production_term(
        reaction::ReactionSpec, target::Int,
        node_to_state::Dict{Int, Int}, scale::Float64)
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
    if _is_input_drive(meta)
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
        node_to_state::Dict{Int, Int}, nn_index::Ref{Int}, scale::Float64)
    meta = reaction.metadata
    if !reaction.known
        isempty(reaction.regulators) &&
            throw(ArgumentError(
                "unknown reaction $(reaction.name) requires one or two regulators"))
        length(reaction.regulators) ≤ 2 ||
            throw(ArgumentError(
                "unknown reaction $(reaction.name) supports at most two regulators"))
        regs = Int[node_to_state[r] for r in reaction.regulators]
        nn_index[] += 1
        return NeuralDestructionTerm(
            target, first(regs), nn_index[], scale, regs)
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

function _edge_production_term(
        edge::EdgeSpec, node_to_state::Dict{Int, Int}, scale::Float64)
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
        edge::EdgeSpec, node_to_state::Dict{Int, Int}, nn_index::Ref{Int},
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
    production_terms = ()
    destruction_terms = ()
    nn_index = Ref(0)
    neural_keys = Set{Tuple{Int, Int}}()

    for reaction in network.reactions
        for (node, coefficient) in reaction.stoichiometry
            haskey(node_to_state, node) || continue
            target = node_to_state[node]
            scale = abs(coefficient)
            if coefficient > 0
                production_terms = _typed_cons(production_terms,
                    _reaction_production_term(
                        reaction, target, node_to_state, scale))
            elseif coefficient < 0
                term = _reaction_destruction_term(
                    reaction, target, node_to_state, nn_index, scale)
                if term isa NeuralDestructionTerm
                    push!(neural_keys, (term.target, term.regulator))
                end
                destruction_terms = _typed_cons(destruction_terms, term)
            end
        end
    end

    for edge in values(network.interactions)
        prod = _edge_production_term(edge, node_to_state, 1.0)
        prod === nothing ||
            (production_terms = _typed_cons(production_terms, prod))
        dest = _edge_destruction_term(edge, node_to_state, nn_index, 1.0)
        if dest !== nothing
            if dest isa NeuralDestructionTerm
                key = (dest.target, dest.regulator)
                key in neural_keys && continue
                push!(neural_keys, key)
            end
            destruction_terms = _typed_cons(destruction_terms, dest)
        end
    end

    isempty(production_terms) &&
        throw(ArgumentError("compiled mechanism has no production terms"))
    isempty(destruction_terms) &&
        throw(ArgumentError("compiled mechanism has no destruction terms"))

    # Duplicate unknown reaction+edge pairs skip the edge after
    # `_edge_destruction_term` has already incremented `nn_index`. A later
    # kept unknown then received a gapped slot (1, 3, …). Multi-head dispatch
    # and `allocate_cache` index by that slot, so `ude_system` / `ude_rhs!`
    # threw BoundsError. Renumber kept heads to 1:n.
    destruction_terms = _reindex_neural_destruction(destruction_terms)

    return CompiledMechanism(
        length(state_ids),
        copy(state_ids),
        node_to_state,
        production_terms,
        destruction_terms)
end

function _reindex_neural_destruction(terms::Tuple)
    acc = Ref(0)
    return map(term -> _reindex_neural_term(term, acc), terms)
end

function _reindex_neural_destruction!(terms::Vector)
    acc = Ref(0)
    for i in eachindex(terms)
        terms[i] = _reindex_neural_term(terms[i], acc)
    end
    return terms
end

@inline function _reindex_neural_term(term::NeuralDestructionTerm, acc::Ref{Int})
    acc[] += 1
    term.nn_index == acc[] && return term
    return NeuralDestructionTerm(
        term.target, term.regulator, acc[], term.scale, term.regulators)
end

@inline _reindex_neural_term(term, ::Ref{Int}) = term

const COMPILE_NETWORK_COUNTER = Ref{Union{Nothing, Base.RefValue{Int}}}(nothing)

function _note_compile_network()
    counter = COMPILE_NETWORK_COUNTER[]
    counter === nothing && return nothing
    counter[] += 1
    return nothing
end

function _neural_layout(compiled)
    n_neural = 0
    max_nn_in = 0
    for term in compiled.destruction_terms
        if term isa NeuralDestructionTerm
            n_neural += 1
            max_nn_in = max(max_nn_in, length(term.regulators))
        end
    end
    return n_neural, max_nn_in
end

function _dummy_packed_params(network, nn, st, compiled)
    impl = UDEModelImpl(network, nn, st, compiled, compiled.state_ids)
    n_neural, max_nn_in = _neural_layout(compiled)
    tmp = UDEModel(
        network, nn, st, compiled, compiled.state_ids,
        impl, compiled.nstates, n_neural, max_nn_in, false, 0, 0, 0)
    schema = parameter_schema(tmp)
    return pack_parameters(default_phys_parameters(schema),
        _nn_parameters_matching(Random.default_rng(), tmp))
end

function _is_linear_ab_compiled(cm)
    cm.nstates == 2 || return false
    prod = cm.production_terms
    dest = cm.destruction_terms
    length(prod) == 1 || return false
    length(dest) == 2 || return false
    p1 = prod[1]
    d1 = dest[1]
    d2 = dest[2]
    p1 isa MassActionProductionTerm{:k_ba} || return false
    d1 isa LinearDestructionTerm{:k_a} || return false
    d2 isa LinearDestructionTerm{:k_b} || return false
    return p1.target == 1 && p1.regulator == 2 &&
           d1.target == 1 && d2.target == 2
end

function _assemble_compiled_model(network::BiologicalNetwork, nn::NN, st::ST) where {NN, ST}
    compiled = compile_mechanism(network)
    unknown = [edge for edge in values(network.interactions) if !edge.known]
    isempty(unknown) &&
        !any(reaction -> !reaction.known, network.reactions) &&
        @debug "Compiled network has no unknown mechanisms."
    impl = UDEModelImpl(network, nn, st, compiled, compiled.state_ids)
    n_neural, max_nn_in = _neural_layout(compiled)
    if _is_linear_ab_compiled(compiled)
        dummy_p = _dummy_packed_params(network, nn, st, compiled)
        idxs = _phys_index_namedtuple(dummy_p)
        return UDEModel(
            network, nn, st, compiled, compiled.state_ids, impl,
            compiled.nstates, n_neural, max_nn_in, true,
            idxs.k_ba, idxs.k_a, idxs.k_b)
    end
    return UDEModel(
        network, nn, st, compiled, compiled.state_ids, impl,
        compiled.nstates, n_neural, max_nn_in, false, 0, 0, 0)
end

function compile_network(network::BiologicalNetwork, nn::NN, st::ST) where {NN, ST}
    _note_compile_network()
    return _assemble_compiled_model(network, nn, st)::UDEModel
end

# Exact architecture used by the SciML allocation fixture. The typeassert
# keeps `compile_network` inferred as the concrete public wrapper.
function compile_network(network::BiologicalNetwork, nn::_DEFAULT_NN_TYPE,
        st::_DEFAULT_ST_TYPE)
    _note_compile_network()
    return _assemble_compiled_model(network, nn, st)::UDEModel
end

"""
    build_ude_model(rng, network=DEFAULT_EXAMPLE_NETWORK) -> (model, params)

Compile `network` into a `UDEModel` and default `ComponentVector` parameters
(`phys` + `nn`).
"""
function build_ude_model(rng::AbstractRNG,
        network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK)
    compiled = compile_mechanism(network)
    nn_terms = [term
                for term in compiled.destruction_terms
                if term isa NeuralDestructionTerm]
    sort!(nn_terms; by = term -> term.nn_index)
    n_heads = max(length(nn_terms), 1)
    input_dims = isempty(nn_terms) ? [1] :
                 [length(term.regulators) for term in nn_terms]
    nn, nn_ps, st = build_ude_nn(rng; n_heads = n_heads, input_dims = input_dims)
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
