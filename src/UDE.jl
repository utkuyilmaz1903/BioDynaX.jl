"""
Positive-by-construction production/destruction UDE. The invariant form
`duᵢ = Pᵢ(u,p,t) - Dᵢ(u,p,t)uᵢ`, with `Pᵢ,Dᵢ ≥ 0`, points inward at every
zero-state boundary and therefore preserves the biological positive orthant.
"""
struct UDEModel{N,NN,ST,C}
    network::N
    nn::NN
    st::ST
    compiled::C
    state_ids::Vector{Int}
end

"""
    pack_parameters(phys, nn_ps) -> ComponentVector

Merge physical kinetic constants and Lux NN parameters into a single
`ComponentVector` so the optimiser sees one flat parameter vector while
user-code keeps name-based access (`p.phys.α_p53`, `p.nn.layer_1.weight`).
"""
function pack_parameters(phys::NamedTuple, nn_ps)
    raw_phys = (; (name => inverse_softplus(value)
                   for (name, value) in pairs(phys))...)
    return ComponentVector(phys = raw_phys, nn = ComponentVector(nn_ps))
end

# Smooth, Zygote-safe non-negativity surrogate.
@inline _nonneg(x) = max(zero(x), x)
@inline positive_parameter(raw; floor = eps(typeof(raw))) =
    softplus(raw) + floor
@inline inverse_softplus(value) =
    value > zero(value) ? value + log(-expm1(-value)) :
    throw(DomainError(value, "positive parameters must be > 0"))
@inline bounded_parameter(raw, lower, upper) =
    lower + (upper - lower) * sigmoid(raw)

"""Independent Lux head per unknown mechanism (Phase 2 multi-NN)."""
struct MultiHeadNetwork{T<:Tuple}
    heads::T
end

function _chain_from_widths(widths, activations)
    layers = Lux.Dense[]
    input = 1
    for (output, activation) in zip(widths, activations)
        push!(layers, Lux.Dense(input => output, activation))
        input = output
    end
    return Lux.Chain(layers...)
end

function _single_head_chain(preset::Symbol = :medium)
    preset == :small &&
        return _chain_from_widths([4, 1], [tanh, softplus])
    preset == :large &&
        return _chain_from_widths([16, 16, 16, 1], [tanh, tanh, tanh, softplus])
    preset == :medium &&
        return _chain_from_widths([8, 8, 1], [tanh, tanh, softplus])
    throw(ArgumentError("unknown NN preset $preset; choose :small, :medium, or :large"))
end

"""
    build_ude_nn(rng; n_heads=1, preset=:medium)

Build one or more independent Lux heads. `preset` selects `:small`, `:medium`,
or `:large` architecture templates.
"""
function build_ude_nn(rng::AbstractRNG; n_heads::Int = 1, preset::Symbol = :medium)
    n_heads ≥ 1 || throw(ArgumentError("n_heads must be ≥ 1"))
    if n_heads == 1
        model = _single_head_chain(preset)
        ps, st = Lux.setup(rng, model)
        return model, ps, st
    end
    heads = ntuple(_ -> _single_head_chain(preset), n_heads)
    ps_pairs = Pair{Symbol,Any}[]
    st_pairs = Pair{Symbol,Any}[]
    for i in 1:n_heads
        ps_i, st_i = Lux.setup(rng, heads[i])
        push!(ps_pairs, Symbol("head_$i") => ps_i)
        push!(st_pairs, Symbol("head_$i") => st_i)
    end
    return MultiHeadNetwork(heads),
           ComponentVector(; ps_pairs...),
           NamedTuple(st_pairs)
end

@inline function _nn_head_params(p_nn, st, slot::Int)
    name = Symbol("head_$slot")
    return getproperty(p_nn, name), getproperty(st, name)
end
