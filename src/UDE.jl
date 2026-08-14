"""
    UDEModel

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

function _chain_from_widths(widths, activations; input::Int = 1)
    layers = Lux.Dense[]
    in_dim = input
    for (output, activation) in zip(widths, activations)
        push!(layers, Lux.Dense(in_dim => output, activation))
        in_dim = output
    end
    return Lux.Chain(layers...)
end

"""Promote Lux parameter leaves to `Float64` (states and ODE solves are Float64)."""
function _float64_param_tree(x)
    if x isa AbstractArray
        return eltype(x) <: AbstractFloat ? Float64.(x) : x
    elseif x isa NamedTuple || x isa Tuple
        return map(_float64_param_tree, x)
    else
        return x
    end
end

function _single_head_chain(preset::Symbol = :medium; input::Int = 1)
    input ≥ 1 || throw(ArgumentError("NN input dimension must be ≥ 1"))
    preset == :small &&
        return _chain_from_widths([4, 1], [tanh, softplus]; input = input)
    preset == :large &&
        return _chain_from_widths([16, 16, 16, 1], [tanh, tanh, tanh, softplus];
                                  input = input)
    preset == :medium &&
        return _chain_from_widths([8, 8, 1], [tanh, tanh, softplus]; input = input)
    throw(ArgumentError("unknown NN preset $preset; choose :small, :medium, or :large"))
end

"""
    build_ude_nn(rng; n_heads=1, preset=:medium, input_dims=nothing)

Build one or more independent Lux heads. `preset` selects `:small`, `:medium`,
or `:large` architecture templates. `input_dims` is the regulator count per
head (default all 1).
"""
function build_ude_nn(rng::AbstractRNG; n_heads::Int = 1, preset::Symbol = :medium,
                      input_dims = nothing)
    n_heads ≥ 1 || throw(ArgumentError("n_heads must be ≥ 1"))
    dims = input_dims === nothing ? ones(Int, n_heads) : collect(Int, input_dims)
    length(dims) == n_heads ||
        throw(ArgumentError("input_dims length must match n_heads"))
    if n_heads == 1
        model = _single_head_chain(preset; input = dims[1])
        ps, st = Lux.setup(rng, model)
        return model, _float64_param_tree(ps), st
    end
    heads = ntuple(i -> _single_head_chain(preset; input = dims[i]), n_heads)
    ps_pairs = Pair{Symbol,Any}[]
    st_pairs = Pair{Symbol,Any}[]
    for i in 1:n_heads
        ps_i, st_i = Lux.setup(rng, heads[i])
        push!(ps_pairs, Symbol("head_$i") => _float64_param_tree(ps_i))
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
