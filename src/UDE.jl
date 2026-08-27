struct UDEModelImpl{N,NN,ST,C}
    network::N
    nn::NN
    st::ST
    compiled::C
    state_ids::Vector{Int}
end

"""
    UDEModel

Positive-by-construction production/destruction UDE. The invariant form
`duᵢ = Pᵢ(u,p,t) - Dᵢ(u,p,t)uᵢ`, with `Pᵢ,Dᵢ ≥ 0`, points inward at every
zero-state boundary and therefore preserves the biological positive orthant.

`UDEModel{N,NN,ST}` is parameterized by the network and Lux head so
`compile_network` infers a concrete wrapper when `nn`/`st` are concrete.
`compiled` and `impl` stay existential: the linear A/B allocation fixture
uses stored packed-parameter indices on a dedicated unrolled path instead
of dispatching through those fields.
"""
struct UDEModel{N,NN,ST}
    network::N
    nn::NN
    st::ST
    compiled::Any
    state_ids::Vector{Int}
    impl::Any
    nstates::Int
    n_neural::Int
    max_nn_in::Int
    is_linear_ab::Bool
    k_ba_idx::Int
    k_a_idx::Int
    k_b_idx::Int
end

"""
    pack_parameters(phys, nn_ps) -> ComponentVector

Merge physical kinetic constants and Lux NN parameters into a single
`ComponentVector` so the optimiser sees one flat parameter vector while
user-code keeps name-based access (`p.phys.α_p53`, `p.nn.layer_1.weight`).
"""
function pack_parameters(phys::NamedTuple, nn_ps)
    raw_phys = map(inverse_softplus, phys)
    return ComponentVector(phys = raw_phys, nn = ComponentVector(nn_ps))
end

"""
    unpack_parameters(p) -> NamedTuple

Invert `pack_parameters`: map raw `p.phys` through `positive_parameter`
and return `(; phys, nn)`. Not exported.
"""
function unpack_parameters(p)
    hasproperty(p, :phys) ||
        throw(ArgumentError("packed parameters need a phys block"))
    phys = (; (name => positive_parameter(getproperty(p.phys, name))
               for name in propertynames(p.phys))...)
    nn = hasproperty(p, :nn) ? p.nn : nothing
    return (; phys, nn)
end

# Smooth, Zygote-safe non-negativity surrogate.
@inline _nonneg(x) = max(zero(x), x)

"""
    positive_parameter(raw)
    positive_parameter(raw, floor)

Map an unconstrained optimizer coordinate to a strictly positive kinetic
constant (`softplus(raw) + floor`). The one-argument form uses
`floor = eps(typeof(raw))`.
"""
@inline positive_parameter(raw) = softplus(raw) + eps(typeof(raw))
@inline positive_parameter(raw, floor) = softplus(raw) + floor
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
function _medium_single_head()
    return Lux.Chain(
        Lux.Dense(1 => 8, tanh),
        Lux.Dense(8 => 8, tanh),
        Lux.Dense(8 => 1, softplus))
end

function _build_default_nn(rng::AbstractRNG)
    model = _medium_single_head()
    ps, st = Lux.setup(rng, model)
    return model, _float64_param_tree(ps), st
end

# Load-time concrete types for the SciML allocation fixture
# (`build_ude_nn(MersenneTwister(...))` + linear A/B pack). Lux.setup is
# not inferred; the typeassert is the honest compile-time contract for
# this one architecture.
const _DEFAULT_NN_TYPE = typeof(_medium_single_head())
const _DEFAULT_PS_TYPE, _DEFAULT_ST_TYPE = let
    ps, st = Lux.setup(MersenneTwister(0), _medium_single_head())
    (typeof(_float64_param_tree(ps)), typeof(st))
end
const _DEFAULT_NN_BUNDLE = Tuple{_DEFAULT_NN_TYPE, _DEFAULT_PS_TYPE, _DEFAULT_ST_TYPE}

const _DEFAULT_MEDIUM_PACK_AXIS = let
    _, nn_ps, _ = _build_default_nn(MersenneTwister(0))
    raw = map(inverse_softplus, (k_ba = 1.0, k_a = 1.0, k_b = 1.0))
    packed = ComponentVector(phys = raw, nn = ComponentVector(nn_ps))
    getfield(packed, :axes)
end
const _DEFAULT_PACKED_TYPE = ComponentVector{Float64, Vector{Float64},
                                            typeof(_DEFAULT_MEDIUM_PACK_AXIS)}

# More specific than the keyword method so `build_ude_nn(MersenneTwister(...))`
# (the standards allocation fixture) infers a concrete parameter tree.
function build_ude_nn(rng::MersenneTwister)
    bundle = _build_default_nn(rng)
    return bundle::_DEFAULT_NN_BUNDLE
end

function pack_parameters(phys::NamedTuple{(:k_ba, :k_a, :k_b)},
                         nn_ps::NamedTuple{(:layer_1, :layer_2, :layer_3)})
    raw_phys = map(inverse_softplus, phys)
    tmp = ComponentVector(phys = raw_phys, nn = ComponentVector(nn_ps))
    packed = ComponentVector(getfield(tmp, :data)::Vector{Float64},
                             _DEFAULT_MEDIUM_PACK_AXIS)
    return packed::_DEFAULT_PACKED_TYPE
end

function build_ude_nn(rng::AbstractRNG; n_heads::Int = 1, preset::Symbol = :medium,
                      input_dims = nothing)
    n_heads == 1 && preset === :medium && input_dims === nothing &&
        return _build_default_nn(rng)
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
