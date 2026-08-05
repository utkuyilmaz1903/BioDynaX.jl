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

function build_ude_nn(rng::AbstractRNG)
    model = Lux.Chain(
        Lux.Dense(1 => 8, tanh),
        Lux.Dense(8 => 8, tanh),
        Lux.Dense(8 => 1, softplus),
    )
    ps, st = Lux.setup(rng, model)
    return model, ps, st
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
# `max(0, x)` is differentiable everywhere except x = 0 (subgradient OK),
# and is exactly the operation Zygote / ChainRules support out of the box.
@inline _nonneg(x) = max(zero(x), x)
@inline positive_parameter(raw; floor = eps(typeof(raw))) =
    softplus(raw) + floor
@inline inverse_softplus(value) =
    value > zero(value) ? value + log(-expm1(-value)) :
    throw(DomainError(value, "positive parameters must be > 0"))
@inline bounded_parameter(raw, lower, upper) =
    lower + (upper - lower) * sigmoid(raw)
