###############################################################################
# UDE.jl — physics-informed Lux network + Zygote-safe out-of-place UDE RHS.
#
# Two physics constraints baked into the model:
#
#   1. The NN output is a *degradation rate*, which is non-negative by
#      definition.  The final layer therefore uses `softplus` so the rate
#      is guaranteed positive without clipping the gradient at zero.
#
#   2. Biological concentrations cannot be negative.  The RHS computes
#      its derivatives off `max(0, x_i)` "safe" surrogates of the state,
#      so a slightly-negative numerical excursion never feeds back into
#      the NN input or the linear decay terms (which would otherwise
#      blow up via negative * negative cross-products).  The biological
#      constraint itself is enforced **differentiably** by a soft barrier
#      term inside the loss function (see `Training.loss_mse`), not by a
#      hard `isoutofdomain` solver interruption — the latter destroys
#      the Zygote gradient chain through `InterpolatingAdjoint`.
###############################################################################

"""
    build_ude_nn(rng) -> (model, ps, st)

Small MLP that maps Mdm2 concentration → effective p53 degradation rate.

# Architecture
- Hidden: two `tanh` layers (smooth, well-behaved gradients).
- Output: `softplus(x) = log(1 + eˣ)`  — strictly positive, smooth, and
  asymptotically linear, so large gradients in the right tail still
  propagate cleanly through reverse-mode AD.
"""
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
    return ComponentVector(phys = phys, nn = ComponentVector(nn_ps))
end

# Smooth, Zygote-safe non-negativity surrogate.
# `max(0, x)` is differentiable everywhere except x = 0 (subgradient OK),
# and is exactly the operation Zygote / ChainRules support out of the box.
@inline _nonneg(x) = max(zero(x), x)

"""
    ude_system(x, p, t, nn, st) -> Vector

Out-of-place UDE right-hand side.

State:   `x = [p53, Mdm2]`
Params:  `p.phys.{α_p53, β_mdm2, γ_mdm2, signal}`, `p.nn.*`

# Physics-informed safeguards
* States are read through `_nonneg` so negative numerical excursions do
  not contaminate the NN input or the linear decay terms.
* The NN itself is `softplus`-headed, so `deg_rate ≥ 0` always.
* All operations are out-of-place — no buffer mutation, no broadcasting
  into pre-allocated arrays — which keeps the function strictly
  Zygote-differentiable for `InterpolatingAdjoint + ZygoteVJP`.
"""
function ude_system(x, p, t, nn, st)
    p53_safe  = _nonneg(x[1])
    mdm2_safe = _nonneg(x[2])

    α_p53  = p.phys.α_p53
    β_mdm2 = p.phys.β_mdm2
    γ_mdm2 = p.phys.γ_mdm2
    signal = p.phys.signal

    nn_out, _ = nn([mdm2_safe], p.nn, st)
    deg_rate  = nn_out[1]                       # ≥ 0 by construction

    dp53  = α_p53  * signal    - deg_rate * p53_safe
    dmdm2 = β_mdm2 * p53_safe  - γ_mdm2  * mdm2_safe

    return [dp53, dmdm2]
end
