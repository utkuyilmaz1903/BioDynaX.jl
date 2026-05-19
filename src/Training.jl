###############################################################################
# Training.jl — fully differentiable training pipeline with a soft barrier.
#
# Why we abandoned the `isoutofdomain` / `AutoTsit5(Rosenbrock23())` design:
#   • `isoutofdomain` interrupts the integrator at the step level.  The
#     resulting non-smooth abort destroys the gradient chain through
#     `InterpolatingAdjoint`, manifesting as zero `Δloss` and silent
#     optimiser stalls.
#   • The stiff Rosenbrock half of `AutoTsit5` triggers `linsolve`/Jacobian
#     paths inside SciMLSensitivity that are not friendly to ZygoteVJP on
#     small toy problems like this one.
#
# What we use instead:
#   • Plain `Tsit5()` — explicit RK, first-class adjoint support, no
#     discrete events to confuse Zygote.
#   • A *differentiable soft barrier* on negative predicted states inside
#     the loss:
#
#         L(p) = MSE(pred, data)  +  w · Σ min(0, pred)²
#
#     Adam sees a smooth gradient that pushes the NN away from negative
#     trajectories — no aborts, no NaNs, just gradient.
#
#   • A side-channel `LossDiagnostics` carries the (mse, penalty)
#     breakdown to the callback so the user can see when the barrier
#     dominates the data-fit term.  The side-channel write is wrapped in
#     `Zygote.@ignore` so AD never traces the mutation.
###############################################################################

using OrdinaryDiffEq: ODEProblem, solve, Tsit5
using SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP

# Default weight for the soft barrier.  Strong enough to dominate when a
# trajectory dips negative, but identically zero once the prediction is
# everywhere non-negative — i.e. no bias against the true optimum.
const DEFAULT_PENALTY_WEIGHT = 1e6

"""
    LossDiagnostics

Mutable side-channel struct: a single shared instance is owned by
`train_ude`, written inside `loss_mse` under `Zygote.@ignore`, and read
in the training callback for diagnostic prints.  Never enters the
autodiff graph.
"""
mutable struct LossDiagnostics
    mse::Float64
    penalty::Float64
    total::Float64
    LossDiagnostics() = new(NaN, NaN, NaN)
end

@inline function _record!(d::LossDiagnostics, mse, penalty, total)
    d.mse     = Float64(mse)
    d.penalty = Float64(penalty)
    d.total   = Float64(total)
    return nothing
end

"""
    predict_ude(p, u0, tspan, saveat, nn, st) -> Matrix

Gradient-friendly forward solve of the UDE for parameters `p`.

* Solver  : `Tsit5()` — explicit RK, the canonical default for smooth
            non-stiff problems with SciMLSensitivity adjoints.
* Sensealg: `InterpolatingAdjoint(autojacvec = ZygoteVJP())` — explicit
            so SciMLSensitivity never falls back to a mutating default
            that Zygote can't differentiate.
* No `isoutofdomain` — negative excursions are handled by the
  differentiable barrier inside `loss_mse`, not by hard solver aborts.
"""
function predict_ude(p, u0, tspan, saveat, nn, st)
    f    = (x, p_, t) -> ude_system(x, p_, t, nn, st)
    prob = ODEProblem(f, u0, tspan, p)
    sol  = solve(
        prob, Tsit5();
        saveat   = saveat,
        abstol   = 1e-6,
        reltol   = 1e-6,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
    )
    return Array(sol)
end

"""
    loss_mse(p, data, t_data, u0, tspan, nn, st;
             penalty_weight = DEFAULT_PENALTY_WEIGHT,
             diagnostics    = nothing)
        -> Real

Composite, fully differentiable loss:

    L(p) = mean(abs2, pred - data)  +  penalty_weight · sum(abs2, min(0, pred))

The penalty term is identically zero when every predicted state is
non-negative, and grows quadratically with any negative excursion — so
Adam receives a smooth, well-conditioned gradient pointing back into the
biologically valid region.

If `diagnostics::LossDiagnostics` is provided, the (mse, penalty, total)
breakdown is written to it under `Zygote.@ignore` for the callback to
inspect.  When the ODE solve terminates early (length mismatch), a
finite fallback `1e3` is returned to keep Adam's state numerically sane.
"""
function loss_mse(p, data, t_data, u0, tspan, nn, st;
                  penalty_weight = DEFAULT_PENALTY_WEIGHT,
                  diagnostics::Union{Nothing,LossDiagnostics} = nothing)
    pred = predict_ude(p, u0, tspan, t_data, nn, st)

    if size(pred, 2) != size(data, 2)
        # Solver terminated early; surface a large finite loss.
        return convert(eltype(p), 1e3)
    end

    mse     = mean(abs2, pred .- data)
    penalty = sum(abs2, min.(zero(eltype(pred)), pred)) * penalty_weight
    total   = mse + penalty

    if diagnostics !== nothing
        Zygote.@ignore _record!(diagnostics, mse, penalty, total)
    end

    return total
end

"""
    train_ude(p_init, data, t_data, u0, tspan, nn, st;
              adam_iters     = 300,
              adam_lr        = 0.01,
              bfgs_iters     = 100,
              penalty_weight = DEFAULT_PENALTY_WEIGHT,
              log_every      = 20,
              verbose        = true)
        -> NamedTuple

Two-stage optimisation with a differentiable soft barrier on negative states.

* **Stage 1 — Adam (≥ 300 iters)**: cheaply escapes pathological NN
  initialisations; the soft barrier prevents the early trajectory from
  ever stranding the optimiser in a negative-state regime.
* **Stage 2 — BFGS (100 iters)**: refines to a tight local optimum once
  Adam has driven the trajectory into the valid region.

The training callback periodically prints the (mse, penalty) split of
the most recent loss evaluation and tags iterations where the barrier
dominates — a clear signal that the NN is still proposing
biologically-invalid trajectories and needs more Adam steps before BFGS.

# Returns
NamedTuple with `params`, `history`, `initial_loss`, `final_loss`, and
`last_breakdown = (mse, penalty, total)`.
"""
function train_ude(p_init, data, t_data, u0, tspan, nn, st;
                   adam_iters::Int  = 300,
                   adam_lr::Float64 = 0.01,
                   bfgs_iters::Int  = 100,
                   penalty_weight   = DEFAULT_PENALTY_WEIGHT,
                   log_every::Int   = 20,
                   verbose::Bool    = true)

    diag = LossDiagnostics()
    loss_closure = (p, _) -> loss_mse(p, data, t_data, u0, tspan, nn, st;
                                      penalty_weight = penalty_weight,
                                      diagnostics    = diag)

    initial_loss = loss_closure(p_init, nothing)
    verbose && println("  → initial loss = ", round(initial_loss; digits = 6),
                       "   (mse = ",     round(diag.mse;     digits = 6),
                       ", penalty = ",   round(diag.penalty; digits = 6), ")")

    history = Float64[]
    cb = function (_state, l)
        push!(history, l)
        if verbose && length(history) % log_every == 0
            mse_now = diag.mse
            pen_now = diag.penalty
            tag = pen_now > mse_now ? "  ⚠ penalty dominates" : ""
            println("       iter ", lpad(length(history), 4),
                    " | total = ",   round(l;        digits = 6),
                    " | mse = ",     round(mse_now;  digits = 6),
                    " | penalty = ", round(pen_now;  digits = 6),
                    tag)
        end
        return false
    end

    optf = Optimization.OptimizationFunction(loss_closure,
                                             Optimization.AutoZygote())
    prob1 = Optimization.OptimizationProblem(optf, p_init)

    verbose && println("  → Stage 1: Adam(lr = $adam_lr) × $adam_iters …")
    res1 = Optimization.solve(prob1,
                              OptimizationOptimisers.Adam(adam_lr);
                              callback = cb, maxiters = adam_iters)
    p_trained = res1.u

    if bfgs_iters > 0
        verbose && println("  → Stage 2: BFGS × $bfgs_iters …")
        prob2 = Optimization.OptimizationProblem(optf, res1.u)
        res2 = try
            Optimization.solve(prob2, OptimizationOptimJL.BFGS();
                               callback = cb, maxiters = bfgs_iters)
        catch err
            @warn "BFGS refinement aborted — keeping Adam result." exception = err
            nothing
        end
        if res2 !== nothing
            p_trained = res2.u
        end
    end

    final_loss = loss_closure(p_trained, nothing)
    verbose && println("  → final loss   = ", round(final_loss;   digits = 6),
                       "   (mse = ",     round(diag.mse;     digits = 6),
                       ", penalty = ",   round(diag.penalty; digits = 6), ")")

    return (params         = p_trained,
            history        = history,
            initial_loss   = initial_loss,
            final_loss     = final_loss,
            last_breakdown = (mse     = diag.mse,
                              penalty = diag.penalty,
                              total   = diag.total))
end
