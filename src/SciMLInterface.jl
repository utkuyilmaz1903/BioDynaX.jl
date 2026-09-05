"""
SciMLBase problem construction for compiled UDE models.

BioDynaX follows the SciML convention: a `UDEModel` is the modeling object,
and `SciMLBase.ODEProblem(model, u0, tspan, p)` is the canonical integration
entry point. Out-of-place dynamics are used for Zygote adjoints; in-place
dynamics with a preallocated cache are used for production forward passes.
"""

struct CompiledOOPRhs{M}
    model::M
end

@inline (f::CompiledOOPRhs)(u, p, t) = ude_system(u, p, t, f.model)

"""
    build_ude_function(model; inplace=false, cache=nothing)

Build an `SciMLBase.ODEFunction` for a compiled `UDEModel`.

- `inplace=false` (default for adjoints): Zygote-safe out-of-place RHS.
- `inplace=true`: allocation-free `ude_rhs!` with a model cache.
"""
function build_ude_function(model::UDEModel, inplace::Bool,
        cache::Union{Nothing, UDEModelCache})
    if inplace
        local_cache = cache === nothing ?
                      allocate_cache(model, Float64) : cache
        return build_ude_rhs(model, local_cache)
    end
    return SciMLBase.ODEFunction{false}(CompiledOOPRhs(model))
end

function build_ude_function(model::UDEModel;
        inplace::Bool = false,
        cache::Union{Nothing, UDEModelCache} = nothing)
    return build_ude_function(model, inplace, cache)
end

"""
    ODEProblem(model::UDEModel, u0, tspan, p; inplace, cache, kwargs...)

Construct a SciML-native `ODEProblem` from a compiled biological UDE.

Use `inplace=true` with `SolverConfig(ad_policy=ProductionAD())` for
allocation-free forward integration. Use the default out-of-place path for
Zygote-based adjoint training.
"""
function SciMLBase.ODEProblem(model::UDEModel, u0, tspan, p;
        inplace::Bool = false,
        cache::Union{Nothing, UDEModelCache} = nothing,
        kwargs...)
    _require_matching_state_length(u0, model.nstates)
    f = build_ude_function(model, inplace, cache)
    return SciMLBase.ODEProblem(f, u0, tspan, p; kwargs...)
end

"""Positional `ODEProblem` used by the default `train_ude` / `predict_ude` path."""
function _odeproblem(model::UDEModel, u0, tspan, p, inplace::Bool)
    _require_matching_state_length(u0, model.nstates)
    f = build_ude_function(model, inplace, nothing)
    return SciMLBase.ODEProblem(f, u0, tspan, p)
end

"""
    recommend_sensealg(model; policy=ZygoteAD(), n_observations=100)

Return a `SensealgRecommendation` describing the chosen adjoint and why.

Mechanistic-only small models prefer `BacksolveAdjoint`; neural unknowns fall
back to checkpointed `InterpolatingAdjoint` with `ZygoteVJP`.
"""
function recommend_sensealg(model::UDEModel;
        policy::AbstractADPolicy = ZygoteAD(),
        n_observations::Int = 100)
    n_observations isa Integer && n_observations ≥ 1 || throw(ArgumentError(
        "n_observations must be an Integer ≥ 1"))
    nn_terms = model.n_neural
    nstates = model.nstates
    if policy isa ProductionAD
        return SensealgRecommendation(
            InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true),
            :interpolating_production,
            "ProductionAD pairs in-place forward with checkpointed InterpolatingAdjoint.")
    end
    if nn_terms == 0 && nstates ≤ 8 && n_observations ≤ 64
        return SensealgRecommendation(
            BacksolveAdjoint(autojacvec = ZygoteVJP()),
            :backsolve_mechanistic,
            "No neural terms and modest state/observation count; BacksolveAdjoint is preferred.")
    end
    if nn_terms > 0
        return SensealgRecommendation(
            InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true),
            :interpolating_neural,
            "Neural destruction terms require reverse-mode VJP adjoints.")
    end
    return SensealgRecommendation(
        InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true),
        :interpolating_default,
        "Default checkpointed InterpolatingAdjoint for general UDE models.")
end

"""Positional `recommend_sensealg` so training locks avoid a keyword sorter."""
function recommend_sensealg(model::UDEModel, policy::AbstractADPolicy,
        n_observations::Integer)
    return recommend_sensealg(
        model; policy = policy, n_observations = Int(n_observations))
end

@inline function _locked_interpolating_adjoint()
    return InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)
end

@inline function _locked_backsolve_adjoint()
    return BacksolveAdjoint(autojacvec = ZygoteVJP())
end

"""Concrete adjoint for the training lock. `n_observations = 100` folds to
`InterpolatingAdjoint` on both neural and mechanistic models."""
function locked_training_sensealg(model::UDEModel, policy::AbstractADPolicy,
        n_observations::Int)
    if policy isa ProductionAD
        return _locked_interpolating_adjoint()
    end
    if model.n_neural == 0 && model.nstates ≤ 8 && n_observations ≤ 64
        return _locked_backsolve_adjoint()
    end
    return _locked_interpolating_adjoint()
end

"""
    auto_sensealg(model; policy=ZygoteAD(), n_observations=100)

Return the recommended SciMLSensitivity adjoint (see `recommend_sensealg`).
"""
function auto_sensealg(model::UDEModel;
        policy::AbstractADPolicy = ZygoteAD(),
        n_observations::Int = 100)
    return recommend_sensealg(
        model; policy = policy, n_observations = n_observations).sensealg
end

"""
    default_solver_config(model; ad_policy=ZygoteAD())

Build a solver configuration with a model-aware adjoint recommendation.
"""
function default_solver_config(model::UDEModel;
        ad_policy::AbstractADPolicy = ZygoteAD(),
        n_observations::Int = 100)
    return SolverConfig(
        ad_policy = ad_policy,
        sensealg = auto_sensealg(
            model; policy = ad_policy, n_observations = n_observations))
end

"""
    solve(model::UDEModel, u0, tspan, p; saveat, solver_config, cache, kwargs...)

Integrate a compiled UDE with SciML defaults.
"""
function SciMLBase.solve(model::UDEModel, u0, tspan, p;
        saveat = nothing,
        solver_config::SolverConfig =
        default_solver_config(model),
        cache::Union{Nothing, UDEModelCache} = nothing,
        kwargs...)
    inplace = solver_config.ad_policy isa ProductionAD &&
              solver_config.sensealg === nothing
    prob = SciMLBase.ODEProblem(
        model, u0, tspan, p; inplace = inplace, cache = cache)
    return SciMLBase.solve(
        prob, solver_config.algorithm;
        saveat = saveat,
        abstol = solver_config.abstol,
        reltol = solver_config.reltol,
        maxiters = solver_config.maxiters,
        sensealg = solver_config.sensealg,
        dense = false,
        save_everystep = false,
        kwargs...)
end
