# SciML integration

BioDynaX follows the SciML modeling pattern: compile a `UDEModel`, wrap it in a
`SciMLBase.ODEProblem`, and integrate with any OrdinaryDiffEq solver.

## Basic usage

```@example sciml
using BioDynaX, SciMLBase, OrdinaryDiffEq, Random

rng = MersenneTwister(0)
network = BiologicalNetwork(
    [NodeSpec(name = :A), NodeSpec(name = :B)],
    EdgeSpec[];
    reactions = [
        ReactionSpec(name = :b_drives_a,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :k_ba)),
        ReactionSpec(name = :a_decay,
                     stoichiometry = Dict(1 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_a)),
        ReactionSpec(name = :b_decay,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_b)),
    ])
model, params = build_ude_model(rng, network)
prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), params)
sol = solve(prob, Tsit5(); saveat = 0:0.5:10.0)
length(sol.t)
```

That snippet constructs an ODE. It is not the unique discovery path.

`generate_from_compiled_model` uses the same entry:
`SciMLBase.ODEProblem(model, u0, tspan, p)`. Remapped multi-head unknowns
and two-regulator `D(S,I)` must match that problem, the in-place cache
path, `remake`, and `SciMLBase.solve(model, ...)`. See
[Compiled experiment path](compiled-path.md).

Training reuses one compiled `UDEModel` and one `TrainingSolveSession`
([Training reuse](training-reuse.md)).
A TrainingSolveSession remakes one SciMLBase.ODEProblem across ICs; it does not compile_network per IC.
Remapped multi-head generate and train_experiments share one compiled tree; train_experiments does not compile per IC.
compose_hybrid_rhs with the neural destruction rate recovers ude_system.

## In-place production integration

For allocation-free forward passes, request an in-place problem and pair it
with `ProductionAD()` during training:

```julia
cache = allocate_cache(model, Float64)
prob = ODEProblem(model, u0, tspan, params; inplace = true, cache = cache)
solver_config = default_solver_config(model; ad_policy = ProductionAD())
prediction = predict_ude(params, u0, tspan, times, model;
                         solver_config = solver_config, cache = cache)
```

The SciML solve surface agrees ude_system, ODEFunction, ODEProblem, remake, inplace cache, SciMLBase.solve, and predict_ude.
See [SciML solve surface](sciml-solve-surface.md). Mechanistic models switch from BacksolveAdjoint to InterpolatingAdjoint when n_observations exceeds 64.

## Adjoints

`auto_sensealg(model)` recommends a SciMLSensitivity adjoint. Neural unknowns
currently require `ZygoteVJP`; both `ZygoteAD()` and `ProductionAD()` use
checkpointed `InterpolatingAdjoint` adjoints today.

## Discovery scaling

Blocked STLSQ reuses one grow-only Gram workspace across bootstrap draws.
Streaming library chunks and blocked STLSQ keep large sample counts tractable.
See [Discovery streaming](discovery-streaming.md) for the workspace types,
`_stlsq_blocked!`, and the chunk-size helper.
Raw trajectories can skip the UDE path:

```julia
dX = estimate_derivatives(X, times)
result = discover_equations(X, times, network; derivatives = dX)
rhs = export_rhs(result)
```

`BioDynaX.select_discovery_config` sweeps AIC/BIC thresholds (internal helper).

## Optimization.jl hook

`train_ude` already uses Optimization.jl for the BFGS refinement step. A
one-shot SciML path is also available as an **unexported** hook so you can
swap `LBFGS` / `Adam` without growing the freeze list:

```julia
prob, objective = BioDynaX.build_optimization_problem(
    model, params, data, times, u0, tspan; config = TrainingConfig())
result = BioDynaX.train_via_optimization(
    model, params, data, times, u0, tspan; maxiters = 50)
```

This is a training alternative, not the unique discovery path. Names stay
`BioDynaX.foo` until a 1.0 freeze decision.

```@docs
build_ude_function
SciMLBase.ODEProblem(::UDEModel, ::Any, ::Any, ::Any)
auto_sensealg
default_solver_config
SciMLBase.solve(::UDEModel, ::Any, ::Any, ::Any)
```
