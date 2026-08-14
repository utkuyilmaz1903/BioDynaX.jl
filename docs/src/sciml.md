# SciML integration

BioDynaX follows the SciML modeling pattern: compile a `UDEModel`, wrap it in a
`SciMLBase.ODEProblem`, and integrate with any OrdinaryDiffEq solver.

## Basic usage

```julia
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
```

That snippet constructs an ODE. It is not the unique discovery path.

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

## Adjoints

`auto_sensealg(model)` recommends a SciMLSensitivity adjoint. Neural unknowns
currently require `ZygoteVJP`; both `ZygoteAD()` and `ProductionAD()` use
checkpointed `InterpolatingAdjoint` adjoints today.

## Discovery scaling

Streaming library chunks and blocked STLSQ keep large sample counts tractable.
Raw trajectories can skip the UDE path:

```julia
dX = estimate_derivatives(X, times)
result = discover_equations(X, times, network; derivatives = dX)
rhs = export_rhs(result)
```

`BioDynaX.select_discovery_config` sweeps AIC/BIC thresholds (internal helper).

```@docs
build_ude_function
SciMLBase.ODEProblem(::UDEModel, ::Any, ::Any, ::Any)
auto_sensealg
default_solver_config
SciMLBase.solve(::UDEModel, ::Any, ::Any, ::Any)
```
