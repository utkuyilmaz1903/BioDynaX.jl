# SciML solve surface

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

The SciML solve surface agrees ude_system, ODEFunction, ODEProblem, remake, inplace cache, SciMLBase.solve, and predict_ude.
Mechanistic models switch from BacksolveAdjoint to InterpolatingAdjoint when n_observations exceeds 64.
An in-place ODEProblem reuses one UDEModelCache across remakes; allocate_cache is not called per IC.
This surface does not add an OrdinaryDiffEq algorithm; SolverConfig.algorithm stays Tsit5.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). The protocol is not made faster by dropping ICs, points, or
seeds.

## What this surface is

`generate_from_compiled_model` already uses
`SciMLBase.ODEProblem(model, u0, tspan, p)`. Training remakes one
`TrainingSolveSession`. Those two paths still need a third check: the
RHS, the in-place cache, and the adjoint recommendation must agree on
the same fixtures, including remapped multi-head and two-regulator
`D(S,I)`.

No new solver is introduced. `Tsit5` remains the algorithm on
`SolverConfig`.

## ODEFunction equals ude_system

```@example solve-odefunction
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
row = BioDynaX.odefunction_rhs_row(model, params, [0.22, 0.14])
(row.holds, row.matches_function, row.matches_inplace, row.no_compile)
```

`SciMLBase.isinplace` is false on the out-of-place function and true on
the cache path.

## Remake of p, u0, tspan

```@example solve-remake-fields
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
row = BioDynaX.remake_field_row(model, params, [0.22, 0.14])
(row.holds, row.matches_p, row.matches_u0, row.matches_tspan, row.no_compile)
```

A remade problem must match a freshly constructed `ODEProblem` with that
field. `compile_network` is not part of remake.

## Cache reuse

```@example solve-cache
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
row = BioDynaX.cache_reuse_row(model, params, [0.22, 0.14])
(row.holds, row.same_du, row.heads, row.no_compile)
```

Remapped two-regulator models keep the same cache pointer across ICs.

```@repl solve-cache-remap
using BioDynaX, Random
net = BioDynaX.build_remapped_two_regulator_network()
model, p = build_ude_model(MersenneTwister(13), net)
packed = pack_parameters(BioDynaX.remapped_two_regulator_phys_truth(), p.nn)
BioDynaX.cache_reuse_row(
    model, packed, BioDynaX.remapped_two_regulator_state()).holds
```

## Observation-count boundary

`recommend_sensealg` prefers `BacksolveAdjoint` only when there are no
neural terms, `nstates ≤ 8`, and `n_observations ≤ 64`. One extra
observation flips a mechanistic model to `InterpolatingAdjoint`. Neural
holes stay interpolating at every width. `ProductionAD` is interpolating
at every width.

```@example solve-boundary
using BioDynaX, Random
linear, _ = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
grid = BioDynaX.sensealg_boundary_grid(linear)
(grid.holds, grid.n64.zygote_name, grid.n65.zygote_name, grid.crossing)
```

```@repl solve-boundary-neural
using BioDynaX, Random
model, _ = build_ude_model(MersenneTwister(11),
    BioDynaX.build_hill_recovery_network(; known = false))
grid = BioDynaX.sensealg_boundary_grid(model)
(grid.neural, grid.n20.zygote_name, grid.n100.zygote_name, grid.crossing)
```

Rationale strings on `SensealgRecommendation` are locked:

```@repl solve-rationale
using BioDynaX, Random
linear, _ = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
BioDynaX.recommend_sensealg_rationale_row(linear; n_observations = 20).name
```

Known Michaelis–Menten and the dimensionless repressilator fixture stay
on the mechanistic side of that boundary. They are not unique-claim
jobs.

The training lock still asks for 100 observations (see
[Training reuse](training-reuse.md)). That is an honesty fact, not a
reason to drop protocol points.

## Multi-IC remake

One template `ODEProblem` remade per IC must match a fresh problem per
IC and must not compile.

```@example solve-multi-ic
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
row = BioDynaX.multi_ic_remake_row(
    model, params, [[0.22, 0.14], [0.30, 0.18]])
(row.holds, row.n_ics, row.no_compile)
```

## ProductionAD in-place

`ProductionAD()` with `sensealg === nothing` uses the in-place RHS.
That trajectory must match the Zygote out-of-place solve at the same
tolerances.

```@example solve-production
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
row = BioDynaX.production_inplace_agreement(model, params, [0.22, 0.14])
(row.holds, row.matches, row.inplace, row.no_compile)
```

## Irregular saveat

`predict_ude` and `SciMLBase.solve` must honour a non-uniform `saveat`.

```@repl solve-irregular
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
BioDynaX.irregular_saveat_row(model, params, [0.22, 0.14]).holds
```

## Invalid solves

NaN initial conditions must throw. A `maxiters = 1` solve records the
SciML retcode; it does not invent a trajectory.

```@repl solve-failed
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
BioDynaX.failed_solve_row(model, params, [0.22, 0.14]).nan_threw
```

## Contract

`sciml_solve_surface_contract_holds()` joins source locks, the 64/65
boundary in `recommend_sensealg`, docs, the export list, and
`RECOVERY_THRESHOLDS`. It does not train the 9-IC protocol.

```@repl solve-surface-contract
using BioDynaX
BioDynaX.sciml_solve_surface_source_holds()
```

## What this page does not claim

- Coefficients are not biological constants when the edge is
  unidentifiable.
- Combined support F1 is not raised to 0.99.
- Hill-from-NN is not opened.
- The 9-IC / 50-point protocol is not shortened.
- `validate_network` does not gain a single-hole gate.
- The public export list is unchanged.
- No new OrdinaryDiffEq algorithm is added.
