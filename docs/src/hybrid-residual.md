# Hybrid residual versus solver

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

hybrid_data_residual agrees with SciMLBase.solve of compose_hybrid_rhs.
At noise 0 the identity residual agrees with predict_ude versus the same observations.
A failed compose path returns Inf or throws; it does not paint UDE F1 as 0.99.
Smoke residual (1 IC / 8 points) is not the seed-103 / 9-IC protocol residual.

The unique-claim residual is a hybrid residual versus observed data on
the current protocol / training IC. It is produced after successful
recovery/discovery. It is not a held-out predictive generalization
metric and not a fully independent validation layer.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Hill-from-NN stays closed.

## What was still open

`HybridCompose` already locked `compose_hybrid_rhs` identity against
`ude_system`. The residual join was still thin: `hybrid_data_residual`
integrates the composed RHS, but tests did not compare that number to
an explicit `SciMLBase.solve` of the same problem, to `predict_ude`,
or to `SciMLBase.ODEProblem(model, u0, tspan, p)`.

Noise-0 identity residual being ~0 does not make a noisy residual ~0.
Smoke generate (1 IC / 8 points) is not the protocol generate
(9 ICs / 50 points). Neither path trains.

A linear destruction term is not a `NeuralDestructionTerm`.
`compose_hybrid_rhs` must reject it. Dual-unknown networks still
compose per head; `only(terms)` still throws.

## Residual solvers

`hybrid_residual_sciml_solve` builds `SciMLBase.ODEProblem` of
`compose_hybrid_rhs` and integrates with `Tsit5`. That is the same
entry `hybrid_data_residual` uses. No new OrdinaryDiffEq algorithm.

`hybrid_residual_model_solve` integrates
`SciMLBase.ODEProblem(model, u0, tspan, p)`. Identity `rate_fn`
must agree with that solve at noise 0.

`hybrid_residual_predict_ude` is the `predict_ude` residual versus
the same observations.

`residual_solver_agreement_row` compares all three. `compile_network`
stays at zero.

```@example hybrid-residual
using BioDynaX

BioDynaX.hybrid_residual_contract()
```

## Noise honesty

`noise0_vs_noisy_residual_row` generates one compiled trajectory at
`noise_σ > 0`. Identity residual versus the clean array is ~0.
Identity residual versus the noisy array is larger. Combined F1
is not painted as 0.99. The skeleton floor stays 0.50.

`noise_grid_residual_row` repeats that at `σ ∈ {0, 0.01, 0.05, 0.10}`.
The zero-noise residual is ~0. The largest-σ residual is larger.

## Smoke versus protocol

`smoke_vs_protocol_residual_row` generates a known-Hill experiment
set at the smoke fingerprint and at the protocol fingerprint.
Smoke is 1 IC / 8 points. Protocol is 9 ICs / 50 points / seed 103.
The unknown-UDE identity residual against known-Hill observations
is finite. It is not a claim that the untrained NN matches Hill.
Protocol ICs are not dropped to make the residual look cheaper.

`smoke_identity_on_self_row` generates from the unknown UDE itself
at the smoke point count. That identity residual is ~0.

`protocol_fingerprint_not_dropped_row` re-reads
`unique_claim_fingerprint()`: 9 ICs, 50 points, seed 103.

## Failed compose paths

`failed_compose_linear_term_row` passes a `LinearDestructionTerm`.
`compose_hybrid_rhs` throws. `validate_network` still returns the
0-hole network.

`failed_compose_empty_terms_row` is known MM. `only(terms)` throws.

`failed_compose_dual_only_row` composes each remapped / dual head.
`only(terms)` throws. Unique-claim recovery does not admit 2 holes.

`failed_compose_export_row` rejects `InsufficientSamples`.
`hybrid_residual_failed_solve_row` returns Inf-or-large for an
exploding rate. `hybrid_residual_shape_guard_row` returns `Inf`
when the observation width does not match.

`failed_compose_wrong_rate_row` shows a constant rate does not
recover `ude_system`.

## Fixture matrix

The identity matrix covers Hill UDE, MM unknown, two-regulator
`D(S,I)`, six-state, the default p53/Mdm2 example, competitive
unknown, and the three-state unknown fixture.

The honesty matrix covers 0-hole linear / known MM / repressilator /
kinetic networks, remapped and skipped-index heads, three-IC
residuals, `TrainingSolveSession`, known-Hill generate versus
unknown compose, noise, and smoke-versus-protocol.

```@example hybrid-residual-names
using BioDynaX

length(BioDynaX.hybrid_residual_fixture_names())
```

## What this page does not do

It does not loosen `RECOVERY_THRESHOLDS`. It does not put a
single-hole gate into `validate_network`. It does not open
Hill-from-NN. It does not drop protocol ICs. It does not add a
solver, Enzyme, MTK, GPU, or SBML product path. Combined support
F1 stays a skeleton floor (0.50).
