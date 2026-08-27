# Hybrid compose path

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

compose_hybrid_rhs with the neural destruction rate recovers ude_system.
hybrid_data_residual versus noise-0 generate_from_compiled_model is a data residual, not an xdot residual.
export_rhs rejects a failed DiscoveryResult; compose_hybrid_rhs is the unknown-edge path.
Remapped multi-head networks compose one NeuralDestructionTerm at a time; only() is not used.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Hill-from-NN stays closed.

## What was still open

`compose_hybrid_rhs` already subtracts the neural destruction and adds
`rate_fn(regulators) * u_target`. `hybrid_data_residual` already
integrates that RHS versus observations. Those two facts were not
locked as a compile-free identity: a `rate_fn` that returns the compiled
neural `D` must recover `ude_system`, and the data residual against
`generate_from_compiled_model` at noise 0 must be ~0.

`export_rhs` already refuses a failed `DiscoveryResult`. That refusal
was not joined to the compose path. A failed discovery must not be
turned into a hybrid RHS.

Remapped multi-head networks have two `NeuralDestructionTerm`s.
`only(terms)` is the unique-claim instrument, not the remapped path.

## Identity rows

`neural_identity_rate` builds the `rate_fn` from
`_destruction_contribution` at the regulator vector.

`neural_identity_rhs_row` compares `ude_system` and
`compose_hybrid_rhs` at two states. `compile_network` stays at zero.

`hybrid_identity_residual_row` compares that hybrid against a
noise-0 generated trajectory. The residual is below `1e-6`. A zero
rate is worse. This is a **data** residual, not an `ẋ` residual.

`hybrid_predict_ude_agreement_row` checks the same residual against
`predict_ude`. At noise 0 the generated trajectory and `predict_ude`
agree.

`hybrid_mask_residual_row` hides one state after the first sample.
An all-false mask returns `Inf`.

The identity matrix covers Hill UDE, MM unknown, two-regulator
`D(S,I)`, six-state, the default p53/Mdm2 example, competitive
unknown, and the three-state unknown fixture.

## Honesty rows

A 0-hole linear network has no `NeuralDestructionTerm`. Known MM and
the repressilator are the same. `validate_network` still returns the
network.

A dual-unknown network has two terms. `only(terms)` throws.
`remapped_compose_row` composes each remapped head separately.

`failed_export_rhs_row` rejects `InsufficientSamples`.
`empty_export_rhs_row` rejects an empty candidate list.

`discover_then_compose_row` samples `D` on a 32-point smoke
trajectory, runs `discover_unknown_rate`, and composes only when
discovery succeeds. Smoke is 1 IC. It is not the seed-103 / 9-IC
protocol.

`hill_known_generate_unknown_identity_row` generates from a known
Hill tree and composes an unknown UDE. The residual is finite. The
truth has 0 holes; the trainer has 1.

`multi_ic_identity_residual_row` repeats the identity residual on
three ICs without compiling.

`sample_unknown_destruction` matches `neural_identity_rate` column
by column.

An irregular `saveat` still has a ~0 identity residual.

`dual_per_term_compose_row` composes each dual-unknown head without
`only()`. `session_predict_hybrid_row` matches a
`TrainingSolveSession` remake against the identity residual.
`normalize_destruction_samples` rescales a sampled `D` to max-abs 1
and does not invent atoms. `equation_to_function` on an explicit
linear candidate stays finite. `compose_hybrid_rhs` itself does not
call `compile_network`.

## Source contract

`compose_hybrid_rhs` calls `ude_system` and
`_destruction_contribution`. It does not call `compile_network`.

`hybrid_data_residual` builds `SciMLBase.ODEProblem` from the
composed RHS and reports RMSE versus data, with an optional mask.

`export_rhs` requires `result.success`.

`residual_shape_guard_row` returns `Inf` when the observation array
or the mask does not match the generated trajectory. That is a data
shape guard, not a painted F1.

`HybridComposeRow` is the typed identity row: name, term count,
residual, compile count. Combined F1 is not a field.

`hybrid_compose_contract` is the landing sentence used by
`docs/src/sciml.md`:

```
compose_hybrid_rhs with the neural destruction rate recovers ude_system.
```

## Unique-claim compose recipe

The golden-path example already does this after discovery. The snippet
below is the same join, not a new claim. It is 1 IC / 8 points when
`BIODYNAX_SMOKE=1` and is not the seed-103 / 9-IC protocol.

```julia
model, params = build_ude_model(rng, network)
trained = train_experiments(params, set, model;
    config = TrainingConfig(adam_iterations = 1, bfgs_iterations = 0))
X = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X)
discovery = discover_unknown_rate(R, times, D; strict = false)
if discovery.success
    rate_fn = equation_to_function(discovery.candidates[1])
    rhs = compose_hybrid_rhs(model, trained.params, term, rate_fn)
    residual = hybrid_data_residual(
        model, trained.params, term, rate_fn, u0, tspan, times, data)
end
```

`hybrid_data_residual` integrates `rhs` with `Tsit5` and reports RMSE
versus `data`. It does not score combined F1. A failed
`DiscoveryResult` must not reach `export_rhs`.

`neural_identity_rate` is the compile-free check that the compose
operator is a replacement, not a second compiled tree. User code
should pass the discovered `rate_fn`, not the identity.

## What this page does not do

It does not drop protocol ICs. It does not open Hill-from-NN. It does
not grow `names(BioDynaX)`. It does not invent a new solver.
