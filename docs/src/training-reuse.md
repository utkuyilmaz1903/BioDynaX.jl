# Training reuse: compiled model, warmup state, locked adjoint

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

A TrainingSolveSession remakes one SciMLBase.ODEProblem across ICs; it does not compile_network per IC.
First-IC warmup hands its Optimisers state to train_experiments; Adam momentum is not discarded.
Neural UDE training locks InterpolatingAdjoint with ZygoteVJP; BacksolveAdjoint is not used on a neural hole.
The Augmented-Lagrangian constraint path calls predict_ude with the compiled model.
with_compile_network_counter fails the suite if the training path compiles per IC.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). The protocol is not made faster by dropping ICs, points, or
seeds.

## What was wasted

`predict_ude` compiled a `UDEModel` whenever `model` was omitted:

```julia
resolved = model === nothing ? compile_network(network, nn, st) : model
```

`train_experiments` already passed `model` into `loss_mse`. The
Augmented-Lagrangian constraint loop did not: it called `predict_ude`
with only `nn, st`, so every IC of every outer iteration rebuilt the
compiled tree.

The unique-claim path trained the first IC with `train_ude`, then called
`train_experiments` on the full set and dropped the Adam
`Optimisers` state. Momentum started from zero on the multi-IC stage.

Neither waste is a weaker experiment. Both are rebuilds of objects the
caller already had.

## Resolve the model once

`resolve_training_model` returns the compiled `UDEModel` or compiles
exactly once. `train_experiments(p, set, model)` must then record
**zero** `compile_network` calls across every IC.

```@example training-resolve
using BioDynaX, Random
net = BioDynaX.build_linear_test_network()
model, p0 = build_ude_model(MersenneTwister(13), net)
set = generate_experiment_set(
    MersenneTwister(13); network = net,
    initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
    tspan = (0.0, 0.8), n_points = 6, noise_σ = 0.0,
    truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
report = BioDynaX.train_experiments_compile_report(init, set, model)
(report.holds, report.with_model, report.with_nn_st, report.n_ics)
```

Passing `nn, st` without `model` is allowed and compiles **once**, not
once per IC.

## Remake session

`training_solve_session` stores one `SciMLBase.ODEProblem`.
`predict_ude_session` remakes `u0`, `tspan`, and `p`. That path is the
forward agreement oracle; Zygote training still constructs a problem
from the stored model (no compile) so adjoints stay on the out-of-place
RHS.

```@example training-remake
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
report = BioDynaX.training_session_remake_agreement(
    model, params, [0.22, 0.14]; tspan = (0.0, 1.0), n_points = 8)
(report.holds, report.no_compile, report.matches)
```

Remapped multi-head and two-regulator fixtures must remake without
compile.

```@repl training-remap-session
using BioDynaX, Random
net = BioDynaX.build_remapped_two_regulator_network()
model, p = build_ude_model(MersenneTwister(13), net)
packed = pack_parameters(BioDynaX.remapped_two_regulator_phys_truth(), p.nn)
BioDynaX.training_session_remake_agreement(
    model, packed, BioDynaX.remapped_two_regulator_state();
    tspan = (0.0, 0.5), n_points = 6).holds
```

## Locked adjoint

`recommend_sensealg` already chooses `InterpolatingAdjoint` for neural
holes and `BacksolveAdjoint` only for small mechanistic models.
`lock_training_solver` writes that choice onto `TrainingConfig`. A
neural hole with `BacksolveAdjoint` fails `assert_training_sensealg`.

`ProductionAD()` with `sensealg === nothing` is the in-place forward
pass and is left alone.

```@repl training-sensealg-neural
using BioDynaX, Random
model, _ = build_ude_model(MersenneTwister(11),
    BioDynaX.build_hill_recovery_network(; known = false))
row = BioDynaX.recommend_sensealg_honesty_row(model)
(row.neural, row.zygote_kind, row.holds)
```

```@repl training-sensealg-linear
using BioDynaX, Random
model, _ = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
BioDynaX.recommend_sensealg_honesty_row(model).neural
```

## Warmup state

`warmup_first_experiment` trains IC 1 with `bfgs_iterations = 0` and
returns the Optimisers state. `train_experiments` accepts that state.
The unique-claim trainer path is `_train_unknown_edge` →
`fit_unknown_destruction` → `train_experiments_with_warmup`.
`_train_unknown_edge` is a Recovery.jl compatibility wrapper: it notes
the train counter, calls `generate_recovery_experiments`, then
`fit_unknown_destruction`. Warmup and `lock_training_config` live in
`fit_unknown_destruction`, not in the wrapper body. The unique-claim
path does not discard Adam momentum.

```@example training-warmup
using BioDynaX, Random
net = BioDynaX.build_linear_test_network()
model, p0 = build_ude_model(MersenneTwister(19), net)
set = generate_experiment_set(
    MersenneTwister(19); network = net,
    initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
    tspan = (0.0, 0.6), n_points = 5, noise_σ = 0.0,
    truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
report = BioDynaX.warmup_state_reuse_report(init, set, model; adam_iterations = 2)
(report.holds, report.warmup_has_state, report.sensealg_locked)
```

`train_ude` stores the same state on `TrainingResult.diagnostics.optimizer_state`
so a caller can resume without a checkpoint file.

## Unique-claim compile check

The smoke unique-claim experiment set is compiled-once data. Warmup on
that set must not call `compile_network`.

```@example training-claim-warmup
using BioDynaX
path = BioDynaX.unique_claim_warmup_compile_path(; smoke = true)
(path.holds, path.compiles, path.compiled_once, path.sensealg.neural)
```

## Generate agreement

predict_ude_session must match generate_from_compiled_model at noise 0 without compile_network.
The session uses the locked adjoint; generate uses `Tsit5` at `1e-9`. Forward
trajectories must still agree.

```@example training-generate-match
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
report = BioDynaX.training_session_matches_generate(
    model, params, [0.22, 0.14]; tspan = (0.0, 0.8), n_points = 8)
(report.holds, report.matches_generate, report.no_compile)
```

## Observation-count honesty

The training lock asks recommend_sensealg for 100 observations; a short mechanistic horizon may still recommend BacksolveAdjoint.
Neural holes stay interpolating at both widths. The lock does not drop
protocol ICs or points to pick a cheaper adjoint.

```@repl training-nobs
using BioDynaX, Random
linear, _ = build_ude_model(MersenneTwister(7), BioDynaX.build_linear_test_network())
row = BioDynaX.sensealg_nobs_honesty_row(linear)
(row.small_name, row.large_name, row.holds)
```

## Counter

`with_compile_network_counter` is the lock. A test that trains a
compiled model and sees `counter[] > 0` means someone rebuilt the tree
on the training path.

```@repl training-counter
using BioDynaX, Random
model, params = build_ude_model(MersenneTwister(0), BioDynaX.build_linear_test_network())
session = BioDynaX.training_solve_session(model, [0.2, 0.1], (0.0, 0.5), params)
BioDynaX.session_predicts_without_compile(session, params, [0.2, 0.1], (0.0, 0.5), 0:0.25:0.5)
```

## Contract

`training_reuse_contract_holds()` joins source locks, the AL model pass,
warmup wiring in `fit_unknown_destruction` (reached through the
`_train_unknown_edge` wrapper), docs, the export list, and
`RECOVERY_THRESHOLDS`. It does not train the 9-IC protocol.

```@repl training-reuse-contract
using BioDynaX
BioDynaX.training_reuse_source_holds()
```

## What this page does not claim

- Coefficients are not biological constants when the edge is
  unidentifiable.
- Combined support F1 is not raised to 0.99.
- Hill-from-NN is not opened.
- The 9-IC / 50-point protocol is not shortened.
- `validate_network` does not gain a single-hole gate.
- The public export list is unchanged.
