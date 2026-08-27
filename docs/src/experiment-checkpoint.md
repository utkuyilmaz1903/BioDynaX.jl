# Experiment fingerprint, batch, and checkpoint

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

experiment_fingerprint hashes times, observations, mask, and u0; metadata is not part of the identity.
experiment_batches partitions every IC; shuffle does not drop or duplicate an experiment.
resume_training from a checkpoint reuses the compiled UDEModel and does not call compile_network.
Remapped multi-head generate and train_experiments share one compiled tree; train_experiments does not compile per IC.
generate_experiment_set_from_compiled_model fingerprints without calling compile_network.
train_experiments_with_warmup on a remapped multi-head set does not call compile_network.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Fingerprints and checkpoints do not make the protocol
faster by dropping ICs, points, or seeds. They avoid hashing metadata
as identity and avoid compiling again on resume.

## What was still open

`generate_experiment_set` already compiles one `GroundTruthModel`.
`TrainingSolveSession` already remakes one `ODEProblem` across ICs.
The remaining join was not locked as a single path:

- a metadata-only `ExperimentSet` change must not change
  `experiment_fingerprint`
- `experiment_batches` must cover every IC, including a short last
  batch, and a shuffled pass must stay reproducible
- `resume_training` must reuse the compiled `UDEModel` and the saved
  Optimisers state
- remapped multi-head generate and `train_experiments` /
  `train_experiments_with_warmup` must share that compiled tree

Those holes lived in `src/Experiments.jl` and `src/Training.jl`. This
page is the contract. Helpers live in `src/ExperimentCheckpoint.jl`
and are not exported.

## Fingerprint rows

`experiment_fingerprint_row` hashes the compiled-network identity
plus the IC, the time grid, and the observation mask. Changing
`metadata` does not change the hash. Changing `u0` or the observation
mask does change the hash. The hash is a 64-character lowercase hex
SHA-256 string.

`ExperimentSet.units` and `state_names` are part of the identity. A
unit relabel or a state-name relabel changes the hash.

An irregular but strictly increasing time grid is a different
experiment from the uniform grid with the same observations.

`data_fingerprint_row` hashes a generated trajectory. Two
`generate_from_compiled_model` calls with the same compiled tree and
the same `u0` at noise 0 agree. A different IC does not. The second
generate does not call `compile_network`.

`unique_claim_fingerprint_set_row` compiles the unique-claim
network through `unique_claim_experiment_set` and fingerprints the
smoke set. Smoke is 1 IC / 8 points. It is not the seed-103 / 9-IC
protocol.

`unique_claim_from_compiled_fingerprint_row` compiles the truth
once, then calls `generate_experiment_set_from_compiled_model`. The
compile counter stays at zero while the set is built and hashed.

Two compiled-once sets that differ only in `noise_σ` do not share a
fingerprint. Zero-noise repeats with the same RNG do.

The committed demo table `examples/data/unknown_inhibition.csv` is a
static fixture. Loading it twice yields the same fingerprint. It is
not a fabricated experimental CSV.

## Batch and weight rows

`experiment_batch_row` covers the requested ICs without padding a
short remainder. A shuffled batch with seed `3` is reproducible.
The same seed on a later call returns the same IC order. A different
seed still covers every IC.

`batch_remainder_row` allows the last batch to be shorter than
`batch_size`. The partition still covers every IC exactly once. No
dummy experiment is appended.

`experiment_weight_row` reads `:weight` and `:noise_σ` from
experiment metadata. Defaults stay `1.0`. Tagged values are used by
`experiment_weights` / heteroskedastic scaling. They are not part of
the fingerprint.

## Checkpoint rows

Checkpoints are Julia `serialize` payloads, not JSON. The schema
version is `CHECKPOINT_SCHEMA_VERSION` (`v1.0.0`).
`save_checkpoint` writes a `Checkpoint`. `load_checkpoint` rejects a
major-version mismatch.

`checkpoint_schema_row` locks the version string in `src/Types.jl`
and the resume assignment `optimizer_state = checkpoint.optimizer_state`
in `src/Training.jl`.

`checkpoint_resume_row` writes the checkpoint, remakes
`train_ude` from the saved model, and asserts the compile counter is
still zero. A diagnostics `optimizer_state` path does the same.

`artifact_roundtrip_row` writes a `TrainingResult` through
`save_result` / `load_result`. The retcode and history length survive.

`resume_source_holds` reads `src/Training.jl`. The resume branch
must restore Adam state, iteration, and Augmented-Lagrangian fields
and must not call `compile_network`.

`frozen_phys_checkpoint_row` keeps a frozen physical parameter
through save and resume.

`resume_equivalence_row` compares a two-step run plus a one-step
resume against a three-step fresh run. Both paths stay compile-free.
The row does not claim bit-identical losses.

## Joint generate and train rows

`remapped_generate_train_row` compiles a remapped two-regulator
network, generates two ICs from that compiled tree, and trains with
`train_experiments`. The compile counter stays at zero. Heads stay
dense with arities `[1, 2]`.

`remapped_warmup_generate_train_row` is the same tree through
`train_experiments_with_warmup`. That is the unique-claim trainer
entry used by `_train_unknown_edge`.

The fixture matrix also covers:

- two-regulator unknown edge
- linear unknown edge
- Hill UDE smoke (known generate, unknown train)
- dual-head unknown edge
- six-state unknown edge
- skipped-duplicate unknown edge
- skipped-middle remapped heads
- MM unknown (known generate, unknown train)
- competitive known and unknown
- default p53/Mdm2 example (duplicate unknown declaration)
- known MM and MM test saturation
- repressilator (generate and fingerprint only)
- zero-hole linear (fingerprint only; recovery rejects, `validate_network` stays open)

Each row is a distinct compiled tree. The remapped row is the joint
path that the compiler remap tests previously covered only as a
compile-time unit.

A masked experiment (one state hidden after the first sample) changes
the fingerprint and still trains without `compile_network`.

## Source contract

`fingerprint_source_holds` is `experiment_fingerprint_source_holds`.
The fingerprint helper must hash `times`, `observations`, `mask`, and
`u0`, and must not hash `exp.metadata`.

`batch_source_holds` is `experiment_batches_source_holds`.
`src/Experiments.jl` must shuffle when `shuffle=true`.

`save_checkpoint_source_holds` requires `serialize` and forbids a
JSON payload.

`experiment_checkpoint_contract` is the one-line landing sentence
used by `docs/src/sciml.md`:

```
Remapped multi-head generate and train_experiments share one compiled tree; train_experiments does not compile per IC.
```

## What this page does not do

It does not drop protocol ICs. It does not paint UDE F1 as 0.99. It
does not put a single-hole gate into `validate_network`. It does not
grow `names(BioDynaX)`. It does not invent a new solver, a coverage
badge, or a fabricated experimental CSV.
