# Out of scope

This page locks what the unique-claim product does **not** claim.
Helpers are **not exported**. `RECOVERY_THRESHOLDS` and the public
export list are unchanged.

## Known graph, one hole, 2–20 states

Unique-claim requires a known graph and one hole; unknown topology, a single noisy CSV, and a general CRN solver are not claimed.

The measured graph fixtures are 2-state, 3-state, and 6-state known
priors. A 20-state known graph is in scope only as the same one-hole
instrument. Unknown topology is not implemented.

## Synthetic CSV, no licensed series

That committed CSV is a synthetic fixture, not a licensed experimental series.
No licensed experimental time series in this repository matches the unique-claim protocol.
That absence is the result.
Do not invent wet-lab recovery.

## Partial observation

UDE training on missing states is not claimed.
Subsampled analytical `D` can feed `hybrid_data_residual`.
A row with `ude_mask_train_claimed = true` fails closed.

## Graph versus global, not versus DataDrivenSparse

The locked prior is library membership of the distractor z, not a F1 gap after Occam.
3-state and 6-state gates are `local_has_true_parent`,
`local_false_parent`, and (at 6 states) `Z_in_local_library`.
DataDrivenSparse could not be resolved against this preview.
A skip is not a win.

## Multi-seed UDE is a report

The red gate remains single-seed 103/104; recovery_seeds.jl --ude is a report, not a gate.
Do not add N × 40 min recovery jobs to CI.

## Also unsupported in this preview

See [v1.0 scientific contract](design/v1_contract.md). These remain
unsupported:

- structural identifiability certificates
- functional identifiability (Q4 remains not implemented)
- trajectory-occupancy discovery
- arbitrary OOD regimes
- unknown topology discovery
- general CRN solving
- arbitrary multi-hole discovery
- multi-hole mechanisms
- canonical Hill recovery from a trained NN
- biological-constant parameter claims under scale non-identifiability
- general missing-state UDE training
- wet-lab decision making
- general experimental-design engine
- LLM integration
- GPU training stack
- broad SBML kinetic parsing

Q4 (practical functional-identifiability diagnostic) remains not
implemented. Q7 is reported held-out generalization evidence, not an
additional success gate. It is not a success gate, not a
model-selection gate, and not a mechanism-identifiability certificate.

## Next milestones (not implemented)

- M3 — Practical Functional Identifiability (pending / future work)
- M4 — Robustness / Trajectory-Context Validation (pending / future work)

These are future milestones, not present capabilities.

Current Q1 is a hybrid residual versus observed data on the training
IC[1], plus separate train/holdout aggregates. Current Q2 is a partial
train-grid `D` diagnostic plus reported holdout neural `D` error, not
proof of uniqueness. Current Q3 `unidentifiable_edge` is a local
practical warning (Fisher condition number **or** `k_prod`/`D` scale
cosine), not a structural certificate. Current Q5 support recovery is
performed on grid-sampled learned `D` and is not canonical Hill
recovery. The train-derived external `D` domain is not global OOD.
