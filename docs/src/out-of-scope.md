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
- Q4 as a success gate or formal identifiability certificate
- public functional-identifiability API
- trajectory-occupancy discovery as a replacement for Q4 or the M1/Q5 composer
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

Q4 is implemented as a practical functional-identifiability diagnostic,
not a gate. It is not a structural identifiability certificate. Q7 is
reported held-out generalization evidence, not an additional success
gate. It is not a success gate, not a model-selection gate, and not a
mechanism-identifiability certificate.

M4 occupancy is an additional sampling/evaluation context, not a replacement for Q4 or the M1/M2 composer.
`functional_identifiability_domain` remains the approved M3 domain.
Q4 is not occupancy-based. Q4 is not a success gate.
Q4 is not structural identifiability.
Q4 does not use M4 trajectory occupancy.
`_evaluate_unknown_rate_recovery` remains unchanged.
The train-derived `_regulator_grid` remains.
Dummy-time discovery remains.
M4 occupancy must not replace the composer.
`evaluate_holdout` remains four-scalar `HoldoutEvidence`.
The 7/2 train/holdout split remains unchanged.
Holdout is not a 0.30 gate.
Occupancy is not added to `HoldoutEvidence`.

The three seed lists are distinct. Do not substitute one for another.
Do not modify the existing M2/M3 seed constants.

- `UNIQUE_CLAIM_PROTOCOL.seed = 103`
- `FUNCTIONAL_ID_RESTART_SEEDS = (201, 202, 203, 204, 205)`
- `ROBUSTNESS_SEEDS = (103, 107, 111, 113, 127)`

M4 must not alter `RECOVERY_THRESHOLDS`,
`FUNCTIONAL_ID_REPORTING_CUTOFFS`, `LOCKED_PUBLIC_EXPORTS`,
`canonical_hill_from_nn == false`, or `unique_claim_kpis_hold`.

## Next milestones (not implemented)

- M4 — Robustness / Trajectory-Context Validation

M4-A1 occupancy runtime exists. M4-A2 is live separation/contract tests.
M4-B remains pending. M4-C remains pending.

occupancy != Q4 domain.z
occupancy != M1 discovery grid
occupancy != M2 holdout evaluator
Occupancy is not part of the recovery result, holdout result, or Q4 diagnostic.

Current Q1 is a hybrid residual versus observed data on the training
IC[1], plus separate train/holdout aggregates. Current Q2 is a partial
train-grid `D` diagnostic plus reported holdout neural `D` error, not
proof of uniqueness. Current Q3 `unidentifiable_edge` is a local
practical warning (Fisher condition number **or** `k_prod`/`D` scale
cosine), not a structural certificate. Current Q5 support recovery is
performed on grid-sampled learned `D` and is not canonical Hill
recovery. The train-derived external `D` domain is not global OOD.
