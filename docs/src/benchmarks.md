# Recovery benchmarks

The scientific wedge is measured by `BioDynaX.run_recovery_suite`,
`benchmark/recovery_suite.jl`, `benchmark/sindy_baseline.jl`,
`benchmark/recovery_seeds.jl`, `benchmark/noise_grid.jl`, and the dedicated
CI `recovery` job.

Discovery targets the **unknown destruction rate** `D(z)`, not the full `ẋ`.
Known production and linear decay stay in compiled IR; STLSQ only sees the
neural edge.

## What is gated

Fast job (`test/test_recovery.jl`):

- **Known linear / MM / Hill / competitive** — `phys` RMSE after `train_ude`.
- **Analytical Hill `D(r)`** — support F1 and rate RMSE from exact samples.
- **Occam prune** — 0.5% noisy analytical Hill must reach combined F1 ≥ 0.99
  on the **same** monomial library (no new atoms).
- **Graph vs global ablation** — same noisy `(r, D)` plus a correlated
  distractor; local F1, false-parent flag, denominator violations. Raw rate
  noise is 0.5% of amplitude. After Occam, graph and global F1 can both be
  1.00; the locked prior is **library membership** of `z`.
- **3-state graph prior** — true parent `R`, no local false parent.
- **6-state graph prior** — same protocol on six dynamic states. Combined F1
  is not the KPI (the target state sits in `local_basis`). The locked
  booleans are `local_has_true_parent`, `local_false_parent`, and
  `Z_in_local_library`. This is the main prior evidence beyond the 3-state
  toy.
- **Wrong-graph negative controls** — 3-state and 6-state. Graph claims
  `Q→S` while the sampled rate is still `D(R)`. Local discovery must miss
  the true parent.
- **Partial observation** — subsampled analytical `D` recall, masked linear
  train, and discovery→`compose_hybrid_rhs` residual versus data. UDE
  training on missing states is **not** claimed (`ude_mask_train_claimed =
  false`).
- **Identifiability interventions** — freeze `k_prod`, normalize sampled
  `D`, and a production-rate change. Jacobian collinearity remains;
  `tradeoff_broken = false` is the locked finding.
- **Competitive unknown parents**, **Elowitz synthetic literature fixture**,
  and a practical `k_prod`↔`D` report. Failures are asserted or documented;
  they are not skipped.

Hard job (`test/run_recovery_hard.jl`):

- **UDE → unknown edge (Hill, zero noise)** — NN must match true `D(r)`
  before discovery runs (`nn_rate_rmse ≤ 0.12`). Multi-IC BFGS refines the
  joint loss over all 9 ICs. Then true-monomial **recall**
  0.99, combined F1 at the UDE skeleton gate (`support_f1_ude = 0.50`, not
  the analytical 0.99), discovered-rate RMSE, hybrid residual **versus data**,
  and `unidentifiable_edge == true` with cosine ≥ 0.95.
- **Same protocol with `σ = 0.02`** (`support_f1_noisy = 0.50`).
- **UDE → unknown edge (MM)** — same training/residual protocol. Canonical MM
  support from the trained NN is **honestly below** the Hill recall gate on this
  budget. NN RMSE and data residual remain gated.

Measured zero-noise Hill UDE (9 ICs, seed 103): NN RMSE ≈ 0.04, recall 1.0,
combined F1 ≈ 0.57 (extras `1` and `r` remain), data residual ≈ 0.003,
`k_prod`↔`D` cosine ≈ 0.997. That F1 is **below** `support_f1_clean`.
`benchmark/ude_f1_attempt.jl` replayed those extras on the same library
(Occam + scale-normalization). Combined F1 stayed 0.57. That script is an
F1 **attempt** (`UNIQUE_CLAIM_F1_ATTEMPT`: `is_protocol = false`,
`trains_ude = false`, `n_ics = 0`, no new atoms), not the seed-103 / 9-IC
recovery protocol and not a reason to tighten `support_f1_ude`. The
product does not claim canonical Hill from a trained NN. Live extras on
the recovery row print through `extras_print_label`; the attempt leftover
pair is not hardcoded when discovery did not score.

`run_recovery_suite` admits unique-claim sections
(`:ude_discovery`, `:mm_unknown`, `:ident_interventions`, `:partial_obs`)
through `admit_recovery_suite_network` before the 9-IC train. Other
sections have an explicit open hole policy; `:ablation` is a library
fixture and does not compile. The matrix is
`recovery_suite_admission_matrix` (not exported).

Thresholds are `RECOVERY_THRESHOLDS`. Loosening them is a breaking change.
v1.0 is not cut until this table stays red when the claim fails.

`support_f1_ude = 0.50` is the **UDE skeleton** combined-support floor.
It does **not** mean “print Hill and stop”.

## Frozen multi-seed analytical Occam

Producer: `benchmark/recovery_seeds.jl`. CI stays on seed 104 for Occam and
seed 103 for UDE. This table is the stability report, not a second red gate.

| seed | F1 | recall | gate |
|------|----|--------|------|
| 103 | 1.00 | 1.00 | yes |
| 107 | 1.00 | 1.00 | yes |
| 111 | 1.00 | 1.00 | yes |
| 113 | 1.00 | 1.00 | yes |
| 127 | 1.00 | 1.00 | yes |

Median / min / max F1 = 1.00 / 1.00 / 1.00. Passed 5/5. Multi-seed UDE is
`julia --project=. benchmark/recovery_seeds.jl --ude` and is **not** a CI job.

## Frozen graph vs global table (2-state toy)

Producer: `benchmark/sindy_baseline.jl`. Same `y`; the only intended difference
is the prior (`basis_scope=:graph` vs `:global`). DataDrivenSparse is optional
and is **not** a CI dependency.

Internal ablation (same `y`, only `basis_scope` differs). After Occam, F1
can match; that is **not** a win. The locked prior is library membership of
the distractor `z`.

| prior | F1 | false parent | den violations | rate RMSE | time (s) |
|-------|----|--------------|----------------|-----------|----------|
| BioDynaX graph | 1.00 | no | 0 | 0.007 | 1.03 |
| BioDynaX global | 1.00 | no | 0 | 0.007 | 0.56 |
| DataDrivenSparse global | unavailable | — | — | — | — |

DataDrivenSparse could not be resolved against this preview
(`DataDrivenDiffEq` requires ModelingToolkit versions that conflict with
BioDynaX weakdep compat `ModelingToolkit = "9, 10"` and `SciMLBase = 3`).
That is an install error, not a skip-as-win and not “we beat them”.
Reproduce the probe with `julia --project=. benchmark/probe_datadriven.jl`.
The pin and the conflict live in `benchmark/external_baseline.md`.

Frozen from `benchmark/sindy_baseline.jl` on the 0.5% Hill + `r^2`-alias distractor
fixture (seed 104). The locked prior difference is library membership of `z`,
not a F1 gap after Occam.

## Frozen 3-state graph-prior table

Producer: `run_recovery_suite(...; sections = (:three_state, :wrong_graph))`.
Synthetic Hill `D(R)` with distractors `Q ≈ R²` and `Z ≈ R`. Combined F1 is
**not** the KPI.

| prior | true parent R | local false parent | notes |
|-------|---------------|--------------------|-------|
| graph (R→S) | yes | no | gated in `test/test_recovery.jl` |
| global | reported | reported | extras on distractors allowed |
| wrong graph (Q→S) | no | — | negative control; must miss R |

## Frozen 6-state graph-prior table (main prior evidence)

Producer: `run_recovery_suite(...; sections = (:six_state, :six_state_wrong_graph))`.
Six dynamic states; one unknown Hill edge; known production/decay on the rest.
Distractor `Z` is absent from the graph-local library and present globally.
Combined F1 is **not** the KPI (measured local F1 ≈ 0.40 because the target
state sits in `local_basis`).

| prior | n | `local_has_true_parent` | `local_false_parent` | `Z_in_local_library` | notes |
|-------|---|-------------------------|----------------------|----------------------|-------|
| graph (R→S) | 6 | yes | no | no | gated |
| global | 6 | reported | yes (this run) | yes (`Z_in_global_library`) | library membership is the prior |
| wrong graph (Q→S) | 6 | no | — | — | negative control; must miss R |

If true parent is missed or a local false parent enters, this table is red.
Do not drop the state count or grow the dictionary to paint it green.

## Identifiability interventions

Producer: `run_recovery_suite(...; sections = (:ident_interventions,))`.

| intervention | result |
|--------------|--------|
| Normalize analytical `D` (`max\|D\|=1`) | Occam F1 still ≥ 0.99 (same library) |
| Freeze `k_prod` during a short train | `k_prod` unchanged; cosine still ≥ 0.95 |
| Production rate 0.9 vs 1.8 | cosine still ≥ 0.95 |
| `tradeoff_broken` | **false** (locked) |

Reporting `unidentifiable_edge` is the product. Claiming the scale is
identified is not.

## Frozen analytical noise grid

Producer: `benchmark/noise_grid.jl` (seed 104). `σ` is a fraction of rate
amplitude (same definition as the 0.5% Occam gate). Not a CI job.

| σ | F1 | recall | den | holds 0.99 gate |
|---|----|--------|-----|-----------------|
| 0.000 | 1.00 | 1.00 | 0 | yes |
| 0.005 | 1.00 | 1.00 | 0 | yes |
| 0.020 | 1.00 | 1.00 | 0 | yes |
| 0.050 | 0.40 | 0.50 | 0 | **no** |
| 0.100 | 0.40 | 0.50 | 0 | **no** |

Analytical Occam holds through σ = 0.02 on this grid and **breaks at σ = 0.05**.
Raw-trajectory discovery via `estimate_derivatives` (central differences) is
claimed only for σ ≤ 0.02. The fast suite asserts that σ = 0.05 **fails** the
0.99 Occam gate (negative control). The UDE protocol is measured at σ = 0.00
and σ = 0.02 in the hard job. It is not claimed at σ ≥ 0.05.

## Published / experimental series

No licensed experimental time series in this repository matches the
unique-claim protocol (known graph, at most one unknown destruction edge,
redistributable license, ≤8 states). That absence is the result. It is not
a silent skip.

Elowitz & Leibler (Nature 403:335–338, 2000) is a **synthetic ODE** fixture
using published dimensionless parameters. It is not experimental CSV.
IRMA / Cantone 2009 is a known 5-gene network with **many** unknown edges;
it is outside the unique-claim protocol and is not a CI gate.

## Running locally

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
julia --project=. benchmark/recovery_suite.jl
julia --project=. benchmark/sindy_baseline.jl
julia --project=. benchmark/recovery_seeds.jl
julia --project=. benchmark/noise_grid.jl
julia --project=. benchmark/ude_f1_attempt.jl
```

## Report fields

| Field | Meaning |
|-------|---------|
| `rmse` / `rel` | physical-parameter relative RMSE (known kinetics) |
| `nn_correlation` / `nn_rate_rmse` | trained `D_nn` vs true Hill/MM rate |
| `support_f1` | implicit numerator+denominator support F1 |
| `support_recall` | fraction of true Hill/MM monomials recovered |
| `discovered_rate_rmse` | `D_hat` vs true rate on a grid |
| `data_residual` | hybrid RHS vs observations (not vs UDE `ẋ`) |
| `normalized_support_f1` | same library after `max\|D\|=1` scaling |
| `local_f1` / `global_f1` | graph vs global discovery on the same `y` (not the prior) |
| `local_false_parent` | whether a distractor entered the local support |
| `local_has_true_parent` | whether the true regulator is in the support |
| `Z_in_local_library` | whether distractor `Z` is in the graph-local monomial library |
| `local_time` / `global_time` | wall time for the two STLSQ runs |
| `identifiability` | practical `k_prod` vs `D` scale collinearity (not structural) |
| `closed_loop_residual` | hybrid RHS vs data after subsampled-`D` discovery |
| `ude_mask_train_claimed` | whether UDE was trained on missing states (false) |

## Honest limits

- Competitive unknown `D(S,I)` is compiled as a 2-input neural head. Parent
  recovery is gated; a canonical competitive equation is **not** claimed.
- No licensed experimental CSV matches the unique-claim protocol.
- UDE combined F1 cannot hold 0.99 on the same library. The public claim is
  recall + data residual. Analytical Occam still gates F1 ≥ 0.99 on 0.5%
  Hill samples.
- Unknown-edge closed loop is **Hill-class**. MM unknown is NN RMSE + data
  residual on this budget.
- Partial observation closed-loop **UDE training** on missing states is not
  claimed. Discovery from subsampled `D` plus hybrid residual versus data
  is gated.
- DataDrivenSparse is never a CI dependency. A skip is not a win.
