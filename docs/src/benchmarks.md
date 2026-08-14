# Recovery benchmarks

The scientific wedge is measured by `run_recovery_suite`,
`benchmark/recovery_suite.jl`, `benchmark/sindy_baseline.jl`, and the dedicated
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
- **3-state graph prior** — true parent `R`, no local false parent. Combined
  F1 is not the 1D Hill analytical gate (the target state sits in
  `local_basis`). This is the main graph-prior table, not the 2-state F1
  comparison.
- **Wrong-graph negative control** — graph claims `Q→S` while the sampled
  rate is still `D(R)`. Local discovery must miss the true parent. Correct
  graphs help because wrong graphs fail.
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
  budget (measured recall ≈ 0.5, F1 ≈ 0.33). NN RMSE and data residual remain
  gated; the Hill 0.99 recall number is not reused silently. Scale-normalizing
  sampled `D` is tried on the same library; it does not promote MM to the
  Hill-class claim.

Measured zero-noise Hill UDE (9 ICs, seed 103): NN RMSE ≈ 0.04, recall 1.0,
combined F1 ≈ 0.57 (extras `1` and `r` remain), data residual ≈ 0.003,
`k_prod`↔`D` cosine ≈ 0.997. That F1 is **below** `support_f1_clean`. The
product does not claim canonical Hill from a trained NN. Scale-normalizing
the sampled NN rate is an extra report field (`normalized_support_f1`); it is
not a new dictionary and not a new public claim.

Thresholds are `RECOVERY_THRESHOLDS`. Loosening them is a breaking change.
Tightening F1 toward `support_f1_clean` is the scientific goal.
v1.0 is not cut until this table stays red when the claim fails.

`support_f1_ude = 0.50` is the **UDE skeleton** combined-support floor applied
to a rate sampled from a trained NN. It does **not** mean “print Hill and stop”.
Canonical Hill combined F1 is `support_f1_clean = 0.99` on analytical samples
after Occam. A green recovery job is necessary, not sufficient, for v1.0.

## Frozen graph vs global table (2-state toy)

Producer: `benchmark/sindy_baseline.jl`. Same `y`; the only intended difference
is the prior (`basis_scope=:graph` vs `:global`). DataDrivenSparse is optional
and is **not** a CI dependency.

| prior | F1 | false parent | den violations | rate RMSE | time (s) |
|-------|----|--------------|----------------|-----------|----------|
| BioDynaX graph | 1.00 | no | 0 | 0.007 | 1.03 |
| BioDynaX global | 1.00 | no | 0 | 0.007 | 0.56 |
| DataDrivenSparse global | skipped (extra not loaded) | — | — | — | — |

Frozen from `benchmark/sindy_baseline.jl` on the 0.5% Hill + `r^2`-alias distractor
fixture (seed 104). The locked prior difference is library membership of `z`,
not a F1 gap after Occam. Re-run the script to refresh wall times; changing
the F1/false-parent contract is breaking. When DataDrivenSparse is loaded, freeze
whatever numbers the script prints (including worse); do not treat a skip as a
win.

Ablation noise stays **0.5%** of rate amplitude. Implicit STLSQ is not claimed
safe at 2% raw `D` noise; that limit is documented rather than hidden.

The locked prior difference is **library membership**: distractor `z` is absent
from the graph-local variables and present in the global library. After Occam
prune, global combined F1 can match the local Hill support on this toy; that
does not remove the prior. `local_false_parent` must stay false. Global false
parent is reported, not required, once nested prune can drop a weak alias.

## Frozen 3-state graph-prior table (main prior evidence)

Producer: `run_recovery_suite(...; sections = (:three_state, :wrong_graph))`.
Synthetic Hill `D(R)` with distractors `Q ≈ R²` and `Z ≈ R`. Combined F1 is
**not** the KPI (target state `S` sits in the local library).

| prior | true parent R | local false parent | notes |
|-------|---------------|--------------------|-------|
| graph (R→S) | yes | no | gated in `test/test_recovery.jl` |
| global | reported | reported | extras on distractors allowed |
| wrong graph (Q→S) | no | — | negative control; must miss R |

## Identifiability interventions

Producer: `run_recovery_suite(...; sections = (:ident_interventions,))`.

| intervention | result |
|--------------|--------|
| Normalize analytical `D` (`max\|D\|=1`) | Occam F1 still ≥ 0.99 (same library) |
| Freeze `k_prod` during a short train | `k_prod` unchanged; cosine still ≥ 0.95 |
| Production rate 0.9 vs 1.8 | cosine still ≥ 0.95 |
| `tradeoff_broken` | **false** (locked) |

The ODE `du = k_prod R − D(R) S` has a production–destruction scale invariance
on observed concentrations (`S ≈ k_prod R / D(R)`). Reporting
`unidentifiable_edge` is the product. Claiming the scale is identified is not.

## Running locally

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
julia --project=. benchmark/recovery_suite.jl
julia --project=. benchmark/sindy_baseline.jl
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
| `local_f1` / `global_f1` | graph vs global discovery on the same `y` |
| `local_false_parent` | whether a distractor entered the local support |
| `local_has_true_parent` | whether the true regulator is in the support |
| `local_time` / `global_time` | wall time for the two STLSQ runs |
| `identifiability` | practical `k_prod` vs `D` scale collinearity (not structural) |
| `closed_loop_residual` | hybrid RHS vs data after subsampled-`D` discovery |
| `ude_mask_train_claimed` | whether UDE was trained on missing states (false) |

## Honest limits

- Competitive unknown `D(S,I)` is compiled as a 2-input neural head. Parent
  recovery is gated; a canonical competitive equation is **not** claimed.
- The Elowitz repressilator fixture uses published dimensionless parameters
  on a synthetic ODE. There is no licensed experimental CSV in this repository;
  that absence is explicit.
- If UDE combined F1 cannot hold 0.99, the public claim is recall + data
  residual, not “Hill keşfi”. **This preview takes that path.** Analytical
  Occam still gates F1 ≥ 0.99 on 0.5% Hill samples.
- Unknown-edge closed loop is **Hill-class**. MM unknown is NN RMSE + data
  residual on this budget.
- Partial observation closed-loop **UDE training** on missing states is red /
  not claimed. Discovery from subsampled `D` plus hybrid residual versus data
  is gated.
- DataDrivenSparse is skipped when the extra is not loaded. A skip is not a
  win.
