# Recovery benchmarks

The scientific wedge is measured by `run_recovery_suite`,
`benchmark/recovery_suite.jl`, and the dedicated CI `recovery` job.

Discovery targets the **unknown destruction rate** `D(z)`, not the full `ẋ`.
Known production and linear decay stay in compiled IR; STLSQ only sees the
neural edge.

## What is gated

Fast job (`test/test_recovery.jl`):

- **Known linear / MM / Hill / competitive** — `phys` RMSE after `train_ude`.
- **Analytical Hill `D(r)`** — support F1 and rate RMSE from exact samples.
- **Graph vs global ablation** — same noisy `(r, D)` plus a correlated
  distractor; local F1, false-parent flag, denominator violations. Raw rate
  noise is 0.5% of amplitude (2% makes implicit STLSQ denominator-unsafe).

Hard job (`test/run_recovery_hard.jl`):

- **UDE → unknown edge (Hill, zero noise)** — NN must match true `D(r)`
  before discovery runs. Then true-monomial recall, support F1, discovered-rate
  RMSE, and hybrid RHS residual **versus data**.
- **Same protocol with `σ = 0.02`**.
- **UDE → unknown edge (MM)**.

Thresholds are `RECOVERY_THRESHOLDS`. Loosening them is a breaking change.
v1.0 is not cut until this table stays red when the claim fails.

## Running locally

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
julia --project=. benchmark/recovery_suite.jl
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
| `local_f1` / `global_f1` | graph vs global discovery on the same `y` |
| `local_false_parent` | whether distractor `z` entered the local support |
| `local_time` / `global_time` | wall time for the two STLSQ runs |
