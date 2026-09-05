# Benchmarks

The recovery benchmarks are the evidence behind the claims in this
documentation. They are run by `BioDynaX.run_recovery_suite`, the scripts in
`benchmark/`, and two test entry points, and they are scored against the
thresholds in `RECOVERY_THRESHOLDS`. Loosening a threshold is treated as a
breaking change.

## What is run

The fast checks run in the default test suite (`Pkg.test()`, file
`test/test_recovery.jl`). They use exact or analytically generated
destruction-rate samples and short training runs:

- **Known kinetics**: linear, Michaelis-Menten, Hill, and competitive
  networks with all terms known; the relative error of the physical
  parameters after `train_ude` must stay below the threshold.
- **Analytical Hill recovery**: support F1 and rate error from exact samples
  of a Hill rate; with 0.5% noise the nested subset selection must reach a
  combined support F1 of at least 0.99 on the same library.
- **Graph-local versus global library**: the same noisy samples plus a
  correlated distractor `z`. The check is library membership: `z` must be
  absent from the graph-local library and present in the global one. After
  subset selection both libraries can reach the same F1; the difference is
  the prior, not the score.
- **Three-state and six-state graph priors** with a wrong-graph negative
  control: the graph-local library must contain the true regulator and no
  false parent; a library built from a deliberately wrong graph must miss the
  true regulator.
- **Partial observation**: recovery from subsampled destruction-rate values
  and the residual of the resulting hybrid model. Training on missing states
  is not part of the benchmark.
- **Identifiability interventions**: freezing `k_prod`, normalizing the
  sampled rate, and changing the production rate; the production-rate versus
  destruction-scale collinearity must persist (cosine at least 0.95).
- **Negative control at 5% noise**: analytical recovery must fail the 0.99
  criterion, so that the noise limit stays visible.

The trained-model checks run in `test/run_recovery_hard.jl` (about 40 minutes)
and follow the [reference protocol](concepts.md#The-reference-protocol):

- **Hill unknown term, no noise**: nine experiments generated once, training
  on experiments 1 to 7. The neural rate must match the true rate on the
  discovery grid (`nn_rate_rmse` at most 0.12) before discovery runs; then
  support recall must be at least 0.99, combined F1 at least 0.50, the hybrid
  residual on experiment 1 at most 0.30, and the scale warning must be raised
  with a cosine of at least 0.95. The residual and the neural-rate error on
  experiments 8 and 9 are reported.
- **The same protocol with 2% observation noise.**
- **Michaelis-Menten unknown term**: neural-rate error and residual only. The
  Hill recall criterion is not applied, and canonical Michaelis-Menten
  support from the trained network is not claimed.

The trained-model library comparison (`BioDynaX.evaluate_trained_graph_local`,
full protocol in `test/run_m4_b_protocol.jl`) runs discovery with the
graph-local, global, and wrong-graph libraries on one trained model's sampled
rate.

## Benchmark scripts

| Script | What it does | CI |
|---|---|---|
| `benchmark/recovery_suite.jl` | fast sections, then the trained-model Hill and Michaelis-Menten sections with the printed report | no |
| `benchmark/sindy_baseline.jl` | graph-local versus global library on the same samples; a DataDrivenSparse row when that package is loaded | no |
| `benchmark/recovery_seeds.jl` | analytical recovery on seeds 103, 107, 111, 113, 127; `--ude` runs the trained-model protocol on each seed | no |
| `benchmark/noise_grid.jl` | analytical Hill recovery at 0, 0.5, 2, 5, and 10% rate noise | no |
| `benchmark/functional_identifiability.jl` | the five-restart functional-identifiability diagnostic | no |
| `benchmark/ude_f1_attempt.jl` | replays the trained-model nuisance terms on the same library to document that they persist | no |
| `benchmark/scale_basis.jl` | library size versus node count | no |
| `benchmark/allocation_gate.jl` | allocation count of the in-place right-hand side | yes |
| `benchmark/probe_datadriven.jl` | checks whether DataDrivenSparse resolves in an isolated environment | yes, allowed to fail |

## Representative results

The numbers below were recorded with the 0.9.2 protocol. They are single
runs at fixed seeds, not success rates. A rerun of `test/run_recovery_hard.jl`
for the 0.10.0 release, with freshly resolved dependency versions, passed
every threshold but did not reproduce the recorded values exactly; the rerun
values are listed under each table. Exact values depend on the dependency
versions in the environment; the thresholds are what the package promises.

Trained-model recovery (`test/run_recovery_hard.jl`, training on experiments
1 to 7, held-out 8 and 9):

| Run | `nn_rate_rmse` | `support_recall` | `support_f1` | `data_residual` | `data_residual_train` | `data_residual_holdout` | `d_rmse_holdout` |
|---|---|---|---|---|---|---|---|
| Hill, seed 103, no noise | 0.046 | 1.0 | 0.571 | 0.0042 | 0.0022 | 0.0041 | 0.012 |
| Hill, seed 113, 2% noise | 0.037 | 1.0 | 0.571 | 0.022 | 0.021 | 0.021 | 0.018 |
| Michaelis-Menten, seed 123 | 0.036 | not applied | 0.667 | 0.0054 | 0.0015 | 0.0032 | 0.0053 |

Rerun for 0.10.0 (September 2026): Hill seed 103 gave `nn_rate_rmse` 0.037,
recall 1.0, F1 0.571, `data_residual` 0.0020, train 0.0010, held-out 0.0015,
`d_rmse_holdout` 0.0046; Hill seed 113 with 2% noise gave 0.098, 1.0, 0.571,
0.022, 0.020, 0.020, 0.072; Michaelis-Menten seed 123 gave 0.028, recall not
applied, F1 0.571, 0.0033, 0.0008, 0.0010, 0.0099.

A support F1 of 0.571 means the true Hill monomials were all recovered
(recall 1.0) together with two nuisance terms, a constant and a linear term.
`benchmark/ude_f1_attempt.jl` shows that subset selection and scale
normalization on the same library do not remove them.

Analytical recovery across seeds (`benchmark/recovery_seeds.jl`, 0.5% noise):
all five seeds reach F1 1.00 and recall 1.00 (reproduced for 0.10.0).

Noise grid (`benchmark/noise_grid.jl`, seed 104; reproduced for 0.10.0, where
the 10% row gave F1 0.50 instead of the recorded 0.40):

| rate noise | F1 | recall | passes 0.99 |
|---|---|---|---|
| 0 | 1.00 | 1.00 | yes |
| 0.5% | 1.00 | 1.00 | yes |
| 2% | 1.00 | 1.00 | yes |
| 5% | 0.40 | 0.50 | no |
| 10% | 0.40 | 0.50 | no |

Analytical recovery holds through 2% noise and breaks at 5%. Discovery from
raw trajectories with finite-difference derivatives is therefore only
claimed up to 2% noise.

Graph-local versus global library (`benchmark/sindy_baseline.jl`, seed 104,
two-state fixture with an `r^2`-like distractor; reproduced for 0.10.0):
both libraries reach F1 1.00 with no false parent after subset selection. The
prior shows up as
library membership of the distractor, not as an F1 gap. The DataDrivenSparse
row is unavailable because that package does not resolve against the
ModelingToolkit versions this package allows; `benchmark/probe_datadriven.jl`
reproduces the resolve error.

Six-state graph prior (`BioDynaX.run_recovery_suite` with
`sections = (:six_state, :six_state_wrong_graph)`): the graph-local library
contains the true regulator and no false parent and excludes the distractor;
the global library includes the distractor and admits a false parent; the
wrong-graph library misses the true regulator. Combined F1 is not scored on
this fixture because the target state itself sits in the local library.

## Report fields

| Field | Meaning |
|---|---|
| `rmse`, `rel` | relative error of the physical parameters (known kinetics) |
| `nn_correlation`, `nn_rate_rmse` | trained neural rate versus the true rate on the discovery grid |
| `support_recall` | fraction of the true monomials recovered |
| `support_f1` | F1 of the recovered numerator and denominator support |
| `discovered_rate_rmse` | discovered rate versus the true rate on the grid |
| `data_residual` | hybrid-model residual on training experiment 1 |
| `data_residual_train`, `data_residual_holdout` | mean residuals on experiments 1 to 7 and on 8 and 9 |
| `d_rmse_holdout`, `d_rmse_holdout_domain` | neural-rate error at held-out regulator values and on the training-derived band |
| `local_f1`, `global_f1` | discovery F1 with the graph-local and the global library |
| `local_has_true_parent`, `local_false_parent`, `Z_in_local_library` | library-membership checks |
| `identifiability` | the production/destruction trade-off report |
| `closed_loop_residual` | hybrid residual after discovery from subsampled rate values |

## Running locally

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. test/run_recovery_hard.jl
julia --project=. test/run_m4_b_protocol.jl
julia --project=. benchmark/recovery_suite.jl
julia --project=. benchmark/sindy_baseline.jl
julia --project=. benchmark/recovery_seeds.jl
julia --project=. benchmark/noise_grid.jl
```
