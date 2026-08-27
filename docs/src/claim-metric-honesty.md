# Claim metric honesty

Helpers on this page are **not exported**. They lock three metric
sentences. They do not raise `RECOVERY_THRESHOLDS.support_f1_ude` and
they do not open Hill-from-NN.

## Combined F1 stays a skeleton floor

`support_f1_ude` stays 0.50; `support_f1_clean` stays 0.99.

The same-library UDE extras probe cannot claim we extracted Hill.
`ude_f1_attempt_live_row` replays the locked surrogate
(`D + const + r`). Extras `1` and `r` remain. Combined F1 does not
reach `support_f1_clean`.

```@example metric-f1
using BioDynaX
row = BioDynaX.ude_f1_attempt_live_row()
(row.extras_one, row.extras_r, row.reaches_clean, row.extracted_hill,
 row.floor, row.clean_gate, row.holds)
```

## Identifiability is Fisher / Jacobian, not structural

unidentifiable_edge is the Fisher/Jacobian cosine or condition-number flag, not StructuralIdentifiability.jl.
coefficients_are_biological_constants is !unidentifiable_edge.

The printed protocol must keep `practical Fisher/Jacobian; not StructuralIdentifiability.jl`.
That is a flag, not a theorem.

```@repl metric-fisher
using BioDynaX
BioDynaX.unidentifiable_edge_from_fisher(;
    condition_number = 1.0e7, collinearity = 0.10)
BioDynaX.unidentifiable_edge_from_fisher(;
    condition_number = 10.0, collinearity = 0.96)
BioDynaX.unidentifiable_edge_from_fisher(;
    condition_number = 10.0, collinearity = 0.50)
BioDynaX.coefficients_are_biological_constants((;
    unidentifiable_edge = true))
```

## MM unknown is not a Hill recall claim

MM unknown gates NN RMSE and hybrid residual; Hill recall 0.99 is not applied.

Hard-job measured values on this budget are recall ~0.5 and F1 ~0.33.
Those numbers are recorded, not painted as Hill 0.99.

```@repl metric-mm
using BioDynaX
g = BioDynaX.mm_unknown_claim_gates()
(g.applies_hill_recall, g.applies_hill_f1_clean, g.nn_rate_rmse,
 g.data_residual, g.measured_recall, g.measured_f1)
BioDynaX.mm_unknown_claim_holds((;
    nn_rate_rmse = 0.04, data_residual = 0.01,
    support_recall = 0.5, support_f1 = 0.33))
```
