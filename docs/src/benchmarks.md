# Recovery benchmarks

The scientific wedge is measured by `run_recovery_suite` and
`benchmark/recovery_suite.jl`.

## What is gated

- **Known linear kinetics** — relative RMSE of recovered `phys` parameters after
  `train_ude` on noise-free compiled data.
- **Known Michaelis–Menten** — same protocol on `build_mm_test_network()`.
- **Known Hill** — same protocol on `build_hill_recovery_network(; known=true)`.
- **Known competitive inhibition** — `build_competitive_test_network()`.
- **UDE → discovery** — train a neural unknown against Hill data, discover from
  the trained RHS, `export_rhs`, report correlation with the UDE derivatives.
- **Ablation** — `local_basis(...; scope = :graph)` vs `scope = :global` term
  counts on a distractor node. This is the “why not DataDrivenDiffEq alone?”
  check: the graph prior drops false-parent terms.

CI thresholds live in `test/test_recovery.jl`. They are the 0.9 contract; they
must stay green before 1.0.

## Running locally

```bash
julia --project=. benchmark/recovery_suite.jl
```

## Report fields

| Field | Meaning |
|-------|---------|
| `rmse` / `rel` | physical-parameter relative RMSE |
| `correlation` | `cor(export_rhs(u), UDE ẋ)` when discovery succeeds |
| `local_terms` / `global_terms` | library sizes in the ablation |
| `retcode` | `DiscoveryRetcode` for the UDE → discovery section |
