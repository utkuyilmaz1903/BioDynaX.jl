# External DataDrivenSparse baseline

This is **not** a skip-as-win. BioDynaX's locked graph-prior evidence is the
internal ablation (`basis_scope=:graph` vs `:global` on the same `y`). An
external DataDrivenSparse row is optional context.

## Why the in-tree row is `unavailable`

BioDynaX weakdeps pin `ModelingToolkit = "9, 10"` and `SciMLBase = "3"`.
`DataDrivenDiffEq` (pulled by `DataDrivenSparse`) historically required
ModelingToolkit versions that do not resolve against that set. The conflict
is an install error in the **package** environment, not a scientific skip.

## Reproduce (isolated temp env, no BioDynaX)

```bash
julia --project=. benchmark/probe_datadriven.jl
```

The probe never loads BioDynaX. A resolve failure prints `UNAVAILABLE` and
the error. A resolve success prints `RESOLVED` and does not claim that
BioDynaX beat DataDrivenSparse.

## Internal ablation (the actual control)

```bash
julia --project=. benchmark/sindy_baseline.jl
```

Same `(r, D)` plus distractor `z`. After Occam, graph and global F1 can both
be 1.00. The locked prior is **library membership** of `z`.
