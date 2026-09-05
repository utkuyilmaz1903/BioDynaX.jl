# External DataDrivenSparse baseline

The evidence for the graph-local library is the internal comparison in
`benchmark/sindy_baseline.jl`: the same samples are fitted with
`basis_scope = :graph` and `basis_scope = :global`, and the difference is
library membership of the distractor, not an F1 gap after subset selection.
A row from DataDrivenSparse is optional context, not part of that evidence.

## Why the DataDrivenSparse row is unavailable

The package allows `ModelingToolkit = "9, 10"` and `SciMLBase = "3"`.
`DataDrivenDiffEq`, which `DataDrivenSparse` depends on, has required
ModelingToolkit versions that do not resolve against that set. The row is
therefore missing because of a dependency conflict, not because of a
benchmark result.

## Reproduce

```bash
julia --project=. benchmark/probe_datadriven.jl   # isolated resolve check, no BioDynaX
julia --project=. benchmark/sindy_baseline.jl     # the internal comparison
```

The probe prints `RESOLVED` or `UNAVAILABLE` with the resolver error. When
DataDrivenSparse is loaded in the session, `sindy_baseline.jl` adds its row.
