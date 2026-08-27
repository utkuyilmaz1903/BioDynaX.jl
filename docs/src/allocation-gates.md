# Allocation and type-stability gates

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

Allocation gates use measured hot-path byte ceilings that can fail.
STLSQWorkspace reuse must not increment resize_count on a same-shape ensure.
unpack_parameters and parameter_schema keep concrete return types.
Allocation smoke is not the seed-103 / 9-IC protocol.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Hill-from-NN stays closed.

## What was still open

`test/test_quality_gates.jl` already measures `ude_rhs!` on the linear
fixture (512 bytes) and the default example (4096 bytes). The
unexported workspaces added after that file — `STLSQWorkspace`,
`TrainingSolveSession`, hybrid residual helpers, graph-local library
rows, denominator splits, pack/unpack, and the executable docs join —
had no `@allocated` ceiling that a regression could fail.

This page is that extension. It does not restate the A–G markdown
pages. It does not claim zero allocation on `pack_parameters`.

## Measured hot path

`allocation_hot` warms a callable, then records two `@allocated`
counts. Thresholds compare the second (`hot`) count to
`ALLOCATION_GATE_LIMITS`. `pack_parameters_allocation_row` and
`positive_parameter_allocation_row` are the cheapest pack/schema
gates. `positive_parameter` and `inverse_softplus` stay at 0 bytes.

```@example allocation-gates
using BioDynaX

BioDynaX.allocation_gates_contract()
```

`stlsq_workspace_reuse_row` calls `ensure_stlsq_workspace!` twice at
the allocated shape. `resize_count` must not increment. A larger
shape must increment. The same grow-only rule applies to
`ImplicitLibraryWorkspace`, `StreamingImplicitWorkspace`, and
`LibraryChunkWorkspace`.

```@example allocation-gates-names
using BioDynaX

length(BioDynaX.allocation_gates_fixture_names())
```

```@example allocation-gates-reuse
using BioDynaX

BioDynaX.stlsq_workspace_reuse_row().same_shape
```

`discovery_workspace_alloc_report` compares a reused
`_stlsq_blocked!` call to a naive allocate-and-fit. The reused path
must stay cheaper. That is 1 IC fixture work, not the seed-103 /
9-IC protocol.

## Type rows

`schema_type_row` requires `ParameterSchema` and a `NamedTuple`
unpack. `allocate_cache_type_row` requires `UDEModelCache`.
`lock_training_solver` stays a `SolverConfig`. Dummy Lux heads on
0-hole models are not `schema.nn_heads`.

`quality_gates_ude_rhs_live_row` re-runs the existing 512 / 4096
`ude_rhs!` ceilings so a silent raise in `test_quality_gates.jl`
still fails here.

## What this page does not do

It does not loosen `RECOVERY_THRESHOLDS`. It does not put a
single-hole gate into `validate_network`. It does not open
Hill-from-NN. Combined support F1 stays a skeleton floor (0.50).
It does not export `allocation_hot`.

`format_allocation_gates_report` prints the measured limits, the
fixture index, and the workspace catalog. Smoke (1 IC) is not the
seed-103 / 9-IC table.
