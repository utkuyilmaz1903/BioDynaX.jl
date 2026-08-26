# Failure-mode instrument

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

DiscoveryRetcode names InsufficientSamples, DenominatorUnsafe, EmptySupport, SingularLibrary, DiscoveryFailed, and DiscoverySuccess.
validate_network stays a topology checker; 0-hole and 2-hole networks still validate.
KPI failure symbols are unidentifiable_edge, data_residual, and support_recall; combined F1 is never a failure symbol.
extras print NA for missing, (none) for an empty collection, and the live leftovers otherwise.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). A failed discovery row is not painted as UDE F1 0.99.

## What was still open

`DiscoveryRetcode` already exists. `_discovery_retcode` already maps
error classes. Those maps were not locked as a catalog that a later
edit can fail.

`validate_network` already stays open. The 0/2-hole admission helper
already rejects unique-claim recovery. The two facts were not joined
on one fixture matrix that a later edit can fail.

`unique_claim_kpi_failures` already omits combined F1. That omission
was not swept across residual / recall / identifiability /
F1 combinations.

`extras_print_label` already prints `NA` / `(none)` / live leftovers.
Missing versus empty was easy to collapse in a later print helper.

## Retcode rows

`discovery_retcode_catalog` lists the six enum values.

`discovery_retcode_mapper_row` feeds `_discovery_retcode`:

- `ArgumentError` containing `insufficient` → `InsufficientSamples`
- `ArgumentError` containing `empty support` → `EmptySupport`
- `DomainError` → `DenominatorUnsafe`
- `LinearAlgebra.SingularException` → `SingularLibrary`
- any other error → `DiscoveryFailed`

`insufficient_samples_row` calls raw-data `discover_equations` with
12 columns. The result is `InsufficientSamples` when `strict=false`.
`strict=true` still throws.

`insufficient_samples_boundary_row` locks the floor: 19 columns fail,
20 columns are admitted to `_run_discovery`.

`n_samples_entry_throws_row` locks a different entry.
`discover_equations(p, nn, st; n_samples=19)` throws even when
`strict=false`.

`empty_support_row` uses `ExplicitSTLSQ(threshold=1e6)` so every term
is wiped. The result is `EmptySupport`.

`explicit_success_row` recovers a support on the linear fixture. That
is not the unique-claim Hill path.

`implicit_insufficient_row` and `discover_unknown_rate_insufficient_row`
fail before a bootstrap Gram is allocated.

`failed_discovery_result_row` builds a `DiscoveryFailed` payload with
no candidates and no `0.99` F1 string.

## 0/2-hole versus validate_network

`hole_validate_matrix` compiles sixteen fixtures. For each row:

- `validate_network(net) === net`
- `count_unknown_destructions` is the compiled head count
- `unique_claim_recovery_admits` is true only when that count is 1
- `assert_single_unknown_destruction` throws unless the count is 1

0-hole and 2-hole (and 3-hole skipped-middle) networks still validate.
Unique-claim recovery still rejects them. Discovery on a 0-hole or
2-hole network is allowed; it is not unique-claim recovery.

The single-hole instrument is not inside `validate_network`.

## KPI failure grid

`kpi_failure_grid` is 72 synthetic rows: two identifiability values,
three residuals, three recalls, four combined-F1 values including
0.10 and 0.99. Combined F1 never appears in
`unique_claim_kpi_failures`.

A row with residual 0.80 fails `:data_residual` only. A row with
recall 0.40 fails `:support_recall` only. A row with
`unidentifiable_edge=false` fails `:unidentifiable_edge` only.
Painting F1 as 0.99 does not create a pass and does not create a
failure symbol.

Thresholds stay `data_residual = 0.30`, `support_recall = 0.99`,
`support_f1_ude = 0.50`.

## extras catalog

`extras_print_catalog_row` locks:

- `nothing` → `NA`
- empty collection → `(none)`
- `("1", "r")` → `1, r`
- the F1-attempt leftover sentence is detected by
  `extras_print_is_hardcoded_attempt` and is not produced by
  `extras_print_label` from a live pair

Missing and empty stay distinct. A failed protocol print with
`extras=nothing` shows `NA`. A failed two-hole print with
`extras=String[]` shows `(none)`. Neither print contains
`support_f1_ude = 0.99`.

## Source contract

`_run_discovery` keeps `size(X, 2) ≥ 20`.
`_discovery_retcode` keeps the five failure maps.
`_format_protocol_extras` keeps `NA` and `(none)`.

`kpi_threshold_boundary_row` locks the numeric edges:
`data_residual = 0.30` still holds, `0.30 + ε` fails;
`support_recall = 0.99` still holds, `0.99 - ε` fails.

`hill_insufficient_samples_row` and `mm_insufficient_samples_row`
are 0-hole known kinetics. They still hit the 20-sample floor.
`hill_known_explicit_success_row` recovers a support on known Hill.
`select_discovery_all_fail_row` is the configuration sweep that
returns `DiscoveryFailed` when every threshold wipes the library.

The six retcode names stay in `LOCKED_PUBLIC_EXPORTS`. The helper
catalog does not.

`failure_mode_contract` is the landing sentence used by
`docs/src/architecture.md`:

```
validate_network stays a topology checker; 0-hole and 2-hole networks still validate.
```

## What this page does not do

It does not drop protocol ICs. It does not put a single-hole gate
into `validate_network`. It does not grow `names(BioDynaX)`. It does
not invent a new solver or a fabricated experimental CSV.
