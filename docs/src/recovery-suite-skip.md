# Recovery suite skip

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

A skipped recovery-suite section does not call _train_unknown_edge.
recovery_suite_plan lists which requested sections train a unique-claim UDE and which requested sections are skipped.
with_train_unknown_edge_counter fails the suite if a skipped unique-claim section still trains.
The default suite still runs :ude_discovery and :mm_unknown; skip is opt-in through the sections keyword.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Skip does not make the protocol faster by dropping ICs,
points, or seeds. It avoids work that was not requested.

## What was wasted

`run_recovery_suite` already gates each section with
`if :name in wanted`. A caller that asked only for `:linear` still had
no instrument that proved `:ude_discovery` and `:mm_unknown` did not
enter `_train_unknown_edge` (9 ICs, protocol points, Adam+BFGS).

Admission (`admit_recovery_suite_network`) already rejects 0/2-hole
networks on unique-claim sections without training. Skip is the other
half: a section that is not in `sections` must not train either.

## Plan

`recovery_suite_plan` is the catalog. Unique-claim trainers are
`:ude_discovery` and `:mm_unknown`. Other unique-claim sections
(`:ident_interventions`, `:partial_obs`) may train a short `train_ude`
or `train_experiments` job; they do not call `_train_unknown_edge`.

```@example suite-plan
using BioDynaX
plan = BioDynaX.recovery_suite_plan((:linear, :ablation))
(isempty(plan.train_unknown_edge),
 :ude_discovery in plan.skipped,
 :mm_unknown in plan.skipped,
 :linear in plan.train_ude)
```

The default plan still includes both unique-claim trainers.

```@repl suite-default-plan
using BioDynaX
plan = BioDynaX.recovery_suite_plan()
(:ude_discovery in plan.train_unknown_edge,
 :mm_unknown in plan.train_unknown_edge,
 length(plan.requested) == 7)
```

## Counter

`_train_unknown_edge` increments `TRAIN_UNKNOWN_EDGE_COUNTER` when a
counter is installed. `with_train_unknown_edge_counter` is the lock.

```@example suite-skip-linear
using BioDynaX, Random
report = BioDynaX.skipped_unique_claim_does_not_train(
    (:linear,);
    rng = MersenneTwister(1),
    linear_adam = 1, linear_bfgs = 0)
(report.holds, report.counter, report.keys)
```

`:ablation` is a library fixture and does not compile or train a UDE.

```@repl suite-skip-ablation
using BioDynaX
BioDynaX.recovery_suite_section_spec(:ablation).trains_unknown_edge
```

## Source gate

Every catalog section is behind `if :name in wanted` in
`run_recovery_suite`. `_train_unknown_edge(` appears only in the
`:ude_discovery` and `:mm_unknown` bodies, and the definition notes the
counter.

```@repl suite-gated
using BioDynaX
BioDynaX.recovery_suite_all_sections_gated()
```

```@repl suite-trainers-source
using BioDynaX
BioDynaX.train_unknown_edge_only_in_unique_claim_source()
```

`:ident_interventions` still runs a short `train_ude` with
`frozen_phys = [:k_prod]`. That is not the 9-IC unique-claim trainer.

```@repl suite-ident-source
using BioDynaX
BioDynaX.ident_interventions_does_not_train_unknown_edge_source()
```

## Cost catalog

`recovery_suite_cost_matrix` lists kind, hole policy, and which trainer
each section is allowed to call. Skip does not loosen
`RECOVERY_THRESHOLDS`.

```@repl suite-cost
using BioDynaX
matrix = BioDynaX.recovery_suite_cost_matrix()
(matrix.holds, matrix.trainer_sections)
```

## Report keys

Each skipped section still writes the keys the catalog names. Literature
is a dimensionless Elowitz fixture, not experimental CSV, and is not the
unique-claim protocol.

```@repl suite-literature-keys
using BioDynaX
keys = BioDynaX.recovery_suite_expected_report_keys(:literature)
(:experimental_csv in keys, :unique_claim_protocol in keys)
```

`:six_state` records `Z_in_local_library`. Combined F1 is not the KPI
on that fixture.

```@repl suite-six-keys
using BioDynaX
:Z_in_local_library in BioDynaX.recovery_suite_expected_report_keys(:six_state)
```

## Contract

`recovery_suite_skip_contract_holds()` joins source gates, the counter
note, docs, the export list, and `RECOVERY_THRESHOLDS`. It does not run
the 9-IC protocol.

```@repl suite-skip-contract
using BioDynaX
BioDynaX.recovery_suite_skip_source_holds()
```

## Benchmark scripts already pass `sections`

`benchmark/recovery_suite.jl` runs the fast known-kinetics + ablation
block, then a second call with only `:ude_discovery` and `:mm_unknown`.
`benchmark/sindy_baseline.jl` requests `:ablation` only.
`benchmark/recovery_seeds.jl` requests `:ude_discovery` only.

```@repl suite-benchmark-skip
using BioDynaX
BioDynaX.recovery_suite_benchmark_skip_source_holds()
```

## Catalog table

```@repl suite-markdown
using BioDynaX
BioDynaX.recovery_suite_skip_markdown_holds()
```

| section | kind | policy | holes | `_train_unknown_edge` | default |
|---|---|---|---|---|---|
| `:linear` | known_kinetics | open | 0 | false | true |
| `:mm` | known_kinetics | open | 0 | false | true |
| `:hill` | known_kinetics | open | 0 | false | true |
| `:competitive` | known_kinetics | open | 0 | false | true |
| `:ude_discovery` | unique_claim | exactly_one | 1 | true | true |
| `:mm_unknown` | unique_claim | exactly_one | 1 | true | true |
| `:ablation` | analytical | library_fixture | NA | false | true |
| `:three_state` | graph_prior | open | 1 | false | false |
| `:wrong_graph` | graph_prior | open | 1 | false | false |
| `:six_state` | graph_prior | open | 1 | false | false |
| `:six_state_wrong_graph` | graph_prior | open | 1 | false | false |
| `:identifiability` | identifiability | open | 0 | false | false |
| `:ident_interventions` | unique_claim | exactly_one | 1 | false | false |
| `:partial_obs` | unique_claim | exactly_one | 1 | false | false |
| `:competitive_unknown` | analytical | open | 1 | false | false |
| `:literature` | literature | open | 0 | false | false |

`:ident_interventions` and `:partial_obs` are unique-claim *admission*
sections. They may train a short `train_ude` / `train_experiments` job.
They do not call `_train_unknown_edge`.

## Skip index

`recovery_suite_skip_index` joins kind, hole policy, trainer flags,
source needles, and expected report keys for every catalog section.
Skipping `:linear` changes the shared RNG stream that `:ude_discovery`
consumes when both run in the default order. That is recorded; the
default runner is not rewritten.

```@repl suite-index
using BioDynaX
index = BioDynaX.recovery_suite_skip_index()
(index.holds, index.n, index.trainers)
```

```@repl suite-shared-rng
using BioDynaX
row = BioDynaX.recovery_suite_shared_rng_honesty()
(row.holds, row.skip_linear_changes_ude_rng)
```

## Protocol width stays on the trainer

Skip does not rewrite `:ude_discovery` to smoke. The gated body still
reads `UNIQUE_CLAIM_PROTOCOL.tspan` and `UNIQUE_CLAIM_PROTOCOL.n_points`.

```@repl suite-protocol-width
using BioDynaX
BioDynaX.unique_claim_trainer_keeps_protocol_source()
```

```@repl suite-needles
using BioDynaX
BioDynaX.recovery_suite_needles_matrix().holds
```

## How to request a subset

```julia
using BioDynaX, Random
# Known kinetics only: _train_unknown_edge is not called.
report = run_recovery_suite(MersenneTwister(1);
    sections = (:linear, :mm, :hill, :competitive),
    linear_adam = 1, linear_bfgs = 0,
    mm_adam = 1, mm_bfgs = 0,
    hill_adam = 1, hill_bfgs = 0,
    competitive_adam = 1, competitive_bfgs = 0)
haskey(report, :ude_discovery)  # false
```

The 9-IC unique-claim job is a separate call with
`sections = (:ude_discovery, :mm_unknown)` and the protocol Adam/BFGS
budget. That call is not made faster by skip.

## What this page does not claim

- Coefficients are not biological constants when the edge is
  unidentifiable.
- Combined support F1 is not raised to 0.99.
- Hill-from-NN is not opened.
- The 9-IC / 50-point protocol is not shortened.
- `validate_network` does not gain a single-hole gate.
- The public export list is unchanged.
- Skipping `:ude_discovery` is not a recovery win.
