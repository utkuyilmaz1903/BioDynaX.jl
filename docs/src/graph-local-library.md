# Graph-local library and ablation

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

local_basis scope=:graph uses graph parents; scope=:global is the ablation.
local_has_true_parent_gate is the recovered-support membership check.
A wrong-graph parent set does not contain the true regulator.
run_recovery_suite graph-prior sections call local_has_true_parent_gate.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Hill-from-NN stays closed. Discovery rows here are 1 IC
and are not the protocol.

## What was still open

`local_basis` already takes `scope=:graph` or `scope=:global`.
`run_recovery_suite` already reported `local_has_true_parent` as a
NamedTuple field. That field was an inline
`support_uses_variable` predicate, not a reusable gate.

A wrong-graph network (`build_wrong_graph_unknown_network`,
`build_six_state_wrong_graph_network`) claims Q→S while the sampled
rate is still D(R). The graph-local library must not contain R.

## Library rows

`graph_vs_global_library_row` compares term counts and variable
sets. The global library is at least as wide as the graph-local
library. Graph parents appear in the graph-local variable set.

`wrong_graph_parent_row` checks that the true regulator is absent
from both `candidate_parents` and the graph-local library.

`ablation_library_row` is the two-state rate network: graph keeps
`r`, global adds distractor `z`.

```@example graph-local-library
using BioDynaX

BioDynaX.graph_local_library_contract()
```

## Parent gates

`local_has_true_parent_gate(candidate; variable)` is false when the
candidate is `nothing` and otherwise delegates to
`support_uses_variable`. `local_has_false_parent_gate` is the
distractor check.

`run_recovery_suite` sections `:three_state`, `:wrong_graph`,
`:six_state`, and `:six_state_wrong_graph` now assign
`local_has_true_parent` from that gate.

Discovery rows (`ablation_discovery_gate_row`,
`three_state_discovery_gate_row`, `wrong_graph_discovery_gate_row`)
run `discover_equations` on 80-point 1-IC samples. They are not
the seed-103 / 9-IC protocol.

analytic library-membership control uses hill_rate_truth and is not trained-UDE evidence.
trained-UDE graph-local evidence samples D from the captured fit_unknown_destruction return params via sample_unknown_destruction.
PR smoke is not trained-UDE scientific acceptance.

## Fixtures

3-state / 6-state true graphs contain R because those fixtures
declare `EdgeSpec`s. Wrong-graph variants omit R. The Hill / MM
recovery networks and two-regulator `D(S,I)` are reaction-only:
`candidate_parents` is empty even though the unknown reaction has
regulators. Remapped and dual
unknowns still expose a graph library per target; unique-claim
recovery does not admit 2 holes. `validate_network` stays open.

```@example graph-local-library-names
using BioDynaX

length(BioDynaX.graph_local_library_fixture_names())
```

## Suite catalog

`suite_section_library_matrix` builds a graph-versus-global library
row for every `run_recovery_suite` section, including `:ablation`.
`validate_network` stays open on each fixture. Graph-prior sections
are `:three_state`, `:wrong_graph`, `:six_state`, and
`:six_state_wrong_graph`.

`format_suite_library_index` prints section, kind, expected holes,
and term counts. Combined F1 is not a column.

`six_state_per_target_library_row` repeats the comparison on every
dynamic state. `screen_variables_bound_row` checks that derivative
correlation screening bounds the candidate set.
`evaluate_graph_library_finite_row` writes the graph-local
numerator through `evaluate_library`.

`suite_parent_set_catalog` lists every suite section and every
dynamic target. `format_suite_parent_catalog` is the markdown
table. Remapped, dual, and the default p53/Mdm2 example repeat
the per-target comparison. `extra_candidates` may widen a
graph-local library; it does not shrink the graph prior.

## What this page does not do

It does not loosen `RECOVERY_THRESHOLDS`. It does not put a
single-hole gate into `validate_network`. It does not open
Hill-from-NN. Combined support F1 stays a skeleton floor (0.50).
It does not drop protocol ICs to make discovery look cheaper.
`local_has_true_parent_gate` is not exported. `local_basis` and
`candidate_parents` stay on the public freeze list. The unique-claim
fingerprint remains seed 103 / 9 ICs / 50 points. Smoke discovery
rows stay 1 IC and do not replace that protocol.

`suite_parent_set_catalog` covers every compiled suite network. A
missing graph-prior section is a product break, not a skipped test.
