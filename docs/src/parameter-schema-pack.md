# Parameter schema and pack

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

parameter_schema collects CustomKineticMetadata.rate_param so :k_custom is present.
unpack_parameters inverts pack_parameters through positive_parameter.
Remapped multi-head pack/unpack keeps one phys block and one compiled NN tree.
frozen_phys zeros the named raw gradient and restores the packed coordinate.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Hill-from-NN stays closed.

## What was still open

`parameter_schema` walked compiled production and destruction terms
and stopped at `CompetitiveDestructionTerm`. `CustomDestructionTerm`
stores an evaluator, not a `rate_param` symbol, so `:k_custom` was
absent. `build_ude_model` then packed a schema that could not
`predict_ude` the kinetic fixture.

`pack_parameters` already maps positive phys values through
`inverse_softplus`. There was no inverse helper. Remapped
`MultiHeadNetwork` trees (`head_1`, `head_2`) were packed by
`build_ude_model` but not locked against `schema.nn_heads`.

0-hole models still construct a dummy Lux head
(`n_heads = max(n_terms, 1)`). `schema.nn_heads` counts compiled
`NeuralDestructionTerm`s and is 0 on those fixtures. That mismatch
is honest, not a second neural hole.

## Pack and unpack

`unpack_parameters` maps raw `p.phys` through `positive_parameter`
and returns `(; phys, nn)`. `positive_parameter_roundtrip_row`
locks `positive_parameter ∘ inverse_softplus` on a positive grid.
`inverse_softplus(0)` throws `DomainError`.

```@example parameter-schema-pack
using BioDynaX

BioDynaX.parameter_schema_pack_contract()
```

## Remapped heads and frozen_phys

`remapped_pack_unpack_row` packs `remapped_two_regulator_phys_truth`
and recovers the same positive constants. The NN tree has `head_1`
and `head_2`. `frozen_phys_zero_gradient_row` zeros the named raw
gradient. `TrainingConfig` copies `frozen_phys`.

```@example parameter-schema-pack-names
using BioDynaX

length(BioDynaX.parameter_schema_pack_fixture_names())
```

## Fixtures

Unknown Hill / MM / two-regulator / remapped / dual / default
example schemas match compiled neural-head counts. Kinetic
`:k_custom` is present and `predict_ude` stays finite. Omitting
`:k_custom` from `validate_phys_parameters` throws. Smoke (1 IC)
is not the seed-103 / 9-IC protocol.

## What this page does not do

It does not loosen `RECOVERY_THRESHOLDS`. It does not put a
single-hole gate into `validate_network`. It does not open
Hill-from-NN. Combined support F1 stays a skeleton floor (0.50).
It does not export `unpack_parameters`.

`schema_vs_compiled_nn_tree_row` locks `schema.nn_heads` against
compiled `NeuralDestructionTerm` count. Dummy Lux heads on 0-hole
models do not count as neural holes.

`schema_name_catalog_row` lists linear `:k_ba/:k_a/:k_b`, kinetic
`:k_custom`, and known-Hill `:vmax`. `bounded_parameter` stays
inside `(lower, upper)`. `pack_parameters` rejects non-positive
phys with `DomainError`. Suite sections that compile contribute
a schema catalog row; production-free library fixtures are
skipped without painting recovery F1.

