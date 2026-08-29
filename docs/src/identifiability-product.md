# Identifiability product rows

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

production_destruction_tradeoff joins UniqueClaimProtocolRow through identifiability_product.
coefficients_are_biological_constants is false exactly when unidentifiable_edge is true.
format_protocol_result prints collinearity only when the cosine is finite.
The tradeoff is a practical Fisher/Jacobian cosine, not StructuralIdentifiability.jl.

`unidentifiable_edge` can still be triggered by the practical Fisher
condition-number threshold **or** that cosine threshold. The flag is a
local practical warning, not structural identifiability.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Hill-from-NN stays closed.

## What was still open

`identifiability_product` already derives
`coefficients_are_biological_constants` from `unidentifiable_edge`.
`format_protocol_result` already prints that boolean. The remaining
drift was the live join: a `production_destruction_tradeoff` report
on a compiled fixture was not turned into a `UniqueClaimProtocolRow`
with the same coefficients boolean, and the collinearity line was
not locked (finite cosine prints; NaN stays silent).

Known Hill and known MM have no `NeuralDestructionTerm`. Their
D-scale cosine is NaN. That is not a claim that the physical
parameters are biological constants. The boolean still follows
`unidentifiable_edge` from the Fisher condition number.

Unknown Hill and unknown MM have a neural destruction head. The
D-scale cosine is finite. Smoke print (1 IC / 8 points) is not the
seed-103 / 9-IC protocol print.

## Live tradeoff

`live_production_destruction_tradeoff` runs the existing
`production_destruction_tradeoff` on a short `predict_ude`
trajectory. It does not train. It does not drop protocol ICs.

`join_tradeoff_protocol_row` builds `identifiability_product`, a
synthetic recovery object, and `UniqueClaimProtocolRow`. Combined
F1 is stored and is not a KPI failure symbol.

```@example identifiability-product
using BioDynaX

BioDynaX.identifiability_product_contract()
```

## Coefficients and print

`coefficients_are_biological_constants_row` covers true / false /
missing / `nothing`. Missing ident still returns true so the
coefficients would *look* identified; `unique_claim_identifiability_holds`
is then false.

`format_protocol_collinearity_row` prints `collinearity: 0.997`
when the cosine is finite. A missing field or `NaN` stays silent.
`coefficients_are_biological_constants` stays `!edge`.

`collinearity_warning_row` locks
`format_production_destruction_warning`: a high cosine is a
practical warning, not StructuralIdentifiability.jl.

`smoke_vs_protocol_print_row` prints both fingerprints. Protocol
keeps 9 ICs / 50 points / seed 103. Smoke is 1 IC / 8 points.

## Fixtures

0-hole fixtures (known Hill, known MM, linear, repressilator) have
NaN D-scale collinearity and still join a protocol row.

Unknown fixtures (Hill, MM, two-regulator, six-state, default
example, competitive, three-state) have a finite cosine when a
neural head exists.

Remapped and dual-unknown networks compose one tradeoff per head.
`only(terms)` still throws. Unique-claim recovery does not admit
2 holes. `validate_network` stays open.

`skipped_middle_tradeoff_path` covers remapped 1:n heads.
`kinetic_known_tradeoff_path` is a 0-hole kinetic network.
`condition_threshold_row` checks that a high Fisher condition
keeps `coefficients_are_biological_constants == false`.
`format_matches_joined_protocol_row` runs
`assert_format_matches_protocol_result` on the joined row.

```@example identifiability-product-names
using BioDynaX

length(BioDynaX.identifiability_product_fixture_names())
```

## What this page does not do

It does not change `production_destruction_tradeoff` return keys.
It does not loosen `RECOVERY_THRESHOLDS`. It does not put a
single-hole gate into `validate_network`. It does not open
Hill-from-NN. Combined support F1 stays a skeleton floor (0.50).
