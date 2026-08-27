# Denominator and domain safety

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

denominator_split_counts walks train, validation, and the orthant domain grid separately.
UDE extras still call denominator_violation_count on the domain grid.
ExplicitCandidate has no rational denominator; the violation count is 0.
A missing candidate records typemax denominator violations.

This page does not change `RECOVERY_THRESHOLDS`, the unique-claim protocol
(seed 103 / 9 ICs), or `validate_network`. Combined F1 stays a skeleton
floor (0.50). Hill-from-NN stays closed. Discovery rows here are 1 IC
and are not the protocol.

## What was still open

`denominator_violation_count` existed only for `ImplicitCandidate`.
The unique-claim extras path counted violations on the sampled
regulator grid and skipped the train / validation / orthant split
that `_discover_implicit` already uses. Explicit STLSQ has no
denominator; that absence was not a typed 0.

`_denominator_domain_grid` already clips the padded lower bound
into the positive orthant and returns an empty matrix when
`domain_samples = 0`. Those properties were not rows.

## Split counts

`denominator_split_counts` calls `denominator_violation_count` on
train, validation, and domain matrices. The UDE extras helper
`ude_extras_denominator_row` builds that split from one regulator
grid and still runs when extras remain. Live extras print their
monomial labels. Unscored extras print `NA`. An empty collection
prints `(none)`. Hardcoded F1-attempt leftover strings stay
rejected.

```@example denominator-domain
using BioDynaX

BioDynaX.denominator_domain_contract()
```

## Synthetic candidates

`synthetic_safe_implicit_candidate` has an identically-1
denominator. `synthetic_unsafe_implicit_candidate` uses
`1 - 2r` and is singular for `r > 0.5`.
`synthetic_near_zero_implicit_candidate` uses `1 - r`.
`domain_grid_nonneg_row` locks the orthant clip.

```@example denominator-domain-names
using BioDynaX

length(BioDynaX.denominator_domain_fixture_names())
```

## Fixtures

Unknown Hill / MM / two-regulator libraries evaluate a
zero-coefficient denominator (identically 1) on a positive
state grid. Remapped and dual unknowns stay 2-hole and do
not admit unique-claim recovery. `validate_network` stays
open. Combined F1 is not painted as 0.99.

Smoke discovery (1 IC / 80 points on a truth rate) is not
the seed-103 / 9-IC protocol.

## What this page does not do

It does not loosen `RECOVERY_THRESHOLDS`. It does not put a
single-hole gate into `validate_network`. It does not open
Hill-from-NN. Combined support F1 stays a skeleton floor (0.50).
