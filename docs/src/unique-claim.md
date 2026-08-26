# Unique claim

Research preview. Not v1.0. Not in General.
Unique claim: recall, hybrid residual versus data, unidentifiable_edge.
Combined support F1 is a skeleton floor (0.50), not the UDE claim.
Canonical Hill from a trained neural rate is closed.
Coefficients are not biological constants when the edge is unidentifiable.
BIODYNAX_SMOKE=1 (1 IC / 8 points) is not the seed-103 / 9-IC protocol.

Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

## Product block order

Stdout and `build_protocol_result` use the same order:
**IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION**.

```@repl claim-blocks
using BioDynaX
BioDynaX.UNIQUE_CLAIM_PRODUCT_BLOCKS
BioDynaX.PROTOCOL_RESULT_FIELDS
```

```@example claim-format
using BioDynaX
ident = (; unidentifiable_edge = true, production_param = :k_prod,
         collinearity = 0.997)
text = BioDynaX.format_protocol_result(ident;
    residual = 0.003, support_recall = 1.0, support_f1 = 0.57,
    extras = ("1", "r"), equations = "D(z) = vmax * r^2 / (K^2 + r^2)",
    unknown_holes = 1, seed = 103, n_ics = 9, n_points = 50,
    adam_iters = 100, bfgs_iters = 50, bootstrap = 8,
    discovery_seed = 3, smoke = false)
BioDynaX.protocol_block_order_holds(text)
```

`coefficients_are_biological_constants` is derived from the edge flag.
Missing ident does **not** satisfy the claim.

```@repl claim-ident
using BioDynaX
ident = (; unidentifiable_edge = true, production_param = :k_prod)
BioDynaX.coefficients_are_biological_constants(ident)
BioDynaX.unique_claim_identifiability_holds(ident)
BioDynaX.identifiability_product(ident).practical_not_structural
BioDynaX.unique_claim_identifiability_holds(nothing)
```

## Protocol fingerprint

```@repl claim-proto
using BioDynaX
p = BioDynaX.UNIQUE_CLAIM_PROTOCOL
(p.seed, p.n_ics, p.n_points, p.smoke_n_ics, p.smoke_n_points)
```

```@example claim-ics
using BioDynaX
protocol_ics = BioDynaX.unique_claim_protocol_ics()
smoke_ics = BioDynaX.unique_claim_protocol_ics(; smoke = true)
(length(protocol_ics), length(smoke_ics),
 BioDynaX.unique_claim_is_protocol(),
 BioDynaX.unique_claim_is_protocol(; smoke = true))
```

```@repl claim-repro
using BioDynaX
repro = BioDynaX.unique_claim_reproduction()
(repro.is_protocol, repro.n_ics, repro.n_points, repro.bfgs_iters)
smoke = BioDynaX.unique_claim_reproduction(; smoke = true)
(smoke.is_protocol, smoke.n_ics, smoke.n_points, smoke.bfgs_iters)
```

The golden-path example reads those helpers. A one-IC
`train_ude` snippet in a notebook is not this fingerprint.

## Fit gates (code, not comments)

`unique_claim_kpis_hold` requires the identifiability edge, hybrid
residual versus data, and true-monomial recall. Combined F1 is not a
hold input.

```@repl claim-gates
using BioDynaX
RECOVERY_THRESHOLDS.data_residual
RECOVERY_THRESHOLDS.support_recall
RECOVERY_THRESHOLDS.support_f1_ude
RECOVERY_THRESHOLDS.support_f1_clean
```

```@example claim-kpi
using BioDynaX
hold = BioDynaX.locked_ude_kpis((;
    data_residual = 0.003, support_recall = 1.0,
    identifiability = (; unidentifiable_edge = true)))
miss = BioDynaX.locked_ude_kpis((;
    data_residual = 0.31, support_recall = 1.0,
    identifiability = (; unidentifiable_edge = true)))
(BioDynaX.unique_claim_kpis_hold(hold),
 BioDynaX.unique_claim_kpi_failures(miss))
```

```@repl claim-f1
using BioDynaX
BioDynaX.unique_claim_f1_meets_skeleton_floor(0.57)
BioDynaX.unique_claim_f1_reaches_analytical_gate(0.57)
```

A residual of `0.31` fails the locked gate. That number is
`RECOVERY_THRESHOLDS.data_residual`, not a comment.

## Live extras vs locked F1 attempt

Analytical Hill on the same monomial library can reach combined F1
0.99. A trained-NN-like rate (`D + const + r`) keeps extras `1` and
`r`. That is why Hill-from-NN stays closed.

```@example claim-extras
using BioDynaX
r = collect(range(0.1, 2.0; length = 120))
times = collect(range(0.0, 1.0; length = length(r)))
D = BioDynaX.hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
clean = discover_unknown_rate(
    reshape(r, 1, :), times, reshape(D, 1, :);
    config = BioDynaX.rate_discovery_config(bootstrap = 0, seed = 1),
    verbose = false, strict = true)
dirty = discover_unknown_rate(
    reshape(r, 1, :), times, reshape(D .+ 0.04 .+ 0.04 .* r, 1, :);
    config = BioDynaX.rate_discovery_config(bootstrap = 0, seed = 103),
    verbose = false, strict = false)
(isempty(BioDynaX.unique_claim_discovery_extras(clean)),
 BioDynaX.unique_claim_discovery_extras(dirty))
```

`benchmark/ude_f1_attempt.jl` replays that surrogate. It is not the
9-IC recovery job. `support_f1_ude` stays 0.50.

## Single-hole instrument vs open compile

`validate_network` checks names, bounds, stoichiometry, and metadata.
Zero or two unknown `D(z)` holes still compile. The unique-claim path
calls `assert_single_unknown_destruction`.

```@example claim-holes
using BioDynaX, Random
zero_net = BioDynaX.build_linear_test_network()
one_net = BioDynaX.build_hill_recovery_network(; known = false)
two_net = BioDynaX.build_dual_unknown_network()
rng = MersenneTwister(0)
zero_m, _ = build_ude_model(rng, zero_net)
one_m, _ = build_ude_model(rng, one_net)
two_m, _ = build_ude_model(rng, two_net)
(validate_network(zero_net) === zero_net,
 validate_network(two_net) === two_net,
 BioDynaX.count_unknown_destructions(zero_m),
 BioDynaX.count_unknown_destructions(one_m),
 BioDynaX.count_unknown_destructions(two_m),
 BioDynaX.assert_single_unknown_destruction(one_m))
```

```@repl claim-holes-throw
using BioDynaX, Random
zero_m, _ = build_ude_model(MersenneTwister(0), BioDynaX.build_linear_test_network())
try
    BioDynaX.assert_single_unknown_destruction(zero_m)
catch e
    e isa ErrorException
end
```

## Protocol result object

```@example claim-result
using BioDynaX
ude = (;
    data_residual = 0.003, support_recall = 1.0, support_f1 = 0.57,
    extras = ["1", "r"],
    identifiability = (; unidentifiable_edge = true))
result = BioDynaX.build_protocol_result(ude)
(Tuple(keys(result)) == BioDynaX.PROTOCOL_RESULT_FIELDS,
 result.coefficients_are_biological_constants,
 result.canonical_hill_from_nn,
 result.claim)
```

`run_recovery_suite` attaches that object on UDE and MM-unknown rows.
It does not replace `locked_ude_kpis`.
Unscored extras print NA; empty extras print (none); live extras are not hardcoded.

```@repl claim-extras-print
using BioDynaX
BioDynaX.extras_print_label(nothing)
BioDynaX.extras_print_label(String[])
BioDynaX.extras_print_label(("1", "r"))
```

Stdout field order matches `PROTOCOL_RESULT_FIELDS`. The object stores
`data_residual`; the printer writes `hybrid_data_residual`.

```@example claim-print-order
using BioDynaX
ident = (; unidentifiable_edge = true, production_param = :k_prod)
ude = (;
    data_residual = 0.003, support_recall = 1.0, support_f1 = 0.57,
    extras = ["1", "r"], identifiability = ident)
result = BioDynaX.build_protocol_result(ude)
text = BioDynaX.format_protocol_result(ident;
    residual = result.data_residual, support_recall = result.support_recall,
    support_f1 = result.support_f1, extras = result.extras,
    unknown_holes = result.unknown_holes, seed = 103, n_ics = 9,
    n_points = 50, adam_iters = 100, bfgs_iters = 50, bootstrap = 8,
    discovery_seed = 3, smoke = false)
(BioDynaX.format_protocol_result_field_order_holds(text),
 BioDynaX.protocol_result_field_to_print_label(:data_residual))
```

## Typed smoke vs protocol fingerprint

`unique_claim_reproduction` is a NamedTuple. `UniqueClaimFingerprint` is
the typed object tests can fail independently. Smoke is not protocol.

```@repl claim-fingerprint
using BioDynaX
fp = BioDynaX.unique_claim_fingerprint()
sm = BioDynaX.unique_claim_fingerprint(; smoke = true)
(BioDynaX.unique_claim_fingerprint_is_protocol(fp),
 BioDynaX.unique_claim_fingerprint_is_smoke(sm),
 sm.n_ics, sm.n_points, sm.bfgs_iterations)
```

The golden-path example reads `unique_claim_fingerprint` and passes it
to `format_protocol_result`.

## Recovery-path hole admission

validate_network stays open; unique-claim recovery admits exactly one unknown D(z).
Zero- and two-hole networks still compile.

```@example claim-recovery-admit
using BioDynaX
zero_net = BioDynaX.build_linear_test_network()
one_net = BioDynaX.build_hill_recovery_network(; known = false)
two_net = BioDynaX.build_dual_unknown_network()
(BioDynaX.unique_claim_compiler_stays_open(zero_net),
 BioDynaX.unique_claim_compiler_stays_open(two_net),
 BioDynaX.unique_claim_recovery_admits(zero_net),
 BioDynaX.unique_claim_recovery_admits(one_net),
 BioDynaX.unique_claim_recovery_admits(two_net),
 BioDynaX.assert_unique_claim_recovery_network(one_net) === one_net)
```

`run_recovery_suite` calls `assert_unique_claim_recovery_network` before
the 9-IC UDE train.

## Compiler remapping (not a unique-claim gate)

compile_mechanism reindexes kept NeuralDestructionTerm heads to 1:n.
A skipped duplicate unknown edge no longer leaves a gapped `nn_index`.
`validate_network` does not own that remapping. Multi-regulator `D(S,I)`
compiles; it is not the unique-claim path.

```@example claim-remap
using BioDynaX, Random
skipped = BioDynaX.build_skipped_duplicate_unknown_network()
two_reg = BioDynaX.build_two_regulator_unknown_network()
snap = BioDynaX.compile_unknown_topology(skipped; rng = MersenneTwister(13))
two = BioDynaX.compile_unknown_topology(two_reg; rng = MersenneTwister(13))
(snap.indices, snap.n_heads, snap.rhs.finite, snap.rhs.parity,
 two.arities, two.rhs.cache_matches)
```

## F1 attempt probe

benchmark/ude_f1_attempt.jl is a same-library probe, not the 9-IC protocol. It does not call `unique_claim_protocol_ics` and does not
train a UDE. `support_f1_ude` stays 0.50.

```@repl claim-f1-attempt
using BioDynaX
c = BioDynaX.unique_claim_f1_attempt_contract()
(c.is_protocol, c.trains_ude, c.n_ics, c.new_atoms,
 c.support_f1_ude, c.support_f1_clean)
```

## What this page is not

- A wet-lab protocol for one noisy CSV and unknown topology.
- A license to call coefficients biological constants.
- A promise that combined F1 from a trained NN is canonical Hill.
- A General-registry install story. Clone and `Pkg.instantiate`.
- GPU / SBML / ModelingToolkit productization. Those names stay
  experimental and unexported.

Gates live in `RECOVERY_THRESHOLDS`. Loosening a number is breaking.
See [Recovery benchmarks](benchmarks.md) and [API stability](stability.md).
