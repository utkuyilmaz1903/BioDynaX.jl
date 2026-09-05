# Unique claim

Research preview. Not v1.0. Not in General.
Unique claim: recall, hybrid residual versus data, unidentifiable_edge.
Combined support F1 is a skeleton floor (0.50), not the UDE claim.
Canonical Hill from a trained neural rate is closed.
Coefficients are not biological constants when the edge is unidentifiable.
BIODYNAX_SMOKE=1 (1 IC / 8 points) is not the seed-103 / 9-IC protocol.

Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

Compiled dynamics are
\(\dot u_i = P_i - D_i u_i\). Current gates map to Q1 (training IC[1]
residual), Q3 (practical scale warning), and Q5 (true-monomial recall).
Q4 is implemented as a practical functional-identifiability diagnostic,
not a gate and not a formal identifiability certificate. Q7 is reported
held-out generalization evidence, not an additional success gate.
Trajectory fit is not proof of mechanism recovery.

Nine experiments are generated once. Training uses ICs 1..7. Holdout
uses ICs 8 and 9. Holdout is observational evidence. It does not gate
0.30. Discovery failure does not erase Q7. Holdout \(D\) metrics come
from the actual neural \(D\), not symbolic reconstruction. Legacy
`data_residual` remains the IC[1] residual.

Current Q1 evidence is the hybrid residual against observed data on the
current protocol, conditionally produced after successful
recovery/discovery, using the reference/training IC. Separate
arithmetic-mean train and holdout residuals are reported beside that
legacy number. The gated Q1 number is still IC[1].

Current Q2 is a partial mechanism-function diagnostic on the
train-derived regulator domain, plus reported holdout neural \(D\)
error. That holdout \(D\) error is evidence, not a uniqueness proof,
and is not a hold input.

Q3 `unidentifiable_edge` can be triggered by the practical condition-number
threshold or the `k_prod`/`D` scale cosine threshold. It is a local
practical warning, not structural identifiability, and not the whole
product.

Current Q5 symbolic support is recovered from grid-sampled learned `D`.
That is not equivalent to canonical Hill recovery.

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
The same-library UDE extras probe cannot claim we extracted Hill.

unidentifiable_edge is the Fisher/Jacobian cosine or condition-number flag, not StructuralIdentifiability.jl.
coefficients_are_biological_constants is `!unidentifiable_edge`.

MM unknown gates NN RMSE and hybrid residual; Hill recall 0.99 is not applied.

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
unique-claim training (nine ICs are generated once; ICs 1–7 are used
for training and ICs 8–9 are held out).

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

## Compiled data generation (joint with remapping)

generate_data uses the compiled NN tree; remapped multi-head and two-regulator D(S,I) are generated together.
generate_experiment_set compiles the ground-truth model once and generates every IC from that stored model.
generate_from_compiled_model integrates SciMLBase.ODEProblem(model, u0, tspan, p).
The joint compiled path is generate_from_compiled_model + remapped heads + admit_recovery_suite_network + UniqueClaimProtocolRow.
`generate_data(::GroundTruthModel)` integrates the stored model through
`generate_from_compiled_model`. A 1-input dummy chain is not restored.
The golden-path example reads ICs and point counts from
`unique_claim_experiment_set`.

```@example claim-datagen-joint
using BioDynaX, Random
net = BioDynaX.build_remapped_two_regulator_network()
row = BioDynaX.joint_datagen_compiler_row(
    net; rng = MersenneTwister(13),
    u0 = BioDynaX.remapped_two_regulator_state(),
    truth_params = BioDynaX.remapped_two_regulator_phys_truth())
(row.joint_holds, row.arities, row.packed_dims, row.recovery_admits,
 row.validate_open)
```

```@repl claim-datagen-fp
using BioDynaX, Random
net = BioDynaX.build_hill_recovery_network(; known = true)
set = BioDynaX.unique_claim_experiment_set(
    MersenneTwister(103), net; smoke = true,
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6))
(length(set.experiments),
 BioDynaX.unique_claim_experiment_set_matches_fingerprint(set; smoke = true))
```

## Recovery-suite admission

run_recovery_suite admits unique-claim sections through admit_recovery_suite_network; 0/2 holes fail closed without training.
Every recovery-suite section has a hole policy; only unique-claim sections reject 0/2 holes before training.
`validate_network` still returns the 0-hole and 2-hole networks.
The full matrix is on [Compiled experiment path](compiled-path.md).

```@example claim-suite-admit
using BioDynaX
closed = BioDynaX.recovery_suite_rejects_zero_and_dual_holes(:ude_discovery)
(closed.holds, closed.zero.admitted, closed.two.admitted, closed.one.admitted,
 BioDynaX.recovery_suite_section_kind(:ude_discovery),
 BioDynaX.recovery_suite_section_kind(:linear))
```

## Protocol row and named KPI failures

UniqueClaimProtocolRow joins UniqueClaimFingerprint, protocol_result, extras_print_label, and named KPI failures.
The named symbols are `:unidentifiable_edge`, `:data_residual`, and
`:support_recall`. Combined F1 is not a failure symbol.

```@repl claim-protocol-row
using BioDynaX
row = BioDynaX.unique_claim_protocol_row_from_fields()
(row.extras_label, row.kpi_failures,
 BioDynaX.format_unique_claim_kpi_failures(row.kpi_failures),
 BioDynaX.unique_claim_kpi_failure_symbols())
```

```@example claim-protocol-row-miss
using BioDynaX
miss = BioDynaX.unique_claim_protocol_row_from_fields(;
    unidentifiable_edge = false, data_residual = 0.31)
(miss.kpi_failures,
 BioDynaX.unique_claim_kpi_failure_message(miss.kpi_failures))
```

## Held-out evidence (Q7, reported, not a gate)

Unique-claim suite sections generate nine ICs once, fit only ICs 1..7,
derive the discovery domain from train only, then evaluate holdout on
ICs 8 and 9. `evaluate_holdout` does not discover. The M1 composer
keeps its existing discovery pipeline.

```
data_residual            = legacy IC[1] hybrid residual
data_residual_train      = (ρ1+ρ2+ρ3+ρ4+ρ5+ρ6+ρ7)/7
data_residual_holdout    = (ρ8+ρ9)/2
```

Those aggregates are arithmetic means. They are not RMS, not
concatenated residuals, not IC[1], and not one holdout experiment.

`d_rmse_holdout` evaluates neural \(D\) at the actual observed
regulator coordinates from experiments 8 and 9.
`d_rmse_holdout_domain` evaluates neural \(D\) on the deterministic
external band derived only from training data. Neither is symbolic
\(D\) reconstruction.

Case A (`training_ok == false`, `discovery === nothing`) leaves
`holdout === nothing`. Cases B and C report holdout even when
symbolic discovery fails. A holdout residual greater than 0.30 does
not suppress that evidence and does not fail the M1 hold.

Q7 is reported held-out generalization evidence, not an additional
success gate. Discovery failure does not erase Q7. Q4 is a practical
functional-identifiability diagnostic, not a success gate and not a
certificate. This page is still a narrow one-hole research preview.
M4-B trained-UDE graph-local validation is implemented. PR smoke is not trained-UDE scientific acceptance. M4-C remains pending future work.

## What this page is not

- A wet-lab protocol for one noisy CSV and unknown topology.
- A general CRN solver.
- A license to call coefficients biological constants.
- A promise that combined F1 from a trained NN is canonical Hill.
- A functional-identifiability or unique-\(D\) certificate.
- A General-registry install story. Clone and `Pkg.instantiate`.
- GPU / SBML / ModelingToolkit productization. Those names stay
  experimental and unexported.

Gates live in `RECOVERY_THRESHOLDS`. Loosening a number is breaking.
See [Recovery benchmarks](benchmarks.md).
