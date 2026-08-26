# Compiled experiment path

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

generate_experiment_set compiles the ground-truth model once and generates every IC from that stored model.
generate_from_compiled_model integrates SciMLBase.ODEProblem(model, u0, tspan, p).
Every recovery-suite section has a hole policy; only unique-claim sections reject 0/2 holes before training.
The joint compiled path is generate_from_compiled_model + remapped heads + admit_recovery_suite_network + UniqueClaimProtocolRow.

`validate_network` stays a topology/metadata checker. Combined F1 is a
skeleton floor (0.50). Canonical Hill from a trained neural rate is closed.
`BIODYNAX_SMOKE=1` (1 IC / 8 points) is not the seed-103 / 9-IC protocol.

## Compile once, then integrate

`#14` stopped `generate_data(::GroundTruthModel)` from rebuilding a
same-network twin. Multi-IC `generate_experiment_set` still rebuilt the
Lux tree on every initial condition. That is closed here:
`compile_ground_truth_model` once, then
`generate_experiment_set_from_compiled_model`.

```@example compiled-once
using BioDynaX, Random
net = BioDynaX.build_dual_unknown_network()
truth = BioDynaX.compile_ground_truth_model(
    MersenneTwister(21), net;
    truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
set = BioDynaX.generate_experiment_set_from_compiled_model(
    truth, MersenneTwister(21);
    initial_conditions = [[0.22, 0.18, 0.16], [0.30, 0.24, 0.20]],
    tspan = (0.0, 1.0), n_points = 8, noise_σ = 0.0)
first_p = first(set.experiments).metadata[:truth_parameters]
(BioDynaX.experiment_set_is_compiled_once(set),
 first_p === last(set.experiments).metadata[:truth_parameters],
 BioDynaX.packed_nn_head_count(truth.parameters),
 length(set.experiments))
```

`unique_claim_experiment_set` reads `UniqueClaimFingerprint` for IC and
point counts and attaches that kind on the set metadata.

```@repl compiled-fp
using BioDynaX, Random
net = BioDynaX.build_hill_recovery_network(; known = true)
set = BioDynaX.unique_claim_experiment_set(
    MersenneTwister(103), net; smoke = true,
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6))
(set.metadata[:compiled_once],
 set.metadata[:unique_claim_fingerprint_kind],
 BioDynaX.unique_claim_experiment_set_matches_fingerprint(set; smoke = true))
```

## SciML agreement

`generate_from_compiled_model` constructs `SciMLBase.ODEProblem(model, u0,
tspan, p)`, not a closure around `ude_system`. Out-of-place, in-place,
`remake`, and `SciMLBase.solve(model, ...)` must match on remapped
multi-head and two-regulator `D(S,I)` fixtures.

```@example compiled-sciml
using BioDynaX, Random
net = BioDynaX.build_remapped_two_regulator_network()
agree = BioDynaX.sciml_compiled_generate_agreement(
    net; rng = Random.MersenneTwister(13),
    u0 = BioDynaX.remapped_two_regulator_state(),
    truth_params = BioDynaX.remapped_two_regulator_phys_truth())
(agree.holds, agree.arch.arities, agree.arch.packed_dims,
 agree.matches_odeproblem, agree.matches_inplace)
```

```@repl compiled-sciml-linear
using BioDynaX, SciMLBase, OrdinaryDiffEq, Random
model, p = build_ude_model(MersenneTwister(0), BioDynaX.build_linear_test_network())
prob = ODEProblem(model, [0.2, 0.1], (0.0, 1.0), p)
sol = solve(prob, Tsit5(); saveat = 0:0.25:1.0)
size(Array(sol), 1)
```

That last snippet is an ODE. It is not the unique-claim protocol.

## Suite hole policy

`admit_recovery_suite_network` always runs `validate_network`. Only
`:unique_claim` sections then require exactly one unknown `D(z)`.
Graph-prior, known-kinetics, literature, and analytical sections stay
open. `:ablation` is a library fixture and does not compile.

```@example compiled-matrix
using BioDynaX
matrix = BioDynaX.recovery_suite_admission_matrix()
zero_dual = BioDynaX.recovery_suite_zero_dual_matrix()
(matrix.holds, matrix.n_sections, matrix.unique_claim,
 zero_dual.holds,
 BioDynaX.recovery_suite_hole_policy(:ude_discovery),
 BioDynaX.recovery_suite_hole_policy(:linear),
 BioDynaX.recovery_suite_hole_policy(:ablation),
 BioDynaX.recovery_suite_expected_holes(:six_state))
```

Zero- and two-hole networks still compile. They fail closed only on
unique-claim sections, and they fail before a 9-IC train.

```@repl compiled-policy
using BioDynaX
BioDynaX.recovery_suite_admits_hole_count(:ude_discovery, 0)
BioDynaX.recovery_suite_admits_hole_count(:ude_discovery, 1)
BioDynaX.recovery_suite_admits_hole_count(:three_state, 2)
```

## Joint path

`joint_compiled_path` is the executable row tests fail independently:
stored model, multi-IC set, SciML agreement, suite admission,
`UniqueClaimProtocolRow`. It does not train a UDE and does not reopen
Hill-from-NN.

```@example compiled-joint
using BioDynaX
path = BioDynaX.remapped_two_regulator_compiled_path()
named = BioDynaX.compiled_path_row_namedtuple(path.row)
(path.holds, named.n_heads, named.arities, named.compiled_once,
 named.recovery_admits, named.sciml_holds)
```

```@repl compiled-claim
using BioDynaX
claim = BioDynaX.unique_claim_compiled_path()
(claim.holds, claim.compiled_once, claim.recovery_admits,
 claim.protocol.extras_label)
```

See [Unique claim](unique-claim.md), [SciML integration](sciml.md), and
[How-to](howto.md). Gates live in `RECOVERY_THRESHOLDS`.
