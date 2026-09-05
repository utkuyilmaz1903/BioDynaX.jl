# Tutorial: unknown inhibition recovery

One command, one product block, one thing this package does not claim.

## One command

```bash
julia --project=. examples/unknown_inhibition.jl
```

That script is the standalone / legacy example: **seed 103**, **9 initial conditions**
generated once, Adam 100 / BFGS 50, `train_experiments` on all nine
generated ICs. This is **not** the M2 recovery-suite train/holdout
protocol. The unique-claim recovery suite generates nine ICs once;
ICs 1–7 are used for training and ICs 8–9 are held out. Then
regulator-grid `sample_unknown_destruction_grid` →
`discover_unknown_rate` (bootstrap 8, seed 3) → `compose_hybrid_rhs`
versus **data**. The suite proof is the `recovery` CI job plus the
protocol string test in `test/test_recovery.jl`. `BIODYNAX_SMOKE=1`
and a shortened single-IC `train_ude` snippet are not the suite
protocol.

```text
CSV / time series
  → BiologicalNetwork (known kinetics + one unknown edge)
  → build_ude_model
  → train_experiments
  → sample_unknown_destruction_grid
  → discover_unknown_rate
  → compose_hybrid_rhs
  → resimulate vs data
  → protocol result (identifiability first)
```

## Product block (expected values)

The example prints identifiability first. Measured Hill UDE on this
standalone / legacy example path (the one command, seed 103, zero
observation noise) still trains all nine generated ICs. That path is
**not** the M2 recovery-suite train/holdout protocol. Unique-claim
suite M2 validated IC[1] `data_residual` = 0.004195 after training
ICs 1–7.

| field | typical value | gated? |
|-------|---------------|--------|
| `unidentifiable_edge` | `true` | yes |
| `coefficients_are_biological_constants` | `false` | derived |
| hybrid residual vs data | ≈ 0.003 (standalone / legacy example that trains all nine generated ICs) | yes (`data_residual`) |
| true-monomial recall | 1.0 | yes (`support_recall`) |
| combined support F1 | ≈ 0.57 | skeleton floor 0.50, **not** 0.99 |
| extras that remain | `1`, `r` | reported, not removed |
| `canonical_hill_from_nn` | `false` | closed |

Canonical Hill combined F1 ≥ 0.99 is gated on **analytical** `D` samples after
Occam (`RECOVERY_THRESHOLDS.support_f1_clean`). It is not the trained-NN claim.

Observed concentrations leave `k_prod` and the scale of `D(z)` collinear.
**Coefficients are not biological constants.** Freeze / `D`-normalization /
production perturbation do not break that Jacobian tradeoff. Zero or
two-or-more unknown `D(z)` holes error on the golden path.

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = false)
println(BioDynaX.format_protocol_result(ident; residual = residual))
```

## We do not claim

- Canonical Hill from a trained neural rate. Extras `1` and `r` remain.
- A wet-lab tool for one noisy CSV and an unknown topology.
- UDE training on missing states. Partial observation is subsampled `D` →
  hybrid residual versus data. The biologist path (train on hidden species)
  is closed.
- A licensed experimental time series that matches the unique-claim protocol.
  Absence is the result. Elowitz is a synthetic ODE fixture.

The example prints that block automatically. Identifiability is the first
stdout section, not a footnote. The same helpers, including smoke versus
the seed-103 / 9-IC fingerprint, are on [Unique claim](unique-claim.md).

```@repl claim-protocol
using BioDynaX
BioDynaX.UNIQUE_CLAIM_PROTOCOL.seed
BioDynaX.UNIQUE_CLAIM_PROTOCOL.n_ics
BioDynaX.unique_claim_is_protocol()
BioDynaX.unique_claim_is_protocol(; smoke = true)
```

## Network (public constructors)

The example does **not** call a recovery fixture. Known graph, one unknown
destruction edge:

```@example tutorial
using BioDynaX
using Random

function unknown_inhibition_network(; known::Bool, hill_order::Int = 2)
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg,
                     stoichiometry = Dict(1 => -1.0), regulators = [2],
                     known = known, family = HILL,
                     metadata = HillMetadata(
                         vmax_param = :vmax, k_param = :K,
                         hill_order = hill_order)),
        ReactionSpec(name = :produce_r,
                     stoichiometry = Dict(2 => 1.0), regulators = [1],
                     metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_r)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

ude_net = unknown_inhibition_network(; known = false, hill_order = 2)
model, params = build_ude_model(Random.MersenneTwister(0), ude_net)
compile_mechanism(ude_net).nstates
```

| Reaction | Role | In the UDE |
|----------|------|------------|
| `R → S` production | known mass action | compiled `P` |
| `R` degrades `S` | **unknown** | `NeuralDestructionTerm` |
| `S → R` production | known mass action | compiled `P` |
| `R` linear decay | known | compiled linear `D` |

CSV import is `experiment_from_csv`. Multi-IC training is
`generate_experiment_set` + `train_experiments` as in the example.

## Discover and resimulate

```julia
X_traj = predict_ude(trained.params, u0, tspan, times, model)
R, D, term = sample_unknown_destruction(model, trained.params, X_traj)
discovery = discover_unknown_rate(R, times, D; verbose = false, strict = true)
rhs = compose_hybrid_rhs(
    model, trained.params, term,
    equation_to_function(discovery.candidates[1]))
residual = hybrid_data_residual(
    model, trained.params, term,
    equation_to_function(discovery.candidates[1]),
    u0, tspan, times, data)
```

That single-trajectory `sample_unknown_destruction` block is a sketch.
The example uses `sample_unknown_destruction_grid` over the nine
generated ICs from `unique_claim_fingerprint()` (legacy full-set grid;
see the example and [Unique claim](unique-claim.md)). The unique-claim
suite derives that grid from train ICs 1–7 only. The example builds the
experiment set with `unique_claim_experiment_set` so IC and point counts
stay on that fingerprint. That helper compiles one ground-truth model
(`compile_ground_truth_model`) and then calls
`generate_experiment_set_from_compiled_model` so every IC shares the
stored NN tree. `BIODYNAX_SMOKE=1` is not that fingerprint.
See [Compiled experiment path](compiled-path.md).

```@repl claim
using BioDynaX
isa(DiscoverySuccess, DiscoveryRetcode)
RECOVERY_THRESHOLDS.data_residual
RECOVERY_THRESHOLDS.support_recall
```

`strict = true` throws. With `strict = false`, check `discovery.retcode`:

| `retcode` | Meaning |
|-----------|---------|
| `DiscoverySuccess` | support recovered; hybrid RHS is allowed |
| `DenominatorUnsafe` | `D(z)` changed sign or hit the floor |
| `EmptySupport` | STLSQ wiped every term |
| `InsufficientSamples` | not enough trajectory points |
| `SingularLibrary` | design matrix was singular |
| `DiscoveryFailed` | anything else |

The recovered object is a rate that must resimulate versus **data**.

Gates live in `RECOVERY_THRESHOLDS`. A green recovery job is necessary, not
sufficient, for v1.0. See [Recovery benchmarks](benchmarks.md).

The executable docs path joins hybrid residual, identifiability product, graph-local library, denominator domain, and parameter schema pack.
`allocation_hot` records a warmed `@allocated` count that can fail.
See [Hybrid residual versus solver](hybrid-residual.md),
[Identifiability product rows](identifiability-product.md),
[Graph-local library and ablation](graph-local-library.md),
[Denominator and domain safety](denominator-domain.md),
[Parameter schema and pack](parameter-schema-pack.md), and
[Allocation and type-stability gates](allocation-gates.md).
tutorial, howto, and sciml must not restate closed H–L holes as current facts.
Smoke (1 IC / 8 points) is not the seed-103 / 9-IC protocol.

Next: [How-to recipes](howto.md), [SciML integration](sciml.md).
