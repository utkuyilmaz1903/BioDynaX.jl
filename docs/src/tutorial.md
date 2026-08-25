# Tutorial: unknown inhibition recovery

One command, one product block, one thing this package does not claim.

## One command

```bash
julia --project=. examples/unknown_inhibition.jl
```

That script is the protocol: **seed 103**, **9 initial conditions**, Adam 100 /
BFGS 50, `train_experiments`, then regulator-grid
`sample_unknown_destruction_grid` → `discover_unknown_rate` (bootstrap 8,
seed 3) → `compose_hybrid_rhs` versus **data**. The proof is the `recovery`
CI job plus the protocol string test in `test/test_recovery.jl`.
`BIODYNAX_SMOKE=1` and a shortened single-IC `train_ude` snippet are not
the protocol.

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

## Product block (expected mertebe)

The example prints identifiability first. Measured Hill UDE on this protocol
(the one command, seed 103, zero observation noise):

| field | mertebe | gated? |
|-------|---------|--------|
| `unidentifiable_edge` | `true` | yes |
| `coefficients_are_biological_constants` | `false` | derived |
| hybrid residual vs data | ≈ 0.003 | yes (`data_residual`) |
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
stdout section, not a footnote.

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
sufficient, for v1.0. See [Recovery benchmarks](benchmarks.md) and
[API stability](stability.md).

Next: [How-to recipes](howto.md), [SciML integration](sciml.md).
