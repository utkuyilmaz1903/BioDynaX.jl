# Tutorial: recovering an unknown inhibition

This page walks through `examples/unknown_inhibition.jl`, the reference
example of the package. The example is also the protocol that the recovery
benchmarks run: seed 103, nine initial conditions generated once, 50 points
each over `t in [0, 8]`, Adam 100 / BFGS 50, and symbolic discovery on a
regulator grid.

Run it with

```bash
julia --project=. examples/unknown_inhibition.jl
```

It takes 10 to 15 minutes. With `BIODYNAX_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0`
in the environment it uses one initial condition with 8 points and two Adam
steps, which checks the installation in about two minutes but does not
produce a meaningful fit.

## The network

Two species. `S` is produced in proportion to `R` (mass action, rate
`k_prod`) and degraded by a Hill-type mechanism driven by `R`; `R` is
produced from `S` (rate `k_rs`) and decays linearly (rate `k_r`). The
network is built twice: once fully known, to generate the synthetic data,
and once with the Hill degradation marked `known = false`, to fit.

```@example tut
using BioDynaX, Random

function unknown_inhibition_network(; known::Bool, hill_order::Int = 2)
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = known, family = HILL,
            metadata = HillMetadata(vmax_param = :vmax, k_param = :K,
                hill_order = hill_order)),
        ReactionSpec(name = :produce_r,
            stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

truth_net = unknown_inhibition_network(; known = true)
ude_net = unknown_inhibition_network(; known = false)
model, params = build_ude_model(MersenneTwister(0), ude_net)
(compile_mechanism(ude_net).nstates, BioDynaX.count_unknown_destructions(model))
```

| Reaction | Role | In the hybrid model |
|---|---|---|
| `R -> S` production | known mass action | compiled production term |
| `R` degrades `S` | unknown | neural destruction term |
| `S -> R` production | known mass action | compiled production term |
| `R` linear decay | known | compiled destruction term |

The recovery workflow requires exactly one unknown destruction term. The
example checks this with `BioDynaX.assert_single_unknown_destruction(model)`
and stops with an error otherwise.

## Synthetic data

The example generates nine experiments from the known network with the
truth parameters `k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6`
and no observation noise. The ground-truth model is compiled once and every
initial condition is integrated from that model. The block below generates
the same set with the one-experiment smoke settings so that it runs quickly
here:

```@example tut
truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
set = BioDynaX.unique_claim_experiment_set(
    MersenneTwister(103), truth_net; smoke = true, truth_params = truth)
(length(set.experiments), size(first(set.experiments).observations))
```

The example also writes the first experiment to a CSV file in a temporary
directory and reads it back with `experiment_from_csv`, to show that
measured data enter through the same path.

## Training

Training has two stages. A warm-up fit on the first experiment uses Adam
with a horizon curriculum (35%, 70%, then 100% of the time span), and the
joint fit over all experiments continues from the warm-up parameters with
Adam followed by BFGS on the full-set loss. The physical parameters start
from a flat guess of 0.8.

```julia
warm = train_ude(pack_parameters(guess, params.nn),
    first_exp.observations, first_exp.times, first_exp.u0, tspan, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 0,
        horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0])))
trained = train_experiments(warm.params, set, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 50))
```

(Illustrative; the docs build does not train.)

## Identifiability

Before discovery, the example computes the production/destruction trade-off
for the first experiment:

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, first_exp.observations, first_exp.times,
    first_exp.u0, tspan; term = term)
```

`ident.unidentifiable_edge` is `true` when the Fisher condition number over
the physical parameters exceeds `1e6` or when the trajectory sensitivity to
`k_prod` is collinear (cosine at least 0.95) with the sensitivity to a
multiplicative rescaling of the unknown term. In this example the cosine is
about 0.997: observed concentrations cannot separate the production rate from
the scale of the destruction term, so the recovered coefficients are not
biological constants. The reference protocol treats a raised warning as
required output, not as a failure.

## Discovery

The learned destruction rate is sampled on a grid of regulator values
derived from the training data, and a rational function of the regulator is
fitted by implicit sparse regression:

```julia
term = only(BioDynaX.neural_destruction_terms(model))
r_range = BioDynaX._regulator_grid(set, term)
R, D, term = BioDynaX.sample_unknown_destruction_grid(model, trained.params, term;
    r_range = r_range)
discovery = discover_unknown_rate(R, range(0.0, 1.0; length = size(R, 2)), D;
    config = BioDynaX.unique_claim_discovery_config(), strict = true)
```

`discover_unknown_rate` treats the samples as a function-regression problem
in the regulator (the time argument is a dummy index). With
`strict = true` a failed discovery throws; with `strict = false` it returns a
`DiscoveryResult` whose `retcode` explains the failure (see
[How-to](howto.md)).

The same regression on exact Hill samples recovers the true monomials with no
nuisance terms, which is what the analytical benchmarks check:

```@example tut
r = collect(range(0.1, 2.0; length = 120))
times = collect(range(0.0, 1.0; length = length(r)))
D = BioDynaX.hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
clean = discover_unknown_rate(reshape(r, 1, :), times, reshape(D, 1, :);
    config = BioDynaX.rate_discovery_config(bootstrap = 0, seed = 1),
    verbose = false, strict = true)
clean.equations
```

## Resimulation and the report

The discovered rate replaces the neural term inside the compiled ODE and the
hybrid model is integrated again from the first initial condition:

```julia
rate_fn = equation_to_function(discovery.candidates[1])
rhs = compose_hybrid_rhs(model, trained.params, term, rate_fn)
residual = hybrid_data_residual(model, trained.params, term, rate_fn,
    first_exp.u0, tspan, first_exp.times, first_exp.observations)
```

`residual` is the root-mean-square difference between that simulation and the
observations. The example then prints a four-section report:

```text
IDENTIFIABILITY
  unknown_holes: 1
  unidentifiable_edge: true
  coefficients_are_biological_constants: false
  production_param: k_prod
  the production rate (k_prod) and the scale of the unknown term are not separately identifiable from these data
  local diagnostic (Fisher condition number and scale collinearity); not a structural identifiability proof
  collinearity: 0.997
FIT
  hybrid_data_residual: 0.001777
  support_recall: not scored (needs the synthetic ground truth)
DISCOVERY
  equations:
dx[1]/dt = (0.24118*1 + -1.3569*x[1] + 7.7609*x[1]^2) / (1 + -0.3862*x[1] + 4.1863*x[1]^2)
  support_f1: not scored (needs the synthetic ground truth)
  extras: 1, r
  canonical_hill_from_nn: false
  acceptance_criteria: scale warning raised, hybrid residual at most 0.3, support recall at least 0.99
REPRODUCTION
  seed: 103
  n_ics: 9
  n_points: 50
  adam_iters: 100
  bfgs_iters: 50
  bootstrap: 8
  discovery_seed: 3
  protocol_kind: protocol
  smoke: false
```

The values above come from a run in September 2026. Read it as follows:

- The scale warning was raised, so the coefficients of the discovered rate
  are not biological constants.
- The hybrid model reproduces the first trajectory with a residual of about
  0.002.
- The rational rate has the `x^2 / (K + x^2)` structure of the true Hill
  term. The `extras` line lists the nuisance monomials that remain (a
  constant and a linear term). Support recall and F1 are not scored by the
  example because they need the synthetic truth; the recovery suite scores
  them (see [Benchmarks](benchmarks.md)).
- The reproduction section records every setting needed to rerun the
  protocol.

## Held-out validation

The example trains on all nine experiments. The recovery suite
(`BioDynaX.run_recovery_suite` with `sections = (:ude_discovery,)`) runs the
same protocol but trains on experiments 1 to 7, derives the discovery grid
from those seven, and then reports the residual and the destruction-rate
error on experiments 8 and 9. Those held-out numbers are reported as
evidence; they are not part of the acceptance criteria. See
[Concepts](concepts.md) for the definitions.

Next: [How-to recipes](howto.md) and [Concepts](concepts.md).
