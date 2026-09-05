# Concepts

## Model form

Every BioDynaX model is a production-destruction system

```math
\frac{du_i}{dt} = P_i(u, p, t) - D_i(u, p, t)\,u_i, \qquad P_i, D_i \ge 0 .
```

`compile_mechanism` lowers reactions and edges into production and
destruction terms. Known kinetics stay symbolic and compiled: mass action,
linear decay, Hill, Michaelis-Menten saturation, competitive binding, an
input drive, or a custom rate expression. Exactly one destruction term may be
marked `known = false`; it becomes a `NeuralDestructionTerm`, a small Lux
multilayer perceptron with a softplus output that maps the regulator
concentrations to a non-negative rate. The unknown term is multiplicative
(`D_i(u) * u_i`), not an additive residual on the right-hand side.

Physical parameters are stored in raw form and mapped through a softplus so
that they stay positive; `positive_parameter` applies the map and
`pack_parameters` builds the `ComponentVector` with `phys` and `nn` axes that
every solver and trainer takes. States pass through `max(0, x)` inside the
right-hand side. These are architectural choices that keep the model in the
positive orthant in practice; they are not a positivity theorem.

`validate_network` checks names, bounds, stoichiometry, and metadata. It does
not count unknown terms: a network with zero or several unknown destruction
terms still compiles. The recovery workflow checks the count separately
(`BioDynaX.assert_single_unknown_destruction`) and raises an error for
anything other than one.

## Networks and metadata

A `BiologicalNetwork` holds `NodeSpec`s and either `ReactionSpec`s (with
stoichiometry and regulator indices) or `EdgeSpec`s (source, target, kind).
Each carries a typed metadata struct naming its rate parameters:

| Metadata | Kinetics | Parameters |
|---|---|---|
| `MassActionMetadata` | rate proportional to the product of the regulators | `rate_param` |
| `LinearDecayMetadata` | first-order decay | `rate_param` |
| `HillMetadata` | `vmax * r^n / (K^n + r^n)` | `vmax_param`, `k_param`, `hill_order` |
| `SaturationMetadata` | Michaelis-Menten saturation | `vmax_param`, `k_param` |
| `CompetitiveMetadata` | competitive binding of two regulators | see the API page |
| `InputDriveMetadata` | production driven by an input node | `rate_param`, `input_param`, `input_node` |
| `CustomKineticMetadata` | user-supplied rate expression | `rate_param` |
| `EmptyMetadata` | no parameters (used for unknown edges) | none |

`Dict{Symbol,Any}` metadata is still accepted for backward compatibility.

```@example concepts
using BioDynaX
ReactionSpec(name = :decay, stoichiometry = Dict(1 => -1.0), regulators = Int[],
    metadata = LinearDecayMetadata(rate_param = :k))
```

## Experiments

An `Experiment` is one time series: `times`, an `observations` matrix
(`nstates x n_points`), the initial state `u0`, an optional `mask` of
observed entries, and metadata such as per-experiment weights. An
`ExperimentSet` is a list of experiments with the state names.
`experiment_from_csv` and `write_experiment_csv` read and write one
experiment; `generate_experiment_set` produces synthetic sets from a known
network by compiling the ground-truth model once and integrating every
initial condition from it.

## Training

`train_ude` fits one experiment and `train_experiments` fits a set. Both
minimize the trajectory mean-squared error with Adam and then, optionally,
refine with BFGS on the full loss. A `TrainingConfig` sets the iteration
counts, learning rate, gradient clipping, a horizon curriculum
(`HorizonCurriculum`, which trains on a growing fraction of the time span),
the constraint strategy (`StructuralPositivity` or
`AugmentedLagrangianConfig`), and `frozen_phys`, a list of physical
parameters held fixed. Gradients come from SciMLSensitivity adjoints:
`auto_sensealg` picks `BacksolveAdjoint` for small purely mechanistic models
with at most 64 observations and `InterpolatingAdjoint` otherwise; a model
with a neural term always uses `InterpolatingAdjoint`.

Training reuses one compiled model and one solver session across initial
conditions. The optimizer state is kept on the result
(`TrainingResult.diagnostics.optimizer_state`), so a first-experiment warm-up
can hand its Adam state to the joint fit, and checkpoints can resume without
recompiling (see [How-to](howto.md)).

## Identifiability diagnostic

With observed concentrations alone, a production rate and the scale of the
destruction term that follows it trade off against each other.
`BioDynaX.production_destruction_tradeoff` quantifies this for a trained
model on one trajectory:

- the Fisher information over the physical parameters (neural weights
  excluded) and its condition number, from a finite-difference trajectory
  Jacobian;
- the cosine between the trajectory sensitivity to the production parameter
  (`k_prod` by default) and the sensitivity to a multiplicative rescaling of
  the unknown term.

`unidentifiable_edge` is `true` when the condition number is at least `1e6`
or the cosine is at least 0.95. `coefficients_are_biological_constants` is
its negation. The diagnostic is local (one trajectory, one parameter point)
and asymptotic; it is not a structural identifiability proof. A raised
warning does not stop the workflow. In the reference protocol it is required
output: the ambiguity must be reported, not hidden.

A second, unexported diagnostic, `BioDynaX.assess_functional_identifiability`,
trains the unknown term independently from five fixed restart seeds
(201 to 205) and compares the learned rate functions pairwise on a shared
domain built from the training and held-out regulator values. It reports
every restart, including failed ones, the scale-normalized disagreement
between rate functions, the agreement between trajectories, and a derived
status. It is a diagnostic; it is not an acceptance criterion. Fisher
information over the physical parameters is also available on its own
through `BioDynaX.assess_identifiability`.

## Symbolic discovery

Discovery fits an implicit rational form

```math
D(z)\,\dot x - N(z) = 0
```

by sequentially thresholded least squares over a monomial library, with the
constant denominator coefficient anchored to one. `ImplicitSINDyPI` is the
default backend; `ExplicitSTLSQ` fits an explicit polynomial right-hand side.
Both are configured through `DiscoveryConfig` (threshold, maximum degree,
bootstrap resamples, validation fraction, domain samples, chunk size).

The library is graph-local: `local_basis(network, target; scope = :graph)`
builds monomials only from the target's parents in the interaction graph.
`scope = :global` uses every dynamic node and is the comparison used in the
benchmarks. For a bounded in-degree `k`, the library size grows with the sum
of `k_i^d` over targets rather than with `n^d`.

```@example concepts
net = BiologicalNetwork(
    [NodeSpec(name = :S), NodeSpec(name = :R), NodeSpec(name = :Z)],
    [EdgeSpec(source = 2, target = 1, kind = INHIBITION, known = false,
        family = HILL, metadata = EmptyMetadata())])
graph = local_basis(net, 1; degree = 2, scope = :graph)
global_lib = local_basis(net, 1; degree = 2, scope = :global)
(graph.variables, length(graph.numerator), global_lib.variables, length(global_lib.numerator))
```

Two entry points share this machinery:

- `discover_unknown_rate(R, times, D)` regresses sampled values of the learned
  destruction rate on the regulator values. This is the path of the reference
  protocol: the neural term is sampled (`sample_unknown_destruction` along
  trajectories, or `BioDynaX.sample_unknown_destruction_grid` on a regulator
  grid) and a rate `D(r)` is fitted.
- `discover_equations(X, times, network)` works on state trajectories and
  their derivatives (`estimate_derivatives`) without a trained model.

Candidates are validated on training columns, on held-out columns, and on a
grid of domain samples in the positive orthant; a candidate whose denominator
changes sign or approaches zero on any of them is rejected
(`DenominatorUnsafe`). Bootstrap resampling reports how often each term is
selected. Failures are reported through `DiscoveryRetcode`
(`InsufficientSamples`, `DenominatorUnsafe`, `EmptySupport`,
`SingularLibrary`, `DiscoveryFailed`); with `strict = true` they throw.

A recovered candidate can be exported as LaTeX (`equation_to_latex`), as a
callable (`equation_to_function`), or as a full right-hand side
(`export_rhs`). `compose_hybrid_rhs` swaps the callable in for the neural
term, and `hybrid_data_residual` integrates the result and compares it with
observations.

### Pruning nuisance terms

Discovery on a learned rate often keeps small terms that fit the neural
network's approximation error rather than the mechanism; on the reference
protocol these are a constant and a linear term next to the true Hill
monomials. The optional stability-selection stage,
`discover_unknown_rate(...; stability_selection = StabilitySelection())`,
resamples the training rows of the regression with replacement `n_boot`
times (default 100), repeats the thresholded fit on every resample, and
keeps a term of the fitted candidate only if it was selected in at least a
fraction `τ` of the resamples (default 0.8), refitting the coefficients of
the kept terms afterwards. Terms are never added, the stage is skipped when
it would remove every numerator term or make the denominator unsafe, and
the selection frequency of every library term is available through
`stability_selection_report` and `format_stability_selection`, so a user can
see why a term was kept or dropped. Its cost is `n_boot` thresholded fits on
the training rows, which for the reference protocol is a fraction of a
second. It is off by default, and with it off the discovery output is
unchanged; the [Benchmarks](benchmarks.md#Stability-selection-on-the-library-comparison-study)
page reports what it does on the library comparison study.

## The reference protocol

The recovery benchmarks and the example share one protocol, stored in
`BioDynaX.UNIQUE_CLAIM_PROTOCOL`:

| Setting | Value |
|---|---|
| seed | 103 |
| initial conditions | 9, generated once |
| points per experiment | 50 over `t in [0, 8]` |
| observation noise | 0 |
| training | Adam 100, then BFGS 50 |
| discovery | bootstrap 8, discovery seed 3 |
| smoke variant | 1 initial condition, 8 points, no BFGS |

In the recovery suite the nine experiments are split 7/2: experiments 1 to 7
are used for training and for deriving the discovery grid; experiments 8 and 9
are held out. The reported quantities are:

| Quantity | Meaning |
|---|---|
| `data_residual` | hybrid residual on training experiment 1 |
| `data_residual_train` | mean of the residuals on experiments 1 to 7 |
| `data_residual_holdout` | mean of the residuals on experiments 8 and 9 |
| `d_rmse_holdout` | error of the neural rate at the observed held-out regulator values |
| `d_rmse_holdout_domain` | error of the neural rate on a band derived from the training data |
| `nn_rate_rmse`, `nn_correlation` | neural rate versus the true rate on the discovery grid |
| `support_recall`, `support_f1` | recovered monomials versus the true Hill support |

The acceptance criteria for the Hill-class recovery are three: the scale
warning is raised, `data_residual` is at most 0.30, and support recall is at
least 0.99. Combined support F1 is reported against a floor of 0.50 and is
not part of the criteria; nuisance terms typically remain. Held-out numbers
are reported and are never compared with a threshold. Michaelis-Menten
unknown terms are checked on the neural-rate error and the residual only.
All thresholds live in `RECOVERY_THRESHOLDS`:

```@example concepts
RECOVERY_THRESHOLDS
```

## Robustness checks

Two further checks exist beyond the single-seed protocol. Both are
unexported.

- `BioDynaX.evaluate_trained_graph_local` trains one model (seed 401, three
  initial conditions), samples its learned rate once, and runs discovery
  three times on the same samples with the graph-local library, a global
  library, and a library from a deliberately wrong graph. The graph-local
  run must keep the true regulator; the wrong-graph run must miss it. A fast
  version runs in the default tests; the full version is
  `test/run_m4_b_protocol.jl`.
- `BioDynaX.TrajectoryOccupancy` collects the observed states of the training
  or held-out experiments as an alternative sampling context for the learned
  rate. It is not used by the functional-identifiability diagnostic or by the
  held-out evaluation.

A multi-seed robustness study of the full protocol is not implemented;
`benchmark/recovery_seeds.jl --ude` runs the trained-model protocol on five
seeds as a report.
