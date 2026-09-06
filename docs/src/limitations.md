# Scope and limitations

BioDynaX is a research tool for a narrow problem: a small biochemical
network whose interaction graph and kinetics are known except for one
destruction term. This page collects every caveat in one place.

## Scope

- **One unknown term.** The recovery workflow requires exactly one unknown
  destruction term. The example and the recovery suite raise an error for
  zero or two or more unknown terms. The compiler accepts other
  configurations, but nothing in the package validates them.
- **Known graph.** The graph-local library is built from the interaction
  graph you supply. Inferring the graph itself is out of scope.
- **Small networks.** The benchmarks cover two-, three-, and six-state
  networks. Larger known graphs use the same machinery, but there is no
  evidence in the repository for them.
- **Validation on synthetic data; two measured datasets.** The benchmarks and
  the tests use data generated from the compiled ground-truth mechanism; the
  CSV in `examples/data/` is synthetic, and the repressilator fixture with
  published dimensionless parameters is an ODE, not a measured series. Two
  case studies on measured data exist: the laccase/ABTS progress curves of
  the [enzyme case study](case-study-laccase.md), nine substrate-depletion
  curves with one observed state, and the single-cell p53 traces of the
  [p53–Mdm2 case study](case-study-p53.md), 40 MCF7 cells with p53 observed
  and Mdm2 an unobserved state. Both are downloaded by scripts with
  checksums (the data do not ship with the package) and run outside the
  tests and CI. They exercise the workflow on real measurements; neither
  has both states of a regulated destruction term measured.

## What the diagnostics do and do not establish

- **The identifiability diagnostic is local and practical.** It flags an
  edge as unidentifiable when the Fisher condition number exceeds `1e6` or
  when the trajectory sensitivities to the production rate and to the scale
  of the unknown term have a cosine of at least 0.95, on one trajectory at
  one parameter point. It is not a structural identifiability proof. A
  passing check does not certify the recovered mechanism, and a raised
  warning does not stop the workflow.
- **A good fit is not mechanism recovery.** A small trajectory residual shows
  that the hybrid model reproduces the data. It does not by itself show that
  the true mechanism was found.
- **Coefficients are not biological constants** unless the scale of
  production or destruction is fixed by outside information. With observed
  concentrations alone, the production rate and the scale of the unknown term
  trade off against each other; freezing the production rate or normalizing
  the sampled rate does not remove this.
- **The functional-identifiability diagnostic is a diagnostic.** Agreement
  between independently trained rate functions across restart seeds is
  evidence, not a proof of uniqueness, and it is not an acceptance criterion.
- **Held-out validation is reported, not enforced.** The reference protocol
  trains on seven of nine initial conditions and reports the residual and the
  destruction-rate error on the other two. Those numbers are evidence; they
  are never compared with a threshold. Two held-out initial conditions are
  not an out-of-distribution test.

## Discovery

- **The discovered form is a rational function, not a canonical Hill law.**
  The reference protocol recovers the true monomials of the Hill term (the
  acceptance criterion is recall of at least 0.99), but nuisance terms remain,
  typically a constant and a linear term. Combined support F1 is scored
  against a floor of 0.50; it is not an acceptance criterion. The package does
  not turn the neural term into a Hill expression with named parameters.
- **Michaelis-Menten unknown terms** are checked on the neural-rate error and
  the residual only; canonical Michaelis-Menten support from the trained
  network is not claimed.
- **Noise.** Analytical recovery with finite-difference derivatives holds up
  to 2% rate noise and fails at 5%. The trained-model protocol is measured at
  0 and 2% observation noise.
- **Partial observation** is limited to discovery from subsampled
  destruction-rate values and the residual of the resulting hybrid model.
  Training on states that are never observed is not supported.

## Robustness

- The trained-model library comparison (graph-local versus global versus
  wrong-graph on one trained model) and the five-restart
  functional-identifiability diagnostic are implemented.
- A multi-seed robustness study of the full protocol is not implemented.
  `benchmark/recovery_seeds.jl --ude` runs the protocol on five seeds as a
  report; the continuous-integration check uses seeds 103 and 104.

## Extensions and integrations

- The GPU extension only transfers arrays; there is no batched GPU training.
- SBML import reads species, reactions, and stoichiometry but does not parse
  kinetic laws; such reactions become unknown neural terms.
- The ModelingToolkit export represents neural terms as placeholder
  variables.
- The DataDrivenSparse backend cannot currently be loaded in the package
  environment because of a dependency version conflict.

## Not in scope

Inferring the interaction graph, general reaction-network solving, several
unknown terms at once, experimental design, wet-lab decision support, and
integration with language models.

## Roadmap

The next items, in no particular order and with no dates: a multi-seed
robustness study of the reference protocol, a broader set of benchmark
networks, and registration in the General registry once the public API has
settled.
