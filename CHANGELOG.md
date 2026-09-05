# Changelog

All notable changes to BioDynaX.jl are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `discover_unknown_term(network, experiments; ...)`, a one-call entry
  point that builds the hybrid model, trains it (warm-up on the first
  experiment, then Adam 100 and BFGS 50), samples the learned rate on the
  regulator grid of the training experiments, discovers a rational rate,
  computes the identifiability diagnostic and the residuals on the first,
  the training, and the held-out experiments (`holdout = 2` by default), and
  returns an `UnknownTermResult`; `report(result)` gives the four-section
  report and `show` prints it. It calls the same functions as the reference
  example, in the same order, with the same defaults, and a test checks
  that its result matches the chained calls field by field.
- An optional stability-selection stage for implicit discovery:
  `discover_unknown_rate(...; stability_selection = StabilitySelection())`
  (also on `discover_equations` and `discover_unknown_term`) resamples the
  training rows of the regression, repeats the thresholded fit on every
  resample, and drops candidate terms selected in fewer than a fraction `τ`
  of the resamples. Terms are never added. `stability_selection_report` and
  `format_stability_selection` show the selection frequency of every library
  term. Off by default; with it off, discovery output is unchanged.
- The library comparison study (`BioDynaX.library_comparison_study`,
  unexported): the trained-model library comparison over seeds and
  observation-noise levels, scoring the graph-local, global, and wrong-graph
  libraries on the same trained model, on the four-state fixture and on the
  two-state reference network, with discovery variants that isolate the
  library construction and the bootstrap. `benchmark/library_comparison_study.jl`
  runs it, appends each row to a CSV as it finishes, and resumes from that
  file; `benchmark/plot_library_comparison.jl` draws the figure. The smoke
  configuration runs in the default test suite and the study runs in the
  weekly heavy CI job.
- `evaluate_trained_graph_local` accepts `seed`, `noise_σ`, and
  `stability_selection` keywords; `build_hill_recovery_network` accepts
  `parent`. All defaults reproduce the previous behaviour.
- `format_protocol_result` accepts `residual_train` and `residual_holdout`
  (printed only when given).

### Changed

- Documentation: the README quick start and the first tutorial section use
  `discover_unknown_term`; the tutorial keeps the step-by-step chain as
  "What the one call does". The benchmarks page has a "Library comparison
  study" section with the figure, the summary tables for both networks, the
  investigation of the recall gap between the study and the reference
  protocol, and the stability-selection comparison; the concepts page
  explains the pruning stage; the how-to page has a one-call recipe.

## [0.10.0] - 2026-09-05

This release turns the repository into a publicly presentable package:
rewritten README and documentation, plain-language printed reports, a
formatted tree, a faster default test suite, working extensions, and a
reorganised continuous-integration workflow. Scientific behaviour (numerics,
thresholds, seeds, protocol settings, library construction) is unchanged.

### Added

- Held-out validation for the reference recovery protocol. Nine initial
  conditions are generated once; the model is trained on the first seven,
  and the residual and the neural destruction-rate error are reported on
  the remaining two. These held-out numbers are reported evidence only and
  are not part of the acceptance criteria.
- A practical functional-identifiability diagnostic
  (`BioDynaX.assess_functional_identifiability`, unexported). Independent
  training restarts on five fixed seeds are compared on a shared domain;
  every restart, including failures, is reported. It is a diagnostic, not
  an acceptance criterion and not a structural identifiability proof.
- Trained-model library validation (`BioDynaX.evaluate_trained_graph_local`,
  unexported): one trained model, its learned destruction rate sampled once,
  and symbolic discovery run with the graph-local library, a global library,
  and a wrong-graph library on the same samples. A fast version runs in the
  default tests; the full protocol is `test/run_m4_b_protocol.jl`.
- Observed-trajectory sampling context (`BioDynaX.TrajectoryOccupancy`,
  unexported) for train or held-out experiments.
- Training reuses one compiled model and one solver session across initial
  conditions; the Adam state from the first-experiment warm-up is carried into
  the joint training.
- Discovery library evaluation reuses grow-only workspaces and streams the
  implicit design matrix in row blocks, reducing peak memory for large sample
  counts.
- `generate_experiment_set` compiles the ground-truth model once and generates
  every initial condition from that model through `SciMLBase.ODEProblem`.
- Experiment fingerprints, batched training, and checkpoint/resume
  (`BioDynaX.save_checkpoint`, `BioDynaX.resume_training`, unexported).
- Parameter packing collects `CustomKineticMetadata.rate_param`;
  `unpack_parameters` inverts `pack_parameters`.
- `denominator_violation_count` is split across training, validation, and the
  orthant domain grid.
- The compiler reindexes kept neural destruction terms to `1:n`, so a skipped
  duplicate unknown edge no longer leaves a gap that crashed `ude_system`.
- Aqua runs with its default `test_all` surface, and a separate `standards` job
  runs JET on `train_ude`, `discover_unknown_rate`, and `compose_hybrid_rhs`.

### Changed

- Recovery paths that require exactly one unknown destruction term check this
  before training and fail early on zero or two unknown terms.
  `validate_network` itself does not count unknown terms.
- The printed recovery report has four sections (identifiability, fit,
  discovery, reproduction) whose order matches the `protocol_result` fields.
  Its explanatory lines are now plain English, floating-point values are
  printed with four significant digits, and the last discovery line reads
  `acceptance_criteria: ...` instead of an internal label.
- The functional-identifiability report and the side-by-side scale-warning
  report use plain section headers.
- The documentation is restructured into ten pages: home, getting started,
  tutorial, concepts, how-to recipes, benchmarks, API reference, extensions,
  scope and limitations, and changelog. Internal design notes and lock lists
  are gone; every remaining code block either runs in the docs build or is
  marked illustrative.
- Source comments and docstrings no longer use internal milestone and
  question labels.
- The two multi-minute training-loop testsets in
  `test/test_experiment_checkpoint.jl` run only with
  `BIODYNAX_TEST_HEAVY=1`; the default `Pkg.test()` is correspondingly
  shorter. No test was removed.
- Continuous integration: the default workflow runs the test suite on Julia
  1.10 and the latest 1.x with coverage, a JuliaFormatter check over the whole
  tree, Aqua and JET, the allocation check, the compat check, and the docs
  build (deployed from `main` and version tags). The trained-model recovery
  protocol, the trained-model library comparison, the heavy test tier, and
  the JET standards run in a weekly scheduled job that can also be started by
  hand.
- The whole tree is formatted with JuliaFormatter (SciML style).
- Every benchmark script starts with a header stating its purpose, runtime,
  output, and how to run it.
- Mechanistic models switch from `BacksolveAdjoint` to `InterpolatingAdjoint`
  when the number of observations exceeds 64. Neural terms always use
  `InterpolatingAdjoint`.
- CompatHelper and TagBot request the write scopes they need.

### Fixed

- The ModelingToolkit extension never loaded (invalid syntax and an
  undeclared dependency on Symbolics) and the SBML extensions could not be
  precompiled because they redefined package methods. All extensions now load;
  `BioDynaX.export_mtk_system` was verified against ModelingToolkit 11.
- `benchmark/scale_basis.jl` could not run because it used unexported names
  without importing them.
- `ModelingToolkit` compat widened to `"9, 10, 11"`; with SciMLBase 3 only
  ModelingToolkit 11 resolves.
- The printed-report consistency check compares support recall and F1 with
  the same rounding that prints them.

### Removed

- Editor and agent configuration (`.cursor/`), internal planning documents
  (`docs/research/`), the debug runner `scripts/run_discovery.jl`, the root
  result figure, and `NEWS.md` (this file is the single changelog).
- Documentation pages that described milestone plans, CI lock lists, and
  API-freeze checklists rather than package behaviour.
- Internal helpers whose only purpose was to read documentation files and
  assert their wording, and the tests that called them.
- Fifty unreferenced leftover helpers from the same layer (source-path,
  test-path, fixture-matrix and lock-row functions that nothing called).

## [0.9.2] - 2026-08-14

### Added

- Six-state graph-prior fixture and a wrong-graph negative control
  (`:six_state`, `:six_state_wrong_graph`) in the recovery suite.
- `benchmark/recovery_seeds.jl` (analytical recovery on five seeds; optional
  `--ude` for the trained-model protocol), `benchmark/noise_grid.jl`, and
  `benchmark/ude_f1_attempt.jl`.
- GitHub Pages workflow that publishes the `gh-pages` branch.
- Practical production/destruction scale warning
  (`BioDynaX.production_destruction_tradeoff` and
  `BioDynaX.report_production_destruction_tradeoff`). The Hill recovery CI job
  requires the warning to be raised.
- `TrainingConfig.frozen_phys` to pin known production parameters during
  training.
- Scale-normalized discovery on the same monomial library
  (`normalize_destruction_samples`).
- Partial-observation path: subsampled destruction-rate samples feed the hybrid
  residual. Training on missing states is not supported.

### Changed

- The export list is reduced to the core types and the functions used in the
  tutorial. Recovery fixtures, Fisher identifiability, and the GPU, SBML, and
  ModelingToolkit extensions are unexported.
- The example builds its network with `ReactionSpec` and `HillMetadata`.
- `Zygote.@ignore` replaced by `ChainRulesCore.ignore_derivatives`.
- `train_experiments` refines the joint loss over every experiment with BFGS;
  Adam may still be minibatched.
- Nested subset selection scores BIC on the fit set and uses held-out residual
  sum of squares only as a safety filter.
- Neural-network parameters are promoted to `Float64` after `Lux.setup`.
- Documenter deploys only from `main`.

## [0.9.1] - 2026-08-13

### Added

- Nested subset pruning and predicted-output implicit refit on the same
  monomial library. Analytical Hill recovery at 0.5% noise reaches a combined
  support F1 of 0.99.
- Graph-local versus global library baseline script
  `benchmark/sindy_baseline.jl` (DataDrivenSparse optional).
- Practical production/destruction trade-off report, three-state graph-prior
  test, partial-observation mask, two-regulator competitive unknown term, and a
  synthetic repressilator fixture.
- TagBot, CompatHelper, Documenter `deploydocs`, and `CONTRIBUTING.md`.

### Changed

- `RECOVERY_THRESHOLDS.nn_rate_rmse` tightened to 0.12.
- Unknown-term training uses nine initial conditions and a 50-point horizon.

## [0.9.0] - 2026-08-13

### Added

- Recovery suite (`BioDynaX.run_recovery_suite`, `benchmark/recovery_suite.jl`)
  with thresholds for linear, Michaelis-Menten, Hill, and competitive parameter
  error, an executable discovered right-hand side, and the graph-local versus
  global library comparison.
- `DiscoveryRetcode` on `DiscoveryResult`; `strict = true` rethrows instead of
  returning a failed result.
- CSV experiment I/O (`experiment_from_csv`, `write_experiment_csv`).
- The unknown-inhibition example and the Documenter tutorial, how-to, and API
  pages.
- Optional `DataDrivenSparseSTLSQ` backend (requires DataDrivenSparse.jl) and
  optional `import_sbmltoolkit_network` (requires SBMLToolkit and Catalyst).
- `local_basis(...; scope = :graph | :global)` for library comparisons.
- `CITATION.cff`.

### Changed

- Default synthetic data uses the compiled mechanism. Pass
  `generator = :hill_p53_fixture` for misspecification studies.
- `ude_system(::SVector)` dispatches through the StaticArrays kernel below
  `STATIC_STATE_THRESHOLD`.
- SBML import no longer guesses Michaelis-Menten kinetics from type names;
  explicit kinetic laws compile as unknown neural terms.

### Fixed

- Julia 1.12 docs and quality load path: `SciMLBase` is a direct `docs/`
  dependency.

## [0.8.0] - 2026-08-12

### Added

- Streaming library evaluation (`evaluate_library_range!`, `LibraryChunks`,
  `each_library_chunk`) and blocked STLSQ with buffer reuse.
- Denominator domain safety: an orthant stress grid (`domain_samples`) checked
  on training, validation, and domain samples.
- Raw-data discovery: `estimate_derivatives` and
  `discover_equations(X, times, network)` without a trained model.
- Equation export: `equation_to_latex`, `equation_to_function`, `export_rhs`.
- Model selection: `information_criterion`, `score_candidate`,
  `select_discovery_config`.

## [0.7.0] - 2026-08-06

### Added

- Optimization.jl training path (`build_optimization_problem`,
  `solve_optimization`, `train_via_optimization`).
- `recommend_sensealg` and `auto_sensealg` selecting `BacksolveAdjoint` for
  mechanistic models and `InterpolatingAdjoint` for neural terms.
- Typed `HorizonCurriculum` for horizon training schedules.
- Per-experiment weighting through `experiment_weight` and
  `experiment_noise_scale` metadata.
- Network size presets `:small`, `:medium`, `:large` for `build_ude_nn`.

## [0.6.0] - 2026-08-06

### Added

- Compiler support for `SATURATION` (Michaelis-Menten) and `CUSTOM_KINETIC`
  reactions with `SaturationMetadata` and `CustomKineticMetadata`.
- Multi-head neural networks (`MultiHeadNetwork`, `build_ude_nn(rng; n_heads)`).
- Stoichiometric scaling on all mechanism terms.
- StaticArrays fast path for networks with at most four states.
- Weak-dependency extensions for ModelingToolkit (`export_mtk_system`) and SBML
  (`import_sbml_network`).

## [0.5.0] - 2026-08-06

### Added

- Fisher-information identifiability (`assess_identifiability`,
  `trajectory_jacobian`, `parameter_credible_intervals`,
  `estimate_parameter_uncertainty`).
- `TrainingRetcode` enum and gradient-norm convergence diagnostics.
- Discovery uncertainty reports and multi-trajectory
  `discover_equations(params, model, set::ExperimentSet)`.
- Benchmark networks including a repressilator.

### Changed

- `TrainingResult.retcode` is a `TrainingRetcode` enum (was a `Symbol`).

## [0.4.0] - 2026-08-06

### Added

- SciML integration: `SciMLBase.ODEProblem(model, u0, tspan, p)`,
  `build_ude_function`, `auto_sensealg`, `default_solver_config`, and
  `SciMLBase.solve(model, ...)`.
- Typed kinetic metadata structs with backward compatibility for
  `Dict{Symbol,Any}` metadata.
- `ExplicitSTLSQ` discovery backend and `ExplicitCandidate` results.

### Changed

- `predict_ude` routes through `SciMLBase.ODEProblem` for both AD policies.
- `RunMetadata` defaults to `BioDynaX.PACKAGE_VERSION`.

[Unreleased]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/releases/tag/v0.4.0
