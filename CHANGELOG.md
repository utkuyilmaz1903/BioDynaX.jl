# Changelog

All notable changes to BioDynaX.jl are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-08-13

### Added

- Scientific recovery suite (`run_recovery_suite`, `benchmark/recovery_suite.jl`)
  with CI gates for linear/MM/Hill/competitive parameter RMSE, UDE→discovery
  executable RHS, and graph-local vs global library ablation.
- `DiscoveryRetcode` on `DiscoveryResult`; `strict=true` rethrows instead of
  returning a failed result.
- CSV experiment I/O (`experiment_from_csv`, `write_experiment_csv`).
- Golden-path example and Documenter tutorial (CSV → train → discover →
  `export_rhs` resimulation), plus How-to and API pages.
- Optional `DataDrivenSparseSTLSQ` backend (requires DataDrivenSparse.jl).
- Optional `import_sbmltoolkit_network` (requires SBMLToolkit + Catalyst).
- `local_basis(...; scope=:graph|:global)` for library ablations.
- Recovery fixtures: `build_mm_test_network`, `build_hill_recovery_network`,
  `build_competitive_test_network`, `build_distractor_network`.
- `CITATION.cff` and API-stability / experimental Documenter pages.

### Changed

- Default synthetic data uses the **compiled mechanism**, not the Hill p53
  fixture. Pass `generator = :hill_p53_fixture` for misspecification studies.
- `STATIC_STATE_THRESHOLD` now dispatches `ude_system(::SVector)` through the
  StaticArrays kernel.
- SBML import no longer guesses Michaelis–Menten from type-name strings;
  explicit kinetic laws compile as unknown neural edges.
- README and architecture docs match the actual GPU/SBML/identifiability
  surface (experimental, not product).
- Package version is 0.9.0.

### Fixed

- Julia 1.12 docs/quality load path: `SciMLBase` is a direct `docs/` dependency;
  quality tests import it via `using BioDynaX: SciMLBase`.

## [0.8.0] - 2026-08-12

### Added

- Streaming library evaluation: `evaluate_library_range!`, `LibraryChunks`,
  `each_library_chunk`, and blocked STLSQ (`_stlsq_blocked`) with buffer reuse.
- Denominator domain safety: orthant stress grid (`domain_samples`) checked on
  train, validation, and biological domain.
- Raw-data discovery: `estimate_derivatives` and
  `discover_equations(X, times, network)` without a trained UDE.
- Equation export: `equation_to_latex`, `equation_to_function`, `export_rhs`.
- Model selection: `information_criterion`, `score_candidate`,
  `select_discovery_config` (AIC/BIC threshold sweep).
- Phase 4 regression tests and expanded `benchmark/scale_basis.jl`.

## [0.7.0] - 2026-08-06

### Added

- `OptimizationInterface`: `build_optimization_problem`, `solve_optimization`,
  `train_via_optimization` for first-class Optimization.jl training.
- `recommend_sensealg` / enhanced `auto_sensealg` with mechanistic
  `BacksolveAdjoint` vs neural `InterpolatingAdjoint` selection.
- Typed `HorizonCurriculum` for horizon training schedules.
- Heteroskedastic multi-experiment weighting via `experiment_weight` and
  `experiment_noise_scale` metadata.
- Lux NN architecture presets `:small`, `:medium`, `:large` for `build_ude_nn`.
- Phase 3 regression tests.

## [0.6.0] - 2026-08-06

### Added

- Full compiler IR for `SATURATION` (Michaelis–Menten) and `CUSTOM_KINETIC` reactions.
- `SaturationMetadata`, `CustomKineticMetadata`, `SaturationProductionTerm`,
  `SaturationDestructionTerm`, `CustomDestructionTerm`.
- Multi-head Lux networks via `MultiHeadNetwork` and `build_ude_nn(rng; n_heads)`.
- Stoichiometric scaling on all mechanism terms (`scale = |coefficient|`).
- StaticArrays fast path for networks with `nstates ≤ 4` (`STATIC_STATE_THRESHOLD`).
- Optional weakdep extensions: `BioDynaXModelingToolkitExt` (`export_mtk_system`),
  `BioDynaXSBMLExt` (`import_sbml_network`).
- Expanded `validate_network` for SATURATION/CUSTOM_KINETIC/UNKNOWN_NN contracts.
- Phase 2 regression tests and fixture networks
  (`build_kinetic_generalization_network`, `build_dual_unknown_network`).

## [0.5.0] - 2026-08-06

### Added

- Fisher information identifiability: `assess_identifiability`, `trajectory_jacobian`,
  `parameter_credible_intervals`, `estimate_parameter_uncertainty`.
- SciML-style `TrainingRetcode` enum and gradient-norm convergence diagnostics.
- Public discovery uncertainty via `uncertainty_reports` and `DiscoveryUncertaintyReport`.
- Multi-trajectory `discover_equations(params, model, set::ExperimentSet)`.
- Benchmark suite: `build_repressilator_network`, `benchmark_networks`,
  `run_benchmark_suite`.
- Phase 1 scientific core regression tests.

### Changed

- `TrainingResult.retcode` is now a `TrainingRetcode` enum (was `Symbol`).

## [0.4.0] - 2026-08-06

### Added

- SciML-native integration: `SciMLBase.ODEProblem(model, u0, tspan, p)`,
  `build_ude_function`, `auto_sensealg`, `default_solver_config`, and
  `SciMLBase.solve(model, ...)`.
- Typed kinetic metadata structs (`InputDriveMetadata`, `MassActionMetadata`,
  `HillMetadata`, `CompetitiveMetadata`, `LinearDecayMetadata`) with backward
  compatibility for `Dict{Symbol,Any}` metadata.
- Working `ExplicitSTLSQ` discovery backend and `ExplicitCandidate` results.
- SciML interface, metadata, and explicit discovery regression tests.
- CI compat downgrade smoke job.
- Expanded Documenter pages (`sciml.md`, `metadata.md`).

### Changed

- Default p53 and linear test networks now use typed metadata.
- `predict_ude` routes through `SciMLBase.ODEProblem` for both Zygote and
  production AD policies.
- `RunMetadata` defaults to `BioDynaX.PACKAGE_VERSION`.
- `ProductionAD` documentation clarifies in-place forward + Zygote adjoint pairing.

### Fixed

- Removed dead `ExplicitSTLSQ` export stub behavior; backend is fully wired.

[0.9.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/releases/tag/v0.4.0
