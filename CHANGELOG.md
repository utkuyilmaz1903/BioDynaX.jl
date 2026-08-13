# Changelog

All notable changes to BioDynaX.jl are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- CI: replace broken exact-version `compat-downgrade` with compat hygiene checks
  that skip Julia stdlibs.
- CI: Documenter no longer duplicates `@autodocs`/`@docs` bindings; `SciMLBase`
  is a direct `docs/` dependency so Julia 1.12 can `using SciMLBase` for method
  docs; missing-docstrings warn instead of failing.
- CI: Aqua piracy policy treats `SciMLBase.ODEProblem`/`solve` as owned via
  `using BioDynaX: SciMLBase` (quality CI has no direct SciMLBase dep); JET
  runs in `:typo` mode.
- Release E2E tests no longer require four Adam steps to strictly decrease loss.
- `_meta_symbol` accepts integer fallbacks and a `KineticMetadata` default method.

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

[Unreleased]: https://github.com/your-org/BioDynaX.jl/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/your-org/BioDynaX.jl/releases/tag/v0.4.0
