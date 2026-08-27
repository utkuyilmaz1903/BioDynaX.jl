# Changelog

All notable changes to BioDynaX.jl are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Claim-scope honesty locks the howto CSV as a synthetic fixture, keeps
  licensed experimental series absent, fails closed if partial observation
  claims masked-state UDE training, names graph-prior parent booleans
  rather than a DataDrivenSparse F1 win, keeps multi-seed UDE as a report,
  and states that unknown topology / general CRN are out of scope
  (`src/ClaimScopeHonesty.jl`, not exported). Docs:
  [out-of-scope](docs/src/out-of-scope.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Claim-metric honesty locks the same-library UDE extras probe below
  `support_f1_clean`, names `unidentifiable_edge` as the Fisher/Jacobian
  cosine or condition-number flag (not StructuralIdentifiability.jl),
  and keeps MM unknown on NN RMSE + residual rather than Hill recall
  0.99 (`src/ClaimMetricHonesty.jl`, not exported). Combined F1 stays
  the skeleton floor 0.50. Docs:
  [claim-metric-honesty](docs/src/claim-metric-honesty.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Allocation / type-stability gates extend `test_quality_gates.jl`
  to the unexported workspaces (`src/AllocationGates.jl`, not
  exported). `allocation_hot` records a warmed `@allocated`
  ceiling that can fail. `STLSQWorkspace` reuse must not increment
  `resize_count` on a same-shape ensure. Docs:
  [allocation-gates](docs/src/allocation-gates.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Docs executable path joins the H–L surfaces (hybrid residual,
  identifiability product, graph-local library, denominator domain,
  parameter schema pack) with leftover scanners on tutorial / howto /
  sciml (`src/DocsExecutable.jl`, not exported). It does not restate
  the A–G pages. Docs:
  [docs-executable](docs/src/docs-executable.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Parameter schema / pack collects `CustomKineticMetadata.rate_param`
  so `:k_custom` is present (`src/ParameterSchemaPack.jl`, not
  exported). `unpack_parameters` inverts `pack_parameters`.
  Remapped multi-head pack/unpack matches the compiled NN tree.
  Docs: [parameter-schema-pack](docs/src/parameter-schema-pack.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Denominator / domain safety splits `denominator_violation_count`
  across train, validation, and the orthant domain grid
  (`src/DenominatorDomain.jl`, not exported). Explicit candidates
  count 0. Failed discovery records `typemax`. UDE extras still
  walk the domain grid. Docs:
  [denominator-domain](docs/src/denominator-domain.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Graph-local library rows lock `local_basis` `scope=:graph` versus
  `scope=:global` and name `local_has_true_parent_gate`
  (`src/GraphLocalLibrary.jl`, not exported). Wrong-graph parent
  sets omit the true regulator. `run_recovery_suite` graph-prior
  sections call that gate. Docs:
  [graph-local-library](docs/src/graph-local-library.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Identifiability product rows join `production_destruction_tradeoff`
  to `UniqueClaimProtocolRow` through `identifiability_product`
  (`src/IdentifiabilityProduct.jl`, not exported). Collinearity
  prints only when the cosine is finite.
  `coefficients_are_biological_constants` follows
  `unidentifiable_edge`. Docs:
  [identifiability-product](docs/src/identifiability-product.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Hybrid residual versus solver agrees `hybrid_data_residual` with
  `SciMLBase.solve` of `compose_hybrid_rhs` and with `predict_ude`
  at noise 0 (`src/HybridResidual.jl`, not exported). Failed compose
  paths return Inf or throw. Smoke residual (1 IC / 8 points) is not
  the seed-103 / 9-IC protocol residual. Docs:
  [hybrid-residual](docs/src/hybrid-residual.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Hybrid compose path locks `compose_hybrid_rhs` identity against
  `ude_system` and `hybrid_data_residual` against noise-0 generated
  data (`src/HybridCompose.jl`, not exported). Failed discovery
  cannot `export_rhs`. Remapped heads compose one term at a time.
  Docs: [hybrid-compose](docs/src/hybrid-compose.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Failure-mode instrument locks `DiscoveryRetcode` maps, the 20-sample
  discovery floor, 0/2-hole `validate_network` openness, KPI failure
  symbols without combined F1, and extras `NA` / `(none)` / live
  leftovers (`src/FailureModes.jl`, not exported). Docs:
  [failure-modes](docs/src/failure-modes.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Experiment fingerprints ignore metadata; batches cover every IC
  without padding; `resume_training` reuses the compiled `UDEModel`
  (`src/ExperimentCheckpoint.jl`, not exported). Remapped multi-head
  generate and `train_experiments` / `train_experiments_with_warmup`
  share one compiled tree. Checkpoints stay Julia `serialize`
  payloads. Docs:
  [experiment-checkpoint](docs/src/experiment-checkpoint.md).
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Unused `run_recovery_suite` sections do not call
  `_train_unknown_edge` (`recovery_suite_plan`,
  `with_train_unknown_edge_counter`; not exported). The default suite
  still runs `:ude_discovery` and `:mm_unknown`. Skip does not drop
  protocol ICs. `benchmark/recovery_suite.jl` keeps the fast block
  separate from the unique-claim trainers.
  `RECOVERY_THRESHOLDS` and the export list are unchanged.
- SciML solve surface agrees `ude_system`, `ODEFunction`,
  `ODEProblem`, `remake`, inplace cache, `SciMLBase.solve`, and
  `predict_ude` (`SolveSurfaceRow`; not exported). Mechanistic
  models switch from `BacksolveAdjoint` to `InterpolatingAdjoint`
  when `n_observations` exceeds 64. No new solver. `RECOVERY_THRESHOLDS`
  and the export list are unchanged.
- Training reuses one compiled `UDEModel` across ICs
  (`TrainingSolveSession`, `train_experiments_with_warmup`; not
  exported). The Augmented-Lagrangian constraint path passes the
  compiled model into `predict_ude`. First-IC warmup hands its
  Optimisers state to `train_experiments`. Neural holes lock
  `InterpolatingAdjoint`; `BacksolveAdjoint` is rejected on a neural
  hole.   `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Discovery library evaluation reuses grow-only STLSQ / implicit-design
  workspaces. `evaluate_library!` writes monomials in place.
  `_fit_implicit` streams row chunks (`_stlsq_blocked!`). Bootstrap
  draws do not rebuild the Gram. `RECOVERY_THRESHOLDS` and the export
  list are unchanged.
- `generate_experiment_set` compiles the ground-truth model once;
  `generate_from_compiled_model` uses `SciMLBase.ODEProblem(model, ...)`.
  Suite hole policy is explicit for every section
  (`recovery_suite_admission_matrix`). Joint compiled path is
  `CompiledPathRow` (not exported). `RECOVERY_THRESHOLDS` and the export
  list are unchanged.
- `generate_from_compiled_model` integrates a stored compiled model;
  remapped multi-head and two-regulator `D(S,I)` generate together.
  `run_recovery_suite` admits unique-claim sections through
  `admit_recovery_suite_network`. `UniqueClaimProtocolRow` names KPI
  failures. `RECOVERY_THRESHOLDS` and the export list are unchanged.
- Unique-claim product helpers name identifiability, fit, discovery, and
  reproduction as one contract (`src/UniqueClaim.jl`, not exported). The
  golden-path example prints live extras and reads `n_ics` from
  `UNIQUE_CLAIM_PROTOCOL`. `validate_network` is unchanged.
- Golden-path example defaults to seed 103, shares `_unknown_edge_ics` with
  the recovery job, and uses regulator-grid discovery
  (`sample_unknown_destruction_grid` + `rate_discovery_config(bootstrap = 8,
  seed = 3)`). Smoke stays 1 IC / 8 points and is not that protocol.
- Deduplicated `.gitignore`. CompatHelper and TagBot now request the write
  scopes they need after GitHub's default read-only `GITHUB_TOKEN`.
- Deduplicated `CITATION.cff` and pointed it at the live `/dev/` docs URL.
  The 0.9.2 preview is not yet in General.
- SciML hardening (research preview, not v1.0): README/docs SciML snippet
  and golden-path example smoke are CI-gated; export docstrings are
  required; doctests cover the 2-node ODE and discovery retcode surface;
  locked UDE KPIs are residual / recall / `unidentifiable_edge`; graph
  prior booleans include `Z_in_local_library`; isolated DataDrivenSparse
  probe is documented (skip is not a win); invariant and σ=0.05 negative
  controls are in the fast suite; macOS×Julia 1 (not 1.10, not recovery),
  coverage upload without a fake coverage badge, SciML formatter config,
  ColPrac templates, and an Optimization.jl hook section.
  `OrdinaryDiffEqTsit5` split was evaluated and deferred
  (`SolverConfig.algorithm` still needs the full OrdinaryDiffEq escape).
  Fast-suite smoke is 1 IC / 8 points and does not overwrite
  `examples/data/unknown_inhibition.csv`. General register and the first
  tag remain maintainer actions.

## [0.9.2] - 2026-08-14

Preview polish. No new science product. Export surface shrinks; docs and
TagBot stay honest; graph prior is measured at 6 states; UDE combined F1
stays a skeleton claim.

### Added

- Six-state graph-prior fixture and wrong-graph negative control
  (`:six_state`, `:six_state_wrong_graph`). Combined F1 is not the KPI.
- `benchmark/recovery_seeds.jl` (analytical Occam on five seeds; optional
  `--ude`). CI remains seed 103 / 104.
- `benchmark/noise_grid.jl` and `benchmark/ude_f1_attempt.jl` (same library;
  no new atoms).
- GitHub Pages workflow that publishes the existing `gh-pages` branch.
  Live URL recorded after HTTP 200:
  https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/

- Golden-path practical `k_prod`↔`D(z)` warning
  (`BioDynaX.production_destruction_tradeoff` / `report_production_destruction_tradeoff`).
  Hill UDE recovery CI now requires `unidentifiable_edge == true` and cosine ≥ 0.95.
- `TrainingConfig.frozen_phys` to pin known production during training. Freeze,
  `D` normalization, and a production-rate perturbation do **not** break the
  Jacobian scale tradeoff; that is the locked finding, not a solved
  identifiability claim.
- Wrong-graph negative control (`:wrong_graph`): a Q→S prior must miss true
  parent `R`. Three-state true-parent / no-local-false-parent remains the
  graph-prior evidence (2-state F1 stays equal after Occam).
- Scale-normalized discovery on the same monomial library
  (`normalize_destruction_samples`); UDE combined F1 is still not the
  analytical Hill gate.
- Partial-observation closed loop: subsampled `D` → hybrid residual versus
  data. UDE training on missing states is **not** claimed
  (`ude_mask_train_claimed = false`).

### Changed

- Public `export` list is the freeze list plus golden-path verbs. Recovery
  fixtures, Fisher, GPU/SBML/MTK, and library internals are `BioDynaX.foo`.
- Golden-path example builds the network with `ReactionSpec` / `HillMetadata`.
- `Zygote.@ignore` replaced by `ChainRulesCore.ignore_derivatives`.
- Tutorial is one command, one table, one warning, one “we do not claim”.
- Experimental page: do not use in a paper or a wet lab.
- Locked UDE claim remains recall + residual after a same-library F1 attempt.
- No licensed experimental time series matches the unique-claim protocol;
  that absence is the result.

- README quick start points at `examples/unknown_inhibition.jl` (adam 100 /
  bfgs 50, 9 ICs). Unknown-edge closed loop is Hill-class; MM unknown remains
  NN RMSE + data residual.
- `train_experiments` BFGS refines the joint loss over every experiment.
  Adam may still be minibatched (`batch_size = 1`); the last IC no longer
  monopolizes the second-order step.
- Nested Occam subset selection scores BIC on the fit set. Held-out RSS is a
  safety filter, not the sparsity score. This keeps analytical 0.5% Hill F1
  at the `support_f1_clean` gate across Julia 1.10 / OpenBLAS.
- UDE NN parameters are promoted to `Float64` after `Lux.setup` so training
  is not mixed-precision `Float32` weights × `Float64` states.
- Documenter `deploydocs` runs only on `main`. The docs job has
  `contents: write` so `GITHUB_TOKEN` can push `gh-pages`. `DOCUMENTER_KEY`
  remains an optional SSH fallback.
- Recovery fixtures in `api.md` are labeled internal (not on the freeze list).

## [0.9.1] - 2026-08-13

### Added

- Research-preview honesty lock: golden path is
  `sample_unknown_destruction` → `discover_unknown_rate` → `compose_hybrid_rhs`.
- Nested Occam prune and predicted-`y` implicit refit on the **same** monomial
  library (no new atoms). 0.5% noisy analytical Hill combined F1 is gated at
  0.99. Trained-NN UDE combined F1 stays below that gate; the locked UDE claim
  is true-monomial recall + hybrid residual versus data.
- Graph vs global baseline producer `benchmark/sindy_baseline.jl` (optional
  DataDrivenSparse; not a CI dependency).
- Practical `k_prod`↔`D` tradeoff report, 3-state graph-prior test, partial
  observation mask, 2-regulator competitive unknown head, and Elowitz
  synthetic repressilator fixture (not experimental CSV).
- TagBot, CompatHelper, Documenter `deploydocs`, and `CONTRIBUTING.md`.

### Changed

- `RECOVERY_THRESHOLDS.nn_rate_rmse` tightened to 0.12. UDE combined F1 stays
  at the skeleton floor 0.50 (not the analytical 0.99). Loosening remains
  breaking; claiming canonical Hill from a trained NN is not this release.
- Unknown-edge training uses 9 ICs and a 50-point horizon.
- v1.0 / JOSS / `]register` wait on scientific gates that can turn CI red.

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

[Unreleased]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.2...HEAD
[0.9.2]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/utkuyilmaz1903/BioDynaX.jl/releases/tag/v0.4.0
