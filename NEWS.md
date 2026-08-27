# BioDynaX.jl Unreleased

- Software hygiene: internal workspace inventory, SciML format list for
  stacked / science files, `scripts/run_discovery.jl` as a debug runner,
  README unique-claim first line. Docs:
  [internal-workspaces](docs/src/internal-workspaces.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Claim-scope honesty (`src/ClaimScopeHonesty.jl`, not exported):
  synthetic howto CSV, no licensed series, no masked-state UDE claim,
  graph-prior booleans, multi-seed UDE as a report, unknown topology
  out of scope. Docs: [out-of-scope](docs/src/out-of-scope.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Claim-metric honesty (`src/ClaimMetricHonesty.jl`, not exported):
  UDE extras stay below `support_f1_clean`; `unidentifiable_edge` is
  Fisher/Jacobian cosine or cond, not StructuralIdentifiability.jl;
  MM unknown gates NN RMSE and residual, not Hill recall 0.99.
  Docs: [claim-metric-honesty](docs/src/claim-metric-honesty.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Allocation / type-stability gates (`src/AllocationGates.jl`, not
  exported): measured `@allocated` ceilings and workspace reuse
  rows that can fail. Docs:
  [allocation-gates](docs/src/allocation-gates.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Docs executable path (`src/DocsExecutable.jl`, not exported):
  joins H–L contract sentences and scans leftover closed-hole
  phrases. Docs: [docs-executable](docs/src/docs-executable.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Parameter schema / pack (`src/ParameterSchemaPack.jl`, not
  exported): `:k_custom` is in `parameter_schema`.
  `unpack_parameters` inverts `pack_parameters`. Docs:
  [parameter-schema-pack](docs/src/parameter-schema-pack.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Denominator / domain safety (`src/DenominatorDomain.jl`, not
  exported): train / validation / orthant split counts. UDE extras
  still call `denominator_violation_count` on the domain grid.
  Docs: [denominator-domain](docs/src/denominator-domain.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Graph-local library (`src/GraphLocalLibrary.jl`, not exported):
  `local_basis` `scope=:graph` versus `scope=:global`.
  `local_has_true_parent_gate` is the recovered-support membership
  check. Docs:
  [graph-local-library](docs/src/graph-local-library.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Identifiability product rows (`src/IdentifiabilityProduct.jl`,
  not exported): `production_destruction_tradeoff` joins
  `UniqueClaimProtocolRow`. Finite collinearity prints; NaN stays
  silent. Docs:
  [identifiability-product](docs/src/identifiability-product.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Hybrid residual versus solver (`src/HybridResidual.jl`, not
  exported): `hybrid_data_residual` agrees with `SciMLBase.solve`
  of `compose_hybrid_rhs`. Noise-0 identity residual is ~0; smoke
  (1 IC / 8 points) is not the seed-103 / 9-IC protocol residual.
  Docs: [hybrid-residual](docs/src/hybrid-residual.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Hybrid compose path (`src/HybridCompose.jl`, not exported):
  `compose_hybrid_rhs` with the neural destruction rate recovers
  `ude_system`. Failed `DiscoveryResult` cannot `export_rhs`.
  Docs: [hybrid-compose](docs/src/hybrid-compose.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Failure-mode instrument (`src/FailureModes.jl`, not exported)
  names every `DiscoveryRetcode`, keeps `validate_network` open on
  0/2-hole networks, and keeps combined F1 out of KPI failure
  symbols. Docs: [failure-modes](docs/src/failure-modes.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Experiment fingerprints, batches, and checkpoint resume stay on one
  compiled tree (`src/ExperimentCheckpoint.jl`, not exported).
  Remapped multi-head generate and `train_experiments` do not
  `compile_network` per IC. Docs:
  [experiment-checkpoint](docs/src/experiment-checkpoint.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Unused recovery-suite sections do not call `_train_unknown_edge`
  (`src/RecoverySuiteSkip.jl`, not exported).
  `recovery_suite_plan` names trainers; the default suite still
  includes `:ude_discovery` and `:mm_unknown`.
  `benchmark/recovery_suite.jl` already splits fast sections from the
  unique-claim trainers. Docs:
  [recovery-suite-skip](docs/src/recovery-suite-skip.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- SciML solve surface (`src/SciMLSolveSurface.jl`, not exported)
  agrees `ude_system` / `ODEFunction` / `ODEProblem` / remake /
  inplace cache / `SciMLBase.solve` / `predict_ude`. The
  `recommend_sensealg` 64/65 observation boundary is locked.
  Docs: [sciml-solve-surface](docs/src/sciml-solve-surface.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Training reuses one compiled `UDEModel` and one
  `TrainingSolveSession` across ICs (`src/TrainingReuse.jl`, not
  exported). `_train_unknown_edge` calls `train_experiments_with_warmup`
  so the first-IC Adam state is not discarded. Neural holes lock
  `InterpolatingAdjoint` with ZygoteVJP. Docs:
  [training-reuse](docs/src/training-reuse.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- Discovery library evaluation reuses grow-only workspaces
  (`STLSQWorkspace`, `StreamingImplicitWorkspace`; not exported).
  `evaluate_library!` writes in place. `_fit_implicit` streams implicit
  design chunks through `_stlsq_blocked!`. `ImplicitSINDyPI.chunk_size`
  is the blocked row width. Docs:
  [discovery-streaming](docs/src/discovery-streaming.md).
  `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- `generate_experiment_set` compiles one `GroundTruthModel` and generates
  every IC from that stored model (`compile_ground_truth_model`,
  `generate_experiment_set_from_compiled_model`; not exported).
  `generate_from_compiled_model` integrates
  `SciMLBase.ODEProblem(model, u0, tspan, p)`. Remapped multi-head and
  two-regulator `D(S,I)` share that SciML path.
- Every `run_recovery_suite` section has a hole policy
  (`recovery_suite_admission_matrix`). Only unique-claim sections reject
  0/2 holes before training; `validate_network` stays open. `:ablation`
  is a library fixture and does not compile.
- `joint_compiled_path` / `CompiledPathRow` (not exported) join compiled
  generate, remapped heads, suite admission, and `UniqueClaimProtocolRow`.
- Docs: [compiled-path](docs/src/compiled-path.md) page with doctested
  snippets. `RECOVERY_THRESHOLDS` and the public export list are unchanged.
- `generate_from_compiled_model` integrates a stored `UDEModel`.
  `generate_data(::GroundTruthModel)` no longer rebuilds a same-network
  twin. Remapped multi-head unknowns and two-regulator `D(S,I)` are
  generated together (`build_remapped_two_regulator_network`,
  `unique_claim_experiment_set`; not exported).
- `run_recovery_suite` admits unique-claim sections through
  `admit_recovery_suite_network`. Zero- and two-hole networks fail closed
  on that path without a 9-IC train; `validate_network` stays open.
  `UniqueClaimProtocolRow` joins `UniqueClaimFingerprint`,
  `protocol_result`, `extras_print_label`, and named KPI failures
  (`:unidentifiable_edge`, `:data_residual`, `:support_recall`).
- Unique-claim protocol surfaces are typed: `UniqueClaimFingerprint`
  distinguishes the seed-103 / 9-IC job from smoke. `format_protocol_result`
  print order is locked to `PROTOCOL_RESULT_FIELDS` (`data_residual` prints
  as `hybrid_data_residual`; `claim` is printed). Unscored extras print
  `NA`; empty extras print `(none)`; the F1-attempt leftover pair is not
  invented. `run_recovery_suite` calls
  `assert_unique_claim_recovery_network` before the UDE train;
  `validate_network` stays open. `UNIQUE_CLAIM_F1_ATTEMPT` records that
  `benchmark/ude_f1_attempt.jl` is a same-library probe, not the protocol.
- Compiler remapping helpers (`assert_dense_neural_index`, skipped-edge
  and two-regulator fixtures) lock the #11 `1:n` reindex without putting
  a single-hole gate into `validate_network`.
- `generate_data` / `generate_experiment_set` and `default_parameters` now
  build NN weights that match the compiled mechanism (multi-head and
  multi-regulator unknowns). A 1-input dummy chain no longer silently
  aliases two unknowns or crash on two-regulator `D(z)`.
  `RECOVERY_THRESHOLDS` is unchanged.
- `compile_mechanism` now reindexes kept `NeuralDestructionTerm` heads to
  `1:n`. A skipped duplicate unknown edge no longer leaves a gapped
  `nn_index` that crashed `ude_system` / `ude_rhs!` on a later unknown.
- Unique-claim product helpers (`src/UniqueClaim.jl`, not exported) name
  the IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION block, split KPI
  failures, live discovery extras, and the seed-103 / 9-IC fingerprint
  versus smoke. `validate_network` stays open. `RECOVERY_THRESHOLDS` and
  the public export list are unchanged.
- `UNIQUE_CLAIM_PROTOCOL` now includes `n_ics`, `smoke_n_ics`, and
  `observation_noise`. The golden-path example reads
  `unique_claim_protocol_ics` / `unique_claim_protocol_n_points` and
  prints live extras instead of a hardcoded `("1", "r")` pair.
- Docs: [unique-claim](docs/src/unique-claim.md) page with doctested
  snippets; stability/CONTRIBUTING drop HTTP/Pages lab notes; landing
  docs stay honest about preview / not in General.
- Fast-suite locks: protocol fingerprint, KPI miss paths, 0/2-hole
  compile, protocol_result field order, formatter config, and the public
  export set.
- README landing page links the live 0.9.2 research-preview documentation.
- Unique-claim hyperparameters live in `UNIQUE_CLAIM_PROTOCOL` (not exported).
  The golden-path example and UDE recovery defaults read that const.
  `RECOVERY_THRESHOLDS` is unchanged.
- Golden-path stdout and docs treat identifiability as the product block
  (`unidentifiable_edge`, `coefficients_are_biological_constants`). The
  example errors unless there is exactly one unknown `D(z)`. No new
  science claim; `RECOVERY_THRESHOLDS` and the export list are unchanged.
- UDE recovery now records live support extras via
  `discovered_support_extras` (not exported). Combined F1 is still a
  skeleton floor. Canonical Hill from a trained NN stays closed.
- `run_recovery_suite` attaches `protocol_result` on UDE and MM-unknown
  rows (`build_protocol_result`, not exported). Existing metric fields
  stay. Canonical Hill from a trained NN stays closed.
- Unique-claim recovery paths require exactly one unknown `D(z)`
  (`assert_single_unknown_destruction`, `only_unknown_destruction`; not
  exported). `validate_network` is unchanged.
- Unique-claim residual and KPI checks are helpers
  (`assert_unique_claim_residual`, `unique_claim_kpis_hold`; not exported).
  Threshold numbers are unchanged.
- `CITATION.cff` is a single CFF 1.2.0 record (no duplicate fields). The
  0.9.2 preview is not yet in General.
- SciML hardening without a v1.0 cut: snippet/example smoke (1 IC in the
  fast suite; does not overwrite the howto CSV), strict export docs,
  doctests, locked KPI order, graph-prior booleans, isolated external
  baseline probe, invariant tests, macOS×Julia 1, coverage without a
  fake badge, formatter config, ColPrac files. Register/tag still unproven.

# BioDynaX.jl 0.9.2

- Public API is the freeze list plus golden-path verbs. Fixtures and
  experimental GPU/SBML/MTK/Fisher names are `BioDynaX.foo`.
- Golden path example uses `ReactionSpec` / `HillMetadata`. One command, one
  table, one `unidentifiable_edge` warning, no canonical-Hill-from-NN claim.
- `Zygote.@ignore` → `ChainRulesCore.ignore_derivatives`.
- 6-state graph prior + wrong-graph are gated. Combined F1 is not the KPI.
- Multi-seed analytical Occam and a noise grid are reported; CI stays
  single-seed. UDE combined F1 attempt on the same library left extras;
  the claim stays recall + residual.
- No licensed experimental series matches the unique-claim protocol. Absence
  is the result. Docs are live at
  https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/. TagBot is configured;
  there is no tag yet.
- Preview register / methods note / v1.0 remain maintainer gates.

# BioDynaX.jl 0.9.1

## Honesty lock

- Research preview. Unique path: `sample_unknown_destruction` →
  `discover_unknown_rate` → `compose_hybrid_rhs`.
- Combined F1 on 0.5% noisy analytical Hill is gated at 0.99 via nested Occam
  prune (same library). UDE recovery gates recall + residual; combined F1 from
  a trained NN is not claimed as canonical Hill.
- v1.0 / JOSS / General wait until those gates can fail CI.

# BioDynaX.jl 0.9.0

## Product contract

- The unique claim is graph-guided hybrid UDE + local rational discovery, now
  gated by `run_recovery_suite` (linear/MM/Hill/competitive RMSE, executable
  discovered RHS, graph-local vs global ablation).
- Default synthetic data comes from the compiled mechanism. The Hill p53 ODE is
  an explicit misspecification fixture.
- GPU, SBML MathML, and Fisher identifiability are documented as experimental.
- Discovery failures set `DiscoveryRetcode`; `strict=true` throws.

## User path

CSV observations → network with unknown edges → `train_ude` →
`sample_unknown_destruction` → `discover_unknown_rate` → `compose_hybrid_rhs`.
See `examples/unknown_inhibition.jl`.

# BioDynaX.jl 0.8.0

## Discovery scaling

- Streaming library chunks (`each_library_chunk`, `evaluate_library_range!`) and
  blocked STLSQ reduce peak memory for large sample counts.
- Denominator safety now stress-tests train, validation, and a biological
  orthant domain grid (`ImplicitSINDyPI.domain_samples`).
- Raw-data discovery: `estimate_derivatives` +
  `discover_equations(X, times, network)` without a trained UDE.
- Export helpers: `equation_to_latex`, `equation_to_function`, `export_rhs`.
- AIC/BIC model selection via `select_discovery_config`.
