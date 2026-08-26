# BioDynaX.jl Unreleased

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
