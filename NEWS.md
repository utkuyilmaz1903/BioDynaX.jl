# BioDynaX.jl Unreleased

- Golden path reports practical `k_prod`↔`D` collinearity. Freeze / normalize /
  production perturbation do not break that scale tradeoff; the warning is the
  finding.
- Wrong-graph negative control and 3-state parent membership are the graph-prior
  evidence. 2-state graph vs global F1 stays equal after Occam.
- Unknown-edge closed loop is Hill-class. MM unknown stays NN RMSE + residual.
- Partial observation: discovery→hybrid residual versus data is gated; UDE
  training on missing states is not claimed.
- README protocol matches `examples/unknown_inhibition.jl` (adam 100 / bfgs 50).

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
