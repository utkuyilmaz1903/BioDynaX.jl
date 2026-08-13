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
`discover_equations` → `export_rhs` → resimulate. See
`examples/unknown_inhibition.jl`.

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
