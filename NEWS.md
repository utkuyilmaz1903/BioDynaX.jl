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

# BioDynaX.jl 0.4.0

## SciML-first integration

BioDynaX now exposes the canonical SciML entry point:

```julia
using BioDynaX, SciMLBase

model, p = build_ude_model(MersenneTwister(0))
prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), p)
sol = solve(prob, Tsit5(); saveat = 0:0.1:10.0)
```

Use `inplace=true` with `ProductionAD()` for allocation-free forward integration
while retaining Zygote adjoints for training.

## Typed metadata

Reaction and edge metadata can be specified with compile-time typed structs
instead of untyped dictionaries. Dict metadata remains supported during the 0.x
migration window.

## Explicit STLSQ discovery

`ExplicitSTLSQ` is now a fully supported discovery backend for polynomial
explicit models (`dx/dt = Φ(x)ξ`), complementing the default implicit rational
`ImplicitSINDyPI` backend.

## Migration notes

- No breaking changes to existing Dict-based network definitions.
- Recommended: migrate fixtures to typed metadata for clearer compile errors.
- `BioDynaX.PACKAGE_VERSION` is now the single source of truth for artifact metadata.
