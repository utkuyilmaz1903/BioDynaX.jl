# Experimental APIs

**Do not use these in a paper or a wet lab.** They are not the unique path
and they are not a product. Names below are **not exported**; call them as
`BioDynaX.foo` if you are extending the package.

## Do not use

- **GPU** — `cu` array copy only. No batched GPU ODE integrator and no GPU
  training loop.
- **ModelingToolkit** — known terms may emit; neural heads are placeholder
  `nn_i(t)` variables, not differentiable MTK models.
- **SBML** — species and stoichiometry only. MathML does not lower to
  Hill/MM metadata. Explicit kinetic laws become `known = false`.
- **Fisher identifiability** — a Gauss–Newton matrix at a fit over physical
  parameters. Not structural identifiability.

## GPU (`CUDA`)

`ExecutionConfig(backend = :gpu)` copies experiment arrays with `cu`. Treat
this as an array-transport helper, not a SciML GPU stack.

## Identifiability

`BioDynaX.assess_identifiability` is experimental Fisher arithmetic.

Unknown-edge recovery reports a practical `k_prod` ↔ `D(z)` scale
collinearity (`BioDynaX.production_destruction_tradeoff`). The golden path
prints that warning. The Hill UDE recovery job requires
`unidentifiable_edge == true`. Pinning `k_prod`, normalizing sampled `D`, or
changing the production rate does **not** remove the Jacobian collinearity.
That is the locked finding. The flag is not a structural certificate and is
not on the freeze list.

`TrainingConfig(frozen_phys = [:k_prod])` pins named physical parameters
during Adam and restores them after BFGS. Use it when a production rate is
known from a separate assay. It is not an identifiability certificate.

## SBML

`BioDynaX.import_sbml_network` maps species and stoichiometry.
`BioDynaX.import_sbmltoolkit_network` lowers through SBMLToolkit + Catalyst
when those packages are loaded. Neither is a kinetic importer.

## ModelingToolkit

`BioDynaX.export_mtk_system` emits known production/destruction terms
symbolically. Neural heads stay placeholders.

## DataDrivenSparse

`BioDynaX.DataDrivenSparseSTLSQ` swaps the explicit coefficient solver.
Graph-local libraries and implicit rational discovery stay in BioDynaX.
This backend is never a CI dependency. A skip is not a win; frozen numbers
live in [Recovery benchmarks](benchmarks.md).
