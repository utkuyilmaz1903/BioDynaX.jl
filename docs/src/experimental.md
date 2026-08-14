# Experimental APIs

These entry points exist so the package can compose with SciML later. They are
**not** the product.

## GPU (`CUDA`)

`ExecutionConfig(backend = :gpu)` copies experiment arrays with `cu`. There is
no batched GPU ODE integrator or GPU training loop. Treat this as an array
transport helper.

## Identifiability

`assess_identifiability` builds a Gauss–Newton Fisher matrix from a
finite-difference trajectory Jacobian over **physical** parameters at a fit.
It is not structural identifiability.

Unknown-edge recovery reports a practical `k_prod` ↔ `D(z)` scale collinearity
(`BioDynaX.production_destruction_tradeoff`). The golden path prints that
warning (`BioDynaX.report_production_destruction_tradeoff`). The Hill UDE
recovery job requires `unidentifiable_edge == true`. Pinning `k_prod` with
`TrainingConfig(frozen_phys = [:k_prod])`, normalizing sampled `D`, or
changing the production rate does **not** remove the Jacobian collinearity;
that is the locked finding. The flag is not a structural certificate and is
not on the freeze list.

`TrainingConfig(frozen_phys = [:k_prod])` pins named physical parameters
during Adam (gradient zeroed) and restores them after BFGS. Use it when a
production rate is known from a separate assay. It is not a structural
identifiability certificate.

## SBML

`import_sbml_network` maps species and stoichiometry. Kinetic MathML is not
parsed into Hill/MM metadata; explicit kinetic laws become `known = false`.
`import_sbmltoolkit_network` lowers through SBMLToolkit + Catalyst when those
packages are loaded.

## ModelingToolkit

`export_mtk_system` emits known production/destruction terms symbolically.
Neural heads are placeholder `nn_i(t)` variables, not differentiable MTK models.

## DataDrivenSparse

`DataDrivenSparseSTLSQ` swaps the explicit coefficient solver for DataDrivenSparse
`STLSQ`. Graph-local libraries and implicit rational discovery stay in BioDynaX.

```@docs
assess_identifiability
IdentifiabilityReport
DataDrivenSparseSTLSQ
GPUBackend
```
