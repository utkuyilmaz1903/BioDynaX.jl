# Extensions

BioDynaX has six package extensions. Each loads automatically when its
trigger package is loaded in the same session. All of them are experimental
and unexported: call them as `BioDynaX.name`, expect their interfaces to
change, and do not rely on them for published results.

| Trigger package | Extension | What it adds |
|---|---|---|
| `CUDA` | `BioDynaXCUDAExt` | `BioDynaX.to_device` and `BioDynaX.gpu_execute` for `ExecutionConfig(backend = :gpu)` |
| `Plots` | `BioDynaXPlotsExt` | `BioDynaX.plot_training` |
| `ModelingToolkit` | `BioDynaXModelingToolkitExt` | `BioDynaX.export_mtk_system` |
| `SBML` | `BioDynaXSBMLExt` | `BioDynaX.import_sbml_network` |
| `SBMLToolkit` and `Catalyst` | `BioDynaXSBMLToolkitExt` | `BioDynaX.import_sbmltoolkit_network` |
| `DataDrivenSparse` | `BioDynaXDataDrivenSparseExt` | the `BioDynaX.DataDrivenSparseSTLSQ` discovery backend |

## GPU

`ExecutionConfig(backend = :gpu)` moves the arrays of each experiment to the
GPU with `cu` before calling the user function. There is no batched GPU ODE
integration and no GPU training loop.

## Plotting

`BioDynaX.plot_training(training, times, observations, dense_times, truth, prediction; state_names)`
draws one panel per state with the ground truth, the observations, and the
model prediction, plus the objective history.

## ModelingToolkit

`BioDynaX.export_mtk_system(model; name)` converts the compiled known
production and destruction terms into a ModelingToolkit `ODESystem`. Neural
terms appear as placeholder variables `nn_i(t)`; they are not differentiable
ModelingToolkit expressions.

## SBML

`BioDynaX.import_sbml_network(path)` builds a `BiologicalNetwork` from the
species, reactions, stoichiometry, and modifiers of an SBML file. Kinetic
laws (MathML) are not parsed; reactions with an explicit kinetic law compile
as unknown neural terms. `BioDynaX.import_sbmltoolkit_network(path)` goes
through SBMLToolkit and Catalyst and can recognize mass-action reactions.
Neither importer produces Hill or Michaelis-Menten metadata.

## DataDrivenSparse

`DiscoveryConfig(backend = BioDynaX.DataDrivenSparseSTLSQ(threshold = 1e-3))`
uses DataDrivenSparse's STLSQ solver for the explicit coefficient fit. The
graph-local library and the implicit rational discovery stay in BioDynaX. At
the time of writing DataDrivenSparse does not resolve against the
ModelingToolkit versions this package allows, so the extension cannot be
loaded in the package environment; `benchmark/probe_datadriven.jl` reports
the current state.

## Fisher identifiability

Not an extension, but also unexported and experimental:
`BioDynaX.assess_identifiability(model, p, data, times, u0, tspan)` computes
a Gauss-Newton Fisher information matrix over the physical parameters at a
fit, with parameter correlations, condition number, and credible intervals
(`BioDynaX.parameter_credible_intervals`,
`BioDynaX.estimate_parameter_uncertainty`). It is local arithmetic at one
parameter point and is not structural identifiability.
