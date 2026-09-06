# Extensions

BioDynaX has nine package extensions. Each loads automatically when its
trigger packages are loaded in the same session. The Catalyst input,
the `Symbolics` output, and the ModelingToolkit export are exported (or
documented) entry points of the package; the others are experimental and
unexported: call them as `BioDynaX.name` and expect their interfaces to
change.

| Trigger package | Extension | What it adds |
|---|---|---|
| `Catalyst` | `BioDynaXCatalystExt` | `network_from_reactionsystem` |
| `Symbolics` | `BioDynaXSymbolicsExt` | `symbolic` |
| `Latexify` (with `Symbolics`) | `BioDynaXLatexifyExt` | `latexify` of discovered rates |
| `ModelingToolkit` | `BioDynaXModelingToolkitExt` | `BioDynaX.export_mtk_system` |
| `CUDA` | `BioDynaXCUDAExt` | `BioDynaX.to_device` and `BioDynaX.gpu_execute` for `ExecutionConfig(backend = :gpu)` |
| `Plots` | `BioDynaXPlotsExt` | `BioDynaX.plot_training` |
| `SBML` | `BioDynaXSBMLExt` | `BioDynaX.import_sbml_network` |
| `SBMLToolkit` and `Catalyst` | `BioDynaXSBMLToolkitExt` | `BioDynaX.import_sbmltoolkit_network` |
| `DataDrivenSparse` | `BioDynaXDataDrivenSparseExt` | the `BioDynaX.DataDrivenSparseSTLSQ` discovery backend |

## Catalyst input

`network_from_reactionsystem(rs; unknown)` converts a Catalyst
`ReactionSystem` into a `BiologicalNetwork`: the same species in Catalyst's
order, the known rate laws compiled to the matching terms, and the reaction
named by `unknown` (an index into `Catalyst.reactions(rs)` or the string of
its `description` metadata) as the one unknown destruction term;
`unknown = nothing` compiles everything as known. Supported rates are a
parameter, `k * Y` for a species that is not a substrate, `hill(Y, v, K, n)`
with a literal integer `n`, and `mm(Y, v, K)`; anything else raises an error
naming the reaction and its rate. The [How-to recipes](howto.md) page has
the full example, and the test suite checks that the tutorial network
converted from Catalyst simulates and discovers exactly as the fixture
written by hand.

## Symbolics output and Latexify

`symbolic(candidate, names)`, `symbolic(result::DiscoveryResult, names)`, and
`symbolic(result::UnknownTermResult)` return the discovered rational rate as
a `Symbolics.Num` in the named variables (an `UnknownTermResult` uses its
network's state names). With `Latexify` loaded, `latexify(result)`,
`latexify(result, names)`, and `latexify(candidate, names)` render the same
expression. The test suite checks that the expression evaluated on the
sample grid matches the numeric candidate to floating-point tolerance.

## GPU

`ExecutionConfig(backend = :gpu)` moves the arrays of each experiment to the
GPU with `cu` before calling the user function. There is no batched GPU ODE
integration and no GPU training loop.

## Plotting

`BioDynaX.plot_training(training, times, observations, dense_times, truth, prediction; state_names)`
draws one panel per state with the ground truth, the observations, and the
model prediction, plus the objective history.

## ModelingToolkit

`BioDynaX.export_mtk_system(model; name, discovered)` converts the compiled
production and destruction terms into a ModelingToolkit `ODESystem` whose
states carry the network's node names. Neural terms appear as placeholder
variables `nn_i(t)` unless `discovered` (a `DiscoveryResult`, a candidate,
or an `UnknownTermResult`) is given, in which case the discovered rational
rate replaces the placeholder of the single unknown term and the system is
complete: `ODEProblem(complete(sys), u0, tspan, p)` solves it. For a fully
known network the exported right-hand sides equal Catalyst's own
`Catalyst.ode_model(rs)` (`convert(ODESystem, rs)` before Catalyst 16), which the test suite checks.

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
