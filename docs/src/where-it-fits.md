# Where BioDynaX fits

BioDynaX sits between three tools a SciML user already knows. The table says
what each takes as input, what it returns, and when to use which. It is
factual; the tools are complementary and BioDynaX depends on two of them.

| | Input | Output | Use it when |
|---|---|---|---|
| **Plain universal differential equation** (the SciMLSensitivity "missing physics" tutorial: an ODE written by hand with a neural network in it, trained with Optimization.jl and the adjoint sensitivities of SciMLSensitivity) | The full ODE written by hand, including where the network sits and what it reads; the data; an optimiser and its schedule | Trained network weights and parameters; whatever you then do with the network is up to you (the tutorial hands its outputs to DataDrivenDiffEq) | The missing part can be anywhere in the model, the model is not a reaction network, or you want full control over the architecture, the loss, and the training loop |
| **DataDrivenDiffEq.jl with DataDrivenSparse.jl** | Trajectories (or a trained network's input–output samples) and a basis of candidate terms; derivative estimates when you fit `dx/dt` | Sparse symbolic equations over the basis (STLSQ, implicit SINDy, and others), as `Symbolics` expressions with coefficients | You already have samples of the quantity to identify and want to choose the basis, the sparsity algorithm, and the validation yourself |
| **Catalyst.jl alone** | A reaction network in its DSL: species, reactions, rate laws, parameters | A `ReactionSystem` that converts to ODE, SDE, and jump problems; parameter estimation through other packages | The mechanism is fully known and the task is simulation, analysis, or fitting the parameters of known rate laws |
| **BioDynaX.jl** | A network with a known interaction graph and known kinetics, written as a `BiologicalNetwork` or as a Catalyst `ReactionSystem` with one reaction named as unknown, plus time series from one or more initial conditions | A trained hybrid model (compiled known terms plus one neural destruction term), a rational rate for the unknown term from a graph-local library, the identifiability diagnostic, residuals on training and held-out experiments, and the completed model as a ModelingToolkit system or a `Symbolics` expression | Exactly one destruction term of a small network is unknown, its regulators are known from the graph, and you want the whole chain (train, sample, discover, check, resimulate) with recorded defaults and a report, rather than assembling it by hand |

What BioDynaX adds to the plain-UDE route is the structure it imposes and
checks: the unknown term is a per-concentration destruction rate `D(z)` of
one species regulated by its graph parents, so the library the sparse
regression searches is small and the recovered term has a fixed meaning;
the production/destruction scale that this structure leaves free is
reported by an identifiability diagnostic; and a rational rate whose
denominator changes sign on the data is refused rather than returned. What
it takes away is generality: no unknown production terms, no unknown term
in more than one species, no rate laws the compiler has no term for (the
[Scope and limitations](limitations.md) page lists them).

Inside, BioDynaX uses the same pieces: OrdinaryDiffEq for the solves,
SciMLSensitivity for the adjoints, Lux for the network, Optimization.jl
with Adam and BFGS for training, and its own implicit sparse regression with
DataDrivenSparse available as an alternative backend through an extension.
Catalyst models come in through `network_from_reactionsystem`, and results
go out through `export_mtk_system` and `symbolic`.

The [Benchmarks](benchmarks.md) page reports what the graph-local library
buys over a global one on the package's own fixtures, and the two case-study
pages report what happened on measured data, including the runs in which no
rational rate was accepted.
