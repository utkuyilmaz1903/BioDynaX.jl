# BioDynaX.jl

**Graph-guided Universal Differential Equations for biological networks.**

BioDynaX compiles a biological interaction graph into a positivity-preserving
UDE, fits it to time series, and recovers **graph-local rational kinetics** on
unknown edges. The unique claim is not “another SINDy” or “another UDE wrapper”:
it is *hybrid known/unknown mechanisms on a graph, with local implicit discovery*.

Requires **Julia ≥ 1.10**.

---

## What it does

1. **Define a network** — nodes, edges, and stoichiometric reactions with kinetic metadata (mass action, Hill, competitive inhibition, saturation, or neural unknowns).
2. **Compile a UDE** — production–destruction RHS `duᵢ = Pᵢ(u,p) − Dᵢ(u,p)·uᵢ` with non-negative rates.
3. **Train** — Adam (optional BFGS), SciMLSensitivity adjoints, optional soft constraints.
4. **Discover** — graph-local implicit SINDy-PI on the **unknown destruction rate** `D(z)`, then `compose_hybrid_rhs` to resimulate.

The default example is the **p53–Mdm2** feedback loop (`build_network()`). Fully known fixtures: `build_linear_test_network()`, `build_mm_test_network()`, `build_hill_recovery_network()`, `build_competitive_test_network()`.

---

## Design highlights

| Topic | Implementation |
|--------|----------------|
| **Dynamics** | Compiled `MechanismCompiler` IR → `ude_system` / `ude_rhs!` |
| **Unknown biology** | `NeuralDestructionTerm` with a softplus-headed Lux MLP |
| **Positivity** | States through `max(0, x)`; optional augmented Lagrangian |
| **Parameters** | `ComponentVector` with `phys` / `nn` axes |
| **Training** | `train_ude`, `train_experiments`, checkpoints |
| **Discovery** | `ImplicitSINDyPI` (default) and `ExplicitSTLSQ`; optional DataDrivenSparse backend |
| **SciML** | `ODEProblem(model, u0, tspan, p)` and OrdinaryDiffEq `solve` |

Synthetic data is generated from the **compiled mechanism** by default. The legacy Hill p53 ODE is available as `generator = :hill_p53_fixture` for misspecification experiments only.

---

## SciML integration

```julia
using BioDynaX, SciMLBase, OrdinaryDiffEq

model, p = build_ude_model(MersenneTwister(0), build_linear_test_network())
prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), p)
sol = solve(prob, Tsit5(); saveat = 0:0.5:10.0)
```

Training uses **`ZygoteAD`** by default. **`ProductionAD`** selects the in-place RHS for forward integration; adjoints remain Zygote-based.

---

## Installation

Until the package is in General, clone and instantiate:

```bash
git clone https://github.com/utkuyilmaz1903/BioDynaX.jl.git
cd BioDynaX.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Optional extensions (load the extra package; they are **not** required):

```julia
using BioDynaX
using CUDA              # experimental: array transfer only
using Plots             # plot_training
using ModelingToolkit   # export_mtk_system (known terms; NN is a placeholder)
using SBML              # import_sbml_network (stoichiometry; unknown kinetics stay unknown)
using SBMLToolkit       # import_sbmltoolkit_network
using DataDrivenSparse  # DataDrivenSparseSTLSQ backend
```

---

## Quick start

```julia
using BioDynaX, Random

rng = MersenneTwister(42)
network = build_linear_test_network()
model, params = build_ude_model(rng, network)

t_data, _, noisy, _ = generate_data(rng; network = network, noise_σ = 0.02)

result = train_ude(
    params, noisy, t_data, [0.2, 0.1], (0.0, 20.0), model;
    adam_iters = 100, bfgs_iters = 0, verbose = true)

discovery = discover_equations(result.params, model; verbose = true, strict = true)
rhs = export_rhs(discovery)
```

Unknown-edge recovery (CSV → train → discover → resimulate) lives in
[`examples/unknown_inhibition.jl`](examples/unknown_inhibition.jl) and
[`docs/src/tutorial.md`](docs/src/tutorial.md).

---

## Limitations (honest)

- **GPU** copies experiment arrays with `cu`. There is no batched GPU ODE/training stack.
- **SBML** import does not parse kinetic MathML into Hill/MM. Use SBMLToolkit for lowering; unrecognized rates become neural unknowns.
- **Identifiability** is a practical Fisher matrix at a fit (physical parameters only), not structural identifiability.
- **NN allocation**: zero-allocation RHS is gated for NN-free linear networks. Lux heads still allocate.
- **Discovery** can fail (denominator sign, empty support). Check `DiscoveryResult.retcode`; pass `strict = true` to throw. Unknown-edge recovery is gated on `D(z)` support F1 and hybrid residual versus **data**, not versus UDE derivatives.
- Target regime is **2–20 states** with a known interaction graph. This is not a general CRN or global SINDy replacement.

---

## Development

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
julia --project=. benchmark/recovery_suite.jl
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
```

CI runs the default test matrix on Windows and Linux (Julia 1.10 and latest),
plus Aqua/JET, docs, the allocation gate, and a dedicated Ubuntu **recovery**
job for unknown-edge Hill/MM `D(z)` gates.

---

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
