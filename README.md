# BioDynaX.jl

**Research preview.** Graph-guided hybrid UDEs: known kinetics stay compiled,
unknown destruction rates `D(z)` are fit with a neural head and recovered by
graph-local implicit SINDy-PI. This is not a general CRN solver or a global
SINDy replacement. The unique path is gated synthetic 2-state unknown-edge
recovery; v1.0 is not this release.

Requires **Julia ≥ 1.10**.

---

## What it does

1. **Define a network** — nodes, edges, and stoichiometric reactions with kinetic metadata (mass action, Hill, competitive inhibition, saturation, or neural unknowns).
2. **Compile a UDE** — production–destruction RHS `duᵢ = Pᵢ(u,p) − Dᵢ(u,p)·uᵢ` with non-negative rates.
3. **Train** — Adam (optional BFGS), SciMLSensitivity adjoints, optional soft constraints.
4. **Discover** — graph-local implicit SINDy-PI on the **unknown destruction rate** `D(z)`, then `compose_hybrid_rhs` to resimulate versus **data**.

The recovered object is a rate that must resimulate. The gated UDE claim is
correct parents (true-monomial **recall**) plus a hybrid RHS that fits
**data**, for **Hill-class** unknown destruction. A printed equation is
canonical Hill only on analytical `D` samples (`support_f1_clean`); trained-NN
rates still keep extra lower-order terms in the same variable, so that
sentence is not the product claim. MM unknown edges gate NN RMSE and data
residual only.

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
using BioDynaX, SciMLBase, OrdinaryDiffEq, Random

model, p = build_ude_model(MersenneTwister(0), build_linear_test_network())
prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), p)
sol = solve(prob, Tsit5(); saveat = 0:0.5:10.0)
```

That snippet constructs an ODE. It is not the unique discovery path.

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

## Quick start (unknown-edge golden path)

CSV / time series → known graph with one unknown edge → `train_experiments`
→ `sample_unknown_destruction` → `discover_unknown_rate` →
`compose_hybrid_rhs` → resimulate versus data, then report practical
`k_prod`↔`D` collinearity.

Runnable copy (same multi-IC protocol as the recovery CI job: **9 ICs**,
**adam 100 / bfgs 50**, residual gate):

```bash
julia --project=. examples/unknown_inhibition.jl
```

Tutorial: [`docs/src/tutorial.md`](docs/src/tutorial.md). Do not use a
shortened single-`train_ude` snippet as the protocol.

After a fit, call (not exported; freeze list unchanged):

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = true)
```

Observed concentrations typically leave `k_prod` and the scale of `D(z)`
collinear (`unidentifiable_edge = true`). That warning is the gated finding.
It is not structural identifiability, and freeze/`D`-normalization do not
remove the Jacobian collinearity.

Do not call `discover_equations` on the full state derivative and treat that as
the product. Known production stays in compiled IR; STLSQ only sees `D(z)`.

---

## Limitations (honest)

- **GPU** copies experiment arrays with `cu`. There is no batched GPU ODE/training stack.
- **SBML** import does not parse kinetic MathML into Hill/MM. Use SBMLToolkit for lowering; unrecognized rates become neural unknowns.
- **Identifiability** is a practical Fisher matrix at a fit (physical parameters only), not structural identifiability. On the unknown-edge path, `k_prod`↔`D(z)` scale collinearity is reported and, for Hill UDE recovery, gated `unidentifiable_edge == true`.
- **NN allocation**: zero-allocation RHS is gated for NN-free linear networks. Lux heads still allocate.
- **Discovery** can fail (denominator sign, empty support). Check `DiscoveryResult.retcode`; pass `strict = true` to throw. Unknown-edge recovery is gated on `D(z)` support, recall, and hybrid residual versus **data**, not versus UDE derivatives. Canonical MM support is not claimed.
- Target regime is **2–20 states** with a known interaction graph. This is not a general CRN or global SINDy replacement.
- A green `recovery` CI job is **necessary, not sufficient** for v1.0. See [API stability](docs/src/stability.md).

---

## Development

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
julia --project=. benchmark/recovery_suite.jl
julia --project=. benchmark/sindy_baseline.jl
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
```

CI runs the default test matrix on Windows and Linux (Julia 1.10 and latest),
plus Aqua/JET, docs, the allocation gate, and a dedicated Ubuntu **recovery**
job for unknown-edge Hill/MM `D(z)` gates. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
