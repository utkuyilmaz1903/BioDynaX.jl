# BioDynaX.jl

**Research preview. Not v1.0.** Graph-guided hybrid UDEs: known kinetics stay
compiled, unknown destruction rates `D(z)` are fit with a neural head and
recovered by graph-local implicit SINDy-PI. This is not a general CRN solver
or a global SINDy replacement.

Requires **Julia ≥ 1.10**.

Docs are configured (`gh-pages` branch exists). The public site was **not
live** on 2026-08-14 (`https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/`
returned 404). A live URL will be written here only after HTTP 200. Until
then use [`docs/src/`](docs/src/). TagBot is configured; there is no tag, so
it is not proven.

---

## What it does

1. **Define a network** — nodes, edges, and stoichiometric reactions with kinetic metadata (mass action, Hill, competitive inhibition, saturation, or neural unknowns).
2. **Compile a UDE** — production–destruction RHS `duᵢ = Pᵢ(u,p) − Dᵢ(u,p)·uᵢ` with non-negative rates.
3. **Train** — Adam (optional BFGS), SciMLSensitivity adjoints, optional soft constraints.
4. **Discover** — graph-local implicit SINDy-PI on the **unknown destruction rate** `D(z)`, then `compose_hybrid_rhs` to resimulate versus **data**.

The gated UDE claim is true-monomial **recall** plus a hybrid RHS that fits
**data**, for **Hill-class** unknown destruction. Combined F1 from a trained
NN is not canonical Hill (extras `1` and `r` remain after a same-library
attempt). MM unknown edges gate NN RMSE and data residual only.

---

## Quick start (15 minutes)

**One command** (same protocol as the recovery CI job: 9 ICs, adam 100 / bfgs 50):

```bash
julia --project=. examples/unknown_inhibition.jl
```

**One table** (seed 103, zero observation noise):

| quantity | mertebe |
|----------|---------|
| hybrid residual vs data | ≈ 0.003 (gated) |
| true-monomial recall | 1.0 (gated) |
| combined support F1 | ≈ 0.57 (skeleton floor 0.50, not 0.99) |
| extras | `1`, `r` remain |

**One warning:** `unidentifiable_edge == true`. Coefficients are not
biological constants. `k_prod` and the scale of `D(z)` stay collinear.

**We do not claim:** canonical Hill from a trained NN; a wet-lab tool for one
noisy CSV and unknown topology; UDE training on missing states; a licensed
experimental series that matches this protocol (absence is the result).

Tutorial: [`docs/src/tutorial.md`](docs/src/tutorial.md).

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = true)
```

---

## Design highlights

| Topic | Implementation |
|--------|----------------|
| **Dynamics** | Compiled `MechanismCompiler` IR → `ude_system` / `ude_rhs!` |
| **Unknown biology** | `NeuralDestructionTerm` with a softplus-headed Lux MLP |
| **Positivity** | States through `max(0, x)`; optional augmented Lagrangian |
| **Parameters** | `ComponentVector` with `phys` / `nn` axes |
| **Training** | `train_ude`, `train_experiments` |
| **Discovery** | `ImplicitSINDyPI` (default) and `ExplicitSTLSQ` |
| **SciML** | `ODEProblem(model, u0, tspan, p)` and OrdinaryDiffEq `solve` |

---

## SciML integration

```julia
using BioDynaX, SciMLBase, OrdinaryDiffEq, Random

network = BiologicalNetwork(
    [NodeSpec(name = :A), NodeSpec(name = :B)],
    EdgeSpec[];
    reactions = [
        ReactionSpec(name = :b_drives_a,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :k_ba)),
        ReactionSpec(name = :a_decay,
                     stoichiometry = Dict(1 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_a)),
        ReactionSpec(name = :b_decay,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_b)),
    ])
model, p = build_ude_model(MersenneTwister(0), network)
prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), p)
sol = solve(prob, Tsit5(); saveat = 0:0.5:10.0)
```

That snippet constructs an ODE. It is not the unique discovery path.

---

## Installation

Until the package is in General, clone and instantiate:

```bash
git clone https://github.com/utkuyilmaz1903/BioDynaX.jl.git
cd BioDynaX.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Optional extensions are **experimental and unexported**. Do not use them in a
paper or a wet lab. Load the extra package only if you are extending the
package (`BioDynaX.export_mtk_system`, `BioDynaX.import_sbml_network`,
`BioDynaX.plot_training`, `BioDynaX.DataDrivenSparseSTLSQ`).

---

## Limitations (honest)

- **GPU** copies experiment arrays with `cu`. There is no batched GPU ODE/training stack.
- **SBML** import does not parse kinetic MathML into Hill/MM.
- **Identifiability** is a practical warning, not a structural certificate.
- **Partial observation:** discovery from subsampled `D` plus hybrid residual versus data is gated. UDE training on missing states is not claimed.
- **No licensed experimental CSV** matches the unique-claim protocol (known graph, ≤1 unknown destruction edge). Elowitz is a synthetic ODE fixture. Absence is the result.
- Target regime is **2–20 states** with a known interaction graph.
- A green `recovery` CI job is **necessary, not sufficient** for v1.0. See [API stability](docs/src/stability.md).

---

## Development

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
julia --project=. benchmark/recovery_suite.jl
julia --project=. benchmark/sindy_baseline.jl
julia --project=. benchmark/recovery_seeds.jl
julia --project=. benchmark/noise_grid.jl
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
