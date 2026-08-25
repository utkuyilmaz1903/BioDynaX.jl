# BioDynaX.jl

[![CI](https://github.com/utkuyilmaz1903/BioDynaX.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/utkuyilmaz1903/BioDynaX.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research preview. Not v1.0.** One-hole instrument: known graph, compiled
known `P` / known `D`, exactly one unknown destruction `D(z)`. The product is
practical identifiability (`unidentifiable_edge`; coefficients are not
biological constants) together with a gated hybrid residual versus data and
true-monomial recall. Canonical Hill from a trained NN is closed. This is not
a general CRN solver or a global SINDy replacement.

Requires **Julia ≥ 1.10**.

Docs: [https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/](https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/)
(HTTP 200 on 2026-08-14 after Pages was pointed at `gh-pages` / `(root)`).
TagBot is configured; there is no git tag, so it is not proven. There is no
version badge until the 0.9.x preview is in General.

---

## Product

1. **Scope** — known graph, compiled known kinetics, exactly one unknown
   destruction `D(z)`. Zero or two-or-more holes are out of claim; the golden
   path errors.
2. **Identifiability (primary, first printed block)** —
   `unidentifiable_edge` and `coefficients_are_biological_constants`. `k_prod`
   and the scale of `D(z)` are not separately identifiable from observed
   concentrations. Practical Fisher/Jacobian, not StructuralIdentifiability.jl.
3. **Fit (gated)** — hybrid residual versus data and true-monomial recall on
   synthetic Hill truth.
4. **Discovery (tertiary)** — a symbolic `D(z)` string. Combined F1 is a
   skeleton floor (0.50), not 0.99. Extras `1` and `r` remain.
   `canonical_hill_from_nn` is false and closed.

MM unknown edges gate NN RMSE and data residual only. Combined F1 from a
trained NN is not canonical Hill.

---

## Quick start (15 minutes)

**One command** (same protocol as the recovery CI job: seed 103, 9 ICs,
adam 100 / bfgs 50, regulator-grid discovery). `BIODYNAX_SMOKE=1` is a
1-IC compile check, not that protocol.

```bash
julia --project=. examples/unknown_inhibition.jl
```

**Product block** (that command prints identifiability first; seed 103, zero
observation noise):

| field | mertebe |
|-------|---------|
| `unidentifiable_edge` | `true` (gated) |
| `coefficients_are_biological_constants` | `false` |
| hybrid residual vs data | ≈ 0.003 (gated) |
| true-monomial recall | 1.0 (gated) |
| combined support F1 | ≈ 0.57 (skeleton floor 0.50, not 0.99) |
| extras | `1`, `r` remain |
| `canonical_hill_from_nn` | `false` (closed) |

**We do not claim:** canonical Hill from a trained NN; a wet-lab tool for one
noisy CSV and unknown topology; UDE training on missing states; a licensed
experimental series that matches this protocol (absence is the result).

Tutorial: [`docs/src/tutorial.md`](docs/src/tutorial.md).

```julia
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, data, times, u0, tspan; term = term, verbose = false)
println(BioDynaX.format_protocol_result(ident; residual = residual))
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
- **Identifiability** is the product (practical Fisher/Jacobian). It is not a structural certificate.
- **Partial observation:** discovery from subsampled `D` plus hybrid residual versus data is gated. UDE training on missing states is not claimed.
- **No licensed experimental CSV** matches the unique-claim protocol (known graph, exactly one unknown destruction edge). Elowitz is a synthetic ODE fixture. Absence is the result.
- Target regime is **2–20 states** with a known interaction graph.
- A green `recovery` CI job is **necessary, not sufficient** for v1.0. See [API stability](docs/src/stability.md).

---

## Development

```bash
julia --project=. test/runtests.jl
BIODYNAX_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0 julia --project=. examples/unknown_inhibition.jl
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

MIT. See [LICENSE](LICENSE). See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
for the ColPrac / Contributor Covenant.
