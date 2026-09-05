# BioDynaX.jl

Research preview. Not v1.0. Unique claim: known graph, one unknown
destruction `D(z)`, practical `unidentifiable_edge`, gated residual and
recall. Not a general network solver.

[![CI](https://github.com/utkuyilmaz1903/BioDynaX.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/utkuyilmaz1903/BioDynaX.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research preview. Not v1.0.** One-hole instrument: known graph, compiled
known `P` / known `D`, exactly one unknown destruction `D(z)`. Compiled
dynamics are `du_i = P_i - D_i * u_i`. The current unique-claim hold is
the Q3 practical scale warning (`unidentifiable_edge`; coefficients are
not biological constants) together with a gated Q1 hybrid residual versus
observed data and Q5 true-monomial recall. Q3 is a local practical warning,
not the whole product and not a structural certificate. Trajectory residual
is not mechanistic recovery. Canonical Hill from a trained NN is closed.
This is not a general CRN solver or a global SINDy replacement. The
scientific contract is
the scientific scope description in the documentation
(Q1–Q7 stay conceptually separate; Q4 is a practical
functional-identifiability diagnostic, not a gate and not a formal
identifiability certificate; Q7 is reported held-out generalization
evidence, not an additional success gate). Nine ICs are generated once;
ICs 1–7 are used for training and ICs 8–9 are held out. M4-B
trained-UDE graph-local validation is implemented. PR smoke is not
trained-UDE scientific acceptance. M4-C remains pending future work.

Requires **Julia ≥ 1.10**.

[Documentation](https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/) for the
0.9.2 research preview.

---

## Product

1. **Scope** — known graph, compiled known kinetics, exactly one unknown
   destruction `D(z)`. Zero or two-or-more holes are out of claim; the golden
   path errors.
2. **Q3 scale warning (first printed block)** —
   `unidentifiable_edge` and `coefficients_are_biological_constants`. The
   flag is a local practical warning: Fisher condition number **or**
   `k_prod`/`D` scale cosine. It is not the product by itself and not a
   structural certificate. `k_prod` and the scale of `D(z)` are not
   separately identifiable from observed concentrations.
3. **Q1 fit (gated)** — hybrid residual versus observed data on the current
   protocol / training IC[1]. Not mechanistic recovery. Separate train
   (ICs 1..7) and holdout (ICs 8, 9) residuals are reported; they are
   not this gate.
4. **Q5 symbolic support (gated)** — true-monomial recall on grid-sampled
   learned `D`. Combined F1 is a skeleton floor (0.50), not 0.99. Extras
   `1` and `r` remain. `canonical_hill_from_nn` is false and closed.
5. **Q7 holdout (reported, not a gate)** — after a train-only fit on ICs
   1..7, residual and neural `D` error on ICs 8 and 9 are reported. The
   0.30 residual gate is not copied to holdout. Q7 is reported held-out
   generalization evidence, not an additional success gate. Q4 is a
   practical functional-identifiability diagnostic, not a success gate.

MM unknown edges gate NN RMSE and data residual only. Combined F1 from a
trained NN is not canonical Hill.

---

## Quick start (15 minutes)

**One command** (standalone / legacy example: seed 103, nine ICs
generated and still trained on this example path. This is **not** the
M2 recovery-suite train/holdout protocol. The unique-claim recovery
suite generates nine ICs once; ICs 1–7 are used for training and
ICs 8–9 are held out. Adam 100 / BFGS 50, regulator-grid discovery).
`BIODYNAX_SMOKE=1` is a 1-IC compile check, not that suite protocol.

```bash
julia --project=. examples/unknown_inhibition.jl
```

**Product block** (that command prints identifiability first; seed 103, zero
observation noise). The residual row is the standalone / legacy example,
which still trains all nine generated ICs. That path is **not** the M2
recovery-suite train/holdout protocol. The unique-claim suite M2
validated IC[1] `data_residual` is 0.004195 after training ICs 1–7.

| field | typical value |
|-------|---------------|
| `unidentifiable_edge` | `true` (gated) |
| `coefficients_are_biological_constants` | `false` |
| hybrid residual vs data | ≈ 0.003 (standalone / legacy example that trains all nine generated ICs; gated) |
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
| **Dynamics** | Compiled `P_i - D_i u_i` IR → `ude_system` / `ude_rhs!` |
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

Synthetic multi-IC data come from `generate_experiment_set`, which
compiles one ground-truth model and then calls
`generate_from_compiled_model` (the same `ODEProblem(model, u0, tspan, p)`
path) on every initial condition. Remapped multi-head and two-regulator
`D(S,I)` fixtures are generated together. Unique-claim suite sections
are admitted through `admit_recovery_suite_network` before training.

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
- **Identifiability (Q3)** is a practical Fisher/Jacobian warning (condition
  number or scale cosine), not a structural certificate and not the whole
  unique-claim hold. Trajectory residual is not mechanistic recovery.
- **Partial observation:** discovery from subsampled `D` plus hybrid residual versus data is gated. UDE training on missing states is not claimed.
- **No licensed experimental CSV** matches the unique-claim protocol (known graph, exactly one unknown destruction edge). Elowitz is a synthetic ODE fixture. Absence is the result.
- Target regime is **2–20 states** with a known interaction graph.
- A green `recovery` CI job is **necessary, not sufficient** for v1.0.
- **Q4** is a practical functional-identifiability diagnostic, not a
  success gate and not a structural identifiability certificate.
- **M4** (pending / future work, not implemented): robustness /
  trajectory-context validation.

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
