# Contributing to BioDynaX.jl

This is a research-preview SciML package. The unique claim is graph-guided
hybrid UDE recovery of an unknown destruction rate `D(z)`, not a general CRN
or global SINDy replacement.

## Tests

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
```

The default test matrix must stay fast: do not put multi-minute UDE trains in
`test/runtests.jl`. Closed-loop unknown-edge trains belong in the dedicated
`recovery` CI job (`test/run_recovery_hard.jl`). `train_experiments` Adam may
be minibatched; BFGS always refines the joint loss over the full set. Do not
loosen `RECOVERY_THRESHOLDS` to paper over a training or Occam bug.

Multi-seed UDE (`benchmark/recovery_seeds.jl --ude`) is a report, not a CI
job. The red gate stays seed 103 / 104.

## Docs, quality, benchmarks

```bash
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=pwd()); Pkg.add(["Aqua", "JET"]); include("test/quality.jl")'
julia --project=. benchmark/sindy_baseline.jl
julia --project=. benchmark/allocation_gate.jl
julia --project=. benchmark/recovery_seeds.jl
julia --project=. benchmark/noise_grid.jl
```

Live docs: https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/
If that 404s, check **Settings → Pages → Deploy from branch `gh-pages` / **
`(root)`. TagBot is configured; do not claim it ran until a tag exists.

## Recovery thresholds

`RECOVERY_THRESHOLDS` in `src/Recovery.jl` is the scientific contract.

- **Loosening** a locked number is a **breaking** change.
- Combined F1 toward `support_f1_clean` was attempted on the same library
  (`benchmark/ude_f1_attempt.jl`). Extras remained. The locked UDE claim is
  recall + data residual until a new major gate.
- Do not grow the STLSQ dictionary to buy F1.
- Do not add DataDrivenDiffEq as a CI dependency.
- Do not write “canonical Hill from a trained NN”.
- Unknown-edge closed loop is **Hill-class**. MM unknown is NN RMSE + residual.
- `unidentifiable_edge == true` on the Hill UDE hard job is a gated warning.

## Public API

Export only the freeze list plus golden-path verbs. Recovery fixtures,
Fisher, GPU/SBML/MTK, and library internals stay `BioDynaX.foo`. Do not grow
the public `export` list to silence Documenter `missing_docs` warnings.
`scripts/_*.jl` debug runners are gitignored; they are not the product.

## Experimental API

PRs that expand GPU, SBML MathML, ModelingToolkit placeholders, or Fisher
identifiability are acceptable only as unexported experimental surface. They
must not join the [stability freeze list](docs/src/stability.md) before v1.0.
Do not use them in a paper or a wet lab.

## v1.0 / JOSS / General

Do not cut v1.0, open a JOSS submission, or `]register` as 1.0. A 0.9.x
research-preview register is a maintainer action after the gates in
`docs/src/stability.md` can turn CI red. README’s first sentence stays
“not v1.0”. A methods note is allowed only when every public sentence
matches a CI gate.
