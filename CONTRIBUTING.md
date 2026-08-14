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
`recovery` CI job (`test/run_recovery_hard.jl`).

## Docs, quality, benchmarks

```bash
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=pwd()); Pkg.add(["Aqua", "JET"]); include("test/quality.jl")'
julia --project=. benchmark/sindy_baseline.jl
julia --project=. benchmark/allocation_gate.jl
```

## Recovery thresholds

`RECOVERY_THRESHOLDS` in `src/Recovery.jl` is the scientific contract.

- **Loosening** a locked number is a **breaking** change.
- **Tightening** combined F1 toward the analytical gate (`support_f1_clean`)
  is the scientific goal.
- Do not grow the STLSQ dictionary to buy F1.
- Do not add DataDrivenDiffEq as a CI dependency.
- Do not write “canonical Hill from a trained NN” while UDE combined F1 sits
  below `support_f1_clean`. The locked UDE claim is recall + data residual.
- Unknown-edge closed loop is **Hill-class**. MM unknown is NN RMSE + residual;
  do not reuse `support_recall = 0.99` there.
- `unidentifiable_edge == true` on the Hill UDE hard job is a gated warning,
  not a claim that production and destruction are identified.

## Experimental API

PRs that expand GPU, SBML MathML, ModelingToolkit placeholders, or Fisher
identifiability are acceptable only as experimental surface. They must not
join the [stability freeze list](docs/src/stability.md) before v1.0.

Do not grow the public `export` list to silence Documenter `missing_docs`
warnings. Freeze-list names get docstrings; `warnonly = [:missing_docs]`
stays while exports remain a superset of the freeze list. `scripts/_*.jl`
debug runners are gitignored; they are not the product.

## v1.0 / JOSS / General

Do not cut v1.0, open a JOSS submission, or `]register` until every item in
`docs/src/stability.md` can turn CI red. Registration is a maintainer action,
never a CI step. A methods note is allowed only when every public sentence
matches a CI gate.
