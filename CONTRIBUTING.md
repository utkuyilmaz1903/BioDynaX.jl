# Contributing to BioDynaX.jl

This is a research-preview SciML package. The unique claim is graph-guided
hybrid UDE recovery of an unknown destruction rate `D(z)`, not a general CRN
or global SINDy replacement. Community norms are in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Tests

```bash
julia --project=. test/runtests.jl
julia --project=. test/run_recovery_hard.jl
BIODYNAX_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0 julia --project=. examples/unknown_inhibition.jl
```

The default test matrix must stay fast: do not put multi-minute UDE trains in
`test/runtests.jl`. `BIODYNAX_SMOKE=1` is **1 IC / 8 points**, not the 9-IC
seed-103 / regulator-grid recovery protocol; that string check stays in
`test/test_recovery.jl`.
The golden path is `examples/unknown_inhibition.jl`.
`scripts/run_discovery.jl` is a debug runner, not the product.
Closed-loop unknown-edge trains belong in the dedicated `recovery` CI job
(`test/run_recovery_hard.jl`). `train_experiments` Adam may be minibatched;
BFGS always refines the joint loss over the full set. Do not loosen
`RECOVERY_THRESHOLDS` to paper over a training or Occam bug.

Multi-seed UDE (`benchmark/recovery_seeds.jl --ude`) is a report, not a CI
job. The red gate stays seed 103 / 104.

## Docs, quality, benchmarks

```bash
julia --project=docs docs/instantiate.jl
julia --project=docs docs/make.jl
julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=pwd()); Pkg.add(["Aqua", "JET"]); include("test/quality.jl")'
julia --project=. benchmark/sindy_baseline.jl
julia --project=. benchmark/probe_datadriven.jl
julia --project=. benchmark/allocation_gate.jl
julia --project=. benchmark/recovery_seeds.jl
julia --project=. benchmark/noise_grid.jl
```

Live docs: https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/
TagBot is configured; do not claim a release tag until one exists.

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
`scripts/run_discovery.jl` is a committed debug runner around
`examples/unknown_inhibition.jl`; do not treat it as the unique-claim
entry point. Unexported lock surfaces versus runtime wiring
(`train_experiments_with_warmup`, `denominator_split_counts`) are listed
in [internal-workspaces](docs/src/internal-workspaces.md).

## Experimental API

PRs that expand GPU, SBML MathML, ModelingToolkit placeholders, or Fisher
identifiability are acceptable only as unexported experimental surface. They
must not join the [stability freeze list](docs/src/stability.md) before v1.0.
Do not use them in a paper or a wet lab.

New files should match `.JuliaFormatter.toml` (SciML style). Do not
reformat the whole tree in a drive-by PR. New Julia files that the
format job lists (stacked workspaces, science-honesty files, and their
tests) must stay `overwrite=false` clean.

`generate_data` must keep the compiled NN tree (multi-head and
multi-regulator). Do not restore a 1-input dummy. `generate_experiment_set`
must compile the ground-truth model once and generate every IC from that
stored model. Unique-claim suite sections go through
`admit_recovery_suite_network`; do not put that single-hole check into
`validate_network`. New files that the format job lists
(`src/CompiledPath.jl`, `test/test_compiled_path.jl`) must stay
`overwrite=false` clean.

## v1.0 / JOSS / General

Do not cut v1.0, open a JOSS submission, or register as 1.0. README’s first
sentence stays “not v1.0”. A methods note is allowed only when every public
sentence matches a CI gate. The 0.9.2 preview is **not yet in General**.
Do not write a General-registry install command until the General PR is
merged and tag `v0.9.2` exists.

Code-side register prerequisites from the SciML hardening pass (strict
`checkdocs`, doctests, example smoke, export docstrings, ColPrac files)
are in the tree. Registrator + first tag are still maintainer actions.

### 0.9.2 preview register (maintainer)

1. Confirm CI on `main` is green, including `recovery`.
2. Install [JuliaRegistrator](https://github.com/apps/juliaregistrator) on
   `utkuyilmaz1903/BioDynaX.jl` only. Fallback:
   [JuliaHub Registrator](https://juliahub.com/ui/Registrator).
3. Open an issue titled exactly `TagBot trigger issue` (leave it open).
4. On the pushed `main` HEAD **commit** page, comment:

```
@JuliaRegistrator register

Release notes:

Research preview 0.9.2, not v1.0. Graph-guided hybrid UDE recovery of unknown destruction D(z). The gated claim is true-monomial recall plus hybrid residual versus data on Hill-class edges, not canonical Hill from a trained NN. See CHANGELOG.md.
```

5. Do not comment on `JuliaRegistries/General`. Do not pass `branch=`.
6. Follow the General PR. New-package AutoMerge waits ~3 days. A comment
   without `[noblock]` blocks AutoMerge.
7. If AutoMerge fails on compat or load: fix, commit, repeat the comment on
   the new commit. Do not loosen `RECOVERY_THRESHOLDS` or grow exports.
8. If the name-similarity check fails: do not rename. Explain on the PR.
9. After merge + tag `v0.9.2` + a clean `]add BioDynaX`, record those facts
   in README and `docs/src/stability.md`. Do not bump the version for that
   docs commit.
