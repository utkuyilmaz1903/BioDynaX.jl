# Contributing to BioDynaX.jl

Thank you for considering a contribution. BioDynaX is a research package
with a deliberately narrow scope: hybrid models of small biochemical
networks with a known interaction graph and exactly one unknown destruction
term. Please read the "Scope and limitations" section of the README before
proposing a feature. Community norms are in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Setting up

```bash
git clone https://github.com/utkuyilmaz1903/BioDynaX.jl.git
cd BioDynaX.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Julia 1.10 or newer is required. Installing the dependencies needs network
access to the Julia package server (`pkg.julialang.org`) or, failing that, to
GitHub for git-based package downloads.

## Running the checks

```bash
julia --project=. -e 'using Pkg; Pkg.test()'                      # default test suite
BIODYNAX_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0 julia --project=. examples/unknown_inhibition.jl
julia --project=. test/run_recovery_hard.jl                       # trained-model recovery (about 40 minutes)
julia --project=docs docs/instantiate.jl && julia --project=docs docs/make.jl
julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=pwd()); Pkg.add(["Aqua", "JET"]); include("test/quality.jl")'
```

The default test suite must stay fast. Do not add multi-minute training runs
to `test/runtests.jl`; put them in `test/run_recovery_hard.jl` or a
`benchmark/` script instead.

## Ground rules for scientific changes

- `RECOVERY_THRESHOLDS` in `src/Recovery.jl` defines the acceptance criteria
  for the recovery benchmarks. Loosening a threshold is a breaking change and
  needs its own pull request with the evidence for it.
- Do not change seeds, initial-condition counts, the train/holdout split, or
  optimizer settings of the reference protocol to make a test pass.
- Do not enlarge the discovery library to raise a support-F1 number.
- Do not add DataDrivenDiffEq or DataDrivenSparse as a test dependency.
- A failing recovery run is a result to report, not to hide.

## Public API

Exported names are documented on the API page of the documentation and are
covered by the test suite. New functionality should start unexported
(`BioDynaX.foo`) and be documented on the extensions or how-to page. Do not
add exports only to silence Documenter warnings. The GPU, SBML,
ModelingToolkit, and DataDrivenSparse extensions are experimental.

## Style

The repository uses [JuliaFormatter](https://github.com/domluna/JuliaFormatter.jl)
with the SciML style (`.JuliaFormatter.toml`). Format the files you touch:

```bash
julia -e 'using JuliaFormatter; format(["src", "ext", "test", "examples", "benchmark", "docs"])'
```

## Pull requests

Keep pull requests focused. Describe what changed and how you checked it.
Update `CHANGELOG.md` under `[Unreleased]` for anything a user would notice.
