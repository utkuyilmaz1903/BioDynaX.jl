# Contributing to BioDynaX.jl

Thank you for considering a contribution. BioDynaX is a research package
with a deliberately narrow scope: hybrid models of small biochemical
networks with a known interaction graph and exactly one unknown destruction
term. Please read the "Scope and limitations" section of the README before
proposing a feature. Community norms are in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

This repository follows the [ColPrac](https://github.com/SciML/ColPrac)
guide on collaborative practices for community packages: open an issue
before a large change, keep pull requests focused, add tests and a
changelog entry with every change, and expect review before a merge. Code
is formatted with JuliaFormatter in the
[SciML style](https://github.com/SciML/SciMLStyle); CI checks that the tree
is formatted.

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
BIODYNAX_TEST_HEAVY=1 julia --project=. -e 'using Pkg; Pkg.test()' # plus the multi-minute training-loop tests
BIODYNAX_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0 julia --project=. examples/unknown_inhibition.jl
julia --project=. test/run_recovery_hard.jl                       # trained-model recovery (about 10 minutes on 4 cores)
julia --project=docs docs/instantiate.jl && julia --project=docs docs/make.jl
julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=pwd()); Pkg.add(["Aqua", "JET"]); include("test/quality.jl")'
```

The default test suite must stay fast. Put multi-minute training runs behind
`BIODYNAX_TEST_HEAVY=1` (see `test/test_experiment_checkpoint.jl`), in
`test/run_recovery_hard.jl`, or in a `benchmark/` script. CI runs the default
suite on every push and pull request and the heavy tier weekly and on demand.

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

The whole tree is formatted; CI checks this on every pull request.

## Pull requests

Keep pull requests focused. Describe what changed and how you checked it.
Update `CHANGELOG.md` under `[Unreleased]` for anything a user would notice.
