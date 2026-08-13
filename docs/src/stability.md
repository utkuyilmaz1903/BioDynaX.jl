# API stability (1.0 freeze)

v0.9 is the honesty + recovery release. **v1.0 is not cut until the recovery
suite is green in CI with locked thresholds** (linear, MM, Hill, competitive,
UDE→discovery, graph-local ablation).

## Frozen before 1.0

These names are the supported surface. Breaking changes require a major bump
after 1.0:

- `BiologicalNetwork`, `NodeSpec`, `EdgeSpec`, `ReactionSpec`
- `UDEModel`, `build_ude_model`, `compile_mechanism`, `ude_system`, `ude_rhs!`
- `SciMLBase.ODEProblem(::UDEModel, ...)`, `SciMLBase.solve(::UDEModel, ...)`
- `train_ude`, `TrainingResult`, `TrainingRetcode`
- `discover_equations`, `DiscoveryResult`, `DiscoveryRetcode`
- `export_rhs`, `equation_to_latex`, `local_basis`
- `Experiment`, `experiment_from_csv`

## Registry

`Project.toml` is General-registry ready (name, uuid, version, compat, extras).
Registration is a maintainer action (`Registrator` / `Pkg.jl`), not part of CI.

## Documenter

`checkdocs = :exports` with `warnonly = [:missing_docs]` until public docstrings
cover the freeze list. Tutorials must not rely on undocumented internals.
