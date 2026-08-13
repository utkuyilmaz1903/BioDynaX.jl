# API stability (1.0 freeze)

v0.9 is the honesty + recovery-infrastructure release. **v1.0 is not cut until
the hard recovery job is green with locked `RECOVERY_THRESHOLDS`** (known-kinetics
RMSE, unknown-edge Hill/MM `D(z)` F1 and data residual, graph vs global F1).

## Frozen before 1.0

These names are the supported surface. Breaking changes require a major bump
after 1.0:

- `BiologicalNetwork`, `NodeSpec`, `EdgeSpec`, `ReactionSpec`
- `UDEModel`, `build_ude_model`, `compile_mechanism`, `ude_system`, `ude_rhs!`
- `SciMLBase.ODEProblem(::UDEModel, ...)`, `SciMLBase.solve(::UDEModel, ...)`
- `train_ude`, `TrainingResult`, `TrainingRetcode`
- `discover_equations`, `discover_unknown_rate`, `DiscoveryResult`, `DiscoveryRetcode`
- `export_rhs`, `compose_hybrid_rhs`, `equation_to_latex`, `local_basis`
- `sample_unknown_destruction`, `Experiment`, `experiment_from_csv`

## Registry

`Project.toml` is General-registry ready (name, uuid, version, compat, extras).
Registration is a maintainer action (`Registrator` / `Pkg.jl`), not part of CI.

## Documenter

`checkdocs = :exports` with `warnonly = [:missing_docs]` until public docstrings
cover the freeze list. Tutorials must not rely on undocumented internals.
