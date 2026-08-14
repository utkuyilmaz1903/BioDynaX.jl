# API stability (1.0 freeze)

v0.9.x is a **research preview**. A green `recovery` CI job is **necessary, not
sufficient** for v1.0. Green CI today proves the method skeleton (hybrid compile,
`D(z)` discovery, graph vs global ablation), not that every printed equation is
canonical Hill, and not that a biologist should run this on an arbitrary CSV.

## v1.0 is not cut until all of the following can fail CI

- Unknown-edge UDE combined support F1 is **not** the analytical Hill gate.
  The locked UDE claim is true-monomial recall + hybrid residual versus data
  on **Hill-class** unknown destruction. Analytical Occam F1
  (`support_f1_clean`) stays 0.99 on exact / 0.5% `D`. Re-opening a
  “canonical Hill from a trained NN” sentence requires a new major scientific
  gate, not a README edit.
- External graph vs global (optional DataDrivenDiffEq) baseline table is frozen
  in `docs/src/benchmarks.md`. Loosening a locked number is breaking. The
  2-state F1 gap is **not** the prior: after Occam both F1s can be 1.00. The
  locked prior is library membership of the distractor, 3-state true-parent
  membership, and a wrong-graph negative control.
- Identifiability of `k_prod` vs `D(z)` is a **user-facing warning**. The Hill
  UDE hard job requires `unidentifiable_edge == true`. Freeze / normalize /
  production perturbation do not claim to break that Jacobian collinearity.
- Partial observation: subsampled `D` → hybrid residual versus data is green
  or honestly red. UDE training on missing states is **not** claimed.
- TagBot, CompatHelper, and Documenter `gh-pages` are **configured**. They are
  not proven live until `DOCUMENTER_KEY` exists and a `gh-pages` site URL is
  recorded here. Until then write “configured, not deployed”.
- `]register` is a maintainer action after those gates, not a CI step.

Loosening `RECOVERY_THRESHOLDS` is a breaking change. Tightening F1 toward the
analytical gate is the scientific goal.

## JOSS / register (maintainer gate, not this work)

All of the following must already be true, and CI must be able to go red if
any of them regresses:

1. The public claim is recall + data residual on Hill-class unknown edges
   (this preview), **or** UDE combined F1 later reaches the analytical gate
   and every sentence is updated.
2. Graph vs global (optional DataDrivenSparse) table is frozen in
   [Recovery benchmarks](benchmarks.md). 3-state + wrong-graph are the prior
   evidence. Loosening a locked number is breaking.
3. Identifiability (`unidentifiable_edge`), 3-state graph-prior, wrong-graph
   negative control, and partial-observation discovery→residual are green or
   honestly red (no silent skip).
4. TagBot, CompatHelper, and Documenter `gh-pages` are configured. Live
   deploy is a secret/`gh-pages` proof, not a YAML file alone.
5. A JOSS paper, if written, is the method plus the closed-loop table. It is
   not a general CRN/SINDy replacement. A methods note is allowed only when
   every sentence matches a CI gate.
6. `]register` is a maintainer action. CI does not register.

Until then this repository stays on the 0.9.x research-preview line. This
preview does **not** cut v1.0, open JOSS, or register.

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

GPU, SBML, ModelingToolkit placeholders, and Fisher identifiability are
[experimental](experimental.md). They are not on the unique path.
`production_destruction_tradeoff` is experimental (call as
`BioDynaX.production_destruction_tradeoff`); it is not on this freeze list.
Recovery suite builders (`run_recovery_suite`, `build_*_recovery_network`)
are internal fixtures, not freeze names.

## Registry

`Project.toml` is General-registry ready (name, uuid, version, compat, extras,
description). Registration is a maintainer action (`Registrator` / `Pkg.jl`),
not part of CI. Do not register v1.0 against this preview bar.

## Documenter

`checkdocs = :exports` with `warnonly = [:missing_docs]` because the public
export list is a **superset** of the freeze list. Freeze-list names have
docstrings. Do not grow the public export list to “fix” missing-docs warnings.
`deploydocs` runs only on `main` when `CI=true`. Without `DOCUMENTER_KEY` the
docs job is **build-only**.
