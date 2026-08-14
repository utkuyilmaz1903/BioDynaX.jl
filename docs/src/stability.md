# API stability (1.0 freeze)

v0.9.x is a **research preview**. A green `recovery` CI job is **necessary, not
sufficient** for v1.0. Green CI today proves the method skeleton (hybrid compile,
`D(z)` discovery, graph vs global ablation, 6-state prior), not that every
printed equation is canonical Hill, and not that a biologist should run this
on an arbitrary CSV.

## Live docs and TagBot (honest)

- Documenter `deploydocs` is configured and pushes `gh-pages` from `main`
  when `CI=true`. The `gh-pages` branch exists.
- Live docs (HTTP 200 on 2026-08-14):
  [https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/](https://utkuyilmaz1903.github.io/BioDynaX.jl/dev/).
  Pages source is `gh-pages` / `(root)`. The repo-root URL without `/dev/`
  is not the Documenter site.
- TagBot and CompatHelper are **configured**. There is **no git tag**, so
  TagBot has not been proven. The first tag after a 0.9.x preview register
  is the proof. Do not write “TagBot ran”.
- `DOCUMENTER_KEY` remains an optional SSH fallback. The docs job uses
  `GITHUB_TOKEN` with `contents: write`.

## v1.0 is not cut until all of the following can fail CI

- Unknown-edge UDE combined support F1 is **not** the analytical Hill gate.
  One same-library attempt (`benchmark/ude_f1_attempt.jl`) left extras `1`
  and `r`. The locked UDE claim is true-monomial recall + hybrid residual
  versus data on **Hill-class** unknown destruction. Re-opening a
  “canonical Hill from a trained NN” sentence requires a new major scientific
  gate, not a README edit.
- External graph vs global (optional DataDrivenSparse) baseline table is frozen
  in `docs/src/benchmarks.md`. Loosening a locked number is breaking. The
  2-state F1 gap is **not** the prior. The locked prior is library membership
  of the distractor, 3-state and **6-state** true-parent membership, and
  wrong-graph negative controls.
- Identifiability of `k_prod` vs `D(z)` is a **user-facing warning**. The Hill
  UDE hard job requires `unidentifiable_edge == true`.
- Partial observation: subsampled `D` → hybrid residual versus data is green
  or honestly red. UDE training on missing states is **not** claimed.
- No licensed experimental time series matches the unique-claim protocol
  (known graph, ≤1 unknown destruction edge, redistributable license).
  Absence is the result. Elowitz is a synthetic ODE fixture.
- TagBot and CompatHelper are configured. Live Pages is proven (HTTP 200).
  A TagBot tag is still unproven until the first 0.9.x preview tag.
- `]register` is a maintainer action after those gates, not a CI step.

Loosening `RECOVERY_THRESHOLDS` is a breaking change. Tightening UDE combined
F1 toward `support_f1_clean` was attempted on the same library and did not
hold. The scientific claim stays recall + residual until a new major gate.

## JOSS / register (maintainer gate, not this work)

All of the following must already be true, and CI must be able to go red if
any of them regresses:

1. The public claim is recall + data residual on Hill-class unknown edges
   (this preview). A canonical-Hill-from-NN sentence is closed until a new
   major gate.
2. Graph vs global (optional DataDrivenSparse) table is frozen in
   [Recovery benchmarks](benchmarks.md). 6-state + wrong-graph are the prior
   evidence beyond the 3-state toy. Loosening a locked number is breaking.
3. Identifiability (`unidentifiable_edge`), graph-prior, wrong-graph
   negative control, and partial-observation discovery→residual are green or
   honestly red (no silent skip).
4. TagBot and CompatHelper are configured. Live Pages is HTTP 200 on
   `/dev/`. A TagBot tag is still required before claiming TagBot ran.
5. A methods note is allowed only when every sentence matches a CI gate.
   JOSS is not this preview.
6. `]register` is a maintainer action for **0.9.x research-preview** only.
   Do not register v1.0 against this bar. README’s first sentence stays
   “not v1.0”.

Until then this repository stays on the 0.9.x research-preview line. This
preview does **not** cut v1.0 or open JOSS. A 0.9.x General preview register
is the next maintainer action and is **not yet in General**. Do not write
`]add BioDynaX` or “TagBot ran” until the General PR is merged and tag
`v0.9.2` exists.

## Frozen before 1.0

These names are the supported surface. Breaking changes require a major bump
after 1.0:

- `BiologicalNetwork`, `NodeSpec`, `EdgeSpec`, `ReactionSpec`
- `UDEModel`, `build_ude_model`, `compile_mechanism`, `ude_system`, `ude_rhs!`
- `SciMLBase.ODEProblem(::UDEModel, ...)`, `SciMLBase.solve(::UDEModel, ...)`
- `train_ude`, `train_experiments`, `TrainingResult`, `TrainingRetcode`
- `discover_equations`, `discover_unknown_rate`, `DiscoveryResult`, `DiscoveryRetcode`
- `export_rhs`, `compose_hybrid_rhs`, `equation_to_latex`, `local_basis`
- `sample_unknown_destruction`, `Experiment`, `experiment_from_csv`

Golden-path verbs (`generate_experiment_set`, `predict_ude`,
`equation_to_function`, `hybrid_data_residual`, `RECOVERY_THRESHOLDS`,
metadata types) are exported so a stranger can run the tutorial. They are
not a promise that every helper is freeze-stable.

GPU, SBML, ModelingToolkit placeholders, and Fisher identifiability are
[experimental](experimental.md) and **unexported**.
`production_destruction_tradeoff` is experimental (call as
`BioDynaX.production_destruction_tradeoff`).
Recovery suite builders (`run_recovery_suite`, `build_*_recovery_network`)
are internal fixtures.

## Registry

`Project.toml` is General-registry ready (name, uuid, version, compat, extras,
description). A 0.9.x preview register is a maintainer action (`Registrator`),
not part of CI, and is **not yet done**. Do not register v1.0 against this
preview bar.

## Documenter

`checkdocs = :exports` with `warnonly = [:missing_docs]`. The public export
list is the freeze list plus golden-path verbs. Do not grow exports to
silence missing-docs warnings. `deploydocs` runs only on `main` when
`CI=true`.
