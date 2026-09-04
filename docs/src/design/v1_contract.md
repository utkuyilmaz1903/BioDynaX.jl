# BioDynaX v1.0 scientific contract

This page is the authoritative scientific contract for BioDynaX **as
implemented today**. It is not the long-term charter in
`docs/research/BIO-DYNAX-VISION.md`.
The package version is still a **0.9.2 research preview**. This contract
does not cut v1.0, register the package, or reopen a closed claim.

The vision form is not the v1.0 product.

## Implemented dynamics

The compiler and `ude_system` / `ude_rhs!` integrate

```math
\dot{u}_i = P_i(u,p,t) - D_i(u,p,t)\, u_i.
```

Unknown biology on the unique-claim path is exactly one multiplicative
destruction rate \(D_i\geq 0\), implemented as a `NeuralDestructionTerm`
(softplus head). It is not an additive residual \(f_{\mathrm{known}}+D\).

Known production and known destruction stay compiled kinetics. They are
not replaced by a neural net.

## Scientific scope (today)

In scope for the unique-claim workflow:

- a user-supplied known biological graph (`BiologicalNetwork`)
- known mechanisms compiled into \(P\) and known \(D\)
- exactly one unknown destruction mechanism
- a nonnegative learned destruction rate
- multi-experiment **training** on unique-claim ICs 1..7
- practical identifiability diagnostics (Q3)
- a practical functional-identifiability diagnostic (Q4; not a gate
  and not a certificate)
- mechanistic recovery diagnostics on a train-derived regulator grid
  (Q2, partial)
- symbolic support recovery (Q5)
- architectural biological / numerical constraints (Q6, partial)
- reported held-out generalization evidence on ICs 8 and 9 (Q7;
  reported, not a gate)
- a conceptual distinction among predictive fit, mechanism-function
  diagnostics, and later validation questions

`validate_network` does not count holes. Zero or two-or-more unknown
destructions still compile. The unique-claim path separately requires
`assert_single_unknown_destruction`.

Held-out validation is **reported evidence**. It is not a success gate
and not a functional-identifiability certificate.

## Q1–Q7 (kept conceptually separate)

These questions must not be collapsed.
Trajectory fit is not proof of mechanism recovery.

The contract distinguishes these questions. That is not a claim that the
implementation already supplies the full operational Q1/Q2/Q4/Q7
separation. Q4 is implemented as a practical diagnostic, not a gate.
Q7 is reported held-out generalization evidence, not an additional success
gate. Do not read the table as a functional-identifiability certificate.

| Id | Question | What it measures | Status |
|----|----------|------------------|--------|
| Q1 | Predictive fit | Hybrid residual of `compose_hybrid_rhs` versus **observations** | `implemented (training IC[1] residual + separate train/holdout evidence)` |
| Q2 | Mechanism function recovery | \(\hat D(z)\) versus \(D_{\mathrm{true}}(z)\) | `partial (regulator-grid D error + reported holdout D error)` |
| Q3 | Scale / parameter practical identifiability | Local Fisher / \(k_{\mathrm{prod}}\leftrightarrow D\) Jacobian cosine | `implemented as practical warning` |
| Q4 | Practical functional-identifiability diagnostic | Agreement of independently trained \(\hat D_i(z)\) versus trajectory agreement | `implemented as a practical diagnostic, not a gate` |
| Q5 | Symbolic support recovery | True-monomial recall; combined F1 as skeleton | `implemented as symbolic support` |
| Q6 | Biological / numerical constraints | \(D\geq 0\), \(P-D\cdot u\) at the axis, denominator sign checks | `partial (architectural, not a theorem)` |
| Q7 | Held-out generalization | Unseen ICs 8,9 and a train-derived external \(r\) band | `reported, not a gate` |

### Q1 Predictive fit

**Implemented** as a hybrid residual against **observed data**, on the
**current protocol**, using the **reference / first training IC**
(`first(experiments)` / training IC[1]), compared with
`RECOVERY_THRESHOLDS.data_residual` (\(0.30\)). Typical printed residuals
can be much smaller.

`result.data_residual` remains that **legacy IC[1]** residual. It is
not the aggregate train residual.

M2 additionally reports arithmetic-mean residuals (not RMS, not
concatenated residuals, not IC[1], not one holdout experiment):

```math
\texttt{data\_residual\_train}
= (\rho_1+\rho_2+\rho_3+\rho_4+\rho_5+\rho_6+\rho_7)/7
```

```math
\texttt{data\_residual\_holdout}
= (\rho_8+\rho_9)/2
```

The gated Q1 number is still the legacy IC[1] residual. Train and
holdout aggregates are separate evidence. Current Q1 evidence is
**not** only a held-out predictive generalization metric (that is Q7).
On the unique-claim path the gated residual is produced
**conditionally** after successful recovery/discovery (failed
neural-rate quality or failed discovery leaves it unset / infinite).
Passing Q1 does not imply Q2–Q7.

Training minimizes trajectory MSE. The residual is versus data, not
versus the trained UDE vector field.

### Q2 Mechanism function recovery

**Partial.** Current Q2 is a partial mechanism-function diagnostic.
The suite reports neural-versus-truth correlation and relative RMSE
for \(D\) on a one-dimensional regulator grid expanded from **training**
experiment extrema (`_regulator_grid` on ICs 1..7). That grid is not
trajectory occupancy. `_evaluate_unknown_rate_recovery` remains
unchanged. The train-derived `_regulator_grid` remains. Dummy-time
discovery remains. M4 occupancy must not replace the composer.

M2 additionally reports holdout \(D\) error from the **actual neural**
\(\hat D\), not from symbolic reconstruction and not from a normalized
symbolic \(D\):

- `d_rmse_holdout` — neural \(D\) at the actual observed regulator
  coordinates from holdout experiments 8 and 9
- `d_rmse_holdout_domain` — neural \(D\) over the deterministic
  external band derived only from training data (not a train-grid-only
  evaluation and not an arbitrary fixed grid)

Those holdout \(D\) numbers are evidence, not proof of uniqueness.
They are **not** inputs to `unique_claim_kpis_hold`. The locked hold
is Q3 + Q1 residual + Q5 recall.

### Q3 Scale / parameter practical identifiability

**Implemented as a practical warning.** `production_destruction_tradeoff`
builds a finite-difference trajectory Jacobian for **physical**
parameters (neural weights excluded) and a cosine between the
\(k_{\mathrm{prod}}\) direction and a multiplicative \(D\)-scale
perturbation.

`unidentifiable_edge` can be triggered by the practical Fisher
condition-number threshold (default \(10^{6}\)) **or** the
\(k_{\mathrm{prod}}/D\) scale cosine threshold (default \(0.95\)).
Do not describe the flag only as the cosine test.

This is a local, asymptotic, single-trajectory practical warning. It is
not a structural identifiability certificate. The unique-claim path
currently **requires** `unidentifiable_edge == true` so the scale
ambiguity is not hidden. `coefficients_are_biological_constants` is
`!unidentifiable_edge` and is not an independent measurement.

Q3 must remain a practical scale/parameter warning.
It is one required conjunct of the current hold, not the whole product
and not proof of mechanism recovery.

### Q4 Practical functional-identifiability diagnostic

**Implemented as a practical diagnostic, not a gate.** The unexported
`assess_functional_identifiability` path compares independently trained
\(\hat D_i(z)\) on a shared observed train-then-holdout domain across
five locked restart seeds `(201, 202, 203, 204, 205)`. It reports every
restart (including failures), every pair, scale-normalized \(D\)
disagreement, trajectory agreement, and a derived status. The result
is not collapsed to a median. `IdentifiabilityReport` remains
parameter-only Fisher information and is not Q4.

Q4 is not a formal identifiability certificate. Do not call a Q4
object “functionally identifiable” or a structural certificate. Q4 is
not a success gate and is not an input to `unique_claim_kpis_hold`.
The hold remains Q3 + Q1 residual + Q5 recall. Q3 remains a separate
scale/parameter practical warning.

`functional_identifiability_domain` remains the approved M3 domain.
Q4 is not occupancy-based. Q4 is not structural identifiability.
Q4 does not use M4 trajectory occupancy. Occupancy-based discovery
and graph-local trained-\(D\) experiments remain M4 future work and
must not rewrite this diagnostic. The five-restart research script
is not a PR gate.

### Q5 Symbolic support recovery

**Implemented as symbolic support.** Unique-claim discovery calls
`discover_unknown_rate` on **grid-sampled learned \(\hat D\)** with a
dummy sample index \(t\in[0,1]\). That is function regression of the
learned destruction rate, not state-derivative SINDy
(\(D(z)\dot x-N(z)=0\) in the trajectory sense).
`_evaluate_unknown_rate_recovery` remains the unique-claim composer.
Dummy-time discovery remains. M4 occupancy must not replace the
composer.

Current symbolic support recovery on that grid is **not** equivalent to
canonical Hill recovery.

The Hill-class gate is true-monomial recall \(\geq 0.99\)
(`RECOVERY_THRESHOLDS.support_recall`). Combined support F1 is a
skeleton floor (\(0.50\)), not the UDE claim. Extra monomials such as
`1` and `r` may remain.

Q5 support recall/F1 is not canonical Hill recovery.
`canonical_hill_from_nn` stays `false`.

### Q6 Biological / numerical constraints

**Partial.** Softplus enforces \(D\geq 0\) and positive physical
parameters. The \(P-D\cdot u\) form points inward at \(u_i=0\); states
pass through `max(0,x)`. Discovery rejects unsafe denominators. These
are architectural and numerical checks, not a positivity theorem and
not a biological-validity certificate.

### Q7 Held-out generalization

**Reported, not a gate.** Unique-claim now reports held-out residual
and neural \(D\) error on ICs 8 and 9 after a train-only fit on ICs
1..7. Q7 is reported held-out generalization evidence, not an additional success gate. It is not a success gate, not a model-selection gate, and
not a mechanism-identifiability certificate. It is not functional
identifiability.

The existing M1 \(0.30\) gate is **not** copied to holdout. A holdout
residual greater than \(0.30\) does not by itself suppress holdout
evidence or set the M1 result to failure.

`ExperimentSet` remains a list. It is not mutated and does not gain
`train` / `holdout` fields. The 7/2 view is an internal
`ExperimentSplit`. Discovery’s internal `validation_fraction` is still
a column split, not this experiment split.

Q7 is not a symbolic holdout discovery benchmark. Holdout evaluation
does not call `discover_unknown_rate`, `discover_unknown`,
`discover_equations`, or `discover_unknown_destruction`. The M1
composer retains its own existing discovery pipeline on the
train-derived grid.

Symbolic discovery failure does not suppress Q7 evidence:

| Case | `training_ok` | `discovery` | `holdout` |
|------|---------------|-------------|-----------|
| A | `false` | `=== nothing` | `=== nothing` |
| B | `true` | `discovery.success == false` | `!== nothing` |
| C | `true` | `discovery.success == true` | `!== nothing` |

This is stronger than the old IC[1]-only evaluation. It does not
establish functional identifiability, canonical Hill reconstruction,
or broad generalization over arbitrary input regimes. The current
holdout is two fixed IC experiments. The train-derived external \(D\)
domain is not global OOD.

`evaluate_holdout` remains four-scalar `HoldoutEvidence`. The 7/2
train/holdout split remains unchanged. Holdout is not a 0.30 gate.
Occupancy is not added to `HoldoutEvidence`.

## Current unique-claim hold (not a Q collapse)

`unique_claim_kpis_hold` is true only when all three hold:

1. Q3: `unidentifiable_edge === true`
2. Q1: hybrid residual \(\leq 0.30\) on training IC[1]
3. Q5: true-monomial recall \(\geq 0.99\) on Hill-class unknown
   destruction

That triple is the present hold. Q3 is the practical
scale/identifiability warning inside the hold: the edge must be
reported unidentifiable so \(k_{\mathrm{prod}}\leftrightarrow D\) scale
ambiguity is not hidden. It is not a structural certificate and is not
by itself “the product.”

The hold is not Q2 uniqueness, not Q4, not Q7, and not canonical Hill.
Q7 holdout numbers are not inputs to this hold.
Combined F1 is not a hold input. The historical protocol-result field
`:recall_plus_data_residual` names Q1+Q5; the live gate still requires
Q3.

## M4 semantic boundary

M4 occupancy is an additional sampling/evaluation context, not a replacement for Q4 or the M1/M2 composer.

M4-A1 occupancy runtime exists. M4-A2 is live separation/contract tests, not production wiring. M4-B remains pending. M4-C remains pending. The lock exists so later slices cannot silently change M2 or M3 semantics.

occupancy != Q4 domain.z
occupancy != M1 discovery grid
occupancy != M2 holdout evaluator
Occupancy is not part of the recovery result, holdout result, or Q4 diagnostic.

### M3 / Q4 stays the approved diagnostic

- `functional_identifiability_domain` remains the approved M3 domain.
- Q4 remains a practical functional-identifiability diagnostic.
- Q4 is not occupancy-based.
- Q4 is not a success gate.
- Q4 is not structural identifiability.
- Q4 does not use M4 trajectory occupancy.

### M1 / Q5 composer stays the discovery owner

- `_evaluate_unknown_rate_recovery` remains unchanged.
- The train-derived `_regulator_grid` remains.
- Dummy-time discovery remains.
- M4 occupancy must not replace the composer.

### M2 holdout stays four scalars

- The 7/2 train/holdout split remains unchanged.
- `evaluate_holdout` remains four-scalar `HoldoutEvidence`.
- Holdout is not a 0.30 gate.
- Occupancy is not added to `HoldoutEvidence`.

### Seed lists stay distinct

These three lists are distinct. Do not substitute one for another.
Do not modify the existing M2/M3 seed constants.

- `UNIQUE_CLAIM_PROTOCOL.seed = 103`
- `FUNCTIONAL_ID_RESTART_SEEDS = (201, 202, 203, 204, 205)`
- `ROBUSTNESS_SEEDS = (103, 107, 111, 113, 127)`

`ROBUSTNESS_SEEDS` is the documented M4-C list. Naming it here does
not add a Julia constant, an export, or a multi-seed product claim.

### Preserved locks

M4 must not alter:

- `RECOVERY_THRESHOLDS`
- `FUNCTIONAL_ID_REPORTING_CUTOFFS`
- `LOCKED_PUBLIC_EXPORTS`
- `canonical_hill_from_nn == false`
- `unique_claim_kpis_hold`

## Out of scope

The following remain unsupported. Naming them here does not implement
them.

- structural identifiability certificates
- Q4 as a success gate or formal identifiability certificate
- public functional-identifiability API
- M4 — Robustness / Trajectory-Context Validation (pending / future work; not implemented)
- trajectory-occupancy discovery as a replacement for Q4 or the M1/Q5 composer
- arbitrary OOD regimes
- unknown topology discovery
- general CRN solving
- arbitrary multi-hole discovery
- canonical Hill recovery from a trained NN
- biological-constant parameter claims under scale non-identifiability
- general missing-state UDE training
- wet-lab decision making
- general experimental-design engine
- LLM integration
- GPU training stack
- broad SBML kinetic parsing

Also closed or deferred: a general SINDy replacement, a general neural
ODE framework, Bayes / OED as product, licensed experimental series
matching this protocol, and treating
`docs/research/BIO-DYNAX-VISION.md` as implementation.

## What this contract does not change

- `RECOVERY_THRESHOLDS` (loosening a number is breaking)
- `FUNCTIONAL_ID_REPORTING_CUTOFFS`
- `LOCKED_PUBLIC_EXPORTS`
- `canonical_hill_from_nn === false`
- `unique_claim_kpis_hold`
- `UNIQUE_CLAIM_PROTOCOL.seed === 103`
- `FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)`
- the compiler’s \(P-D\cdot u\) formulation
- the public export / freeze list
- `validate_network` as a topology/metadata checker

Green `recovery` CI remains necessary, not sufficient, for a scientific
v1.0 tag.
