# Architecture

BioDynaX separates biological semantics, numerical execution, optimization and
equation discovery.

The authoritative scientific contract is
[v1.0 scientific contract](design/v1_contract.md). Compiled dynamics are
\(\dot u_i = P_i(u,p,t) - D_i(u,p,t)\, u_i\). The vision form
\(\dot x = f_{\mathrm{known}} + D\) is not the v1.0 product.

## Data flow

1. `BiologicalNetwork` defines typed nodes, interactions and reactions.
2. `compile_mechanism` lowers reactions and edges to production–destruction
   IR. Unknown edges become `NeuralDestructionTerm`. Duplicate unknown
   reaction+edge pairs skip the edge; kept heads are reindexed to `1:n`
   so `ude_system` / `ude_rhs!` / `allocate_cache` stay in bounds. That
   remapping is a compiler contract, not a unique-claim hole gate.
3. `compile_network` wraps the IR in a `UDEModel`.
4. `ExperimentSet` carries replicates, irregular samples and observation masks.
5. `train_ude` or `train_experiments` returns a versioned `TrainingResult`.
   A `TrainingSolveSession` remakes one `ODEProblem` across ICs and does
   not call `compile_network` per IC. First-IC warmup hands Optimisers
   state to the multi-IC stage. See [Training reuse](training-reuse.md).
6. `local_basis` derives candidate variables from each target's graph parents
   (`scope = :graph`, or `:global` for ablations).
7. `discover_equations` fits `D(z)ẋ-N(z)=0` on a **state-derivative** path,
   validates denominators and reports bootstrap term-selection frequencies.
   Failures set `DiscoveryRetcode`. Unique-claim
   `discover_unknown_rate` instead regresses sampled \(\hat D\) with a
   dummy sample index \(t\in[0,1]\). That is function regression of the
   learned destruction rate, not trajectory \(\dot x\)-SINDy.

The unique-claim hold is printed and stored in that same order:

1. **IDENTIFIABILITY (Q3)** — `unidentifiable_edge` and
   `coefficients_are_biological_constants`. Local practical warning
   (Fisher condition number **or** \(k_{\mathrm{prod}}/D\) scale cosine),
   not the whole product and not a structural certificate.
2. **FIT (Q1 + Q5)** — hybrid residual versus observed data on the
   current training IC[1] (`first(experiments)`), and true-monomial
   recall. Residual is not mechanistic recovery and is not held-out
   generalization.
3. **DISCOVERY** — symbolic `D(z)` from grid-sampled learned \(D\), live
   extras, skeleton combined F1. `canonical_hill_from_nn` stays false.
4. **REPRODUCTION** — seed 103, 9 ICs, 50 points. Smoke is labeled
   separately and is not this fingerprint.

## Unique-claim recovery pipeline

`run_recovery_suite` is a gated `Dict{Symbol,Any}` dispatcher. A
requested section runs only inside `if :name in wanted`. Unique-claim
sections (`:ude_discovery`, `:mm_unknown`) do not sample, discover, or
score in the suite body. They call `_train_unknown_edge` then
`_evaluate_unknown_rate_recovery`, then the existing identifiability
path, then `report_recovery`.

`_evaluate_unknown_rate_recovery` owns unique-claim control flow:
sample the learned destruction rate on the regulator grid (the existing
`(R, D, term)` path), apply the neural-rate training gate, and either
return the early NamedTuple (`discovery === nothing`, residual `Inf`,
no discovery) or run raw then normalized `discover_unknown_rate` and
metric-only `evaluate_recovery`.

`report_recovery` builds the internal `MechanismRecoveryResult` and
always fills `locked_kpis` and `protocol_result`. The type is not
exported. `haskey` / `hasproperty` mean the field exists, not that
discovery ran or that a KPI is meaningful.

Unique-claim shells consume the shared suite RNG through
`consume_shared_suite_rng!` (the discarded dummy `build_ude_model`
draw). That path does not construct a per-section
`MersenneTwister(seed)`.

Q1 remains the hybrid residual on the current reference IC
(`first(experiments)` / training IC[1]). It is not held-out
generalization.

The SciML solve surface agrees `ude_system`, `ODEFunction`,
`ODEProblem`, `remake`, inplace cache, `SciMLBase.solve`, and
`predict_ude`. Mechanistic models switch adjoints at 64 observations.
See [SciML solve surface](sciml-solve-surface.md).

A skipped recovery-suite section does not call _train_unknown_edge.
See [Recovery suite skip](recovery-suite-skip.md).

`experiment_fingerprint` hashes times, observations, mask, and `u0`.
`resume_training` reuses the compiled `UDEModel` and does not call
`compile_network`. Remapped multi-head generate and
`train_experiments` share one compiled tree. See
[Experiment fingerprint and checkpoint](experiment-checkpoint.md).

validate_network stays a topology checker; 0-hole and 2-hole networks still validate.
See [Failure modes](failure-modes.md).

compose_hybrid_rhs with the neural destruction rate recovers ude_system.
See [Hybrid compose path](hybrid-compose.md).
hybrid_data_residual agrees with SciMLBase.solve of compose_hybrid_rhs.
See [Hybrid residual versus solver](hybrid-residual.md).
production_destruction_tradeoff joins UniqueClaimProtocolRow through identifiability_product.
See [Identifiability product rows](identifiability-product.md).
local_basis scope=:graph uses graph parents; scope=:global is the ablation.
See [Graph-local library and ablation](graph-local-library.md).
denominator_split_counts walks train, validation, and the orthant domain grid separately.
See [Denominator and domain safety](denominator-domain.md).
parameter_schema collects CustomKineticMetadata.rate_param so :k_custom is present.
See [Parameter schema and pack](parameter-schema-pack.md).
The executable docs path joins hybrid residual, identifiability product, graph-local library, denominator domain, and parameter schema pack.
See [Docs executable path](docs-executable.md).
Allocation gates use measured hot-path byte ceilings that can fail.
See [Allocation and type-stability gates](allocation-gates.md).

See [Unique claim](unique-claim.md). `validate_network` stays a
topology/metadata checker; the single-hole instrument is
`assert_single_unknown_destruction` on the golden path only.
`UniqueClaimFingerprint` types smoke versus the seed-103 / 9-IC job.
Unscored extras print NA; they are not the F1-attempt leftover pair.

`generate_data` uses the compiled NN tree. Remapped multi-head unknowns
and two-regulator `D(S,I)` are generated together;
`generate_from_compiled_model` integrates a stored `UDEModel` through
`SciMLBase.ODEProblem(model, u0, tspan, p)`.
`generate_experiment_set` compiles that model once.
`run_recovery_suite` admits unique-claim sections through
`admit_recovery_suite_network`. Zero- and two-hole networks fail closed
on that path without training. Every suite section has a hole policy;
only unique-claim sections reject 0/2 holes. `UniqueClaimProtocolRow`
stores named KPI failures (`:unidentifiable_edge`, `:data_residual`,
`:support_recall`). See [Compiled experiment path](compiled-path.md).

## Positivity and constraints

The default UDE uses

```math
\dot{x}_i = P_i(x,p,t) - D_i(x,p,t)x_i,\qquad P_i,D_i\geq 0.
```

This points inward at `x_i=0`. That is an architectural / numerical
constraint, not a formal positivity theorem. Non-structural inequalities
use a smooth Powell–Hestenes–Rockafellar Augmented Lagrangian.

## Rational discovery

For each target and graph-local regulator set, BioDynaX identifies:

```math
D(z)\dot{x}_i-N(z)=0.
```

The constant denominator coefficient is anchored to one. Numerator and
denominator coefficients are selected jointly with blocked ridge STLSQ
(`_stlsq_blocked!` on a grow-only Gram workspace). Dense QR `_stlsq` is
the agreement oracle, not the bootstrap factorisation. Contiguous
hold-out blocks and bootstrap consensus reject unstable supports.

The product path (`sample_unknown_destruction` → `discover_unknown_rate`)
applies that implicit problem to the **unknown destruction rate**, not the
full state derivative. Known production and linear decay stay in compiled IR.
`compose_hybrid_rhs` stitches the recovered rate back into the ODE.
`local_basis(...; scope=:graph)` versus `scope=:global` is the same solver
with a different library — that is the graph-prior ablation.

## Scaling

Libraries are generated per target from graph parents. For bounded indegree `k`,
the library scales with `Σ O(k_i^d)` rather than global `O(n^d)`.

Streaming chunks (`each_reusable_library_chunk`) overwrite one library
buffer. Blocked STLSQ reuses a Gram workspace across bootstrap draws
and can fill implicit design in row chunks without a full `n × p`
matrix. Implicit candidates are stress-tested on train, validation and an
orthant grid (`domain_samples`). See [Discovery streaming](discovery-streaming.md).

Raw trajectories can enter discovery via `estimate_derivatives` and
`discover_equations(X, times, network)`. Recovered candidates export to LaTeX
or callable RHS closures (`export_rhs`).

## Execution

Serial, threaded, and distributed backends are supported. The `:gpu` backend is
**experimental array transfer only** (see [Experimental](experimental.md)).
