# API

Names on the [stability freeze](stability.md) plus the golden-path verbs a
stranger types in the [tutorial](tutorial.md). Experimental entry points are
**not exported**; see [Experimental](experimental.md).

## Network and compile

```@docs
BiologicalNetwork
NodeSpec
EdgeSpec
ReactionSpec
EdgeKind
ACTIVATION
INHIBITION
UNKNOWN_NN
NodeKind
STATE
INPUT
LATENT
KineticFamily
MASS_ACTION
SATURATION
HILL
COMPETITIVE
CUSTOM_KINETIC
validate_network
state_nodes
candidate_parents
compile_mechanism
build_ude_model
UDEModel
ude_system
ude_rhs!
NeuralDestructionTerm
```

## Metadata

Typed kinetic metadata, including `MetadataLike`, lives on the
[Metadata](metadata.md) page.

## Training

```@docs
train_ude
train_experiments
TrainingResult
TrainingRetcode
TrainingConfig
HorizonCurriculum
SolverConfig
StructuralPositivity
AugmentedLagrangianConfig
AbstractConstraintStrategy
AbstractADPolicy
ZygoteAD
ProductionAD
predict_ude
```

## Discovery

```@docs
discover_equations
discover_unknown_rate
DiscoveryResult
DiscoveryRetcode
DiscoverySuccess
InsufficientSamples
DenominatorUnsafe
EmptySupport
SingularLibrary
DiscoveryFailed
DiscoveryConfig
ImplicitSINDyPI
ExplicitSTLSQ
ImplicitCandidate
ExplicitCandidate
local_basis
export_rhs
equation_to_latex
equation_to_function
estimate_derivatives
sample_unknown_destruction
compose_hybrid_rhs
hybrid_data_residual
```

## Experiments

```@docs
Experiment
ExperimentSet
experiment_from_csv
write_experiment_csv
generate_experiment_set
```

## Parameters and SciML

`build_ude_function`, `auto_sensealg`, and `default_solver_config` are
documented on [SciML Integration](sciml.md).

```@docs
pack_parameters
parameter_schema
ParameterSchema
allocate_cache
positive_parameter
```

## Contract

```@docs
RECOVERY_THRESHOLDS
```

Recovery fixtures (`run_recovery_suite`, `build_*_recovery_network`) are
internal. Call them as `BioDynaX.run_recovery_suite` from tests and
benchmarks. They are not on the freeze list.

Unique-claim product helpers are also internal (`BioDynaX.foo`):
`UNIQUE_CLAIM_PROTOCOL`, `UniqueClaimFingerprint`,
`unique_claim_fingerprint`, `format_protocol_result`,
`build_protocol_result`, `assert_single_unknown_destruction`,
`assert_unique_claim_recovery_network`, `unique_claim_kpis_hold`,
`unique_claim_discovery_extras`, `unique_claim_protocol_ics`,
`extras_print_label`, `UNIQUE_CLAIM_F1_ATTEMPT`. Compiler remapping
helpers (`assert_dense_neural_index`,
`build_skipped_duplicate_unknown_network`,
`build_two_regulator_unknown_network`,
`build_remapped_two_regulator_network`,
`generate_from_compiled_model`,
`unique_claim_experiment_set`) live in the same unexported
surface. Recovery admission (`admit_recovery_suite_network`,
`UniqueClaimProtocolRow`, `unique_claim_kpi_failure_symbols`) is
documented with executable snippets on
[Unique claim](unique-claim.md). Do not export them to silence Documenter.
