# API reference

Every exported name is documented here. Unexported helpers mentioned in the
guide (for example `BioDynaX.report_production_destruction_tradeoff`,
`BioDynaX.run_recovery_suite`, `BioDynaX.assess_functional_identifiability`)
are internal and may change between minor versions.

## Network specification

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
```

## Kinetic metadata

```@docs
KineticMetadata
MetadataLike
EmptyMetadata
InputDriveMetadata
MassActionMetadata
HillMetadata
CompetitiveMetadata
LinearDecayMetadata
SaturationMetadata
CustomKineticMetadata
```

## Compiled models

```@docs
compile_mechanism
build_ude_model
UDEModel
NeuralDestructionTerm
ude_system
ude_rhs!
allocate_cache
pack_parameters
parameter_schema
ParameterSchema
positive_parameter
```

## SciML interface

```@docs
build_ude_function
SciMLBase.ODEProblem(::UDEModel, ::Any, ::Any, ::Any)
SciMLBase.solve(::UDEModel, ::Any, ::Any, ::Any)
auto_sensealg
default_solver_config
SolverConfig
AbstractADPolicy
ZygoteAD
ProductionAD
```

## Experiments

```@docs
Experiment
ExperimentSet
experiment_from_csv
write_experiment_csv
generate_experiment_set
```

## Training

```@docs
train_ude
train_experiments
predict_ude
TrainingConfig
HorizonCurriculum
TrainingResult
TrainingRetcode
AbstractConstraintStrategy
StructuralPositivity
AugmentedLagrangianConfig
```

## Symbolic discovery

```@docs
discover_unknown_rate
discover_equations
sample_unknown_destruction
estimate_derivatives
local_basis
DiscoveryConfig
ImplicitSINDyPI
ExplicitSTLSQ
DiscoveryResult
ImplicitCandidate
ExplicitCandidate
DiscoveryRetcode
DiscoverySuccess
InsufficientSamples
DenominatorUnsafe
EmptySupport
SingularLibrary
DiscoveryFailed
equation_to_function
equation_to_latex
export_rhs
compose_hybrid_rhs
hybrid_data_residual
```

## Benchmark thresholds

```@docs
RECOVERY_THRESHOLDS
```
