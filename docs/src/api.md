# API reference

Every exported name is documented here. Unexported helpers mentioned in the
guide (for example `HybridKinetics.report_production_destruction_tradeoff`,
`HybridKinetics.run_recovery_suite`, `HybridKinetics.assess_functional_identifiability`)
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

## One-call workflow

```@docs
discover_unknown_term
UnknownTermResult
report_unknown_term
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
StabilitySelection
stability_selection_report
format_stability_selection
```

## Catalyst input and symbolic output

Exported wrappers whose implementations live in the extensions
`HybridKineticsCatalystExt` (`using Catalyst`) and `HybridKineticsSymbolicsExt`
(`using Symbolics`); see the [Extensions](extensions.md) page.

```@docs
network_from_reactionsystem
symbolic
```

## Benchmark thresholds

```@docs
RECOVERY_THRESHOLDS
```

## Library comparison study

Unexported. The multi-seed, multi-noise comparison of the graph-local, global,
and wrong-graph discovery libraries described on the
[Benchmarks](benchmarks.md#Library-comparison-study) page.

```@docs
HybridKinetics.library_comparison_study
HybridKinetics.library_comparison_run
HybridKinetics.library_comparison_smoke
HybridKinetics.library_study_summary
HybridKinetics.format_library_study_summary
HybridKinetics.append_library_study_row
HybridKinetics.read_library_study_csv
HybridKinetics.library_study_training_set
HybridKinetics.LIBRARY_STUDY_DESIGNS
HybridKinetics.library_study_default_design
HybridKinetics.designed_trained_graph_local_coordinates
```
