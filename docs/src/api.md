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
compile_mechanism
build_ude_model
UDEModel
ude_system
ude_rhs!
```

## Training

```@docs
train_ude
train_experiments
TrainingResult
TrainingRetcode
TrainingConfig
HorizonCurriculum
predict_ude
```

## Discovery

```@docs
discover_equations
discover_unknown_rate
DiscoveryResult
DiscoveryRetcode
local_basis
export_rhs
equation_to_latex
equation_to_function
sample_unknown_destruction
compose_hybrid_rhs
hybrid_data_residual
NeuralDestructionTerm
```

## Experiments

```@docs
Experiment
experiment_from_csv
write_experiment_csv
generate_experiment_set
```

## Config and helpers

```@docs
pack_parameters
ParameterSchema
ZygoteAD
ProductionAD
AugmentedLagrangianConfig
ImplicitSINDyPI
ExplicitSTLSQ
estimate_derivatives
candidate_parents
```

## Contract

```@docs
RECOVERY_THRESHOLDS
```

Recovery fixtures (`run_recovery_suite`, `build_*_recovery_network`) are
internal. Call them as `BioDynaX.run_recovery_suite` from tests and
benchmarks. They are not on the freeze list.
