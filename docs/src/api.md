# API

Names on the [stability freeze](stability.md). Experimental entry points are
documented on [Experimental](experimental.md).

## Network and compile

```@docs
BiologicalNetwork
compile_mechanism
build_ude_model
UDEModel
ude_system
STATIC_STATE_THRESHOLD
```

## Training

```@docs
train_ude
TrainingResult
```

## Discovery

```@docs
discover_equations
DiscoveryResult
DiscoveryRetcode
local_basis
export_rhs
equation_to_latex
discover_unknown_rate
sample_unknown_destruction
compose_hybrid_rhs
NeuralDestructionTerm
```

## Experiments

```@docs
Experiment
experiment_from_csv
```

## Recovery

```@docs
run_recovery_suite
RECOVERY_THRESHOLDS
build_mm_test_network
build_hill_recovery_network
build_mm_recovery_network
build_competitive_test_network
```
