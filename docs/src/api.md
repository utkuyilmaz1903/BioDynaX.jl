# API

Names on the [stability freeze](stability.md). Experimental entry points are
documented on [Experimental](experimental.md).

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
STATIC_STATE_THRESHOLD
```

## Training

```@docs
train_ude
TrainingResult
TrainingRetcode
```

## Discovery

```@docs
discover_equations
DiscoveryResult
DiscoveryRetcode
local_basis
export_rhs
equation_to_latex
equation_to_function
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

## Recovery fixtures (not on the freeze list)

These names are CI / benchmark helpers. They are exported for
`run_recovery_suite` but they are **not** the stability freeze. See
[Experimental](experimental.md) for GPU, SBML, MTK, and identifiability.

```@docs
run_recovery_suite
RECOVERY_THRESHOLDS
build_mm_test_network
build_hill_recovery_network
build_mm_recovery_network
build_competitive_test_network
```
