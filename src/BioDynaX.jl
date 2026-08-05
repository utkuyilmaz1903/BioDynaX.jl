###############################################################################
# BioDynaX.jl — Universal Differential Biological Network Solver.
#
# Top-level module: brings in all submodule files (single shared namespace,
# the standard Julia package pattern) and curates the public API.
###############################################################################
module BioDynaX

# -- External dependencies ----------------------------------------------------
using Dates
using Distributed
using Graphs
using LinearAlgebra
using Lux
using NNlib: sigmoid, softplus
using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity
using SciMLBase
using Optimization
using OptimizationOptimJL
using Optimisers
using PrecompileTools
using Zygote
using Random
using Serialization
using SHA
using Statistics
using StaticArrays

# -- Source files --------------------------------------------------------------
include("Types.jl")
include("Config.jl")
include("Network.jl")
include("Experiments.jl")
include("UDE.jl")
include("ModelCache.jl")
include("ParameterSchema.jl")
include("MechanismCompiler.jl")
include("DataGen.jl")
include("Training.jl")
include("BasisFactory.jl")
include("Discovery.jl")
include("Execution.jl")
include("Visualization.jl")
include("Precompile.jl")

# -- Public API ---------------------------------------------------------------
# Network layer
export RunMetadata, TrainingResult, DiscoveryResult, Checkpoint,
       data_fingerprint, save_result, load_result
export AbstractConstraintStrategy, StructuralPositivity,
       AugmentedLagrangianConfig, SolverConfig, TrainingConfig,
       AbstractDiscoveryBackend, ExplicitSTLSQ, ImplicitSINDyPI,
       DiscoveryConfig, ExecutionConfig
export BiologicalNetwork, NodeSpec, EdgeSpec, ReactionSpec,
       EdgeKind, NodeKind, KineticFamily,
       ACTIVATION, INHIBITION, UNKNOWN_NN,
       STATE, INPUT, LATENT,
       MASS_ACTION, SATURATION, HILL, COMPETITIVE, CUSTOM_KINETIC,
       build_network, build_linear_test_network, DEFAULT_EXAMPLE_NETWORK,
       describe_network, validate_network, state_nodes,
       candidate_parents

# UDE / NN layer
export UDEModel, UDEModelCache, build_ude_nn, build_ude_model, compile_network,
       compile_mechanism, CompiledMechanism, pack_parameters, ude_system,
       ude_rhs!, allocate_cache, build_ude_rhs, parameter_schema,
       default_parameters, default_phys_parameters, ParameterSchema,
       positive_parameter, bounded_parameter
export AbstractADPolicy, ZygoteAD, ProductionAD, sensealg

# Experiment layer
export Experiment, DeviceExperiment, ExperimentSet, as_experiment_set,
       experiment_fingerprint, experiment_batches

# Synthetic data layer
export GroundTruthModel, ground_truth!, generate_data, generate_experiment_set,
       default_truth_params

# Training layer
export predict_ude, loss_mse, train_ude, save_checkpoint, load_checkpoint,
       resume_training, train_experiments

# Symbolic discovery layer
export MonomialTerm, LocalBasisSpec, local_basis, candidate_count,
       evaluate_library!, evaluate_library,
       ImplicitCandidate, discover_equations, sample_learned_function,
       format_equation

# Execution layer
export execute_experiments, gpu_available, to_device,
       SerialBackend, ThreadsBackend, DistributedBackend, GPUBackend
export plot_training

end # module BioDynaX
