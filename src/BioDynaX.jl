###############################################################################
# BioDynaX.jl — graph-guided biological UDEs with local rational discovery.
#
# Top-level module: brings in all submodule files (single shared namespace,
# the standard Julia package pattern) and curates the public API.
###############################################################################
module BioDynaX

const PACKAGE_VERSION = v"0.9.1"

# -- External dependencies ----------------------------------------------------
using Dates
using DelimitedFiles
using Distributed
using Graphs
using LinearAlgebra
using Lux
using NNlib: sigmoid, softplus
using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity: InterpolatingAdjoint, BacksolveAdjoint, ZygoteVJP
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
include("ScientificCore.jl")
include("Config.jl")
include("Metadata.jl")
include("Network.jl")
include("Experiments.jl")
include("UDE.jl")
include("ModelCache.jl")
include("ParameterSchema.jl")
include("MechanismCompiler.jl")
include("SciMLInterface.jl")
include("DataGen.jl")
include("Training.jl")
include("OptimizationInterface.jl")
include("BasisFactory.jl")
include("Discovery.jl")
include("Identifiability.jl")
include("BenchmarkNetworks.jl")
include("Recovery.jl")
include("Bridge.jl")
include("Execution.jl")
include("Visualization.jl")
include("Precompile.jl")

# -- Public API ---------------------------------------------------------------
# Network layer
export RunMetadata, TrainingResult, DiscoveryResult, Checkpoint,
       TrainingRetcode, DiscoveryRetcode, TrainingDiagnostics, ParameterUncertainty,
       DiscoveryUncertaintyReport, IdentifiabilityReport, BenchmarkOutcome,
       DiscoverySuccess, InsufficientSamples, DenominatorUnsafe, EmptySupport,
       SingularLibrary, DiscoveryFailed,
       data_fingerprint, save_result, load_result
export AbstractConstraintStrategy, StructuralPositivity,
       AugmentedLagrangianConfig, SolverConfig, TrainingConfig,
       HorizonCurriculum, SensealgRecommendation,
       AbstractDiscoveryBackend, ExplicitSTLSQ, ImplicitSINDyPI,
       DataDrivenSparseSTLSQ, DiscoveryConfig, ExecutionConfig
export BiologicalNetwork, NodeSpec, EdgeSpec, ReactionSpec,
       EdgeKind, NodeKind, KineticFamily,
       ACTIVATION, INHIBITION, UNKNOWN_NN,
       STATE, INPUT, LATENT,
       MASS_ACTION, SATURATION, HILL, COMPETITIVE, CUSTOM_KINETIC,
       build_network, build_linear_test_network, build_repressilator_network,
       build_dual_unknown_network, build_kinetic_generalization_network,
       build_mm_test_network, build_hill_recovery_network,
       build_mm_recovery_network, build_competitive_test_network,
       build_distractor_network, build_rate_discovery_network,
       build_rate_ablation_network,
       DEFAULT_EXAMPLE_NETWORK, benchmark_networks, run_benchmark_suite,
       run_recovery_suite, relative_parameter_error, RECOVERY_THRESHOLDS,
       term_key, active_support, support_f1, rate_rel_rmse,
       denominator_violation_count, hill_rate_support, mm_rate_support,
       hill_rate_truth, mm_rate_truth, sample_unknown_destruction,
       sample_unknown_destruction_grid, discover_unknown_rate,
       compose_hybrid_rhs, hybrid_data_residual, neural_destruction_terms, rate_discovery_config,
       describe_network, validate_network, state_nodes,
       candidate_parents

# UDE / NN layer
export UDEModel, UDEModelCache, build_ude_nn, build_ude_model, compile_network,
       compile_mechanism, CompiledMechanism, pack_parameters, ude_system,
       ude_rhs!, allocate_cache, build_ude_rhs, parameter_schema,
       default_parameters, default_phys_parameters, ParameterSchema,
       positive_parameter, bounded_parameter
export KineticMetadata, EmptyMetadata, InputDriveMetadata, MassActionMetadata,
       HillMetadata, CompetitiveMetadata, LinearDecayMetadata, MetadataLike,
       SaturationMetadata, CustomKineticMetadata, metadata_summary
export MultiHeadNetwork
export SaturationDestructionTerm, SaturationProductionTerm, CustomDestructionTerm,
       NeuralDestructionTerm,
       STATIC_STATE_THRESHOLD, export_mtk_system, import_sbml_network,
       import_sbmltoolkit_network
export build_ude_function, auto_sensealg, recommend_sensealg,
       default_solver_config
export AbstractADPolicy, ZygoteAD, ProductionAD, sensealg

# Experiment layer
export Experiment, DeviceExperiment, ExperimentSet, as_experiment_set,
       experiment_fingerprint, experiment_batches,
       experiment_weight, experiment_noise_scale,
       experiment_from_csv, write_experiment_csv

# Synthetic data layer
export GroundTruthModel, ground_truth!, generate_data, generate_experiment_set,
       default_truth_params

# Training layer
export predict_ude, loss_mse, train_ude, save_checkpoint, load_checkpoint,
       resume_training, train_experiments, estimate_parameter_uncertainty,
       build_optimization_problem, solve_optimization, train_via_optimization
export assess_identifiability, fisher_information_matrix, trajectory_jacobian,
       parameter_credible_intervals

# Symbolic discovery layer
export MonomialTerm, LocalBasisSpec, local_basis, candidate_count,
       evaluate_library!, evaluate_library, evaluate_library_range!,
       LibraryChunks, each_library_chunk,
       ImplicitCandidate, ExplicitCandidate, discover_equations,
       format_equation, equation_to_latex, equation_to_function, export_rhs,
       estimate_derivatives, information_criterion, score_candidate,
       select_discovery_config, uncertainty_reports

# Execution layer
export execute_experiments, gpu_available, to_device,
       SerialBackend, ThreadsBackend, DistributedBackend, GPUBackend
export plot_training

end # module BioDynaX
