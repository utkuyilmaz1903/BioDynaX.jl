###############################################################################
# BioDynaX.jl — graph-guided biological UDEs with local rational discovery.
#
# Top-level module: brings in all submodule files (single shared namespace,
# the standard Julia package pattern) and curates the public API.
###############################################################################
module BioDynaX

const PACKAGE_VERSION = v"0.9.2"

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
using ChainRulesCore: ignore_derivatives
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
include("DiscoveryWorkspace.jl")
include("Discovery.jl")
include("Identifiability.jl")
include("BenchmarkNetworks.jl")
include("Recovery.jl")
include("UniqueClaim.jl")
include("CompilerContract.jl")
include("DataGenContract.jl")
include("RecoveryAdmission.jl")
include("CompiledPath.jl")
include("TrainingReuse.jl")
include("SciMLSolveSurface.jl")
include("RecoverySuiteSkip.jl")
include("ExperimentCheckpoint.jl")
include("FailureModes.jl")
include("Bridge.jl")
include("Execution.jl")
include("Visualization.jl")
include("Precompile.jl")

# -- Public API ---------------------------------------------------------------
# Freeze list + golden-path verbs. Fixtures, Fisher, GPU/SBML/MTK, and
# library internals are `BioDynaX.foo` (not exported).
export BiologicalNetwork, NodeSpec, EdgeSpec, ReactionSpec,
       EdgeKind, NodeKind, KineticFamily,
       ACTIVATION, INHIBITION, UNKNOWN_NN,
       STATE, INPUT, LATENT,
       MASS_ACTION, SATURATION, HILL, COMPETITIVE, CUSTOM_KINETIC
export KineticMetadata, EmptyMetadata, InputDriveMetadata, MassActionMetadata,
       HillMetadata, CompetitiveMetadata, LinearDecayMetadata, MetadataLike,
       SaturationMetadata, CustomKineticMetadata
export UDEModel, build_ude_model, compile_mechanism, ude_system, ude_rhs!,
       pack_parameters, parameter_schema, ParameterSchema, allocate_cache,
       positive_parameter
export TrainingResult, TrainingRetcode, TrainingConfig, HorizonCurriculum,
       SolverConfig, StructuralPositivity, AugmentedLagrangianConfig,
       AbstractConstraintStrategy, AbstractADPolicy, ZygoteAD, ProductionAD,
       train_ude, train_experiments, predict_ude
export Experiment, ExperimentSet, experiment_from_csv, write_experiment_csv,
       generate_experiment_set
export discover_equations, discover_unknown_rate, DiscoveryResult, DiscoveryRetcode,
       DiscoverySuccess, InsufficientSamples, DenominatorUnsafe, EmptySupport,
       SingularLibrary, DiscoveryFailed,
       DiscoveryConfig, ImplicitSINDyPI, ExplicitSTLSQ,
       ImplicitCandidate, ExplicitCandidate,
       local_basis, export_rhs, equation_to_latex, equation_to_function,
       estimate_derivatives, compose_hybrid_rhs, sample_unknown_destruction,
       hybrid_data_residual, NeuralDestructionTerm
export RECOVERY_THRESHOLDS
export validate_network, state_nodes, candidate_parents
export build_ude_function, auto_sensealg, default_solver_config

end # module BioDynaX
