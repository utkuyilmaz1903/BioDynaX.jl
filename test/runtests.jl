using BioDynaX
include("internals.jl")
using ComponentArrays
using LinearAlgebra
using Lux
using Optimization
using OrdinaryDiffEq: Tsit5, solve
using SciMLBase
using SciMLSensitivity: InterpolatingAdjoint, BacksolveAdjoint
using Random
using Statistics
using StaticArrays
using Test
using Zygote

LinearAlgebra.BLAS.set_num_threads(1)

include("test_baseline.jl")
include("test_network.jl")
include("test_ude.jl")
include("test_training.jl")
include("test_discovery.jl")
include("test_experiments.jl")
include("test_compiler.jl")
include("test_quality_gates.jl")
include("test_release.jl")
include("test_sciml_interface.jl")
include("test_metadata.jl")
include("test_explicit_discovery.jl")
include("test_phase1.jl")
include("test_phase2.jl")
include("test_phase3.jl")
include("test_phase4.jl")
include("test_recovery.jl")
include("test_golden_path.jl")
include("test_invariants.jl")
include("test_example_smoke.jl")
include("test_unique_claim_product.jl")
include("test_protocol_fingerprint.jl")
include("test_docs_honesty.jl")
include("test_compiler_contract.jl")
include("test_protocol_surface.jl")
include("test_datagen_contract.jl")
include("test_recovery_admission.jl")
include("test_compiled_path.jl")
include("test_discovery_workspace.jl")
include("test_training_reuse.jl")
include("test_sciml_solve_surface.jl")
include("test_recovery_suite_skip.jl")
include("test_experiment_checkpoint.jl")
include("test_failure_modes.jl")
include("test_hybrid_compose.jl")
include("test_hybrid_residual.jl")
include("test_identifiability_product.jl")
include("test_graph_local_library.jl")
include("test_denominator_domain.jl")
include("test_parameter_schema_pack.jl")
include("test_docs_executable.jl")
include("test_allocation_gates.jl")
include("test_claim_metric_honesty.jl")
