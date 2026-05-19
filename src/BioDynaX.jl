###############################################################################
# BioDynaX.jl — Universal Differential Biological Network Solver.
#
# Top-level module: brings in all submodule files (single shared namespace,
# the standard Julia package pattern) and curates the public API.
###############################################################################
module BioDynaX

# -- External dependencies ----------------------------------------------------
using Graphs
using Lux
using NNlib: softplus
using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using Zygote
using Random
using Statistics
using DataDrivenDiffEq
using DataDrivenSparse
using ModelingToolkit: @variables, equations

# -- Source files (order matters: Network → UDE → DataGen → Training → Discovery)
include("Network.jl")
include("UDE.jl")
include("DataGen.jl")
include("Training.jl")
include("Discovery.jl")

# -- Public API ---------------------------------------------------------------
# Network layer
export BiologicalNetwork, EdgeKind,
       ACTIVATION, INHIBITION, UNKNOWN_NN,
       build_network, describe_network

# UDE / NN layer
export build_ude_nn, pack_parameters, ude_system

# Synthetic data layer
export ground_truth!, generate_data, default_truth_params

# Training layer
export predict_ude, loss_mse, train_ude

# Symbolic discovery layer
export discover_equations, sample_learned_function

end # module BioDynaX
