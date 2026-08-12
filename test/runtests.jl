using BioDynaX
using ComponentArrays
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
