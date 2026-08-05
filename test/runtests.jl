using BioDynaX
using ComponentArrays
using Lux
using Random
using Statistics
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
