#!/usr/bin/env julia
# protocol runner. Not a CI job. PR smoke is not trained-UDE
# scientific acceptance.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
include(joinpath(@__DIR__, "internals.jl"))
using LinearAlgebra
using Random
using Statistics
using Test

LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "test_trained_library_comparison_protocol.jl"))
