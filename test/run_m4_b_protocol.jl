#!/usr/bin/env julia
# M4-B protocol runner. Not a CI job. PR smoke is not trained-UDE
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

include(joinpath(@__DIR__, "test_m4_b_trained_graph_local_protocol.jl"))
