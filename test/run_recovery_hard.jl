#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
include(joinpath(@__DIR__, "internals.jl"))
using LinearAlgebra
using OrdinaryDiffEq: Tsit5, solve
using SciMLBase
using Random
using Statistics
using Test

LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "test_recovery_hard.jl"))
