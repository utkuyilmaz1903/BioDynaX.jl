#!/usr/bin/env julia
# Isolated industry-bar runner. Not invoked from runtests.jl.
# CI job: `standards`. Failures are the threshold; do not mark them broken.

using BioDynaX
include(joinpath(@__DIR__, "internals.jl"))
using ComponentArrays
using ForwardDiff
using JET
using LinearAlgebra
using OrdinaryDiffEq: Tsit5, solve
using Random
using SciMLBase
using StaticArrays
using Test

LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "test_standards.jl"))
include(joinpath(@__DIR__, "test_standards_jet.jl"))
