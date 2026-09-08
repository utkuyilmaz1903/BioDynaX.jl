#!/usr/bin/env julia
# Isolated industry-bar runner. Not invoked from runtests.jl.
# CI job: `standards`. Failures are the threshold; do not mark them broken.
#
# Solver / array names are imported from HybridKinetics so this file can run in
# the same temp environment as quality.jl (develop + JET + ForwardDiff).

using HybridKinetics
include(joinpath(@__DIR__, "internals.jl"))
using HybridKinetics: SA, SciMLBase, SVector, Tsit5, solve
using ForwardDiff
using JET
using LinearAlgebra
using Random
using Test

LinearAlgebra.BLAS.set_num_threads(1)

@testset "industry standards" begin
    include(joinpath(@__DIR__, "test_standards.jl"))
    include(joinpath(@__DIR__, "test_standards_jet.jl"))
end
