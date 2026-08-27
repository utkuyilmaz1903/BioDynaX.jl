#!/usr/bin/env julia
###############################################################################
# Debug runner, not the product.
# Thin wrapper around the golden path in examples/unknown_inhibition.jl.
# The unique-claim path is that example (seed 103, 9 ICs), not this script.
# Discovery failures throw (`strict = true`); they are not warned away.
###############################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "examples", "unknown_inhibition.jl"))

discovery, sol = main()
discovery.success ||
    error("discovery failed ($(discovery.retcode)): $(discovery.message)")
println("retcode = ", discovery.retcode)
println("equations:\n", discovery.equations)
