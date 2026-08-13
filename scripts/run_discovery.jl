#!/usr/bin/env julia
###############################################################################
# Thin wrapper around the golden path in examples/unknown_inhibition.jl.
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
