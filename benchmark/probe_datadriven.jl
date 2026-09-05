#!/usr/bin/env julia
# Checks whether DataDrivenSparse and DataDrivenDiffEq resolve in an isolated
# temporary environment (BioDynaX is not loaded). Prints RESOLVED or
# UNAVAILABLE with the resolver error. Runs in CI (job "external-baseline",
# allowed to fail); a failure documents the dependency conflict and is not a
# comparison result. Runtime: a few minutes (package resolution only).
# Run:  julia --project=. benchmark/probe_datadriven.jl

using Pkg

println("DataDrivenSparse / DataDrivenDiffEq isolated resolve probe")
println("This environment does not load BioDynaX.")
try
    Pkg.activate(; temp = true)
    Pkg.add(["DataDrivenSparse", "DataDrivenDiffEq"])
    println("RESOLVED")
catch error
    println("UNAVAILABLE")
    showerror(stdout, error)
    println()
end
