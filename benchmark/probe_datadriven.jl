#!/usr/bin/env julia
# Isolated resolve probe. Never a skip-as-win and never a CI red/green claim.
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
