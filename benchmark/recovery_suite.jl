#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using Printf
using Random

report = run_recovery_suite(MersenneTwister(1))
println("BioDynaX recovery suite")
if haskey(report, :linear)
    @printf "  linear RMSE        %.4f\n" report[:linear].rmse
end
if haskey(report, :mm)
    @printf "  MM RMSE            %.4f\n" report[:mm].rmse
end
if haskey(report, :hill)
    @printf "  Hill RMSE          %.4f\n" report[:hill].rmse
end
if haskey(report, :competitive)
    @printf "  competitive RMSE   %.4f\n" report[:competitive].rmse
end
if haskey(report, :ude_discovery)
    ude = report[:ude_discovery]
    println("  UDE discovery      ", ude.retcode, "  corr=", ude.correlation)
end
if haskey(report, :ablation)
    a = report[:ablation]
    println("  local/global terms ", a.local_terms, "/", a.global_terms)
end
