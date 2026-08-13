#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using Printf
using Random

report = run_recovery_suite(MersenneTwister(1);
                            sections = (:linear, :mm, :hill, :competitive, :ablation))
println("BioDynaX recovery suite (fast sections)")
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
if haskey(report, :ablation)
    a = report[:ablation]
    println("  local/global terms ", a.local_terms, "/", a.global_terms)
    println("  local/global F1    ", a.local_f1, "/", a.global_f1)
    println("  false parent       local=", a.local_false_parent,
            " global=", a.global_false_parent)
    println("  den violations     local=", a.local_denominator_violations,
            " global=", a.global_denominator_violations)
    println("  wall time (s)      local=", a.local_time, " global=", a.global_time)
end

hard = run_recovery_suite(MersenneTwister(103);
                          sections = (:ude_discovery, :mm_unknown))
println("BioDynaX recovery suite (unknown-edge UDE)")
for key in (:ude_discovery, :mm_unknown)
    haskey(hard, key) || continue
    u = hard[key]
    println("  ", key, "  retcode=", u.retcode,
            "  nn_corr=", u.nn_correlation,
            "  F1=", u.support_f1,
            "  recall=", u.support_recall,
            "  data_resid=", u.data_residual)
end
