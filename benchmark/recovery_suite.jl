#!/usr/bin/env julia
# Recovery benchmark report. First the fast sections (known linear, Michaelis-
# Menten, Hill and competitive kinetics; graph-local versus global library),
# then the trained-model sections (:ude_discovery and :mm_unknown) on the
# reference protocol, printing the full report for each. Not run in CI; the
# CI equivalents are test/test_recovery.jl and test/run_recovery_hard.jl.
# Runtime: not measured for this release; the two trained-model sections take
# about as long as test/run_recovery_hard.jl (about 10 minutes on 4 cores).
# Run:  julia --project=. benchmark/recovery_suite.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using BioDynaX: run_recovery_suite
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

hard = run_recovery_suite(MersenneTwister(BioDynaX.REFERENCE_PROTOCOL.seed);
    sections = (:ude_discovery, :mm_unknown))
println("BioDynaX recovery suite (unknown-edge UDE)")
fingerprint = BioDynaX.reference_protocol_fingerprint()
for key in (:ude_discovery, :mm_unknown)
    haskey(hard, key) || continue
    u = hard[key]
    extras = hasproperty(u, :protocol_result) ? u.protocol_result.extras :
             (hasproperty(u, :extras) ? u.extras : nothing)
    println("  ", key, "  retcode=", u.retcode,
        "  nn_corr=", u.nn_correlation,
        "  F1=", u.support_f1,
        "  recall=", u.support_recall,
        "  data_resid=", u.data_residual,
        "  extras=", BioDynaX.extras_print_label(extras))
    println(BioDynaX.format_recovery_protocol(u, fingerprint))
end
