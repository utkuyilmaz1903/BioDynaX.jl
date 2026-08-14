#!/usr/bin/env julia
# Analytical Hill D(r) noise grid. Not a CI job.
# σ is a fraction of rate amplitude (same definition as the 0.5% Occam gate).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using BioDynaX:
    hill_rate_truth, hill_rate_support, support_f1, rate_discovery_config,
    discover_unknown_rate, denominator_violation_count, RECOVERY_THRESHOLDS
using Printf
using Random

const SIGMAS = (0.0, 0.005, 0.02, 0.05, 0.10)
const SEED = 104

function run_sigma(σ::Float64)
    r = collect(range(0.1, 2.0; length = 180))
    rng = MersenneTwister(SEED)
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    amp = max(maximum(abs, D), eps(Float64))
    D_obs = D .+ σ .* amp .* randn(rng, length(r))
    result = discover_unknown_rate(
        reshape(r, 1, :), collect(range(0.0, 1.0; length = length(r))),
        reshape(D_obs, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = SEED),
        verbose = false, strict = false)
    truth = hill_rate_support(2)
    metrics = result.success ?
        support_f1(result.candidates[1], truth.numerator, truth.denominator) :
        nothing
    den = result.success ?
        denominator_violation_count(result.candidates[1], reshape(r, 1, :)) :
        typemax(Int)
    f1 = metrics === nothing ? 0.0 : metrics.combined.f1
    recall = metrics === nothing ? 0.0 : metrics.combined.recall
    holds = result.success &&
        f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean &&
        recall ≥ RECOVERY_THRESHOLDS.support_recall &&
        den == 0
    return (; σ, success = result.success, f1, recall, den, holds)
end

function main()
    println("Analytical Hill noise grid (seed 104; σ × rate amplitude)")
    @printf "  %8s %8s %8s %8s %6s\n" "σ" "F1" "recall" "den" "holds"
    rows = [run_sigma(σ) for σ in SIGMAS]
    for row in rows
        @printf "  %8.3f %8.3f %8.3f %8s %6s\n" row.σ row.f1 row.recall string(row.den) string(row.holds)
    end
    first_break = findfirst(row -> !row.holds, rows)
    if first_break === nothing
        println("  holds on the whole grid")
    else
        println("  first break at σ = ", rows[first_break].σ,
                " (Occam 0.99 / recall 0.99 / den=0)")
    end
    println("  UDE protocol is measured at σ = 0.00 and σ = 0.02 in the hard job;",
            " it is not claimed at σ ≥ 0.05.")
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
