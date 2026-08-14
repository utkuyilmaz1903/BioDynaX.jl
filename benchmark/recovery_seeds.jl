#!/usr/bin/env julia
# Multi-seed recovery report. CI stays on seeds 103/104.
# Default: cheap analytical Occam on five seeds.
# Optional: `--ude` runs the Hill UDE hard protocol (not a CI job).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using BioDynaX:
    run_recovery_suite, hill_rate_truth, hill_rate_support, support_f1,
    rate_discovery_config, discover_unknown_rate, RECOVERY_THRESHOLDS
using Printf
using Random
using Statistics

const SEEDS = (103, 107, 111, 113, 127)

function analytical_occam(seed::Int)
    r = collect(range(0.1, 2.0; length = 180))
    rng = MersenneTwister(seed)
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    amp = max(maximum(abs, D), eps(Float64))
    D_noisy = D .+ 0.005 .* amp .* randn(rng, length(r))
    result = discover_unknown_rate(
        reshape(r, 1, :), collect(range(0.0, 1.0; length = length(r))),
        reshape(D_noisy, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = seed),
        verbose = false, strict = false)
    truth = hill_rate_support(2)
    metrics = result.success ?
        support_f1(result.candidates[1], truth.numerator, truth.denominator) :
        nothing
    return (;
        seed,
        success = result.success,
        f1 = metrics === nothing ? 0.0 : metrics.combined.f1,
        recall = metrics === nothing ? 0.0 : metrics.combined.recall,
        gate = metrics !== nothing &&
            metrics.combined.f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean)
end

function ude_hill(seed::Int)
    report = run_recovery_suite(MersenneTwister(seed);
                                sections = (:ude_discovery,))
    u = report[:ude_discovery]
    return (;
        seed,
        nn_rate_rmse = u.nn_rate_rmse,
        support_recall = u.support_recall,
        support_f1 = u.support_f1,
        data_residual = u.data_residual,
        unidentifiable_edge = u.identifiability.unidentifiable_edge,
        gate = u.nn_rate_rmse ≤ RECOVERY_THRESHOLDS.nn_rate_rmse &&
            u.support_recall ≥ RECOVERY_THRESHOLDS.support_recall &&
            u.support_f1 ≥ RECOVERY_THRESHOLDS.support_f1_ude &&
            u.data_residual ≤ RECOVERY_THRESHOLDS.data_residual &&
            u.identifiability.unidentifiable_edge)
end

function _summarize(name, rows, fields)
    println(name)
    @printf "  %-6s" "seed"
    for field in fields
        @printf " %14s" field
    end
    println("     gate")
    for row in rows
        @printf "  %-6d" row.seed
        for field in fields
            val = getfield(row, field)
            if val isa Bool
                @printf " %14s" string(val)
            else
                @printf " %14.4f" val
            end
        end
        println(row.gate ? "      yes" : "       no")
    end
    for field in fields
        vals = [getfield(row, field) for row in rows]
        eltype(vals) <: Number || continue
        @printf "  %-6s %14.4f %14.4f %14.4f\n" field median(vals) minimum(vals) maximum(vals)
    end
    println("  passed ", count(row -> row.gate, rows), "/", length(rows),
            "  (CI remains a single-seed red gate)")
end

function main(args = ARGS)
    rows = [analytical_occam(seed) for seed in SEEDS]
    _summarize("Analytical Occam (0.5% Hill, same library)", rows,
               (:f1, :recall))
    if "--ude" in args
        ude_rows = [ude_hill(seed) for seed in SEEDS]
        _summarize("UDE Hill (9 ICs; not a CI job)", ude_rows,
                   (:nn_rate_rmse, :support_recall, :support_f1,
                    :data_residual))
    end
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
