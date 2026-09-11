#!/usr/bin/env julia
# Multi-term study (0.16): two unknown destruction terms on non-adjacent nodes
# (fixture `separate`), on adjacent nodes (`coupled`), and three unknown terms
# (`three`), each with its single-unknown controls, on the study grid of seeds
# 103, 107, 111, 113, 127 and observation noise 0.0, 0.02, 0.05, with the
# discovery run without and with stability selection on the same trained
# model. Rows are appended to benchmark/results/multi_term_study.csv as each
# run finishes and a rerun skips the runs already in the file.
# Run:  julia --project=. benchmark/multi_term_study.jl [options]
#   --fixtures separate,coupled   fixtures to run (default: separate,coupled)
#   --seeds 103,107               comma-separated seeds
#   --noise 0.0,0.02              comma-separated noise levels
#   --no-controls                 skip the single-unknown controls
#   --out PATH                    CSV path (default: benchmark/results/multi_term_study.csv)
#   --timed                       one run (separate, all unknown, seed 103, noise 0.0),
#                                 prints its wall time and exits without writing
using HybridKinetics
using Random

function _option(args, name, default)
    i = findfirst(==(name), args)
    return i === nothing || i == length(args) ? default : args[i + 1]
end
const ARGS_ = copy(ARGS)
fixtures = Tuple(Symbol.(split(_option(ARGS_, "--fixtures", "separate,coupled"), ",")))
seeds = Tuple(parse.(Int,
    split(_option(ARGS_, "--seeds",
            join(HybridKinetics.MULTI_TERM_STUDY_SEEDS, ",")), ",")))
noise = Tuple(parse.(Float64,
    split(
        _option(ARGS_, "--noise",
            join(HybridKinetics.MULTI_TERM_STUDY_NOISE_LEVELS, ",")),
        ",")))
out = _option(ARGS_, "--out", joinpath(@__DIR__, "results", "multi_term_study.csv"))
controls = !("--no-controls" in ARGS_)

if "--timed" in ARGS_
    t = @elapsed rows = HybridKinetics.multi_term_study_run(;
        fixture = :separate, seed = 103, noise_σ = 0.0, unknown = :all)
    println("one run (separate, S1+S2 unknown, seed 103, noise 0.0): ",
        round(t; digits = 1), " s; training ", round(rows[1].train_time_s; digits = 1), " s")
    for r in rows
        println("  ", r.variant, " ", r.node, ": F1 ", round(r.support_f1; digits = 2),
            ", rate rmse ", round(r.nn_rate_rmse; digits = 3), ", bias ",
            round(r.nn_rate_bias; digits = 3), ", cross-term ", round(
                r.cross_term_max; digits = 3))
    end
    exit()
end

HybridKinetics.multi_term_study(; fixtures = fixtures, seeds = seeds, noise_levels = noise,
    controls = controls, out = out)
println("Julia ", VERSION, ", HybridKinetics ", HybridKinetics.PACKAGE_VERSION)
