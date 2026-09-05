#!/usr/bin/env julia
# Library comparison study: trains the four-state trained-model library check
# once per seed and observation-noise level, then scores the graph-local, the
# global, and the wrong-graph discovery library on the same learned rate.
# Default study: seeds 103, 107, 111, 113, 127; noise 0.0, 0.02, 0.05; all
# three libraries (15 trainings). Rows are appended to
# benchmark/results/library_comparison.csv as they finish, and a rerun skips
# the rows already in the file, so an interrupted study resumes where it
# stopped. Prints the median and interquartile range per library and noise
# level and the dependency versions of the run.
# Runs in the weekly heavy CI job.
# Runtime: about one hour on 4 cores for the default study. Measured for
# 0.11.0: 207 s for one run (188 s of training) with `--timed`.
# Run:  julia --project=. benchmark/library_comparison_study.jl [options]
#   --timed               one seed, one noise level; prints the extrapolated
#                         wall time of the default study and exits
#   --seeds 103,107       comma-separated seeds (default: the study seeds)
#   --noise 0.0,0.02      comma-separated noise levels (default: 0.0,0.02,0.05)
#   --out PATH            CSV path (default: benchmark/results/library_comparison.csv)
#   --fresh               ignore rows already in the CSV (the file is replaced)
#   --pruning             graph-local library only, with the stability-selection
#                         stage on (StabilitySelection() defaults); rows go to
#                         benchmark/results/library_comparison_pruned.csv
#   --fixture two_state   the two-state reference-protocol network instead of
#                         the four-state network; rows go to
#                         benchmark/results/library_comparison_two_state.csv
#   --variants all        every discovery variant (study, bootstrap, parents,
#                         reference) from the same trainings; rows go to
#                         benchmark/results/library_comparison_variants.csv
#                         (two_state: library_comparison_two_state_variants.csv)

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using LinearAlgebra
using Statistics

LinearAlgebra.BLAS.set_num_threads(1)

function _option(args, name, default)
    index = findfirst(==(name), args)
    index === nothing && return default
    index < length(args) || error("$(name) needs a value")
    return args[index + 1]
end

const ARGS_ = copy(ARGS)
const TIMED = "--timed" in ARGS_
const FRESH = "--fresh" in ARGS_
const PRUNING = "--pruning" in ARGS_
const FIXTURE = Symbol(_option(ARGS_, "--fixture", "four_state"))
const VARIANTS = _option(ARGS_, "--variants", "study") == "all" ?
                 BioDynaX.LIBRARY_STUDY_VARIANTS :
                 Tuple(Symbol.(split(_option(ARGS_, "--variants", "study"), ",")))
const OUT = _option(ARGS_, "--out",
    joinpath(@__DIR__, "results",
        string("library_comparison",
            FIXTURE === :four_state ? "" : "_" * string(FIXTURE),
            length(VARIANTS) > 1 ? "_variants" : "",
            PRUNING ? "_pruned" : "", ".csv")))
const LIBRARIES = PRUNING ? (:graph_local,) : BioDynaX.LIBRARY_STUDY_LIBRARIES
const SELECTION = PRUNING ? StabilitySelection() : nothing
const SEEDS = TIMED ? (first(BioDynaX.LIBRARY_STUDY_SEEDS),) :
              Tuple(parse.(
    Int, split(_option(ARGS_, "--seeds",
            join(BioDynaX.LIBRARY_STUDY_SEEDS, ",")), ",")))
const NOISE = TIMED ? (first(BioDynaX.LIBRARY_STUDY_NOISE_LEVELS),) :
              Tuple(parse.(Float64,
    split(_option(ARGS_, "--noise",
            join(BioDynaX.LIBRARY_STUDY_NOISE_LEVELS, ",")), ",")))

"""Versions of the packages that determine the numerical results."""
function dependency_versions(names = ("OrdinaryDiffEq", "SciMLSensitivity", "Lux",
        "Optimization", "Zygote", "SciMLBase"))
    versions = Dict(info.name => info.version for info in values(Pkg.dependencies()))
    return join((string(name, " ", get(versions, name, "not loaded")) for name in names),
        ", ")
end

function main()
    println("Library comparison study")
    println("Julia ", VERSION, "; ", dependency_versions())
    println("fixture: ", FIXTURE, "; variants: ", join(VARIANTS, ", "))
    println("seeds: ", join(SEEDS, ", "), "; noise: ", join(NOISE, ", "),
        "; libraries: ", join(LIBRARIES, ", "),
        "; stability selection: ", SELECTION === nothing ? "off" :
                                   string("n_boot ", SELECTION.n_boot, ", τ ", SELECTION.τ))
    if FRESH && isfile(OUT)
        rm(OUT)
    end
    existing = TIMED ? NamedTuple[] : BioDynaX.read_library_study_csv(OUT)
    done = BioDynaX.library_study_keys(existing)
    isempty(done) || println("resuming: ", length(done), " rows already in ", OUT)
    started = time()
    rows = BioDynaX.library_comparison_study(;
        seeds = SEEDS, noise_levels = NOISE, libraries = LIBRARIES,
        fixture = FIXTURE, variants = VARIANTS,
        stability_selection = SELECTION,
        skip = (seed, noise, library, variant) -> (seed, noise, library, variant) in done,
        on_row = row -> begin
            TIMED || BioDynaX.append_library_study_row(OUT, row)
            println("  ", row.seed, "  ", row.noise, "  ", rpad(string(row.variant), 10),
                rpad(string(row.library), 12),
                " F1 ", round(row.support_f1; digits = 3),
                "  recall ", round(row.support_recall; digits = 3),
                "  extras ", row.extra_terms, isempty(row.extras) ? "" :
                                              " (" * row.extras * ")",
                "  holdout ", round(row.holdout_residual; sigdigits = 4),
                "  nn_rmse ", round(row.nn_rate_rmse; sigdigits = 4),
                "  train ", round(row.train_time_s; digits = 1), " s")
        end,
        verbose = true)
    elapsed = time() - started
    if TIMED
        n_default = length(BioDynaX.LIBRARY_STUDY_SEEDS) *
                    length(BioDynaX.LIBRARY_STUDY_NOISE_LEVELS)
        println("one run (training, sampling, three discoveries, residuals): ",
            round(elapsed; digits = 1), " s")
        println("extrapolated default study (", n_default, " runs): ",
            round(elapsed * n_default / 60; digits = 1), " min")
        return
    end
    all_rows = vcat(existing, rows)
    println()
    println("new rows: ", length(rows), " in ", round(elapsed / 60; digits = 1),
        " min; total rows in ", OUT, ": ", length(all_rows))
    println()
    println("Summary (median [q25, q75] over seeds; fixture ", FIXTURE, "):")
    print(BioDynaX.format_library_study_summary(
        BioDynaX.library_study_summary(all_rows;
            metrics = (:support_f1, :support_recall, :extra_terms, :holdout_residual,
                :nn_rate_rmse));
        metrics = (:support_f1, :support_recall, :extra_terms, :holdout_residual,
            :nn_rate_rmse)))
    println()
    println("Environment: Julia ", VERSION, "; ", dependency_versions())
end

main()
