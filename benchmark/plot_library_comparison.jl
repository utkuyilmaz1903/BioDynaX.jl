#!/usr/bin/env julia
# Figure of the library comparison study: support F1 against observation
# noise, one line per discovery library (graph-local, global, wrong graph),
# median over seeds with the interquartile range as a band. Reads
# benchmark/results/library_comparison.csv written by
# benchmark/library_comparison_study.jl and writes
# docs/src/assets/library_comparison.png.
#
# Requires Plots.jl in the active environment, for example:
#   julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=".");
#             Pkg.add("Plots"); include("benchmark/plot_library_comparison.jl")'
# Runtime: under a minute after precompilation. Not run in CI.
# Run:  julia benchmark/plot_library_comparison.jl [csv path] [png path]

using BioDynaX
using Plots

const CSV_PATH = length(ARGS) >= 1 ? ARGS[1] :
                 joinpath(@__DIR__, "results", "library_comparison.csv")
const PNG_PATH = length(ARGS) >= 2 ? ARGS[2] :
                 joinpath(@__DIR__, "..", "docs", "src", "assets",
    "library_comparison.png")

const LABELS = Dict(
    :graph_local => "graph-local library",
    :global => "global library",
    :wrong_graph => "wrong-graph library")

function main()
    rows = BioDynaX.read_library_study_csv(CSV_PATH)
    isempty(rows) && error("no rows in $(CSV_PATH); run library_comparison_study.jl first")
    summary = BioDynaX.library_study_summary(rows; metrics = (:support_f1,))
    seeds = length(unique(row.seed for row in rows))
    figure = plot(;
        xlabel = "observation noise (standard deviation)",
        ylabel = "support F1 (median, interquartile band)",
        title = "Discovery library comparison, $(seeds) seeds",
        titlefontsize = 10, legend = :topright, ylims = (-0.02, 1.02),
        size = (640, 420), dpi = 150, margin = 5Plots.mm)
    for library in BioDynaX.LIBRARY_STUDY_LIBRARIES
        entries = [entry for entry in summary if entry.library === library]
        isempty(entries) && continue
        x = [entry.noise for entry in entries]
        y = [entry.support_f1_median for entry in entries]
        lo = [entry.support_f1_median - entry.support_f1_q25 for entry in entries]
        hi = [entry.support_f1_q75 - entry.support_f1_median for entry in entries]
        plot!(figure, x, y; ribbon = (lo, hi), fillalpha = 0.15,
            marker = :circle, markersize = 4, linewidth = 2,
            label = LABELS[library])
    end
    xticks!(figure, sort(unique(row.noise for row in rows)))
    mkpath(dirname(abspath(PNG_PATH)))
    savefig(figure, PNG_PATH)
    println("wrote ", PNG_PATH)
end

main()
