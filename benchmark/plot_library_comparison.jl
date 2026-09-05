#!/usr/bin/env julia
# Figure of the library comparison study: support F1 against observation
# noise, one line per discovery library (graph-local, global, wrong graph),
# median over seeds with the interquartile range as a band. One panel per
# fixture: the four-state network (benchmark/results/library_comparison.csv,
# or library_comparison_variants.csv) and the two-state network
# (library_comparison_two_state.csv, or library_comparison_two_state_variants.csv),
# all written by benchmark/library_comparison_study.jl. Only the :study
# variant is drawn. A panel whose file is missing is skipped. Writes
# docs/src/assets/library_comparison.png.
#
# Requires Plots.jl in the active environment, for example:
#   julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=".");
#             Pkg.add("Plots"); include("benchmark/plot_library_comparison.jl")'
# Runtime: under a minute after precompilation. Not run in CI.
# Run:  julia benchmark/plot_library_comparison.jl [png path]

using BioDynaX
using Plots

const PNG_PATH = length(ARGS) >= 1 ? ARGS[1] :
                 joinpath(@__DIR__, "..", "docs", "src", "assets",
    "library_comparison.png")

const PANELS = (
    (fixture = :four_state, title = "four-state network",
        csv = (joinpath(@__DIR__, "results", "library_comparison.csv"),
            joinpath(@__DIR__, "results", "library_comparison_variants.csv"))),
    (fixture = :two_state, title = "two-state network",
        csv = (joinpath(@__DIR__, "results", "library_comparison_two_state.csv"),
            joinpath(@__DIR__, "results", "library_comparison_two_state_variants.csv"))))

const LABELS = Dict(
    :graph_local => "graph-local library",
    :global => "global library",
    :wrong_graph => "wrong-graph library")

# Distinct styles so that coinciding lines (graph-local and global on a
# two-state network) stay visible.
const STYLES = Dict(
    :graph_local => (linestyle = :solid, linewidth = 4, marker = :circle, markersize = 6),
    :global => (linestyle = :dash, linewidth = 2, marker = :square, markersize = 4),
    :wrong_graph => (linestyle = :dot, linewidth = 2, marker = :diamond, markersize = 4))

function panel(rows, title)
    rows = [row for row in rows if row.variant === :study]
    summary = BioDynaX.library_study_summary(rows; metrics = (:support_f1,))
    seeds = length(unique(row.seed for row in rows))
    figure = plot(;
        xlabel = "observation noise (standard deviation)",
        ylabel = "support F1 (median, interquartile band)",
        title = "$(title), $(seeds) seeds",
        titlefontsize = 10, legend = :topright, ylims = (-0.02, 1.02),
        margin = 5Plots.mm)
    for library in BioDynaX.LIBRARY_STUDY_LIBRARIES
        entries = [entry for entry in summary if entry.library === library]
        isempty(entries) && continue
        x = [entry.noise for entry in entries]
        y = [entry.support_f1_median for entry in entries]
        lo = [entry.support_f1_median - entry.support_f1_q25 for entry in entries]
        hi = [entry.support_f1_q75 - entry.support_f1_median for entry in entries]
        plot!(figure, x, y; ribbon = (lo, hi), fillalpha = 0.15,
            STYLES[library]..., label = LABELS[library])
    end
    xticks!(figure, sort(unique(row.noise for row in rows)))
    return figure
end

function main()
    panels = []
    for spec in PANELS
        path = findfirst(isfile, spec.csv)
        rows = path === nothing ? NamedTuple[] :
               BioDynaX.read_library_study_csv(spec.csv[path])
        if isempty(rows)
            println("no rows for the ", spec.title, "; panel skipped")
            continue
        end
        push!(panels, panel(rows, spec.title))
    end
    isempty(panels) && error("no study CSV found; run library_comparison_study.jl first")
    figure = plot(panels...; layout = (1, length(panels)),
        size = (640 * length(panels), 420), dpi = 150)
    mkpath(dirname(abspath(PNG_PATH)))
    savefig(figure, PNG_PATH)
    println("wrote ", PNG_PATH)
end

main()
