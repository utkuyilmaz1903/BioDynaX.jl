#!/usr/bin/env julia
# Figure of the p53–Mdm2 case study from the files written by run_case_study.jl:
# left, four cells (two training, two held out): normalised p53 (points), the
# hybrid model's p53 (solid) and its unobserved Mdm2 (dashed); right, the
# learned per-concentration destruction rate of p53 against the model's Mdm2
# with the discovered rational form. Writes docs/src/assets/p53_mdm2.png.
#
# Requires Plots.jl in the active environment, for example:
#   julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=".");
#             Pkg.add("Plots"); include("examples/p53_mdm2/plot_case_study.jl")'
# Not run by the test suite or CI.

using DelimitedFiles
using Plots

const RESULTS_DIR = joinpath(@__DIR__, "data", "results")
const PNG_PATH = length(ARGS) >= 1 ? ARGS[1] :
                 joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "p53_mdm2.png")

function read_table(path)
    table, header = readdlm(path, ','; header = true)
    return Dict(String(name) => table[:, i] for (i, name) in enumerate(vec(header)))
end

function main()
    fits = read_table(joinpath(RESULTS_DIR, "fits.csv"))
    rates = read_table(joinpath(RESULTS_DIR, "rate_samples.csv"))
    cells = unique(String.(fits["cell"]))
    held = Dict(c => fits["held_out"][findfirst(==(c), String.(fits["cell"]))] in (
                    true, "true")
    for c in cells)
    chosen = vcat(
        first([c for c in cells if !held[c]], 2), first([c for c in cells if held[c]], 2))
    panels = []
    for c in chosen
        rows = findall(==(c), String.(fits["cell"]))
        t = fits["time_h"][rows]
        panel = plot(; xlabel = "time (h)", ylabel = "fraction of the cell's maximum",
            title = string(c, held[c] ? " (held out)" : " (training)"), titlefontsize = 9,
            legend = c == chosen[1] ? :topright : false, ylims = (0, 1.4), margin = 3Plots.mm)
        scatter!(
            panel, t, fits["p53_normalised"][rows]; markersize = 2, markerstrokewidth = 0,
            color = :grey, label = "p53-YFP")
        source = all(isfinite, fits["hybrid_p53"][rows]) ? "hybrid" : "model"
        plot!(panel, t, fits[source * "_p53"][rows]; color = 1, linewidth = 2,
            label = source == "hybrid" ? "hybrid model p53" : "trained model p53")
        plot!(panel, t, fits[source * "_mdm2"][rows]; color = 2, linewidth = 2,
            linestyle = :dash, label = "model Mdm2 (unobserved)")
        push!(panels, panel)
    end
    right = plot(; xlabel = "Mdm2 (model units)", ylabel = "destruction rate of p53 (1/h)",
        title = "learned rate and discovered form", titlefontsize = 9, legend = :topleft,
        margin = 3Plots.mm)
    plot!(right, rates["mdm2_model_units"], rates["learned_rate_per_h"]; linewidth = 3,
        label = "learned D(Mdm2)")
    if all(isfinite, rates["discovered_rate_per_h"])
        plot!(
            right, rates["mdm2_model_units"], rates["discovered_rate_per_h"]; linewidth = 2,
            linestyle = :dash, label = "discovered rational form")
    end
    if all(isfinite, rates["discovered_stable_rate_per_h"])
        plot!(right, rates["mdm2_model_units"], rates["discovered_stable_rate_per_h"];
            linewidth = 2, linestyle = :dot, label = "with stability selection")
    end
    figure = plot(panels..., right; layout = @layout([a b; c d; e{0.4h}]),
        size = (1000, 1000), dpi = 150)
    mkpath(dirname(abspath(PNG_PATH)))
    savefig(figure, PNG_PATH)
    println("wrote ", PNG_PATH)
end

main()
