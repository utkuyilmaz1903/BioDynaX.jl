#!/usr/bin/env julia
# Figure of the laccase/ABTS case study from the files written by
# run_case_study.jl: left, the measured progress curves (points) with the
# hybrid model that uses the discovered rate (lines; the trained model when
# discovery failed), training curves in grey and held-out curves in colour;
# right, the learned per-concentration destruction rate D(ABTS) sampled on
# the regulator grid and the discovered rational form when there is one.
# Writes docs/src/assets/laccase_abts.png.
#
# Requires Plots.jl in the active environment, for example:
#   julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=".");
#             Pkg.add("Plots"); include("examples/laccase_abts/plot_case_study.jl")'
# Not run by the test suite or CI.

using DelimitedFiles
using Plots

const RESULTS_DIR = joinpath(@__DIR__, "data", "results")
const PNG_PATH = length(ARGS) >= 1 ? ARGS[1] :
                 joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "laccase_abts.png")

function read_table(path)
    table, header = readdlm(path, ','; header = true)
    columns = Dict(String(name) => table[:, i] for (i, name) in enumerate(vec(header)))
    return columns, size(table, 1)
end

function main()
    fits, n = read_table(joinpath(RESULTS_DIR, "fits.csv"))
    rates, _ = read_table(joinpath(RESULTS_DIR, "rate_samples.csv"))
    hybrid_available = all(isfinite, fits["hybrid_model_uM"])
    model_column = hybrid_available ? "hybrid_model_uM" : "trained_model_uM"
    left = plot(; xlabel = "time (s)", ylabel = "ABTS (µM)",
        title = hybrid_available ? "progress curves and hybrid model" :
                "progress curves and trained model (discovery failed)",
        titlefontsize = 10, legend = :topright, margin = 5Plots.mm)
    experiments = sort(unique(Int.(fits["experiment"])))
    held_colours = Dict{Int, Int}()
    for e in experiments
        rows = findall(==(e), Int.(fits["experiment"]))
        held = fits["held_out"][rows[1]] in (true, "true")
        curve = Int(fits["curve"][rows[1]])
        colour = held ? get!(held_colours, curve, length(held_colours) + 1) : :grey
        label = held && !(curve in keys(held_colours)) ? "" : ""
        scatter!(left, fits["time_s"][rows], fits["abts_uM"][rows];
            color = colour, markersize = 2.5, markerstrokewidth = 0, label = "")
        plot!(left, fits["time_s"][rows], fits[model_column][rows];
            color = colour, linewidth = held ? 2 : 1, label = "")
    end
    plot!(left, Float64[], Float64[]; color = :grey, label = "training curves")
    plot!(left, Float64[], Float64[]; color = 1, label = "held-out curves (25 and 75 µM)")
    right = plot(; xlabel = "ABTS (µM)", ylabel = "destruction rate D (1/s)",
        title = all(isfinite, rates["discovered_rate_per_s"]) ?
                "learned rate and discovered form" :
                "learned rate (no rational form accepted)", titlefontsize = 10,
        legend = :topright, margin = 5Plots.mm)
    plot!(right, rates["abts_uM"], rates["learned_rate_per_s"]; linewidth = 3,
        label = "learned D(ABTS)")
    if all(isfinite, rates["discovered_rate_per_s"])
        plot!(right, rates["abts_uM"], rates["discovered_rate_per_s"]; linewidth = 2,
            linestyle = :dash, label = "discovered rational form")
    end
    if all(isfinite, rates["discovered_stable_rate_per_s"])
        plot!(right, rates["abts_uM"], rates["discovered_stable_rate_per_s"];
            linewidth = 2, linestyle = :dot, label = "with stability selection")
    end
    figure = plot(left, right; layout = (1, 2), size = (1280, 420), dpi = 150)
    mkpath(dirname(abspath(PNG_PATH)))
    savefig(figure, PNG_PATH)
    println("wrote ", PNG_PATH)
end

main()
