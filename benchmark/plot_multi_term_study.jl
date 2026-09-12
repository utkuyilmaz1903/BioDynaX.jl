#!/usr/bin/env julia
# Figure of the multi-term study: per-term support F1 (left) and relative
# RMSE of the learned rate (right), medians over seeds, against observation
# noise, for the single-unknown control and the two-unknown run of the
# separate and the coupled fixture, without stability selection. Series are
# offset slightly along the noise axis so that coinciding medians stay
# visible.
# Run:  julia --project=. benchmark/plot_multi_term_study.jl [CSV] [PNG]
# (Plots must be installed in the active environment.)
using HybridKinetics, Plots, Statistics
csv = length(ARGS) ≥ 1 ? ARGS[1] : joinpath(@__DIR__, "results", "multi_term_study.csv")
png = length(ARGS) ≥ 2 ? ARGS[2] :
      joinpath(@__DIR__, "..", "docs", "src", "assets", "multi_term_f1.png")
rows = HybridKinetics.read_multi_term_csv(csv)
isempty(rows) && error("no rows in $(csv)")
summary = HybridKinetics.multi_term_study_summary(rows)
noise_levels = sort(unique(r.noise for r in rows))
step = length(noise_levels) > 1 ? minimum(diff(noise_levels)) : 0.02
series = []
for fixture in (:separate, :coupled)
    fx = HybridKinetics.multi_term_fixture(fixture)
    label_all = Symbol(join(string.(fx.nodes), "+"))
    for node in fx.nodes,
        (unknown, kind) in ((label_all, "both unknown"), (node, "control"))

        pts = [(s.noise, s.f1_median, s.rmse_median)
               for s in summary
               if s.fixture == fixture && s.unknown == unknown && s.variant == :plain &&
                  s.node == node]
        isempty(pts) || push!(series, (fixture, node, kind, sort(pts)))
    end
end
offsets = range(-0.12step, 0.12step; length = max(length(series), 2))
markers = Dict(:separate => :circle, :coupled => :square)
function panel(index, ylabel; ylims)
    plt = plot(; xlabel = "observation noise", ylabel = ylabel, ylims = ylims,
        legend = index == 2 ? :bottomright : :bottomleft, legendfontsize = 7)
    for (k, (fixture, node, kind, pts)) in enumerate(series)
        x = first.(pts) .+ offsets[k]
        y = index == 1 ? getindex.(pts, 2) : getindex.(pts, 3)
        plot!(plt, x, y; marker = markers[fixture], markersize = 5,
            linestyle = kind == "control" ? :dot : :solid, alpha = kind == "control" ? 0.7 :
                                                                   1.0,
            label = "$(fixture), $(node), $(kind)")
    end
    return plt
end
fig = plot(panel(1, "support F1 (median over seeds)"; ylims = (0, 1.05)),
    panel(2, "learned rate, relative RMSE (median)"; ylims = (0, :auto));
    layout = (1, 2), size = (1100, 430), dpi = 150, left_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm)
mkpath(dirname(png))
savefig(fig, png)
println("wrote ", png)
