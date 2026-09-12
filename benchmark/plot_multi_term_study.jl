#!/usr/bin/env julia
# Figure of the multi-term study: per-term support F1 (median over seeds)
# against observation noise, for the single-unknown control, the separate
# fixture and the coupled fixture, without stability selection.
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
plt = plot(; xlabel = "observation noise", ylabel = "support F1 (median over seeds)",
    legend = :bottomleft, ylims = (0, 1.05), size = (720, 420), dpi = 150)
styles = Dict(:separate => (:circle, :solid), :coupled => (:square, :dash))
for fixture in (:separate, :coupled)
    fx = HybridKinetics.multi_term_fixture(fixture)
    label_all = Symbol(join(string.(fx.nodes), "+"))
    for node in fx.nodes
        both = [(s.noise, s.f1_median)
                for s in summary
                if s.fixture == fixture && s.unknown == label_all && s.variant == :plain &&
                   s.node == node]
        ctrl = [(s.noise, s.f1_median)
                for s in summary
                if s.fixture == fixture && s.unknown == node && s.variant == :plain &&
                   s.node == node]
        isempty(both) || plot!(plt, first.(both), last.(both); marker = styles[fixture][1],
            linestyle = styles[fixture][2], label = "$(fixture), $(node), both unknown")
        isempty(ctrl) || plot!(plt, first.(ctrl), last.(ctrl); marker = styles[fixture][1],
            linestyle = :dot, alpha = 0.6, label = "$(fixture), $(node), single-unknown control")
    end
end
mkpath(dirname(png))
savefig(plt, png)
println("wrote ", png)
