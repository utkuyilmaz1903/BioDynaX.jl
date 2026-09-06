#!/usr/bin/env julia
# Preprocessing of the single-cell p53 traces for the p53–Mdm2 case study.
#
# Reads `dataset.csv` and `classes.csv` unpacked by download_data.jl: one row
# per cell, a class (cell line and dose, for example `MCF7_4` is MCF7 at
# 4 Gy), and 96 p53-YFP values at 15 min intervals from 0 to 1425 min.
# Builds a BioDynaX `ExperimentSet` with two states, P (p53, observed) and M
# (Mdm2, never observed in this dataset), one experiment per cell.
#
# Choices, all fixed before the first run:
#   1. Cell line MCF7, the line of Geva-Zatorsky et al. (2006), at 4 Gy: the
#      2006 study irradiated with 5 Gy, which is not among the deposit's
#      doses (1, 2, 4, 6, 8 Gy); of the two nearest doses the lower one is
#      taken because the 2017 study reports pulses broadening into a single
#      peak at the higher doses.
#   2. Oscillating cells: at least P53_MIN_PEAKS (3) peaks within the 24 h, a
#      peak being a local maximum that exceeds both neighbouring minima by at
#      least P53_MIN_PROMINENCE (0.10) of the trace's maximum and lies at
#      least P53_MIN_SEPARATION_MIN (120 min) from the previous peak. The
#      2017 study reports autocorrelations, not a per-cell criterion, so this
#      written rule is applied uniformly to every cell.
#   3. Each trace is divided by its own maximum, so P is in units of the
#      cell's maximum p53-YFP fluorescence (arbitrary units). Time is in
#      hours. Nothing is smoothed, trimmed, or interpolated; the 96 points
#      enter as measured (the grid is already common to all cells).
#   4. M is unobserved: its observation row is NaN and masked out of the loss;
#      its initial value is set equal to the cell's normalised p53 at t = 0
#      (M and P start at the same normalised level), a stated assumption.
#   5. At most P53_MAX_CELLS (40) selected cells, in the file's order; every
#      fifth selected cell is held out (8 of 40, 20%), and the held-out cells
#      are placed last so that `discover_unknown_term(...; holdout = n)`
#      holds them out.
#
# Run:  julia --project=. examples/p53_mdm2/preprocess.jl
# writes data/p53_selected.csv (the selected cells, normalised) and prints a
# summary.

using BioDynaX
using DelimitedFiles
using Statistics

const P53_CELL_LINE = "MCF7"
const P53_DOSE_GY = 4
const P53_MIN_PEAKS = 3
const P53_MIN_PROMINENCE = 0.10
const P53_MIN_SEPARATION_MIN = 120.0
const P53_MAX_CELLS = 40
const P53_HOLDOUT_EVERY = 5

"""
    read_p53_dataset(dir) -> NamedTuple

All traces of `dataset.csv` in `dir`: ids, class names, the time vector in
minutes, and the value matrix (cells × times).
"""
function read_p53_dataset(dir::AbstractString)
    table, header = readdlm(joinpath(dir, "dataset.csv"), ','; header = true)
    header = vec(String.(header))
    header[1] == "ID" && header[2] == "class" || error("unexpected dataset.csv header")
    times = [parse(Float64, replace(name, "P53_" => "")) for name in header[3:end]]
    classes = Dict{Int, String}()
    ctable, _ = readdlm(joinpath(dir, "classes.csv"), ','; header = true)
    for i in 1:size(ctable, 1)
        classes[Int(ctable[i, 1])] = String(ctable[i, 2])
    end
    ids = String.(table[:, 1])
    class_names = [classes[Int(table[i, 2])] for i in 1:size(table, 1)]
    values = Float64.(table[:, 3:end])
    return (; ids, class_names, times, values)
end

"""
    count_peaks(trace, times; min_prominence, min_separation) -> Int

Local maxima that exceed both neighbouring minima by `min_prominence` times
the trace maximum and lie `min_separation` after the previous counted peak.
"""
function count_peaks(trace::AbstractVector, times::AbstractVector;
        min_prominence = P53_MIN_PROMINENCE, min_separation = P53_MIN_SEPARATION_MIN)
    n = length(trace)
    threshold = min_prominence * maximum(trace)
    peaks = 0
    last_peak = -Inf
    for i in 2:(n - 1)
        trace[i] > trace[i - 1] && trace[i] >= trace[i + 1] || continue
        left = minimum(@view trace[1:i])
        right = minimum(@view trace[i:n])
        trace[i] - left >= threshold && trace[i] - right >= threshold || continue
        times[i] - last_peak >= min_separation || continue
        peaks += 1
        last_peak = times[i]
    end
    return peaks
end

"""
    select_p53_cells(data; cell_line, dose_gy, min_peaks, max_cells) -> Vector{Int}

Row indices of the cells of `cell_line` at `dose_gy` with at least `min_peaks`
peaks, in file order, at most `max_cells` of them.
"""
function select_p53_cells(data; cell_line = P53_CELL_LINE, dose_gy = P53_DOSE_GY,
        min_peaks = P53_MIN_PEAKS, max_cells = P53_MAX_CELLS)
    class_name = string(cell_line, "_", dose_gy)
    selected = Int[]
    for i in eachindex(data.ids)
        data.class_names[i] == class_name || continue
        count_peaks(view(data.values, i, :), data.times) >= min_peaks || continue
        push!(selected, i)
        length(selected) >= max_cells && break
    end
    return selected
end

"""
    p53_experiment_set(dir; kwargs...) -> (set, info)

`ExperimentSet` of the selected cells: states `[:P, :M]`, P normalised to the
cell's maximum, M unobserved (NaN, masked), time in hours; training cells
first, held-out cells (every `P53_HOLDOUT_EVERY`-th selected cell) last.
"""
function p53_experiment_set(dir::AbstractString; holdout_every = P53_HOLDOUT_EVERY,
        kwargs...)
    data = read_p53_dataset(dir)
    selected = select_p53_cells(data; kwargs...)
    isempty(selected) && error("no cell passed the selection rule")
    held = [selected[k] for k in eachindex(selected) if k % holdout_every == 0]
    train = [selected[k] for k in eachindex(selected) if k % holdout_every != 0]
    order = vcat(train, held)
    hours = data.times ./ 60
    experiments = Experiment{Float64, Matrix{Float64}, Vector{Float64}}[]
    for i in order
        trace = data.values[i, :]
        p = trace ./ maximum(trace)
        observations = vcat(reshape(p, 1, :), fill(NaN, 1, length(p)))
        push!(experiments, Experiment(Symbol(data.ids[i]), hours, observations,
            [p[1], p[1]];
            metadata = Dict{Symbol, Any}(:cell => data.ids[i],
                :class => data.class_names[i], :maximum_yfp => maximum(trace),
                :peaks => count_peaks(trace, data.times), :held_out => i in held)))
    end
    set = ExperimentSet(experiments, [:P, :M]; units = [:fraction_of_maximum, :unobserved],
        metadata = Dict{Symbol, Any}(:time_unit => "h",
            :source => "Mendeley Data 10.17632/4vnndy59fp.2, p53_DoseCellLine"))
    candidates = count(==(string(P53_CELL_LINE, "_", P53_DOSE_GY)), data.class_names)
    passing = count(i -> data.class_names[i] == string(P53_CELL_LINE, "_", P53_DOSE_GY) &&
                         count_peaks(view(data.values, i, :), data.times) >= P53_MIN_PEAKS,
        eachindex(data.ids))
    info = (; n_cells_in_class = candidates, n_passing = passing,
        n_selected = length(selected), n_train = length(train), holdout = length(held),
        n_points = length(hours), duration_h = last(hours),
        interval_min = data.times[2] - data.times[1],
        selected_ids = data.ids[order])
    return set, info
end

"""Write the selected cells: cell, held_out, time_min, p53_yfp, p53_normalised."""
function write_p53_selected(path::AbstractString, set::ExperimentSet)
    open(path, "w") do io
        println(io, "cell,class,held_out,peaks,time_min,p53_yfp,p53_normalised")
        for e in set.experiments, (j, t) in enumerate(e.times)
            p = e.observations[1, j]
            println(io, e.metadata[:cell], ",", e.metadata[:class], ",",
                e.metadata[:held_out], ",", e.metadata[:peaks], ",", round(t * 60; digits = 3),
                ",", p * e.metadata[:maximum_yfp], ",", p)
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    include(joinpath(@__DIR__, "download_data.jl"))
    dir = download_p53_data()
    set, info = p53_experiment_set(dir)
    path = write_p53_selected(joinpath(P53_DATA_DIR, "p53_selected.csv"), set)
    println("wrote ", path)
    println(P53_CELL_LINE, " at ", P53_DOSE_GY, " Gy: ", info.n_cells_in_class, " cells, ",
        info.n_passing, " with at least ", P53_MIN_PEAKS, " peaks, ", info.n_selected,
        " selected (", info.n_train, " training, ", info.holdout, " held out); ",
        info.n_points, " points at ", info.interval_min, " min over ", info.duration_h, " h")
    for e in set.experiments[1:3]
        println("  ", e.name, ": peaks ", e.metadata[:peaks], ", maximum YFP ",
            round(e.metadata[:maximum_yfp]; digits = 3), ", u0 ", e.u0)
    end
end
