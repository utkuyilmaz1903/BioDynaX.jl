#!/usr/bin/env julia
# Preprocessing of the laccase/ABTS progress curves for the enzyme case study.
#
# Reads the unpacked EnzymeML document written by download_data.jl: nine
# measurements ("Oxidation 1" to "Oxidation 9"), each a CSV with a time column
# (seconds) and three replicate columns of the ABTS concentration (µmol/l),
# with the initial concentrations and the enzyme concentration declared in
# experiment.xml. Builds a BioDynaX `ExperimentSet` in which every replicate
# curve is one experiment.
#
# Preprocessing, and nothing else:
#   1. Concentrations are divided by ABTS_SCALE (100 µmol/l) and times by
#      TIME_SCALE (1000 s), so the model state is ABTS/100 µM and its time unit
#      is 1000 s. Nothing is smoothed, trimmed, or interpolated; the 21 points
#      of every curve enter as measured.
#   2. The initial condition of a replicate is its own measured value at t = 0.
#   3. The curves listed in `holdout_curves` (by measurement number, 1 to 9,
#      in the document's order of increasing initial ABTS) are placed last so
#      that `discover_unknown_term(...; holdout = 3 * length(holdout_curves))`
#      holds out all their replicates.
#
# Run:  julia --project=. examples/laccase_abts/preprocess.jl
# writes data/abts_tidy.csv (one row per measurement point) and prints a summary.

using BioDynaX
using DelimitedFiles
using Statistics

const ABTS_SCALE = 100.0    # µmol/l per model unit
const TIME_SCALE = 1000.0   # seconds per model time unit
const ABTS_HOLDOUT_CURVES = (3, 7)   # 25 µM and 75 µM nominal initial ABTS

"""
    read_abts_document(dir) -> NamedTuple

Measurements of the unpacked EnzymeML document in `dir`: for each, the id,
name, nominal initial ABTS (µmol/l), enzyme concentration (µmol/l), time
vector (s), and replicate matrix (3 × n, µmol/l).
"""
function read_abts_document(dir::AbstractString)
    xml = read(joinpath(dir, "experiment.xml"), String)
    measurements = NamedTuple[]
    for block in eachmatch(
        r"<enzymeml:measurement file=\"(file\d+)\" id=\"(m\d+)\" name=\"([^\"]*)\">(.*?)</enzymeml:measurement>"s,
        xml)
        file_id, id, name, body = block.captures
        enzyme = parse(Float64,
            match(r"<enzymeml:initConc protein=\"p0\" value=\"([^\"]+)\"", body).captures[1])
        substrate = parse(Float64,
            match(r"<enzymeml:initConc reactant=\"s0\" value=\"([^\"]+)\"", body).captures[1])
        file = match(Regex("<enzymeml:file file=\"([^\"]+)\" format=\"format\\d+\" id=\"$(file_id)\""),
        xml).captures[1]
        table = readdlm(joinpath(dir, file), ',', Float64)
        push!(measurements,
            (; id = String(id), name = String(name),
                initial_abts = substrate, enzyme,
                times = table[:, 1], replicates = permutedims(table[:, 2:end])))
    end
    sort!(measurements; by = m -> m.initial_abts)
    units = Dict(m.captures[1] => m.captures[2]
    for m in eachmatch(
        r"<unitDefinition metaid=\"[^\"]+\" id=\"(u\d)\" name=\"([^\"]+)\"", xml))
    return (; measurements, units)
end

"""
    abts_experiment_set(dir; holdout_curves=ABTS_HOLDOUT_CURVES) -> (set, info)

`ExperimentSet` with one experiment per replicate curve (27 experiments), the
training curves first and the replicates of `holdout_curves` last, plus a
summary `info` (number of held-out experiments, scales, curve order).
"""
function abts_experiment_set(dir::AbstractString;
        holdout_curves = ABTS_HOLDOUT_CURVES)
    document = read_abts_document(dir)
    n = length(document.measurements)
    all(1 <= c <= n for c in holdout_curves) || throw(ArgumentError(
        "holdout_curves must index the $(n) measurements"))
    order = vcat([i for i in 1:n if !(i in holdout_curves)], collect(holdout_curves))
    experiments = Experiment{Float64, Matrix{Float64}, Vector{Float64}}[]
    for i in order
        m = document.measurements[i]
        for r in 1:size(m.replicates, 1)
            values = m.replicates[r, :] ./ ABTS_SCALE
            push!(experiments,
                Experiment(
                    Symbol(m.id, "_replicate_", r), m.times ./ TIME_SCALE,
                    reshape(values, 1, :), [values[1]];
                    metadata = Dict{Symbol, Any}(:curve => i, :replicate => r,
                        :initial_abts_uM => m.initial_abts, :enzyme_uM => m.enzyme)))
        end
    end
    set = ExperimentSet(experiments, [:ABTS]; units = [:per_100_uM],
        metadata = Dict{Symbol, Any}(:time_unit => "1000 s",
            :concentration_unit => "100 µmol/l", :source => "EnzymeML/Lauterbach_2022 Scenario4"))
    holdout = 3 * length(holdout_curves)
    info = (; n_curves = n, n_experiments = length(experiments), holdout,
        curve_order = order, abts_scale = ABTS_SCALE, time_scale = TIME_SCALE,
        enzyme_uM = unique(m.enzyme for m in document.measurements),
        initial_abts_uM = [m.initial_abts for m in document.measurements],
        n_points = length(first(document.measurements).times),
        duration_s = last(first(document.measurements).times))
    return set, info
end

"""Write one row per measured point: curve, replicate, time_s, abts_uM."""
function write_abts_tidy(path::AbstractString, dir::AbstractString)
    document = read_abts_document(dir)
    open(path, "w") do io
        println(io, "curve,measurement,initial_abts_uM,enzyme_uM,replicate,time_s,abts_uM")
        for (i, m) in enumerate(document.measurements), r in 1:size(m.replicates, 1),
            (j, t) in enumerate(m.times)

            println(io, i, ",", m.id, ",", m.initial_abts, ",", m.enzyme, ",", r, ",",
                t, ",", m.replicates[r, j])
        end
    end
    return path
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    include(joinpath(@__DIR__, "download_data.jl"))
    dir = download_abts_data()
    set, info = abts_experiment_set(dir)
    tidy = write_abts_tidy(joinpath(ABTS_DATA_DIR, "abts_tidy.csv"), dir)
    println("wrote ", tidy)
    println("curves: ", info.n_curves, " (initial ABTS ", join(info.initial_abts_uM, ", "),
        " µM; enzyme ", join(info.enzyme_uM, ", "), " µM); ", info.n_points,
        " points over ", info.duration_s, " s; ", info.n_experiments,
        " replicate experiments, ", info.holdout, " held out (curves ",
        join(ABTS_HOLDOUT_CURVES, " and "), ")")
    for e in set.experiments[1:3]
        println("  ", e.name, ": u0 ", e.u0, ", ", length(e.times), " points, range ",
            extrema(e.observations))
    end
end
