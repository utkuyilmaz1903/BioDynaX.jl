#!/usr/bin/env julia
# p53–Mdm2 case study on measured single-cell data (download_data.jl,
# preprocess.jl): MCF7 cells after 4 Gy of ionizing radiation, one
# experiment per cell, p53 observed, Mdm2 never observed. Two states, P (p53)
# and M (Mdm2); known: constant production of P, production of M
# proportional to P (the linear form of the Geva-Zatorsky 2006 models),
# linear degradation of M; unknown: the per-concentration destruction rate of
# P regulated by M, declared as a reaction with `known = false` and the edge
# P ← M:
#
#     dP/dt = k_prod - D(M) P,      dM/dt = k_m P - k_dm M.
#
# The workflow is `discover_unknown_term` with the reference defaults (warm-up
# on the first training cell, Adam 100 then BFGS 50, bootstrap 8, discovery
# seed 3), once without and once with stability selection. Because M is not
# observed, the learned rate is sampled on the range of the model's own M
# trajectories over the training cells (`regulator_grid`), 80 points, with
# the same margins as the observed grid. Nothing is tuned per cell.
#
# Writes to examples/p53_mdm2/data/results/ (not committed):
#   report_default.txt, report_stability.txt   four-section reports
#   selection_frequencies.txt                  stability-selection table
#   rate_samples.csv                           M, learned D, discovered D
#   fits.csv                                   data, trained model, hybrid model
#   summary.txt                                residuals, coefficients, environment
# Run:  julia --project=. examples/p53_mdm2/run_case_study.jl
# Smoke check (2 Adam steps, no BFGS, 4 resamples, 6 cells):
#   BIODYNAX_SMOKE=1 julia --project=. examples/p53_mdm2/run_case_study.jl
# Not run by the test suite or CI.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using BioDynaX
using LinearAlgebra
using OrdinaryDiffEq
using Random
using SciMLBase
using Statistics

LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "download_data.jl"))
include(joinpath(@__DIR__, "preprocess.jl"))

const SMOKE = get(ENV, "BIODYNAX_SMOKE", "") == "1"
const RESULTS_DIR = joinpath(P53_DATA_DIR, "results")
const P53_GRID_POINTS = 80

"""
Two states, P (p53, observed) and M (Mdm2, latent): constant production of
P (`k_prod`, with the input factor frozen at 1), production of M proportional
to P (`k_m`), linear degradation of M (`k_dm`), and the unknown destruction
of P regulated by M, declared as an unknown reaction and as the edge P ← M.
"""
function p53_mdm2_network()
    nodes = [NodeSpec(name = :P), NodeSpec(name = :M, kind = LATENT, observed = false)]
    hill = HillMetadata(vmax_param = :vmax, k_param = :K, hill_order = 2)
    edges = [EdgeSpec(source = 2, target = 1, kind = UNKNOWN_NN, known = false,
        family = HILL, metadata = hill)]
    reactions = [
        ReactionSpec(name = :produce_p, stoichiometry = Dict(1 => 1.0),
            regulators = Int[],
            metadata = InputDriveMetadata(rate_param = :k_prod, input_param = :input)),
        ReactionSpec(name = :mdm2_degrades_p53, stoichiometry = Dict(1 => -1.0),
            regulators = [2], known = false, family = HILL, metadata = hill),
        ReactionSpec(name = :produce_m, stoichiometry = Dict(2 => 1.0),
            regulators = [1], metadata = MassActionMetadata(rate_param = :k_m)),
        ReactionSpec(name = :decay_m, stoichiometry = Dict(2 => -1.0),
            regulators = Int[], metadata = LinearDecayMetadata(rate_param = :k_dm))]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

const P53_FROZEN = [:input]

"""Flat 0.8 for every fitted parameter, 1.0 for the frozen input factor."""
function p53_phys_init(network)
    model, _ = build_ude_model(MersenneTwister(0), network)
    names = Tuple(parameter_schema(model).phys_names)
    return NamedTuple{names}(ntuple(i -> names[i] === :input ? 1.0 : 0.8, length(names)))
end

"""
Regulator grid for the unobserved M: the range of the trained model's M
trajectories over the training cells, widened by 10% on each side (floor
0.05), `P53_GRID_POINTS` points, as `_regulator_grid` does for observed data.
"""
function latent_mdm2_grid(model, params, train_set, term)
    values = Float64[]
    for e in train_set.experiments
        X = predict_ude(params, e.u0, (first(e.times), last(e.times)), e.times, model)
        append!(values, X[term.regulator, :])
    end
    lo, hi = extrema(values)
    span = max(hi - lo, 0.1)
    return range(max(0.05, lo - 0.1 * span), hi + 0.1 * span; length = P53_GRID_POINTS)
end

function dependency_versions(names = ("OrdinaryDiffEq", "SciMLSensitivity", "Lux",
        "Optimization", "Zygote", "SciMLBase"))
    versions = Dict(info.name => info.version for info in values(Pkg.dependencies()))
    return join((string(name, " ", get(versions, name, "not loaded")) for name in names),
        ", ")
end

function simulate(rhs, u0, times)
    prob = SciMLBase.ODEProblem(rhs, u0, (first(times), last(times)))
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return fill(NaN, length(u0), length(times))
    return Array(sol)
end

function cell_rows(result, set)
    rate_fn = result.discovery.success ?
              equation_to_function(result.discovery.candidates[1]) : nothing
    hybrid = rate_fn === nothing ? nothing :
             compose_hybrid_rhs(result.model, result.params, result.term, rate_fn)
    ude = (u, p, t) -> ude_system(u, result.params, t, result.model)
    rows = NamedTuple[]
    for (i, e) in enumerate(set.experiments)
        pred = simulate(ude, e.u0, e.times)
        hyb = hybrid === nothing ? fill(NaN, size(pred)) : simulate(hybrid, e.u0, e.times)
        data = vec(e.observations[1, :])
        push!(rows, (; index = i, name = e.name, held_out = i in result.holdout_indices,
            peaks = e.metadata[:peaks],
            ude_rmse = sqrt(mean(abs2, vec(pred[1, :]) .- data)),
            hybrid_rmse = sqrt(mean(abs2, vec(hyb[1, :]) .- data)),
            times = e.times, data, model_p = vec(pred[1, :]), model_m = vec(pred[2, :]),
            hybrid_p = vec(hyb[1, :]), hybrid_m = vec(hyb[2, :])))
    end
    return rows
end

function main()
    mkpath(RESULTS_DIR)
    dir = download_p53_data()
    set, info = SMOKE ? p53_experiment_set(dir; max_cells = 6) : p53_experiment_set(dir)
    net = p53_mdm2_network()
    BioDynaX.count_unknown_destructions(net) == 1 || error("expected one unknown term")
    training = TrainingConfig(
        adam_iterations = SMOKE ? 2 : BioDynaX.REFERENCE_PROTOCOL.adam_iterations,
        bfgs_iterations = SMOKE ? 0 : BioDynaX.REFERENCE_PROTOCOL.bfgs_iterations,
        log_every = 10^6, frozen_phys = P53_FROZEN)
    selection = SMOKE ? StabilitySelection(n_boot = 4) : StabilitySelection()
    phys_init = p53_phys_init(net)

    println("p53–Mdm2 case study (Mdm2 unobserved)")
    println("Julia ", VERSION, "; ", dependency_versions())
    println(P53_CELL_LINE, " at ", P53_DOSE_GY, " Gy: ", info.n_cells_in_class, " cells, ",
        info.n_passing, " with at least ", P53_MIN_PEAKS, " peaks, ", info.n_selected,
        " used (", info.n_train, " training, ", info.holdout, " held out); ", info.n_points,
        " points at ", info.interval_min, " min over ", info.duration_h, " h")
    started = time()
    println("\n== discover_unknown_term, reference defaults ==")
    default = discover_unknown_term(net, set; training, holdout = info.holdout,
        rng = MersenneTwister(0), phys_init, regulator_grid = latent_mdm2_grid,
        seed = 0, verbose = true)
    t_default = time() - started
    started = time()
    println("\n== discover_unknown_term with stability selection (n_boot ",
        selection.n_boot, ", τ ", selection.τ, ") ==")
    stable = discover_unknown_term(net, set; training, holdout = info.holdout,
        rng = MersenneTwister(0), phys_init, regulator_grid = latent_mdm2_grid,
        seed = 0, stability_selection = selection, verbose = true)
    t_stable = time() - started
    println("\n", format_stability_selection(stable.discovery))

    write(joinpath(RESULTS_DIR, "report_default.txt"), report_unknown_term(default))
    write(joinpath(RESULTS_DIR, "report_stability.txt"), report_unknown_term(stable))
    write(joinpath(RESULTS_DIR, "selection_frequencies.txt"),
        format_stability_selection(stable.discovery))

    R = vec(default.samples.R)
    D = vec(default.samples.D)
    fn_default = default.discovery.success ?
                 equation_to_function(default.discovery.candidates[1]) : nothing
    fn_stable = stable.discovery.success ?
                equation_to_function(stable.discovery.candidates[1]) : nothing
    open(joinpath(RESULTS_DIR, "rate_samples.csv"), "w") do io
        println(io, "mdm2_model_units,learned_rate_per_h,discovered_rate_per_h,",
            "discovered_stable_rate_per_h")
        for (r, d) in zip(R, D)
            println(io, r, ",", d, ",", fn_default === nothing ? NaN : fn_default([r]), ",",
                fn_stable === nothing ? NaN : fn_stable([r]))
        end
    end

    rows = cell_rows(default, set)
    open(joinpath(RESULTS_DIR, "fits.csv"), "w") do io
        println(io, "cell,held_out,peaks,time_h,p53_normalised,model_p53,model_mdm2,",
            "hybrid_p53,hybrid_mdm2")
        for row in rows, j in eachindex(row.times)
            println(io, row.name, ",", row.held_out, ",", row.peaks, ",", row.times[j], ",",
                row.data[j], ",", row.model_p[j], ",", row.model_m[j], ",",
                row.hybrid_p[j], ",", row.hybrid_m[j])
        end
    end

    open(joinpath(RESULTS_DIR, "summary.txt"), "w") do io
        println(io, "p53–Mdm2 case study, ", SMOKE ? "smoke run" : "full run")
        println(io, "Julia ", VERSION, "; ", dependency_versions())
        println(io, "cells: ", info.n_selected, " (", info.n_train, " training, ",
            info.holdout, " held out) of ", info.n_passing, " passing of ",
            info.n_cells_in_class, " in ", P53_CELL_LINE, "_", P53_DOSE_GY)
        println(io, "training: Adam ", training.adam_iterations, ", BFGS ",
            training.bfgs_iterations, "; wall time default ", round(t_default; digits = 1),
            " s, with stability selection ", round(t_stable; digits = 1), " s")
        println(io, "regulator grid (M, model units): ", first(R), " to ", last(R), ", ",
            length(R), " points")
        for (label, result) in (("default", default), ("stability selection", stable))
            println(io, "\n[", label, "]")
            println(io, "discovery success: ", result.discovery.success, " (",
                result.discovery.retcode, "): ", result.discovery.message)
            println(io, "equation (P and M in fractions of the cell's maximum p53; rates per hour): ",
                result.discovery.equations)
            println(io, "training final loss: ", result.training.final_loss)
            names = parameter_schema(result.model).phys_names
            println(io, "physical parameters after training: ",
                join((string(n, " = ", round(BioDynaX.positive_parameter(v); sigdigits = 4))
                for (n, v) in zip(names, collect(result.params.phys))), ", "))
            println(io, "hybrid residual, first training cell: ",
                result.residuals.data_residual)
            println(io, "hybrid residual, mean over training cells: ",
                result.residuals.data_residual_train)
            println(io, "hybrid residual, mean over held-out cells: ",
                result.residuals.data_residual_holdout)
            println(io, "identifiability: condition number ",
                result.identifiability.condition_number, ", production correlation ",
                result.identifiability.production_correlation, ", collinearity ",
                result.identifiability.collinearity, ", flagged ",
                result.identifiability.unidentifiable_edge)
            if result.discovery.success
                c = result.discovery.candidates[1]
                println(io, "numerator terms: ",
                    [t.label for t in c.specification.numerator], " coefficients ",
                    c.numerator_coefficients)
                println(io, "denominator terms: ",
                    [t.label for t in c.specification.denominator], " coefficients ",
                    c.denominator_coefficients)
            end
        end
        println(io, "\n[per-cell RMSE of p53, fractions of the cell's maximum]")
        for row in rows
            println(io, row.name, " (", row.peaks, " peaks) ",
                row.held_out ? "held out" : "training", ": trained model ",
                round(row.ude_rmse; sigdigits = 4), ", hybrid ",
                round(row.hybrid_rmse; sigdigits = 4))
        end
        train = [r.ude_rmse for r in rows if !r.held_out]
        held = [r.ude_rmse for r in rows if r.held_out]
        println(io, "trained-model RMSE mean: training ", mean(train), ", held out ",
            mean(held))
        htrain = [r.hybrid_rmse for r in rows if !r.held_out]
        hheld = [r.hybrid_rmse for r in rows if r.held_out]
        println(io, "hybrid RMSE mean: training ", mean(htrain), ", held out ", mean(hheld))
        println(io, "learned rate on the grid (M, D per hour): ",
            join((string(round(r; digits = 3), " ", round(d; digits = 4))
            for (r, d) in zip(R[1:10:end], D[1:10:end])), "; "))
    end
    println("\nwrote ", RESULTS_DIR)
    print(read(joinpath(RESULTS_DIR, "summary.txt"), String))
end

main()
