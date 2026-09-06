#!/usr/bin/env julia
# Enzyme case study on measured data: laccase-catalysed oxidation of ABTS,
# nine substrate-depletion progress curves in triplicate (download_data.jl,
# preprocess.jl). One observed state, ABTS; the enzyme concentration is the
# same in every curve and is not a model state. The unknown term is the
# per-concentration destruction rate D(ABTS) in
#
#     d[ABTS]/dt = -D([ABTS]) [ABTS],
#
# so that Michaelis-Menten depletion, -Vmax [ABTS] / (K + [ABTS]), corresponds to
# D = Vmax / (K + [ABTS]): a constant numerator and a linear denominator.
# The workflow is `discover_unknown_term` with the reference defaults (Adam 100
# then BFGS 50, bootstrap 8, discovery seed 3), once without and once with
# stability selection; six of the 27 replicate curves (the 25 µM and 75 µM
# curves) are held out. Nothing is tuned per curve.
#
# Writes to examples/laccase_abts/data/results/ (not committed):
#   report_default.txt, report_stability.txt   four-section reports
#   selection_frequencies.txt                  stability-selection table
#   rate_samples.csv                           S, learned D, discovered D
#   fits.csv                                   data, trained model, hybrid model
#   summary.txt                                residuals, coefficients, environment
# Run:  julia --project=. examples/laccase_abts/run_case_study.jl
# Runtime: about 10 minutes on 4 cores. Needs the data (download_data.jl).
# Smoke check (2 Adam steps, no BFGS, 4 resamples):
#   BIODYNAX_SMOKE=1 julia --project=. examples/laccase_abts/run_case_study.jl
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
const RESULTS_DIR = joinpath(ABTS_DATA_DIR, "results")

"""
One observed state, ABTS, whose destruction is the unknown term regulated by
ABTS itself. The edge and the reaction both declare the term unknown; the
saturation family is the literature form the discovered rate is compared to.
The compiled mechanism needs a production term, so a basal production of
ABTS is declared with both of its parameters (`k_prod`, `input`) frozen
during training at 1e-8 (`ABTS_FROZEN`, `ABTS_PHYS_INIT`; the positivity
constraint of the reference defaults does not admit zero), a production of
1e-16 model units per model time unit: ABTS is only consumed.
"""
function laccase_abts_network()
    nodes = [NodeSpec(name = :ABTS)]
    metadata = SaturationMetadata(vmax_param = :vmax, km_param = :K)
    edges = [EdgeSpec(source = 1, target = 1, kind = UNKNOWN_NN, family = SATURATION,
        known = false, metadata = metadata)]
    reactions = [
        ReactionSpec(name = :no_production, stoichiometry = Dict(1 => 1.0),
            regulators = Int[],
            metadata = InputDriveMetadata(rate_param = :k_prod, input_param = :input)),
        ReactionSpec(name = :oxidation, stoichiometry = Dict(1 => -1.0),
            regulators = [1], known = false, family = SATURATION, metadata = metadata)]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

const ABTS_FROZEN = [:k_prod, :input]
const ABTS_PHYS_INIT = (k_prod = 1e-8, input = 1e-8)

"""Implicit support of D = Vmax / (K + S): constant numerator, linear denominator."""
michaelis_menten_self_support() = (numerator = Set([((), ())]),
    denominator = Set([((1,), (1,))]))

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

function experiment_residuals(result, set)
    hybrid = nothing
    if result.discovery.success && !isempty(result.discovery.candidates)
        rate_fn = equation_to_function(result.discovery.candidates[1])
        hybrid = compose_hybrid_rhs(result.model, result.params, result.term, rate_fn)
    end
    ude = (u, p, t) -> ude_system(u, result.params, t, result.model)
    rows = NamedTuple[]
    for (i, e) in enumerate(set.experiments)
        pred_ude = simulate(ude, e.u0, e.times)
        pred_hybrid = hybrid === nothing ? fill(NaN, size(pred_ude)) :
                      simulate(hybrid, e.u0, e.times)
        push!(rows,
            (; index = i, name = e.name,
                curve = e.metadata[:curve], replicate = e.metadata[:replicate],
                initial_abts_uM = e.metadata[:initial_abts_uM],
                held_out = i in result.holdout_indices,
                ude_rmse = sqrt(mean(abs2, pred_ude .- e.observations)),
                hybrid_rmse = sqrt(mean(abs2, pred_hybrid .- e.observations)),
                times = e.times, data = vec(e.observations),
                ude = vec(pred_ude), hybrid = vec(pred_hybrid)))
    end
    return rows
end

function main()
    mkpath(RESULTS_DIR)
    dir = download_abts_data()
    set, info = abts_experiment_set(dir)
    net = laccase_abts_network()
    BioDynaX.count_unknown_destructions(net) == 1 || error("expected one unknown term")
    training = TrainingConfig(
        adam_iterations = SMOKE ? 2 : BioDynaX.REFERENCE_PROTOCOL.adam_iterations,
        bfgs_iterations = SMOKE ? 0 : BioDynaX.REFERENCE_PROTOCOL.bfgs_iterations,
        log_every = 10^6, frozen_phys = ABTS_FROZEN)
    selection = SMOKE ? StabilitySelection(n_boot = 4) : StabilitySelection()
    support = michaelis_menten_self_support()

    println("Laccase/ABTS case study")
    println("Julia ", VERSION, "; ", dependency_versions())
    println(info.n_experiments, " replicate experiments (", info.n_curves, " curves, ",
        info.n_points, " points over ", info.duration_s, " s); ", info.holdout,
        " held out; ABTS in units of ", info.abts_scale, " µM, time in units of ",
        info.time_scale, " s")
    started = time()
    println("\n== discover_unknown_term, reference defaults ==")
    default = discover_unknown_term(net, set; training, holdout = info.holdout,
        rng = MersenneTwister(0), phys_init = ABTS_PHYS_INIT, known_support = support,
        seed = 0, verbose = true)
    t_default = time() - started
    started = time()
    println("\n== discover_unknown_term with stability selection (n_boot ",
        selection.n_boot, ", τ ", selection.τ, ") ==")
    stable = discover_unknown_term(net, set; training, holdout = info.holdout,
        rng = MersenneTwister(0), phys_init = ABTS_PHYS_INIT, known_support = support,
        seed = 0, stability_selection = selection, verbose = true)
    t_stable = time() - started
    println("\n", format_stability_selection(stable.discovery))

    write(joinpath(RESULTS_DIR, "report_default.txt"), report_unknown_term(default))
    write(joinpath(RESULTS_DIR, "report_stability.txt"), report_unknown_term(stable))
    write(joinpath(RESULTS_DIR, "selection_frequencies.txt"),
        format_stability_selection(stable.discovery))

    # Learned rate and discovered rates on the regulator grid, in data units.
    R = vec(default.samples.R)
    D = vec(default.samples.D)
    fn_default = default.discovery.success ?
                 equation_to_function(default.discovery.candidates[1]) : nothing
    fn_stable = stable.discovery.success ?
                equation_to_function(stable.discovery.candidates[1]) : nothing
    open(joinpath(RESULTS_DIR, "rate_samples.csv"), "w") do io
        println(io, "abts_model_units,abts_uM,learned_rate_per_model_time,",
            "learned_rate_per_s,discovered_rate_per_s,discovered_stable_rate_per_s")
        for (r, d) in zip(R, D)
            fd = fn_default === nothing ? NaN : fn_default([r])
            fs = fn_stable === nothing ? NaN : fn_stable([r])
            println(io, r, ",", r * info.abts_scale, ",", d, ",", d / info.time_scale, ",",
                fd / info.time_scale, ",", fs / info.time_scale)
        end
    end

    rows = experiment_residuals(default, set)
    open(joinpath(RESULTS_DIR, "fits.csv"), "w") do io
        println(io, "experiment,curve,replicate,initial_abts_uM,held_out,time_s,",
            "abts_uM,trained_model_uM,hybrid_model_uM")
        for row in rows, (t, y, u, h) in zip(row.times, row.data, row.ude, row.hybrid)
            println(io, row.index, ",", row.curve, ",", row.replicate, ",",
                row.initial_abts_uM, ",", row.held_out, ",", t * info.time_scale, ",",
                y * info.abts_scale, ",", u * info.abts_scale, ",", h * info.abts_scale)
        end
    end

    open(joinpath(RESULTS_DIR, "summary.txt"), "w") do io
        println(io, "Laccase/ABTS case study, ", SMOKE ? "smoke run" : "full run")
        println(io, "Julia ", VERSION, "; ", dependency_versions())
        println(io, "training: Adam ", training.adam_iterations, ", BFGS ",
            training.bfgs_iterations, "; wall time default ", round(t_default; digits = 1),
            " s, with stability selection ", round(t_stable; digits = 1), " s")
        for (label, result) in (("default", default), ("stability selection", stable))
            println(io, "\n[", label, "]")
            println(io, "discovery success: ", result.discovery.success, " (",
                result.discovery.retcode, "): ", result.discovery.message)
            println(io, "equation (model units, D per 1000 s, S per 100 µM): ",
                result.discovery.equations)
            println(io, "extras beyond D = a / (1 + b S): ",
                result.extras === nothing ? "NA" : join(result.extras, ", "))
            println(io, "training final loss: ", result.training.final_loss)
            println(io, "physical parameters after training (frozen at 1e-8): ",
                [BioDynaX.positive_parameter(v) for v in collect(result.params.phys)])
            println(io, "hybrid residual, first training experiment: ",
                result.residuals.data_residual)
            println(io, "hybrid residual, mean over training experiments: ",
                result.residuals.data_residual_train)
            println(io, "hybrid residual, mean over held-out experiments: ",
                result.residuals.data_residual_holdout)
            println(io, "identifiability: condition number ",
                result.identifiability.condition_number, ", collinearity ",
                result.identifiability.collinearity, ", flagged ",
                result.identifiability.unidentifiable_edge)
            if result.discovery.success
                c = result.discovery.candidates[1]
                println(io, "numerator coefficients: ", c.numerator_coefficients)
                println(io, "denominator coefficients: ", c.denominator_coefficients)
                println(io, "numerator terms: ",
                    [t.label for t in c.specification.numerator])
                println(io, "denominator terms: ",
                    [t.label for t in c.specification.denominator])
            end
        end
        if !isempty(rows)
            println(io, "\n[per-experiment RMSE, model units (100 µM)]")
            for row in rows
                println(io, row.name, " curve ", row.curve, " (", row.initial_abts_uM,
                    " µM) ", row.held_out ? "held out" : "training",
                    ": trained model ", round(row.ude_rmse; sigdigits = 4),
                    ", hybrid ", round(row.hybrid_rmse; sigdigits = 4))
            end
            for (label, key) in (("trained model", :ude_rmse), ("hybrid", :hybrid_rmse))
                train = [getproperty(r, key) for r in rows if !r.held_out]
                held = [getproperty(r, key) for r in rows if r.held_out]
                println(io, label, " RMSE mean: training ", mean(train), ", held out ",
                    mean(held), " (model units); in µM: ", mean(train) * info.abts_scale,
                    " and ", mean(held) * info.abts_scale)
            end
        end
        println(io, "\n[learned rate on the regulator grid, every tenth point]")
        for i in 1:10:length(R)
            println(io, "ABTS ", round(R[i] * info.abts_scale; digits = 1), " µM: D ",
                round(D[i] / info.time_scale; sigdigits = 4), " per s")
        end
    end
    println("\nwrote ", RESULTS_DIR)
    print(read(joinpath(RESULTS_DIR, "summary.txt"), String))
end

main()
