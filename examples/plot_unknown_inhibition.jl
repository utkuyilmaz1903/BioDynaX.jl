#!/usr/bin/env julia
# Figure for the README and docs: reruns the reference protocol from
# examples/unknown_inhibition.jl (seed 103, 9 initial conditions, Adam 100 /
# BFGS 50, regulator-grid discovery) and draws two panels: observed S and R
# for the first initial condition against the hybrid model that uses the
# discovered symbolic rate, and the true, learned, and discovered
# destruction rate D(R).
#
# Requires Plots.jl in the active environment, for example:
#   julia -e 'using Pkg; Pkg.activate(; temp=true); Pkg.develop(path=".");
#             Pkg.add(["Plots", "OrdinaryDiffEq", "SciMLBase"]);
#             include("examples/plot_unknown_inhibition.jl")'
# Runtime: about 10 minutes (the training is the same as the example).
# Writes docs/src/assets/unknown_inhibition.png.

using BioDynaX
using Plots
using Random

include(joinpath(@__DIR__, "unknown_inhibition.jl"))

const _PROTOCOL = BioDynaX.UNIQUE_CLAIM_PROTOCOL

function _train_reference(; seed = _PROTOCOL.seed)
    rng = MersenneTwister(seed)
    truth_net = unknown_inhibition_network(; known = true, hill_order = 2)
    ude_net = unknown_inhibition_network(; known = false, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    set = BioDynaX.unique_claim_experiment_set(rng, truth_net; truth_params = truth)
    model, params = build_ude_model(rng, ude_net)
    phys_names = Tuple(parameter_schema(model).phys_names)
    guess = NamedTuple{phys_names}(ntuple(_ -> 0.8, length(phys_names)))
    first_exp = first(set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    warm = train_ude(
        pack_parameters(guess, params.nn), first_exp.observations, first_exp.times,
        first_exp.u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = _PROTOCOL.adam_iterations, bfgs_iterations = 0,
            horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0]),
            log_every = 10^6), verbose = false)
    trained = train_experiments(
        warm.params, set, model;
        config = TrainingConfig(
            adam_iterations = _PROTOCOL.adam_iterations,
            bfgs_iterations = _PROTOCOL.bfgs_iterations, log_every = 10^6),
        verbose = false)
    term = only(BioDynaX.neural_destruction_terms(model))
    r_range = BioDynaX._regulator_grid(set, term)
    R, D, term = BioDynaX.sample_unknown_destruction_grid(
        model, trained.params, term; r_range = r_range)
    times_grid = collect(range(0.0, 1.0; length = size(R, 2)))
    discovery = discover_unknown_rate(
        R, times_grid, D; config = BioDynaX.unique_claim_discovery_config(),
        verbose = false, strict = true)
    return (; set, model, trained, term, truth, R, D, discovery, tspan)
end

function make_figure(path = joinpath(@__DIR__, "..", "docs", "src", "assets",
        "unknown_inhibition.png"))
    res = _train_reference()
    rate_fn = equation_to_function(res.discovery.candidates[1])
    rhs = compose_hybrid_rhs(res.model, res.trained.params, res.term, rate_fn)
    e = first(res.set.experiments)
    dense = collect(range(res.tspan[1], res.tspan[2]; length = 200))
    prob = SciMLBase.ODEProblem(rhs, e.u0, res.tspan)
    sol = solve(prob, Tsit5(); saveat = dense)
    pred = Array(sol)

    p1 = plot(xlabel = "time", ylabel = "concentration", legend = :topright,
        title = "Hybrid model with discovered rate, first IC")
    scatter!(p1, e.times, e.observations[1, :]; label = "S observed", markersize = 3)
    scatter!(p1, e.times, e.observations[2, :]; label = "R observed", markersize = 3)
    plot!(p1, dense, pred[1, :]; label = "S hybrid", linewidth = 2)
    plot!(p1, dense, pred[2, :]; label = "R hybrid", linewidth = 2)

    r = vec(res.R)
    hill = BioDynaX.hill_rate_truth(r; vmax = res.truth.vmax, K = res.truth.K, n = 2)
    found = [rate_fn([x]) for x in r]
    p2 = plot(xlabel = "R", ylabel = "destruction rate D(R)", legend = :topleft,
        title = "Unknown destruction rate D(R)")
    plot!(p2, r, hill; label = "true Hill", linewidth = 2)
    plot!(p2, r, vec(res.D); label = "learned (neural)", linewidth = 2, linestyle = :dash)
    plot!(p2, r, found; label = "discovered (rational)", linewidth = 2, linestyle = :dot)

    figure = plot(p1, p2; layout = (1, 2), size = (1100, 400), left_margin = 8Plots.mm, bottom_margin = 6Plots.mm)
    mkpath(dirname(path))
    savefig(figure, path)
    println("wrote ", path)
    return figure
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    make_figure()
end
