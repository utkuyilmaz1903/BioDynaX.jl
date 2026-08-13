#!/usr/bin/env julia
# Golden path: CSV observations → unknown-edge UDE → discovery → resimulate.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using OrdinaryDiffEq
using Random
using SciMLBase

function main(; seed::Int = 7,
                adam_iters::Int = 80,
                bfgs_iters::Int = 20)
    rng = MersenneTwister(seed)
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    truth_model, truth_p0 = build_ude_model(rng, truth_net)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    p_true = pack_parameters(truth, truth_p0.nn)
    u0 = [0.3, 0.25]
    tspan = (0.0, 6.0)
    times, clean, noisy, _ = generate_data(
        rng; network = truth_net, u0 = u0, tspan = tspan,
        n_points = 40, noise_σ = 0.01, truth_params = p_true)

    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)
    csv_path = joinpath(data_dir, "unknown_inhibition.csv")
    write_experiment_csv(
        csv_path,
        Experiment(:unknown_inhibition, times, noisy, u0);
        state_names = [:S, :R])

    experiment, names = experiment_from_csv(csv_path)
    @assert names == [:S, :R]
    model, params = build_ude_model(rng, ude_net)
    trained = train_ude(
        params, experiment.observations, experiment.times,
        experiment.u0, tspan, model;
        adam_iters = adam_iters, bfgs_iters = bfgs_iters, verbose = true)
    discovery = discover_equations(
        trained.params, model;
        u0 = experiment.u0, tspan = tspan, n_samples = 80,
        verbose = true, strict = true)
    rhs = export_rhs(discovery)
    prob = ODEProblem((u, p, t) -> rhs(u), experiment.u0, tspan)
    sol = solve(prob, Tsit5(); saveat = experiment.times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) ||
        error("recovered RHS failed to integrate: $(sol.retcode)")
    println("Recovered equations:\n", discovery.equations)
    println("CSV: ", csv_path)
    return discovery, sol
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
