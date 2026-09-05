#!/usr/bin/env julia
# Reference example: recover an unknown Hill-type inhibition in a two-species
# network. Same protocol as the recovery benchmark job: nine initial
# conditions generated once (seed 103), Adam 100 / BFGS 50, symbolic discovery
# on a regulator grid, then resimulation of the hybrid model against the data.
# Runtime: about 10 to 15 minutes. Prints a four-section report
# (identifiability, fit, discovery, reproduction).
# Run:  julia --project=. examples/unknown_inhibition.jl
# Fast installation check (1 initial condition, 8 points, 2 Adam steps):
#   BIODYNAX_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0 julia --project=. examples/unknown_inhibition.jl
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using BioDynaX
using OrdinaryDiffEq
using Random
using SciMLBase
using Statistics

function unknown_inhibition_network(; known::Bool, hill_order::Int = 2)
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :hill_deg,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = known, family = HILL,
            metadata = HillMetadata(
                vmax_param = :vmax, k_param = :K,
                hill_order = hill_order)),
        ReactionSpec(name = :produce_r,
            stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

function main(; seed::Int = BioDynaX.UNIQUE_CLAIM_PROTOCOL.seed,
        adam_iters::Int = BioDynaX.UNIQUE_CLAIM_PROTOCOL.adam_iterations,
        bfgs_iters::Int = BioDynaX.UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
        noise_σ::Float64 = 0.0,
        smoke::Bool = false)
    protocol = BioDynaX.UNIQUE_CLAIM_PROTOCOL
    fingerprint = BioDynaX.unique_claim_fingerprint(; smoke)
    rng = MersenneTwister(seed)
    truth_net = unknown_inhibition_network(; known = true, hill_order = 2)
    ude_net = unknown_inhibition_network(; known = false, hill_order = 2)
    BioDynaX.assert_unique_claim_recovery_network(ude_net)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    tspan = fingerprint.tspan
    ics = BioDynaX.unique_claim_protocol_ics(; smoke)
    n_points = BioDynaX.unique_claim_protocol_n_points(; smoke)
    set = BioDynaX.unique_claim_experiment_set(
        rng, truth_net; smoke, truth_params = truth, noise_σ = noise_σ)
    length(set.experiments) == length(ics) ||
        error("unique_claim_experiment_set IC count must match fingerprint")
    size(first(set.experiments).observations, 2) == n_points ||
        error("unique_claim_experiment_set point count must match fingerprint")

    # Never overwrite the committed howto fixture.
    data_dir = mktempdir()
    csv_path = joinpath(data_dir, "unknown_inhibition.csv")
    first_exp = first(set.experiments)
    write_experiment_csv(
        csv_path, first_exp; state_names = [:S, :R])
    loaded, names = experiment_from_csv(csv_path)
    @assert names == [:S, :R]
    @assert loaded.times ≈ first_exp.times

    model, params = build_ude_model(rng, ude_net)
    BioDynaX.assert_single_unknown_destruction(model)
    phys_names = Tuple(parameter_schema(model).phys_names)
    guess = NamedTuple{phys_names}(ntuple(_ -> 0.8, length(phys_names)))
    ude_init = pack_parameters(guess, params.nn)
    if smoke
        trained = train_experiments(
            ude_init, set, model;
            config = TrainingConfig(
                adam_iterations = adam_iters, bfgs_iterations = 0,
                log_every = 10^6),
            verbose = false)
    else
        warm = train_ude(
            ude_init, first_exp.observations, first_exp.times, first_exp.u0,
            (first(first_exp.times), last(first_exp.times)), model;
            config = TrainingConfig(
                adam_iterations = adam_iters, bfgs_iterations = 0,
                horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0]),
                log_every = 10^6),
            verbose = true)
        trained = train_experiments(
            warm.params, set, model;
            config = TrainingConfig(
                adam_iterations = adam_iters, bfgs_iterations = bfgs_iters,
                log_every = 10^6),
            verbose = true)
    end

    if smoke
        X_traj = predict_ude(
            trained.params, first_exp.u0, tspan, first_exp.times, model)
        R, D, term = sample_unknown_destruction(model, trained.params, X_traj)
        discovery = discover_unknown_rate(
            R, first_exp.times, D; verbose = false, strict = false)
    else
        term = only(BioDynaX.neural_destruction_terms(model))
        r_range = BioDynaX._regulator_grid(set, term)
        R, D, term = BioDynaX.sample_unknown_destruction_grid(
            model, trained.params, term; r_range = r_range)
        times_grid = collect(range(0.0, 1.0; length = size(R, 2)))
        discovery = discover_unknown_rate(
            R, times_grid, D;
            config = BioDynaX.unique_claim_discovery_config(),
            verbose = true, strict = true)
    end
    ident = BioDynaX.report_production_destruction_tradeoff(
        model, trained.params, first_exp.observations, first_exp.times,
        first_exp.u0, tspan; term = term, verbose = false)
    residual = Inf
    rhs = nothing
    extras = nothing
    if discovery.success && !isempty(discovery.candidates)
        rate_fn = equation_to_function(discovery.candidates[1])
        rhs = compose_hybrid_rhs(model, trained.params, term, rate_fn)
        residual = hybrid_data_residual(
            model, trained.params, term, rate_fn,
            first_exp.u0, tspan, first_exp.times, first_exp.observations)
        extras = BioDynaX.unique_claim_discovery_extras(discovery.candidates[1])
    elseif !smoke
        error("discovery failed ($(discovery.retcode)): $(discovery.message)")
    end
    println(BioDynaX.format_protocol_result(ident, fingerprint;
        residual = residual,
        equations = discovery.equations,
        extras = extras,
        unknown_holes = BioDynaX.count_unknown_destructions(model),
        seed = seed,
        n_ics = length(ics),
        n_points = n_points,
        adam_iters = adam_iters,
        bfgs_iters = smoke ? fingerprint.bfgs_iterations : bfgs_iters))
    println("Hybrid right-hand side constructed: ", rhs === nothing ? "no" : "yes")
    println("CSV (first IC): ", csv_path)
    if !smoke
        BioDynaX.assert_unique_claim_residual(residual)
    end
    return discovery, residual, ident
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    smoke = get(ENV, "BIODYNAX_SMOKE", "0") == "1"
    adam = parse(
        Int, get(ENV, "ADAM_ITERS",
            string(BioDynaX.UNIQUE_CLAIM_PROTOCOL.adam_iterations)))
    bfgs = parse(
        Int, get(ENV, "BFGS_ITERS",
            string(BioDynaX.UNIQUE_CLAIM_PROTOCOL.bfgs_iterations)))
    main(; adam_iters = adam, bfgs_iters = bfgs, smoke = smoke)
end
