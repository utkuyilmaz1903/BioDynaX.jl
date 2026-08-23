#!/usr/bin/env julia
# Golden path (same protocol as the recovery CI job):
# multi-IC experiments → unknown-edge UDE → D(z) discovery → hybrid resimulate.
# The network is built from the public constructors, not a recovery fixture.
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using BioDynaX
using OrdinaryDiffEq
using Random
using SciMLBase
using Statistics

const UNKNOWN_EDGE_ICS = [
    [0.25, 0.20], [0.80, 0.35], [0.40, 1.10], [1.20, 0.70], [0.15, 0.90],
    [0.50, 0.15], [0.90, 1.50], [0.20, 0.50], [1.50, 1.20]
]

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

function main(; seed::Int = 7,
        adam_iters::Int = 100,
        bfgs_iters::Int = 50,
        noise_σ::Float64 = 0.0,
        smoke::Bool = false)
    rng = MersenneTwister(seed)
    truth_net = unknown_inhibition_network(; known = true, hill_order = 2)
    ude_net = unknown_inhibition_network(; known = false, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    tspan = (0.0, 8.0)
    ics = smoke ? UNKNOWN_EDGE_ICS[1:1] : UNKNOWN_EDGE_ICS
    n_points = smoke ? 8 : 50
    set = generate_experiment_set(
        rng; network = truth_net, initial_conditions = ics,
        tspan = tspan, n_points = n_points, noise_σ = noise_σ,
        truth_params = truth)

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

    X_traj = predict_ude(
        trained.params, first_exp.u0, tspan, first_exp.times, model)
    R, D, term = sample_unknown_destruction(model, trained.params, X_traj)
    discovery = discover_unknown_rate(
        R, first_exp.times, D; verbose = !smoke, strict = !smoke)
    ident = BioDynaX.report_production_destruction_tradeoff(
        model, trained.params, first_exp.observations, first_exp.times,
        first_exp.u0, tspan; term = term, verbose = true)
    residual = Inf
    rhs = nothing
    if discovery.success && !isempty(discovery.candidates)
        rate_fn = equation_to_function(discovery.candidates[1])
        rhs = compose_hybrid_rhs(model, trained.params, term, rate_fn)
        residual = hybrid_data_residual(
            model, trained.params, term, rate_fn,
            first_exp.u0, tspan, first_exp.times, first_exp.observations)
    elseif !smoke
        error("discovery failed ($(discovery.retcode)): $(discovery.message)")
    end
    println("Recovered equations:\n", discovery.equations)
    println("Hybrid RHS constructed: ", typeof(rhs))
    println("Locked UDE claim (order matters):")
    println("  hybrid residual vs data: ", residual)
    println("  unidentifiable_edge: ", ident.unidentifiable_edge)
    println("  true-monomial recall is a CI gate on synthetic Hill truth")
    println("CSV (first IC): ", csv_path)
    if !smoke
        residual ≤ RECOVERY_THRESHOLDS.data_residual ||
            error("hybrid residual $(residual) exceeds RECOVERY_THRESHOLDS.data_residual")
    end
    ident.unidentifiable_edge && println(
        "Note: unidentifiable_edge=$(ident.unidentifiable_edge); ",
        "coefficients are not biological constants. ",
        "This preview does not claim canonical Hill from a trained NN. ",
        "The example prints this warning automatically.")
    return discovery, residual, ident
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    smoke = get(ENV, "BIODYNAX_SMOKE", "0") == "1"
    adam = parse(Int, get(ENV, "ADAM_ITERS", "100"))
    bfgs = parse(Int, get(ENV, "BFGS_ITERS", "50"))
    main(; adam_iters = adam, bfgs_iters = bfgs, smoke = smoke)
end
