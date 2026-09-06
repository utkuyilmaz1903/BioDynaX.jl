#!/usr/bin/env julia
# Two model variants for the laccase/ABTS progress curves, following the
# one-state case study (run_case_study.jl), in which no rational rate was
# accepted and the learned rate was non-monotone in the substrate. Each
# variant is one extra run on the same data, split, budget, and discovery
# configuration; nothing is tuned per curve and no default changes.
#
#   (a) product inhibition: states S (ABTS, observed) and C, the initial
#       substrate of the curve, a state with no reactions (dC/dt = 0) so that
#       the product Q = C - S is a function of the two; the unknown
#       destruction of S reads S and C (edges S → S and C → S), which lets the
#       rate depend on the product (the edges are derived from the reaction);
#   (b) enzyme inactivation: states S (observed) and E, the active enzyme
#       (unobserved, in µmol/l, E(0) = 0.93 as declared in the source
#       document), with first-order inactivation (`k_inact`, fitted); the
#       unknown destruction of S reads S and E (edges S → S and E → S).
#
# Both variants have two regulators, so the one-regulator entry point is not
# used; the same steps are run in the same order: warm-up on the first
# training curve, Adam 100 then BFGS 50, samples of the learned rate, the
# reference discovery configuration (bootstrap 8, discovery seed 3), hybrid
# residuals on the training and the held-out curves. (a) samples the rate on
# a design over (S, C): for each of the nine initial concentrations, 10 values
# of S from that curve's last measured value to its first (90 points); (b)
# samples it along the trained model's trajectories of the training curves
# (every second point, 231 points), since E is unobserved.
#
# Writes to examples/laccase_abts/data/results/variants_summary.txt and
# variant_<name>_rate_samples.csv.
# Run:  julia --project=. examples/laccase_abts/run_model_variants.jl
# Smoke check: BIODYNAX_SMOKE=1 julia --project=. examples/laccase_abts/run_model_variants.jl
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
const ENZYME_UM = 0.93

function basal_production()
    return ReactionSpec(name = :no_production, stoichiometry = Dict(1 => 1.0),
        regulators = Int[],
        metadata = InputDriveMetadata(rate_param = :k_prod, input_param = :input))
end

"""(a) ABTS and its initial concentration C; the unknown rate reads both."""
function product_inhibition_network()
    nodes = [NodeSpec(name = :ABTS), NodeSpec(name = :C)]
    reactions = [basal_production(),
        ReactionSpec(name = :oxidation, stoichiometry = Dict(1 => -1.0),
            regulators = [1, 2], known = false, family = BioDynaX.COMPETITIVE,
            metadata = HillMetadata())]
    # The graph edges ABTS -> ABTS and C -> ABTS are derived from the unknown
    # reaction's regulators (0.12); an explicit UNKNOWN_NN edge per regulator
    # would compile to a second and third neural term.
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""(b) ABTS and the active enzyme E with first-order inactivation."""
function enzyme_inactivation_network()
    nodes = [NodeSpec(name = :ABTS), NodeSpec(name = :E, kind = LATENT, observed = false)]
    reactions = [basal_production(),
        ReactionSpec(name = :oxidation, stoichiometry = Dict(1 => -1.0),
            regulators = [1, 2], known = false, family = BioDynaX.COMPETITIVE,
            metadata = HillMetadata()),
        ReactionSpec(name = :inactivation, stoichiometry = Dict(2 => -1.0),
            regulators = Int[], metadata = LinearDecayMetadata(rate_param = :k_inact))]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""The one-state set with a second state row: C = S(0) (observed, constant) or E (unobserved)."""
function two_state_set(set::ExperimentSet, variant::Symbol)
    experiments = map(set.experiments) do e
        s = e.observations[1, :]
        second = variant === :product ? fill(s[1], 1, length(s)) : fill(NaN, 1, length(s))
        u0 = variant === :product ? [s[1], s[1]] : [s[1], ENZYME_UM]
        Experiment(e.name, e.times, vcat(reshape(s, 1, :), second), u0;
            metadata = e.metadata)
    end
    names = variant === :product ? [:ABTS, :C] : [:ABTS, :E]
    return ExperimentSet(experiments, names; metadata = set.metadata)
end

function phys_init_for(model)
    names = Tuple(parameter_schema(model).phys_names)
    return NamedTuple{names}(ntuple(i -> names[i] in (:k_prod, :input) ? 1e-8 : 0.8,
        length(names)))
end

function train_variant(network, set, holdout, training)
    n = length(set.experiments)
    train_set = ExperimentSet(set.experiments[1:(n - holdout)], set.state_names;
        units = set.units, metadata = set.metadata)
    holdout_set = ExperimentSet(set.experiments[(n - holdout + 1):n], set.state_names;
        units = set.units, metadata = set.metadata)
    model, p0 = build_ude_model(MersenneTwister(0), network)
    ude_init = pack_parameters(phys_init_for(model), p0.nn)
    first_exp = first(train_set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    warm = train_ude(ude_init, first_exp.observations, first_exp.times, first_exp.u0,
        tspan, model;
        config = TrainingConfig(training; bfgs_iterations = 0,
            horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0])),
        verbose = false)
    trained = train_experiments(warm.params, train_set, model; config = training,
        verbose = false)
    return (; model, params = trained.params, training = trained, train_set, holdout_set,
        term = only(BioDynaX.neural_destruction_terms(model)))
end

function simulate(rhs, u0, times)
    prob = SciMLBase.ODEProblem(rhs, u0, (first(times), last(times)))
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return fill(NaN, length(u0), length(times))
    return Array(sol)
end

"""Root-mean-square error of a right-hand side against the observed row of a set."""
function rmse_over(rhs, set)
    values = map(set.experiments) do e
        pred = simulate(rhs, e.u0, e.times)
        mask = e.mask
        n = count(mask)
        n == 0 && return NaN
        sqrt(sum(abs2, ifelse.(mask, pred .- e.observations, 0.0)) / n)
    end
    return mean(values)
end

"""(a): design over (S, C); (b): the trained model's trajectories."""
function rate_samples(variant, fit)
    if variant === :product
        columns = Vector{Float64}[]
        for e in fit.train_set.experiments
            c = e.u0[2]
            s_last = minimum(e.observations[1, :])
            for s in range(s_last, c; length = 10)
                push!(columns, [s, c])
            end
        end
        X = reduce(hcat, columns)
    else
        columns = Vector{Float64}[]
        for e in fit.train_set.experiments
            traj = simulate(
                (u, p, t) -> ude_system(u, fit.params, t, fit.model), e.u0, e.times)
            for j in 1:2:size(traj, 2)
                push!(columns, traj[:, j])
            end
        end
        X = reduce(hcat, columns)
    end
    R, D, _ = sample_unknown_destruction(fit.model, fit.params, X)
    return Matrix{Float64}(R), Matrix{Float64}(D)
end

function run_variant(variant, set, info, training, io)
    network = variant === :product ? product_inhibition_network() :
              enzyme_inactivation_network()
    BioDynaX.count_unknown_destructions(network) == 1 || error("expected one unknown term")
    vset = two_state_set(set, variant)
    started = time()
    fit = train_variant(network, vset, info.holdout, training)
    train_time = time() - started
    R, D = rate_samples(variant, fit)
    times = collect(range(0.0, 1.0; length = size(R, 2)))
    discovery = discover_unknown_rate(
        R, times, D; config = BioDynaX.rate_discovery_config(),
        verbose = false, strict = false)
    ude = (u, p, t) -> ude_system(u, fit.params, t, fit.model)
    ude_train = rmse_over(ude, fit.train_set)
    ude_hold = rmse_over(ude, fit.holdout_set)
    hybrid_train = hybrid_hold = NaN
    if discovery.success
        rate_fn = equation_to_function(discovery.candidates[1])
        hybrid = compose_hybrid_rhs(fit.model, fit.params, fit.term, rate_fn)
        hybrid_train = rmse_over(hybrid, fit.train_set)
        hybrid_hold = rmse_over(hybrid, fit.holdout_set)
    end
    names = parameter_schema(fit.model).phys_names
    println(io,
        "\n[",
        variant === :product ? "(a) product inhibition, states ABTS and C" :
        "(b) enzyme inactivation, states ABTS and E",
        "]")
    println(io, "training wall time ", round(train_time; digits = 1), " s; final loss ",
        fit.training.final_loss)
    println(io, "physical parameters after training (per 1000 s): ",
        join(
            (string(n, " = ", round(BioDynaX.positive_parameter(v); sigdigits = 4))
            for (n, v) in zip(names, collect(fit.params.phys))),
            ", "))
    println(io, "rate samples: ", size(R, 2), " points; regulator ranges ",
        [extrema(R[i, :]) for i in 1:size(R, 1)])
    println(io, "discovery success: ", discovery.success, " (", discovery.retcode, "): ",
        discovery.message)
    println(io, "equation (regulators x[1] = ABTS per 100 µM, x[2] = ",
        variant === :product ? "C per 100 µM" : "E per µM", "; rate per 1000 s): ",
        discovery.equations)
    if discovery.success
        c = discovery.candidates[1]
        println(io, "numerator terms: ", [t.label for t in c.specification.numerator],
            " coefficients ", c.numerator_coefficients)
        println(io, "denominator terms: ", [t.label for t in c.specification.denominator],
            " coefficients ", c.denominator_coefficients)
    end
    println(io, "trained-model RMSE of ABTS (per 100 µM): training ", ude_train,
        ", held out ", ude_hold, "; in µM ", ude_train * ABTS_SCALE, " and ",
        ude_hold * ABTS_SCALE)
    println(io, "hybrid-model RMSE of ABTS (per 100 µM): training ", hybrid_train,
        ", held out ", hybrid_hold, "; in µM ", hybrid_train * ABTS_SCALE, " and ",
        hybrid_hold * ABTS_SCALE)
    open(joinpath(RESULTS_DIR, string("variant_", variant, "_rate_samples.csv")), "w") do f
        println(f,
            "regulator_1,regulator_2,learned_rate_per_model_time,discovered_rate_per_model_time")
        fn = discovery.success ? equation_to_function(discovery.candidates[1]) : nothing
        for j in 1:size(R, 2)
            println(f, R[1, j], ",", R[2, j], ",", D[1, j], ",",
                fn === nothing ? NaN : fn(R[:, j]))
        end
    end
    return (; discovery, ude_train, ude_hold, hybrid_train, hybrid_hold, fit)
end

function main()
    mkpath(RESULTS_DIR)
    dir = download_abts_data()
    set, info = abts_experiment_set(dir)
    training = TrainingConfig(
        adam_iterations = SMOKE ? 2 : BioDynaX.REFERENCE_PROTOCOL.adam_iterations,
        bfgs_iterations = SMOKE ? 0 : BioDynaX.REFERENCE_PROTOCOL.bfgs_iterations,
        log_every = 10^6, frozen_phys = [:k_prod, :input])
    path = joinpath(RESULTS_DIR, "variants_summary.txt")
    open(path, "w") do io
        println(io, "Laccase/ABTS model variants, ", SMOKE ? "smoke run" : "full run")
        println(io,
            "Julia ",
            VERSION,
            "; ",
            join(
                (string(i.name, " ", i.version)
                for i in values(Pkg.dependencies())
                if i.name in ("OrdinaryDiffEq", "SciMLSensitivity", "Lux",
                    "Optimization", "Zygote", "SciMLBase")),
                ", "))
        println(io, info.n_experiments, " replicate experiments, ", info.holdout,
            " held out; training Adam ", training.adam_iterations, ", BFGS ",
            training.bfgs_iterations)
        for variant in (:product, :enzyme)
            run_variant(variant, set, info, training, io)
        end
    end
    print(read(path, String))
end

main()
