###############################################################################
# Trained-UDE graph-local library validation (not exported).
#
# One real `fit_unknown_destruction` call → captured TrainingResult.params →
# one `sample_unknown_destruction` → three `discover_equations` executions
# (graph / global / wrong-graph) on the same learned D.
#
# analytic library-membership control uses hill_rate_truth and is not trained-UDE evidence.
# trained-UDE graph-local evidence samples D from the captured fit_unknown_destruction return params via sample_unknown_destruction.
# PR smoke is not trained-UDE scientific acceptance.
#
# This file does not change RECOVERY_THRESHOLDS, LOCKED_PUBLIC_EXPORTS,
# evaluate_holdout, the functional-identifiability diagnostic, occupancy, or UNIQUE_CLAIM_PROTOCOL. No public
# export is added. training_call is an unexported composition argument.
###############################################################################

const M4B_PROTOCOL = (
    seed = 401,
    n_ics = 3,
    n_points = 40,
    tspan = (0.0, 8.0),
    noise_σ = 0.0,
    adam_iterations = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
    bfgs_iterations = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
    discovery_seed = 11,
    bootstrap = 0,
    n_sample_points = 80,
    x_seed = 619)

const M4B_SMOKE = (
    seed = 401,
    n_ics = 1,
    n_points = 8,
    tspan = (0.0, 8.0),
    noise_σ = 0.0,
    adam_iterations = 2,
    bfgs_iterations = 0,
    discovery_seed = 11,
    bootstrap = 0,
    n_sample_points = 24,
    x_seed = 619)

const M4B_SCOPE_PLAN = (
    (name = :graph, network = :ude, basis_scope = :graph),
    (name = :global, network = :ude, basis_scope = :global),
    (name = :wrong_graph, network = :wrong, basis_scope = :graph))

"""Truth parameters of the trained-model library check for data generation. Not discovery D."""
m4b_truth_params() = (
    k_prod = 0.9, vmax = 1.7, K = 0.6,
    k_rq = 1.0, k_r = 0.6, k_qs = 0.8, k_q = 0.5, k_z = 0.4)

function m4b_budget(kind::Symbol)
    kind === :smoke && return M4B_SMOKE
    kind === :protocol && return M4B_PROTOCOL
    throw(ArgumentError("kind must be :smoke or :protocol; got $(kind)"))
end

function m4b_initial_conditions(kind::Symbol)
    if kind === :smoke
        return [[0.30, 0.25, 0.20, 0.15]]
    elseif kind === :protocol
        return [
            [0.30, 0.25, 0.20, 0.15],
            [0.80, 0.40, 0.35, 0.20],
            [0.45, 1.10, 0.50, 0.30]]
    end
    throw(ArgumentError("kind must be :smoke or :protocol; got $(kind)"))
end

"""
    designed_trained_graph_local_coordinates(n_sample_points; x_seed=619)

Locked 4×n designed coordinates (S, R, Q, Z). Not occupancy, not a fill
grid, and not functional-identifiability `domain.z`. Independent of any stored evidence.X.
"""
function designed_trained_graph_local_coordinates(n_sample_points::Integer;
        x_seed::Integer = 619)
    n = Int(n_sample_points)
    r = collect(range(0.1, 2.0; length = n))
    s = fill(0.4, n)
    q = r .^ 2 .+ 0.08 .* maximum(r .^ 2) .* randn(MersenneTwister(x_seed), n)
    z = r .+ 0.10 .* (maximum(r) - minimum(r)) .*
             randn(MersenneTwister(x_seed), n)
    return permutedims(hcat(s, r, q, z))
end

function dummy_trained_graph_local_times(n::Integer)
    return collect(range(0.0, 1.0; length = Int(n)))
end

"""
    TrainedGraphLocalEvidence

Unexported output of the trained-model library check. `training` is the stored fit
result and is not the provenance oracle; the oracle is the captured
`fit_unknown_destruction` return params on the test side.
"""
struct TrainedGraphLocalEvidence
    kind::Symbol
    training::TrainingResult
    model::UDEModel
    term::NeuralDestructionTerm
    params_nn_fingerprint::UInt64
    X::Matrix{Float64}
    D::Matrix{Float64}
    times::Vector{Float64}
    graph_discovery::DiscoveryResult
    global_discovery::DiscoveryResult
    wrong_graph_discovery::DiscoveryResult
    function TrainedGraphLocalEvidence(
            kind::Symbol,
            training::TrainingResult,
            model::UDEModel,
            term::NeuralDestructionTerm,
            params_nn_fingerprint::UInt64,
            X::AbstractMatrix,
            D::AbstractMatrix,
            times::AbstractVector,
            graph_discovery::DiscoveryResult,
            global_discovery::DiscoveryResult,
            wrong_graph_discovery::DiscoveryResult)
        kind === :smoke || kind === :protocol ||
            throw(ArgumentError(
                "TrainedGraphLocalEvidence.kind must be :smoke or :protocol"))
        size(D, 1) == 1 || throw(DimensionMismatch("learned D must be 1×N"))
        size(X, 2) == size(D, 2) == length(times) || throw(DimensionMismatch(
            "X columns, D columns, and times must match"))
        return new(kind, training, model, term, params_nn_fingerprint,
            Matrix{Float64}(X), Matrix{Float64}(D), Vector{Float64}(times),
            graph_discovery, global_discovery, wrong_graph_discovery)
    end
end

function _m4b_scope_network(name::Symbol, ude_net, wrong_net)
    name === :ude && return ude_net
    name === :wrong && return wrong_net
    throw(ArgumentError("scope network must be :ude or :wrong; got $(name)"))
end

function _m4b_discovery_config(budget, scope::Symbol)
    return rate_discovery_config(
        scope = scope,
        bootstrap = budget.bootstrap,
        seed = budget.discovery_seed)
end

"""
    library_study_training_set(kind; seed, noise_σ, initial_conditions)

Experiment set of the trained-model library check: the four-state network with
the true Hill term, integrated from `initial_conditions` with the budget's time
span and point count, and additive Gaussian observation noise of standard
deviation `noise_σ` drawn from `MersenneTwister(seed)`. The defaults are the
budget's own seed, noise, and initial conditions, so
`library_study_training_set(kind)` is the set `evaluate_trained_graph_local`
trains on. Not exported.
"""
function library_study_training_set(kind::Symbol;
        seed::Integer = m4b_budget(kind).seed,
        noise_σ::Real = m4b_budget(kind).noise_σ,
        initial_conditions = m4b_initial_conditions(kind))
    budget = m4b_budget(kind)
    truth_net = build_three_state_unknown_network(;
        known = true, with_distractor = true, parent = 2)
    return generate_experiment_set(
        MersenneTwister(seed);
        network = truth_net,
        initial_conditions = initial_conditions,
        tspan = budget.tspan,
        n_points = budget.n_points,
        noise_σ = Float64(noise_σ),
        truth_params = m4b_truth_params(),
        generator = :compiled_mechanism)
end

"""
    evaluate_trained_graph_local(; kind, training_call=fit_unknown_destruction,
                                 seed=m4b_budget(kind).seed,
                                 noise_σ=m4b_budget(kind).noise_σ)

Unexported orchestrator of the trained-model library check. Exactly one `training_call`, one learned-D
sample, and three `discover_equations` executions. Holdout does not
select the optimizer, initialization, or scope.

`seed` seeds both the data generation and the network initialisation;
`noise_σ` is the observation-noise standard deviation of the generated
experiments. Their defaults are the budget's own values (seed 401, no noise),
so a call without them reproduces the original single run. The library
comparison study varies them.
"""
function evaluate_trained_graph_local(;
        kind::Symbol,
        training_call = fit_unknown_destruction,
        seed::Integer = m4b_budget(kind).seed,
        noise_σ::Real = m4b_budget(kind).noise_σ)
    budget = m4b_budget(kind)
    ude_net = build_three_state_unknown_network(;
        known = false, with_distractor = true, parent = 2)
    wrong_net = build_wrong_graph_unknown_network(;
        known = false, with_distractor = true)
    train_set = library_study_training_set(kind; seed, noise_σ)
    model, p0 = build_ude_model(MersenneTwister(seed), ude_net)
    training = training_call(
        model, p0, train_set;
        adam = budget.adam_iterations,
        bfgs = budget.bfgs_iterations)
    training isa TrainingResult || throw(ArgumentError(
        "training_call must return a TrainingResult"))
    X = designed_trained_graph_local_coordinates(
        budget.n_sample_points; x_seed = budget.x_seed)
    (_, D, term) = sample_unknown_destruction(model, training.params, X)
    D = Matrix{Float64}(D)
    X = Matrix{Float64}(X)
    dX = zeros(Float64, size(X, 1), size(X, 2))
    dX[1, :] .= vec(D)
    times = dummy_trained_graph_local_times(size(X, 2))
    discoveries = DiscoveryResult[]
    for scope in M4B_SCOPE_PLAN
        network = _m4b_scope_network(scope.network, ude_net, wrong_net)
        push!(discoveries,
            discover_equations(
                X, times, network;
                derivatives = dX,
                targets = 1,
                config = _m4b_discovery_config(budget, scope.basis_scope),
                verbose = false))
    end
    return TrainedGraphLocalEvidence(
        kind,
        training,
        model,
        term,
        nn_parameter_fingerprint(training.params.nn),
        X,
        D,
        times,
        discoveries[1],
        discoveries[2],
        discoveries[3])
end
