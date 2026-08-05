###############################################################################
# DataGen.jl — synthetic-data generation from ground-truth mechanisms.
###############################################################################

"""Explicit ground-truth contract for synthetic experiment generation."""
struct GroundTruthModel
    network::BiologicalNetwork
    model::UDEModel
    parameters
    generator::Symbol
end

function GroundTruthModel(rng::AbstractRNG, network::BiologicalNetwork;
                          parameters = nothing)
    compiled, default_params = build_ude_model(rng, network)
    params = parameters === nothing ? default_params : parameters
    generator = network === DEFAULT_EXAMPLE_NETWORK ?
        :hill_p53_fixture : :compiled_mechanism
    return GroundTruthModel(network, compiled, params, generator)
end

"""
    ground_truth!(dx, x, p, t)

Mechanistic ground-truth p53 / Mdm2 model with Hill-saturation kinetics on
the Mdm2 → p53 degradation term. Used only for the default example fixture.
"""
function ground_truth!(dx, x, p, t)
    p53, mdm2 = x[1], x[2]
    deg = p.γ_p53 * mdm2^p.n / (p.K^p.n + mdm2^p.n)
    dx[1] = p.α_p53 * p.signal - deg * p53
    dx[2] = p.β_mdm2 * p53 - p.γ_mdm2 * mdm2
    return nothing
end

function _ground_truth_rhs(x, p, t, truth::GroundTruthModel)
    truth.generator == :hill_p53_fixture &&
        return _ground_truth_rhs_p53(x, p, t)
    return ude_system(x, p, t, truth.model)
end

function _ground_truth_rhs_p53(x, p, t)
    dx = similar(x)
    ground_truth!(dx, x, p, t)
    return dx
end

function generate_experiment_set(rng::AbstractRNG;
                                 network::BiologicalNetwork =
                                     DEFAULT_EXAMPLE_NETWORK,
                                 initial_conditions = [[0.2, 0.1]],
                                 kwargs...)
    experiments = Experiment[]
    truth = nothing
    for (index, u0) in pairs(initial_conditions)
        times, _, noisy, parameters =
            generate_data(rng; u0 = Float64.(u0), network = network, kwargs...)
        truth = parameters
        push!(experiments, Experiment(
            Symbol("experiment_$index"), times, noisy, Float64.(u0);
            metadata = Dict(:truth_parameters => parameters)))
    end
    state_names = [node.name for node in network.nodes if node.kind != INPUT]
    set = ExperimentSet(experiments, state_names;
        metadata = Dict(:generator => network === DEFAULT_EXAMPLE_NETWORK ?
                                       :hill_ground_truth : :compiled_ground_truth,
                          :truth_parameters => truth))
    return set
end

default_truth_params() = (
    α_p53 = 0.9,
    β_mdm2 = 1.1,
    γ_mdm2 = 1.5,
    signal = 1.0,
    γ_p53 = 2.0,
    K = 0.5,
    n = 4,
)

function generate_data(rng::AbstractRNG;
                       network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
                       u0::Vector{Float64} = [0.2, 0.1],
                       tspan::Tuple{Float64,Float64} = (0.0, 20.0),
                       n_points::Int = 40,
                       noise_σ::Float64 = 0.05,
                       truth_params::Union{Nothing,NamedTuple} = nothing)

    n_points ≥ 2 || throw(ArgumentError("n_points must be ≥ 2 (got $n_points)"))
    noise_σ ≥ 0 || throw(ArgumentError("noise_σ must be ≥ 0 (got $noise_σ)"))
    tspan[2] > tspan[1] ||
        throw(ArgumentError("tspan must be increasing (got $tspan)"))

    t_data = collect(range(tspan[1], tspan[2]; length = n_points))

    if network === DEFAULT_EXAMPLE_NETWORK
        params = truth_params === nothing ? default_truth_params() : truth_params
        prob = ODEProblem(ground_truth!, u0, tspan, params)
    else
        if truth_params === nothing
            truth = GroundTruthModel(rng, network)
        else
            nn, nn_ps, st = build_ude_nn(rng)
            model = compile_network(network, nn, st)
            params = truth_params isa ComponentVector ?
                truth_params : pack_parameters(truth_params, nn_ps)
            truth = GroundTruthModel(network, model, params, :compiled_mechanism)
        end
        params = truth.parameters
        rhs = (x, p, t) -> ude_system(x, p, t, truth.model)
        prob = ODEProblem(rhs, u0, tspan, params)
    end

    sol = solve(prob, Tsit5(); saveat = t_data,
                abstol = 1e-9, reltol = 1e-9)
    clean_data = Array(sol)
    noisy_data = clean_data .+ noise_σ .* randn(rng, size(clean_data))
    return t_data, clean_data, noisy_data, params
end

ground_truth!(truth::GroundTruthModel, dx, x, p, t) =
    _ground_truth_rhs(x, p, t, truth)

function generate_data(truth::GroundTruthModel, rng::AbstractRNG; kwargs...)
    return generate_data(rng; network = truth.network,
                       truth_params = truth.parameters, kwargs...)
end
