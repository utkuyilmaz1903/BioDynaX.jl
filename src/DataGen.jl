###############################################################################
# DataGen.jl — synthetic-data generation from ground-truth mechanisms.
###############################################################################

"""Explicit ground-truth contract for synthetic experiment generation."""
struct GroundTruthModel
    network::BiologicalNetwork
    model::UDEModel
    parameters::Any
    generator::Symbol
end

function GroundTruthModel(rng::AbstractRNG, network::BiologicalNetwork;
        parameters = nothing,
        generator::Symbol = :compiled_mechanism)
    compiled, default_params = build_ude_model(rng, network)
    params = parameters === nothing ? default_params : parameters
    resolved = generator
    if generator === :hill_p53_fixture && network !== DEFAULT_EXAMPLE_NETWORK
        throw(ArgumentError(":hill_p53_fixture is only defined for DEFAULT_EXAMPLE_NETWORK"))
    end
    return GroundTruthModel(network, compiled, params, resolved)
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

"""
    compile_ground_truth_model(rng, network; truth_params, generator)

Build one `GroundTruthModel`. Multi-IC generation must call this once so
every trajectory shares the same compiled NN tree.
"""
function compile_ground_truth_model(rng::AbstractRNG, network::BiologicalNetwork;
        truth_params::Union{Nothing, NamedTuple, ComponentVector} = nothing,
        generator::Symbol = :compiled_mechanism)
    if generator === :hill_p53_fixture
        network === DEFAULT_EXAMPLE_NETWORK ||
            throw(ArgumentError(":hill_p53_fixture requires DEFAULT_EXAMPLE_NETWORK"))
        params = truth_params === nothing ? default_truth_params() : truth_params
        params isa NamedTuple ||
            throw(ArgumentError(":hill_p53_fixture expects NamedTuple truth_params"))
        model, _ = build_ude_model(rng, network)
        return GroundTruthModel(network, model, params, :hill_p53_fixture)
    end
    generator === :compiled_mechanism ||
        throw(ArgumentError("unknown generator $generator"))
    if truth_params === nothing
        return GroundTruthModel(rng, network; generator = :compiled_mechanism)
    end
    model, default_params = build_ude_model(rng, network)
    schema = parameter_schema(model)
    if truth_params isa ComponentVector
        validate_phys_parameters(unpack_parameters(truth_params).phys, schema)
        params_cv = truth_params
    else
        validate_phys_parameters(truth_params, schema)
        params_cv = pack_parameters(truth_params, default_params.nn)
    end
    return GroundTruthModel(network, model, params_cv, :compiled_mechanism)
end

"""
    generate_experiment_set_from_compiled_model(truth, rng; initial_conditions, ...)

Integrate each IC on the stored `GroundTruthModel`. Does not call
`build_ude_model` again.
"""
function generate_experiment_set_from_compiled_model(truth::GroundTruthModel,
        rng::AbstractRNG;
        initial_conditions,
        tspan::Tuple{Float64, Float64} = (0.0, 20.0),
        n_points::Int = 40,
        noise_σ::Float64 = 0.05)
    isempty(initial_conditions) &&
        throw(ArgumentError("generate_experiment_set needs at least one IC"))
    experiments = Experiment[]
    for (index, u0) in pairs(initial_conditions)
        times, _, noisy, parameters = generate_data(
            truth, rng; u0 = Float64.(u0), tspan, n_points, noise_σ)
        push!(experiments,
            Experiment(
                Symbol("experiment_$index"), times, noisy, Float64.(u0);
                metadata = Dict{Symbol, Any}(
                    :truth_parameters => parameters,
                    :generator => truth.generator,
                    :compiled_nstates => truth.model.compiled.nstates)))
    end
    state_names = [node.name for node in truth.network.nodes if node.kind != INPUT]
    generator_label = truth.generator === :compiled_mechanism ?
                      :compiled_ground_truth : truth.generator
    return ExperimentSet(experiments, state_names;
        metadata = Dict{Symbol, Any}(
            :generator => generator_label,
            :truth_parameters => truth.parameters,
            :compiled_once => true,
            :n_ics => length(initial_conditions),
            :n_points => n_points,
            :tspan => tspan,
            :noise_σ => noise_σ))
end

"""
    generate_experiment_set(rng; network, initial_conditions, tspan, n_points, ...)

Build one compiled ground-truth model, then an `ExperimentSet` from that
stored model. This is the multi-IC entry used by the golden path.
"""
function generate_experiment_set(rng::AbstractRNG;
        network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
        initial_conditions = [[0.2, 0.1]],
        tspan::Tuple{Float64, Float64} = (0.0, 20.0),
        n_points::Int = 40,
        noise_σ::Float64 = 0.05,
        truth_params::Union{Nothing, NamedTuple, ComponentVector} = nothing,
        generator::Symbol = :compiled_mechanism)
    truth = compile_ground_truth_model(
        rng, network; truth_params = truth_params, generator = generator)
    return generate_experiment_set_from_compiled_model(
        truth, rng; initial_conditions, tspan, n_points, noise_σ)
end

function default_truth_params()
    (
        α_p53 = 0.9,
        β_mdm2 = 1.1,
        γ_mdm2 = 1.5,
        signal = 1.0,
        γ_p53 = 2.0,
        K = 0.5,
        n = 4
    )
end

function generate_data(rng::AbstractRNG;
        network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
        u0::Vector{Float64} = [0.2, 0.1],
        tspan::Tuple{Float64, Float64} = (0.0, 20.0),
        n_points::Int = 40,
        noise_σ::Float64 = 0.05,
        truth_params::Union{Nothing, NamedTuple, ComponentVector} = nothing,
        generator::Symbol = :compiled_mechanism)
    n_points ≥ 2 || throw(ArgumentError("n_points must be ≥ 2 (got $n_points)"))
    noise_σ ≥ 0 || throw(ArgumentError("noise_σ must be ≥ 0 (got $noise_σ)"))
    tspan[2] > tspan[1] ||
        throw(ArgumentError("tspan must be increasing (got $tspan)"))

    if generator === :hill_p53_fixture
        network === DEFAULT_EXAMPLE_NETWORK ||
            throw(ArgumentError(":hill_p53_fixture requires DEFAULT_EXAMPLE_NETWORK"))
        params = truth_params === nothing ? default_truth_params() : truth_params
        params isa NamedTuple ||
            throw(ArgumentError(":hill_p53_fixture expects NamedTuple truth_params"))
        t_data = collect(range(tspan[1], tspan[2]; length = n_points))
        prob = ODEProblem(ground_truth!, u0, tspan, params)
        sol = solve(prob, Tsit5(); saveat = t_data,
            abstol = 1e-9, reltol = 1e-9)
        clean_data = Array(sol)
        noisy_data = clean_data .+ noise_σ .* randn(rng, size(clean_data))
        return t_data, clean_data, noisy_data, params
    end

    truth = compile_ground_truth_model(
        rng, network; truth_params = truth_params, generator = generator)
    return generate_data(truth, rng; u0, tspan, n_points, noise_σ)
end

"""
    generate_from_compiled_model(model, params, rng; u0, tspan, n_points, noise_σ)

Integrate the supplied `UDEModel` through `SciMLBase.ODEProblem(model, u0,
tspan, p)`. Does not rebuild the Lux tree. `generate_data(::GroundTruthModel)`
uses this so a stored compiled model is the generator, not a same-network twin.
"""
function generate_from_compiled_model(model::UDEModel, params, rng::AbstractRNG;
        u0::Vector{Float64} = [0.2, 0.1],
        tspan::Tuple{Float64, Float64} = (0.0, 20.0),
        n_points::Int = 40,
        noise_σ::Float64 = 0.05)
    n_points ≥ 2 || throw(ArgumentError("n_points must be ≥ 2 (got $n_points)"))
    noise_σ ≥ 0 || throw(ArgumentError("noise_σ must be ≥ 0 (got $noise_σ)"))
    tspan[2] > tspan[1] ||
        throw(ArgumentError("tspan must be increasing (got $tspan)"))
    length(u0) == model.compiled.nstates || throw(ArgumentError(
        "u0 length $(length(u0)) does not match compiled nstates $(model.compiled.nstates)"))
    t_data = collect(range(tspan[1], tspan[2]; length = n_points))
    # Same SciML entry as `ODEProblem(model, u0, tspan, p)` / `solve(model, ...)`.
    prob = SciMLBase.ODEProblem(model, u0, tspan, params)
    sol = solve(prob, Tsit5(); saveat = t_data,
        abstol = 1e-9, reltol = 1e-9)
    clean_data = Array(sol)
    noisy_data = clean_data .+ noise_σ .* randn(rng, size(clean_data))
    return t_data, clean_data, noisy_data, params
end

ground_truth!(truth::GroundTruthModel, dx, x, p, t) = _ground_truth_rhs(x, p, t, truth)

function generate_data(truth::GroundTruthModel, rng::AbstractRNG;
        u0::Vector{Float64} = [0.2, 0.1],
        tspan::Tuple{Float64, Float64} = (0.0, 20.0),
        n_points::Int = 40,
        noise_σ::Float64 = 0.05,
        kwargs...)
    if truth.generator === :hill_p53_fixture
        return generate_data(rng; network = truth.network,
            truth_params = truth.parameters,
            generator = :hill_p53_fixture, u0, tspan, n_points, noise_σ)
    end
    truth.generator === :compiled_mechanism ||
        throw(ArgumentError("unknown generator $(truth.generator)"))
    return generate_from_compiled_model(
        truth.model, truth.parameters, rng; u0, tspan, n_points, noise_σ)
end
