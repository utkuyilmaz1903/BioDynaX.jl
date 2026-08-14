"""
Standard benchmark networks for scientific regression and recovery tests.
"""
function build_repressilator_network(; hill_order::Int = 4)::BiologicalNetwork
    nodes = [
        NodeSpec(name = :A),
        NodeSpec(name = :B),
        NodeSpec(name = :C),
    ]
    reactions = [
        ReactionSpec(name = :basal_a,
                     stoichiometry = Dict(1 => 1.0), regulators = Int[],
                     metadata = InputDriveMetadata(
                         rate_param = :basal, input_param = :unit)),
        ReactionSpec(name = :b_inhibits_a,
                     stoichiometry = Dict(1 => -1.0), regulators = [3],
                     known = true, family = HILL,
                     metadata = HillMetadata(
                         vmax_param = :vmax_a, k_param = :K_c, hill_order = hill_order)),
        ReactionSpec(name = :basal_b,
                     stoichiometry = Dict(2 => 1.0), regulators = Int[],
                     metadata = InputDriveMetadata(
                         rate_param = :basal, input_param = :unit)),
        ReactionSpec(name = :c_inhibits_b,
                     stoichiometry = Dict(2 => -1.0), regulators = [1],
                     known = true, family = HILL,
                     metadata = HillMetadata(
                         vmax_param = :vmax_b, k_param = :K_a, hill_order = hill_order)),
        ReactionSpec(name = :basal_c,
                     stoichiometry = Dict(3 => 1.0), regulators = Int[],
                     metadata = InputDriveMetadata(
                         rate_param = :basal, input_param = :unit)),
        ReactionSpec(name = :a_inhibits_c,
                     stoichiometry = Dict(3 => -1.0), regulators = [2],
                     known = true, family = HILL,
                     metadata = HillMetadata(
                         vmax_param = :vmax_c, k_param = :K_b, hill_order = hill_order)),
        ReactionSpec(name = :decay_a,
                     stoichiometry = Dict(1 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :γ)),
        ReactionSpec(name = :decay_b,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :γ)),
        ReactionSpec(name = :decay_c,
                     stoichiometry = Dict(3 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :γ)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""Return the canonical benchmark topologies exercised in CI."""
benchmark_networks() = (
    p53 = DEFAULT_EXAMPLE_NETWORK,
    linear = build_linear_test_network(),
    repressilator = build_repressilator_network(),
)

struct BenchmarkOutcome
    name::Symbol
    initial_loss::Float64
    final_loss::Float64
    converged::Bool
    retcode::TrainingRetcode
    discovery_success::Bool
end

"""
    run_benchmark_suite(rng; train_kwargs..., discovery=true)

End-to-end smoke benchmark across standard network topologies.
"""
function run_benchmark_suite(rng::AbstractRNG = Random.default_rng();
                             discovery::Bool = true,
                             noise_σ::Real = 0.02,
                             n_points::Int = 16,
                             train_kwargs...)
    outcomes = BenchmarkOutcome[]
    for (name, network) in pairs(benchmark_networks())
        model, params = build_ude_model(rng, network)
        u0 = name == :repressilator ? [0.3, 0.1, 0.2] :
             name == :linear ? [0.2, 0.1] : [0.2, 0.1]
        tspan = (0.0, name == :repressilator ? 8.0 : 4.0)
        times, _, noisy, _ = generate_data(
            rng; network = network, u0 = u0, tspan = tspan,
            n_points = n_points, noise_σ = float(noise_σ))
        trained = train_ude(
            params, noisy, times, u0, tspan, model;
            adam_iters = get(train_kwargs, :adam_iters, 6),
            bfgs_iters = get(train_kwargs, :bfgs_iters, 0),
            verbose = false, log_every = 100)
        disc_ok = true
        if discovery && name != :p53
            result = discover_equations(
                trained.params, model;
                u0 = u0, tspan = tspan, n_samples = min(80, 4 * n_points),
                config = DiscoveryConfig(backend = ExplicitSTLSQ()),
                verbose = false)
            disc_ok = result.success
        end
        push!(outcomes, BenchmarkOutcome(
            name, trained.initial_loss, trained.final_loss,
            trained.converged, trained.retcode, disc_ok))
    end
    return outcomes
end
