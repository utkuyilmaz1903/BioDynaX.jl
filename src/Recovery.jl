"""Recovery fixtures and metrics for the scientific wedge (graph-local UDE + discovery)."""

function relative_parameter_error(estimate, truth::NamedTuple)
    names = collect(keys(truth))
    rel = Dict{Symbol,Float64}()
    squares = Float64[]
    for name in names
        fitted = positive_parameter(getproperty(estimate.phys, name))
        target = float(getproperty(truth, name))
        err = abs(fitted - target) / max(abs(target), eps(target))
        rel[name] = err
        push!(squares, abs2(err))
    end
    return sqrt(mean(squares)), rel
end

"""Fully known Michaelis–Menten production network (two states)."""
function build_mm_test_network()::BiologicalNetwork
    nodes = [NodeSpec(name = :S), NodeSpec(name = :E)]
    reactions = [
        ReactionSpec(name = :sat_prod,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     known = true, family = SATURATION,
                     metadata = SaturationMetadata(
                         vmax_param = :vmax, km_param = :km)),
        ReactionSpec(name = :s_decay,
                     stoichiometry = Dict(1 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_s)),
        ReactionSpec(name = :e_prod,
                     stoichiometry = Dict(2 => 1.0), regulators = [1],
                     metadata = MassActionMetadata(rate_param = :k_se)),
        ReactionSpec(name = :e_decay,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_e)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""
Two-state Hill degradation of `S` by `R`. Set `known=false` to replace the Hill
edge with a neural unknown for the UDE → discovery path.
"""
function build_hill_recovery_network(; known::Bool = true,
                                     hill_order::Int = 2)::BiologicalNetwork
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
                     metadata = LinearDecayMetadata(rate_param = :k_r)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""Fully known competitive-inhibition destruction of enzyme `E` by inhibitor `I`."""
function build_competitive_test_network()::BiologicalNetwork
    nodes = [NodeSpec(name = :E), NodeSpec(name = :S), NodeSpec(name = :I)]
    reactions = [
        ReactionSpec(name = :source,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :k_in)),
        ReactionSpec(name = :competitive,
                     stoichiometry = Dict(1 => -1.0),
                     regulators = [2, 3], known = true, family = COMPETITIVE,
                     metadata = CompetitiveMetadata(
                         vmax_param = :vmax, km_param = :km, ki_param = :ki)),
        ReactionSpec(name = :substrate_decay,
                     stoichiometry = Dict(2 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_s)),
        ReactionSpec(name = :inhibitor_decay,
                     stoichiometry = Dict(3 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_i)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""Chain plus a distractor node for graph-local vs global library ablation."""
function build_distractor_network()::BiologicalNetwork
    nodes = [NodeSpec(name = :x), NodeSpec(name = :reg), NodeSpec(name = :z)]
    edges = [
        EdgeSpec(source = 2, target = 1, kind = INHIBITION, family = HILL),
    ]
    reactions = [
        ReactionSpec(name = :z_decay,
                     stoichiometry = Dict(3 => -1.0), regulators = Int[],
                     metadata = LinearDecayMetadata(rate_param = :k_z)),
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

function _rhs_correlation(rhs, X, dX)
    pred = reduce(hcat, (rhs(X[:, j]) for j in axes(X, 2)))
    c = cor(vec(pred), vec(dX))
    return isnan(c) ? 0.0 : c
end

"""
    run_recovery_suite(rng=MersenneTwister(1); kwargs...)

Scientific recovery report used by CI and `benchmark/recovery_suite.jl`.
"""
function run_recovery_suite(rng::AbstractRNG = MersenneTwister(1);
                            linear_adam::Int = 40,
                            linear_bfgs::Int = 20,
                            mm_adam::Int = 50,
                            mm_bfgs::Int = 25,
                            ude_adam::Int = 50,
                            ude_bfgs::Int = 20,
                            hill_adam::Int = 40,
                            hill_bfgs::Int = 20,
                            competitive_adam::Int = 40,
                            competitive_bfgs::Int = 20,
                            sections = (:linear, :mm, :hill, :competitive,
                                        :ude_discovery, :ablation))
    report = Dict{Symbol,Any}()
    wanted = Set(sections)

    if :linear in wanted
    linear_net = build_linear_test_network()
    linear_model, linear_p0 = build_ude_model(rng, linear_net)
    linear_truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    linear_true = pack_parameters(linear_truth, linear_p0.nn)
    u0 = [0.35, 0.25]
    tspan = (0.0, 8.0)
    times, clean, _, _ = generate_data(
        rng; network = linear_net, u0 = u0, tspan = tspan,
        n_points = 40, noise_σ = 0.0, truth_params = linear_true)
    linear_init = pack_parameters((k_ba = 1.15, k_a = 0.85, k_b = 0.75), linear_p0.nn)
    linear_fit = train_ude(
        linear_init, clean, times, u0, tspan, linear_model;
        adam_iters = linear_adam, bfgs_iters = linear_bfgs, verbose = false)
    linear_rmse, linear_rel = relative_parameter_error(linear_fit.params, linear_truth)
    report[:linear] = (; rmse = linear_rmse, rel = linear_rel,
                       final_loss = linear_fit.final_loss)
    end

    if :mm in wanted
    mm_net = build_mm_test_network()
    mm_model, mm_p0 = build_ude_model(rng, mm_net)
    mm_truth = (vmax = 1.6, km = 0.45, k_s = 0.7, k_se = 0.9, k_e = 0.55)
    mm_true = pack_parameters(mm_truth, mm_p0.nn)
    mm_u0 = [0.4, 0.3]
    mm_tspan = (0.0, 8.0)
    mm_times, mm_clean, _, _ = generate_data(
        rng; network = mm_net, u0 = mm_u0, tspan = mm_tspan,
        n_points = 50, noise_σ = 0.0, truth_params = mm_true)
    mm_init = pack_parameters(
        (vmax = 1.1, km = 0.7, k_s = 1.0, k_se = 0.6, k_e = 0.8), mm_p0.nn)
    mm_fit = train_ude(
        mm_init, mm_clean, mm_times, mm_u0, mm_tspan, mm_model;
        adam_iters = mm_adam, bfgs_iters = mm_bfgs, verbose = false)
    mm_rmse, mm_rel = relative_parameter_error(mm_fit.params, mm_truth)
    report[:mm] = (; rmse = mm_rmse, rel = mm_rel, final_loss = mm_fit.final_loss)
    end

    if :hill in wanted
    hill_net = build_hill_recovery_network(; known = true, hill_order = 2)
    hill_model, hill_p0 = build_ude_model(rng, hill_net)
    hill_truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    hill_true = pack_parameters(hill_truth, hill_p0.nn)
    hill_u0 = [0.3, 0.25]
    hill_tspan = (0.0, 8.0)
    hill_times, hill_clean, _, _ = generate_data(
        rng; network = hill_net, u0 = hill_u0, tspan = hill_tspan,
        n_points = 50, noise_σ = 0.0, truth_params = hill_true)
    hill_init = pack_parameters(
        (k_prod = 0.7, vmax = 1.2, K = 0.8, k_rs = 0.75, k_r = 0.85), hill_p0.nn)
    hill_fit = train_ude(
        hill_init, hill_clean, hill_times, hill_u0, hill_tspan, hill_model;
        adam_iters = hill_adam, bfgs_iters = hill_bfgs, verbose = false)
    hill_rmse, hill_rel = relative_parameter_error(hill_fit.params, hill_truth)
    report[:hill] = (; rmse = hill_rmse, rel = hill_rel,
                     final_loss = hill_fit.final_loss)
    end

    if :competitive in wanted
    comp_net = build_competitive_test_network()
    comp_model, comp_p0 = build_ude_model(rng, comp_net)
    comp_truth = (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6, k_s = 0.8, k_i = 0.5)
    comp_true = pack_parameters(comp_truth, comp_p0.nn)
    comp_u0 = [0.25, 0.45, 0.2]
    comp_tspan = (0.0, 8.0)
    comp_times, comp_clean, _, _ = generate_data(
        rng; network = comp_net, u0 = comp_u0, tspan = comp_tspan,
        n_points = 55, noise_σ = 0.0, truth_params = comp_true)
    comp_init = pack_parameters(
        (k_in = 0.65, vmax = 1.1, km = 0.65, ki = 0.9, k_s = 1.05, k_i = 0.75),
        comp_p0.nn)
    comp_fit = train_ude(
        comp_init, comp_clean, comp_times, comp_u0, comp_tspan, comp_model;
        adam_iters = competitive_adam, bfgs_iters = competitive_bfgs,
        verbose = false)
    comp_rmse, comp_rel = relative_parameter_error(comp_fit.params, comp_truth)
    report[:competitive] = (; rmse = comp_rmse, rel = comp_rel,
                            final_loss = comp_fit.final_loss)
    end

    if :ude_discovery in wanted
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    truth_model, truth_p0 = build_ude_model(rng, truth_net)
    hill_truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    hill_true = pack_parameters(hill_truth, truth_p0.nn)
    hill_u0 = [0.3, 0.25]
    hill_times, hill_clean, _, _ = generate_data(
        rng; network = truth_net, u0 = hill_u0, tspan = (0.0, 6.0),
        n_points = 40, noise_σ = 0.0, truth_params = hill_true)
    ude_model, ude_p0 = build_ude_model(rng, ude_net)
    ude_init = pack_parameters(
        (k_prod = 0.7, k_rs = 0.8, k_r = 0.8), ude_p0.nn)
    ude_fit = train_ude(
        ude_init, hill_clean, hill_times, hill_u0, (0.0, 6.0), ude_model;
        adam_iters = ude_adam, bfgs_iters = ude_bfgs, verbose = false)
    ude_X, ude_dX, _ = _collect_trajectory_data(
        ude_fit.params, ude_model, hill_u0, (0.0, 6.0), 80)
    discovery = discover_equations(
        ude_X, collect(range(0.0, 6.0; length = size(ude_X, 2))), ude_net;
        derivatives = ude_dX,
        config = DiscoveryConfig(
            backend = ImplicitSINDyPI(
                threshold = 1e-3, max_degree = 2, max_hill_degree = 2,
                bootstrap_samples = 4, validation_fraction = 0.2,
                domain_samples = 32, chunk_size = 32),
            include_interactions = false, seed = 3),
        verbose = false, strict = false)
    corr = NaN
    if discovery.success
        rhs = export_rhs(discovery)
        corr = _rhs_correlation(rhs, ude_X, ude_dX)
    end
    report[:ude_discovery] = (;
        success = discovery.success,
        retcode = discovery.retcode,
        correlation = corr,
        message = discovery.message)
    end

    if :ablation in wanted
    x = collect(range(0.1, 2.0; length = 180))
    z = collect(range(0.4, 1.2; length = 180))
    X_ab = permutedims(hcat(x, x ./ 2 .+ 0.2, z))
    vmax, k = 1.7, 0.6
    d1 = vmax .* (X_ab[2, :] .^ 2) ./ (k^2 .+ X_ab[2, :] .^ 2)
    dX_ab = vcat(reshape(d1, 1, :), zeros(2, length(x)))
    distractor = build_distractor_network()
    local_count = candidate_count(local_basis(
        distractor, 1; degree = 2, include_interactions = false, scope = :graph))
    global_count = candidate_count(local_basis(
        distractor, 1; degree = 2, include_interactions = false, scope = :global,
        X = X_ab, derivative = vec(dX_ab[1, :])))
    local_spec = local_basis(
        distractor, 1; degree = 2, include_interactions = false, scope = :graph,
        X = X_ab, derivative = vec(dX_ab[1, :]))
    global_spec = local_basis(
        distractor, 1; degree = 2, include_interactions = false, scope = :global,
        X = X_ab, derivative = vec(dX_ab[1, :]))
    report[:ablation] = (;
        local_terms = local_count,
        global_terms = global_count,
        local_variables = copy(local_spec.variables),
        global_variables = copy(global_spec.variables))
    end
    return report
end
