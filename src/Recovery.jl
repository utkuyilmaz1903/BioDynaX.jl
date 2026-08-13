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

"""Locked scientific-recovery contract. Loosening a threshold is a breaking change."""
const RECOVERY_THRESHOLDS = (
    nn_correlation = 0.90,
    nn_rate_rmse = 0.25,
    support_f1_clean = 0.99,
    support_f1_ude = 0.50,
    support_f1_noisy = 0.50,
    support_recall = 0.99,
    discovered_rate_rmse = 0.20,
    data_residual = 0.30,
)

term_key(term::MonomialTerm) = (Tuple(term.variables), Tuple(term.powers))

function active_support(candidate::ImplicitCandidate; atol::Real = 1e-8)
    num = Set{Tuple{Tuple{Vararg{Int}},Tuple{Vararg{Int}}}}()
    den = Set{Tuple{Tuple{Vararg{Int}},Tuple{Vararg{Int}}}}()
    spec = candidate.specification
    for (coefficient, term) in zip(candidate.numerator_coefficients, spec.numerator)
        abs(coefficient) > atol && push!(num, term_key(term))
    end
    for (coefficient, term) in zip(candidate.denominator_coefficients, spec.denominator)
        abs(coefficient) > atol && push!(den, term_key(term))
    end
    return (numerator = num, denominator = den)
end

function active_support(candidate::ExplicitCandidate; atol::Real = 1e-8)
    num = Set{Tuple{Tuple{Vararg{Int}},Tuple{Vararg{Int}}}}()
    for (coefficient, term) in zip(candidate.coefficients,
                                   candidate.specification.numerator)
        abs(coefficient) > atol && push!(num, term_key(term))
    end
    return (numerator = num, denominator = Set{eltype(num)}())
end

function support_f1(recovered::Set, truth::Set)
    isempty(truth) && isempty(recovered) &&
        return (; precision = 1.0, recall = 1.0, f1 = 1.0, tp = 0, fp = 0, fn = 0)
    tp = length(intersect(recovered, truth))
    fp = length(setdiff(recovered, truth))
    fn = length(setdiff(truth, recovered))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall)
    return (; precision, recall, f1, tp, fp, fn)
end

function support_f1(candidate, truth_num::Set, truth_den::Set; atol::Real = 1e-8)
    recovered = active_support(candidate; atol = atol)
    combined_rec = union(
        Set((:n, key) for key in recovered.numerator),
        Set((:d, key) for key in recovered.denominator))
    combined_truth = union(
        Set((:n, key) for key in truth_num),
        Set((:d, key) for key in truth_den))
    return (;
        numerator = support_f1(recovered.numerator, truth_num),
        denominator = support_f1(recovered.denominator, truth_den),
        combined = support_f1(combined_rec, combined_truth))
end

function rate_rel_rmse(estimate, truth)
    estimate_vec = vec(Float64.(estimate))
    truth_vec = vec(Float64.(truth))
    length(estimate_vec) == length(truth_vec) ||
        throw(DimensionMismatch("estimate and truth must have the same length"))
    scale = max(sqrt(mean(abs2, truth_vec)), eps(Float64))
    return sqrt(mean(abs2, estimate_vec .- truth_vec)) / scale
end

function denominator_violation_count(candidate::ImplicitCandidate, X;
                                     floor::Real = 1e-8)
    _, denominator = _evaluate_candidate(
        candidate.specification,
        candidate.numerator_coefficients,
        candidate.denominator_coefficients, X)
    return count(<(floor), denominator)
end

function support_uses_variable(candidate; variable::Int, atol::Real = 1e-8)
    recovered = active_support(candidate; atol = atol)
    keys = union(recovered.numerator, recovered.denominator)
    return any(key -> variable in key[1], keys)
end

"""True implicit support for `D = vmax r^n / (K^n + r^n)` on variable `variable`."""
function hill_rate_support(order::Int; variable::Int = 1)
    key = ((variable,), (order,))
    return (numerator = Set([key]), denominator = Set([key]))
end

"""True implicit support for `D = vmax r / (km + r)` on variable `variable`."""
function mm_rate_support(; variable::Int = 1)
    key = ((variable,), (1,))
    return (numerator = Set([key]), denominator = Set([key]))
end

hill_rate_truth(r; vmax, K, n) = vmax .* (r .^ n) ./ (K^n .+ r .^ n)
mm_rate_truth(r; vmax, km) = vmax .* r ./ (km .+ r)

neural_destruction_terms(model::UDEModel) =
    [term for term in model.compiled.destruction_terms if term isa NeuralDestructionTerm]

"""
    sample_unknown_destruction(model, p, X; term=nothing)

Evaluate compiled neural destruction `D` at trajectory columns of `X`
(the rate used inside `du = P - D·u`, not a raw Lux call).
"""
function sample_unknown_destruction(model::UDEModel, p, X::AbstractMatrix;
                                    term = nothing)
    terms = neural_destruction_terms(model)
    chosen = term === nothing ? only(terms) : term
    n = size(X, 2)
    rates = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        x = @view X[:, j]
        rates[j] = _destruction_contribution(
            chosen, chosen.target, x, p, model.nn, model.st)
    end
    R = reshape(Float64.(X[chosen.regulator, :]), 1, :)
    return R, reshape(rates, 1, :), chosen
end

function sample_unknown_destruction_grid(model::UDEModel, p, term;
                                         r_range = range(0.05, 2.0; length = 80),
                                         fill_value = 0.3)
    nstates = model.compiled.nstates
    r = collect(r_range)
    X = fill(float(fill_value), nstates, length(r))
    X[term.regulator, :] .= r
    return sample_unknown_destruction(model, p, X; term = term)
end

"""Single-state network used to discover a scalar rate `D(r)`."""
build_rate_discovery_network() =
    BiologicalNetwork([NodeSpec(name = :r)], EdgeSpec[])

"""Two-state rate network: `r` plus unused distractor `z` for global ablations."""
build_rate_ablation_network() =
    BiologicalNetwork([NodeSpec(name = :r), NodeSpec(name = :z)], EdgeSpec[])

function rate_discovery_config(; threshold = 1e-3, degree = 2, bootstrap = 8,
                               scope::Symbol = :graph, seed = 3)
    return DiscoveryConfig(
        backend = ImplicitSINDyPI(
            threshold = threshold, max_degree = degree, max_hill_degree = degree,
            bootstrap_samples = bootstrap, validation_fraction = 0.2,
            domain_samples = 32, chunk_size = 32),
        include_interactions = false, seed = seed, basis_scope = scope)
end

function _permute_rate_samples(X::AbstractMatrix, dX::AbstractMatrix, seed)
    n = size(X, 2)
    perm = randperm(MersenneTwister(UInt64(seed) ⊻ 0x9e3779b97f4a7c15), n)
    return X[:, perm], dX[:, perm]
end

"""
    discover_unknown_rate(R, times, D; network, config, ...)

Discover a scalar destruction rate `D(r)` with graph-local implicit SINDy-PI.
`R` and `D` are `1 × n` (or matching) sample matrices.
"""
function discover_unknown_rate(R::AbstractMatrix, times, D::AbstractMatrix;
                               network = build_rate_discovery_network(),
                               config = rate_discovery_config(),
                               verbose::Bool = false, strict::Bool = false)
    R_perm, D_perm = _permute_rate_samples(R, D, config.seed)
    return discover_equations(
        R_perm, times, network; derivatives = D_perm, targets = 1,
        config = config, verbose = verbose, strict = strict)
end

"""
    compose_hybrid_rhs(model, p, term, rate_fn)

ODE right-hand side that keeps compiled known terms and replaces neural
destruction `term` with `rate_fn([regulator])`.
"""
function compose_hybrid_rhs(model::UDEModel, p, term::NeuralDestructionTerm, rate_fn)
    return function (u, _, t)
        du = ude_system(u, p, t, model)
        nn_D = _destruction_contribution(
            term, term.target, u, p, model.nn, model.st)
        hat_D = rate_fn([u[term.regulator]])
        du[term.target] += (nn_D - hat_D) * u[term.target]
        return du
    end
end

function hybrid_data_residual(model, p, term, rate_fn, u0, tspan, times, data)
    rhs = compose_hybrid_rhs(model, p, term, rate_fn)
    prob = SciMLBase.ODEProblem(rhs, u0, tspan)
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return Inf
    pred = Array(sol)
    size(pred) == size(data) || return Inf
    return sqrt(mean(abs2, pred .- data))
end

function _unknown_edge_ics()
    return [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10], [1.20, 0.70], [0.15, 0.90]]
end

function _train_unknown_edge(rng, ude_model, ude_p0, truth_net, truth_params;
                             adam, bfgs, noise_σ, tspan, n_points)
    set = generate_experiment_set(
        rng; network = truth_net, initial_conditions = _unknown_edge_ics(),
        tspan = tspan, n_points = n_points, noise_σ = noise_σ,
        truth_params = truth_params)
    names = Tuple(parameter_schema(ude_model).phys_names)
    guess = NamedTuple{names}(ntuple(_ -> 0.8, length(names)))
    ude_init = pack_parameters(guess, ude_p0.nn)
    first_exp = first(set.experiments)
    warm = train_ude(
        ude_init, first_exp.observations, first_exp.times, first_exp.u0,
        (first(first_exp.times), last(first_exp.times)), ude_model;
        config = TrainingConfig(
            adam_iterations = adam,
            bfgs_iterations = 0,
            horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0]),
            log_every = 10^6),
        verbose = false)
    fit = train_experiments(
        warm.params, set, ude_model;
        config = TrainingConfig(
            adam_iterations = adam,
            bfgs_iterations = bfgs,
            log_every = 10^6),
        verbose = false)
    return fit, set
end

function _regulator_grid(set::ExperimentSet, term; npoints::Int = 80)
    values = reduce(vcat, (exp.observations[term.regulator, :]
                           for exp in set.experiments))
    lo, hi = extrema(values)
    span = max(hi - lo, 0.1)
    start = max(0.05, lo - 0.1 * span)
    stop = hi + 0.1 * span
    return range(start, stop; length = npoints)
end

function _evaluate_unknown_rate_recovery(ude_model, ude_params, term, truth_rate;
                                         order, family::Symbol, noise_σ,
                                         data_residual_fn,
                                         r_range = range(0.05, 2.0; length = 80))
    R_grid, D_nn, _ = sample_unknown_destruction_grid(
        ude_model, ude_params, term; r_range = r_range)
    r = vec(R_grid)
    D_true = truth_rate(r)
    nn_corr = cor(vec(D_nn), D_true)
    nn_corr = isnan(nn_corr) ? 0.0 : nn_corr
    nn_rmse = rate_rel_rmse(D_nn, D_true)
    training_ok = nn_corr ≥ RECOVERY_THRESHOLDS.nn_correlation &&
                  nn_rmse ≤ RECOVERY_THRESHOLDS.nn_rate_rmse
    if !training_ok
        return (;
            nn_correlation = nn_corr,
            nn_rate_rmse = nn_rmse,
            success = false,
            retcode = DiscoveryFailed,
            message = "training did not identify the unknown edge",
            support_f1 = 0.0,
            support_recall = 0.0,
            discovered_rate_rmse = Inf,
            data_residual = Inf,
            denominator_violations = typemax(Int),
            discovery = nothing,
            term = term)
    end
    times = collect(range(0.0, 1.0; length = length(r)))
    truth_support = family === :hill ? hill_rate_support(order) : mm_rate_support()
    discovery = discover_unknown_rate(
        R_grid, times, D_nn;
        config = rate_discovery_config(bootstrap = 8, seed = 3),
        verbose = false, strict = false)
    f1 = 0.0
    recall = 0.0
    rate_rmse = Inf
    residual = Inf
    den_violations = typemax(Int)
    if discovery.success
        candidate = discovery.candidates[1]
        metrics = support_f1(candidate, truth_support.numerator,
                             truth_support.denominator)
        f1 = metrics.combined.f1
        recall = metrics.combined.recall
        d_hat = equation_to_function(candidate)
        D_hat = [d_hat([rj]) for rj in r]
        rate_rmse = rate_rel_rmse(D_hat, D_true)
        den_violations = denominator_violation_count(candidate, R_grid)
        residual = data_residual_fn(d_hat)
    end
    return (;
        nn_correlation = nn_corr,
        nn_rate_rmse = nn_rmse,
        success = discovery.success,
        retcode = discovery.retcode,
        message = discovery.message,
        support_f1 = f1,
        support_recall = recall,
        discovered_rate_rmse = rate_rmse,
        data_residual = residual,
        denominator_violations = den_violations,
        discovery = discovery,
        term = term)
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

"""Two-state MM degradation of `S` by `R`. `known=false` compiles a neural unknown."""
function build_mm_recovery_network(; known::Bool = true)::BiologicalNetwork
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s,
                     stoichiometry = Dict(1 => 1.0), regulators = [2],
                     metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :mm_deg,
                     stoichiometry = Dict(1 => -1.0), regulators = [2],
                     known = known, family = SATURATION,
                     metadata = SaturationMetadata(
                         vmax_param = :vmax, km_param = :km)),
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
                            ude_adam::Int = 80,
                            ude_bfgs::Int = 40,
                            hill_adam::Int = 40,
                            hill_bfgs::Int = 20,
                            competitive_adam::Int = 40,
                            competitive_bfgs::Int = 20,
                            ude_noise_σ::Float64 = 0.0,
                            sections = (:linear, :mm, :hill, :competitive,
                                        :ude_discovery, :mm_unknown, :ablation))
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
    # Consume the same RNG stream as known-kinetics fixtures so UDE init stays stable.
    build_ude_model(rng, truth_net)
    hill_truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    ude_model, ude_p0 = build_ude_model(rng, ude_net)
    ude_fit, ude_set = _train_unknown_edge(
        rng, ude_model, ude_p0, truth_net, hill_truth;
        adam = ude_adam, bfgs = ude_bfgs, noise_σ = ude_noise_σ,
        tspan = (0.0, 8.0), n_points = 40)
    term = only(neural_destruction_terms(ude_model))
    ref_exp = first(ude_set.experiments)
    evaled = _evaluate_unknown_rate_recovery(
        ude_model, ude_fit.params, term,
        r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2);
        order = 2, family = :hill, noise_σ = ude_noise_σ,
        r_range = _regulator_grid(ude_set, term),
        data_residual_fn = d_hat -> hybrid_data_residual(
            ude_model, ude_fit.params, term, d_hat,
            ref_exp.u0, (first(ref_exp.times), last(ref_exp.times)),
            ref_exp.times, ref_exp.observations))
    report[:ude_discovery] = evaled
    end

    if :mm_unknown in wanted
    truth_net = build_mm_recovery_network(; known = true)
    ude_net = build_mm_recovery_network(; known = false)
    build_ude_model(rng, truth_net)
    mm_truth = (k_prod = 0.9, vmax = 1.6, km = 0.45, k_rs = 1.0, k_r = 0.6)
    ude_model, ude_p0 = build_ude_model(rng, ude_net)
    ude_fit, ude_set = _train_unknown_edge(
        rng, ude_model, ude_p0, truth_net, mm_truth;
        adam = ude_adam, bfgs = ude_bfgs, noise_σ = ude_noise_σ,
        tspan = (0.0, 8.0), n_points = 40)
    term = only(neural_destruction_terms(ude_model))
    ref_exp = first(ude_set.experiments)
    evaled = _evaluate_unknown_rate_recovery(
        ude_model, ude_fit.params, term,
        r -> mm_rate_truth(r; vmax = 1.6, km = 0.45);
        order = 1, family = :mm, noise_σ = ude_noise_σ,
        r_range = _regulator_grid(ude_set, term),
        data_residual_fn = d_hat -> hybrid_data_residual(
            ude_model, ude_fit.params, term, d_hat,
            ref_exp.u0, (first(ref_exp.times), last(ref_exp.times)),
            ref_exp.times, ref_exp.observations))
    report[:mm_unknown] = evaled
    end

    if :ablation in wanted
    r = collect(range(0.1, 2.0; length = 180))
    rng_ab = MersenneTwister(104)
    vmax, k = 1.7, 0.6
    D = hill_rate_truth(r; vmax = vmax, K = k, n = 2)
    amp = max(maximum(abs, D), eps(Float64))
    D_noisy = D .+ 0.005 .* amp .* randn(rng_ab, length(r))
    z = D .+ 0.15 .* amp .* randn(rng_ab, length(r))
    X_ab = permutedims(hcat(r, z))
    dX_ab = vcat(reshape(D_noisy, 1, :), reshape(-0.5 .* z, 1, :))
    X_ab, dX_ab = _permute_rate_samples(X_ab, dX_ab, 104)
    times_ab = collect(range(0.0, 1.0; length = length(r)))
    net_ab = build_rate_ablation_network()
    truth = hill_rate_support(2; variable = 1)
    local_time = @elapsed local_disc = discover_equations(
        X_ab, times_ab, net_ab; derivatives = dX_ab, targets = 1,
        config = rate_discovery_config(scope = :graph, bootstrap = 8, seed = 4),
        verbose = false, strict = false)
    global_time = @elapsed global_disc = discover_equations(
        X_ab, times_ab, net_ab; derivatives = dX_ab, targets = 1,
        config = rate_discovery_config(scope = :global, bootstrap = 8, seed = 4),
        verbose = false, strict = false)
    local_idx = local_disc.success ?
        findfirst(c -> c.target == 1, local_disc.candidates) : nothing
    local_cand = local_idx === nothing ? nothing : local_disc.candidates[local_idx]
    global_idx = global_disc.success ?
        findfirst(c -> c.target == 1, global_disc.candidates) : nothing
    global_cand = global_idx === nothing ? nothing : global_disc.candidates[global_idx]
    local_f1 = local_cand === nothing ? 0.0 :
        support_f1(local_cand, truth.numerator, truth.denominator).combined.f1
    global_f1 = global_cand === nothing ? 0.0 :
        support_f1(global_cand, truth.numerator, truth.denominator).combined.f1
    local_fp = local_cand !== nothing &&
        support_uses_variable(local_cand; variable = 2)
    global_fp = global_cand !== nothing &&
        support_uses_variable(global_cand; variable = 2)
    local_den = local_cand === nothing ? typemax(Int) :
        denominator_violation_count(local_cand, X_ab)
    global_den = global_cand === nothing ? typemax(Int) :
        denominator_violation_count(global_cand, X_ab)
    local_spec = local_basis(
        net_ab, 1; degree = 2, include_interactions = false, scope = :graph)
    global_spec = local_basis(
        net_ab, 1; degree = 2, include_interactions = false, scope = :global)
    report[:ablation] = (;
        local_terms = candidate_count(local_spec),
        global_terms = candidate_count(global_spec),
        local_variables = copy(local_spec.variables),
        global_variables = copy(global_spec.variables),
        local_success = local_disc.success,
        global_success = global_disc.success,
        local_f1 = local_f1,
        global_f1 = global_f1,
        local_false_parent = local_fp,
        global_false_parent = global_fp,
        local_denominator_violations = local_den,
        global_denominator_violations = global_den,
        local_time = local_time,
        global_time = global_time)
    end
    return report
end
