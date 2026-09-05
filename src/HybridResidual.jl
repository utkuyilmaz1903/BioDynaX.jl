###############################################################################
# Hybrid residual versus solver (not exported).
#
# HybridCompose locked compose_hybrid_rhs identity against ude_system.
# This file locks the remaining residual join: hybrid_data_residual versus
# SciMLBase.solve of the composed RHS, versus predict_ude, versus
# SciMLBase.ODEProblem(model, ...). Failed compose paths stay failed.
# Noise-0 identity residual is ~0; smoke (1 IC / 8 points) is not the
# seed-103 / 9-IC protocol residual.
#
# Does not drop protocol ICs. Does not grow exports. Does not open
# Hill-from-NN. Combined F1 stays a skeleton floor.
###############################################################################

const HYBRID_RESIDUAL_MUST_CONTAIN = (
    "function hybrid_residual_sciml_solve",
    "function hybrid_residual_model_solve",
    "function hybrid_residual_predict_ude",
    "function residual_solver_agreement_row",
    "function noise0_vs_noisy_residual_row",
    "function smoke_vs_protocol_residual_row",
    "function failed_compose_linear_term_row",
    "struct HybridResidualRow",
    "function hybrid_residual_failed_solve_row")

const HYBRID_RESIDUAL_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function hybrid_residual_source_path()
    joinpath(pkgdir(BioDynaX), "src", "HybridResidual.jl")
end

# -- Residual solvers ---------------------------------------------------------

"""
    array_data_rmse(pred, data)

RMSE of two observation arrays. Shape mismatch is `Inf`.
"""
function array_data_rmse(pred, data)
    size(pred) == size(data) || return Inf
    return sqrt(mean(abs2, pred .- data))
end

"""
    hybrid_residual_sciml_solve(model, p, term, rate_fn, u0, tspan, times, data)

`SciMLBase.ODEProblem` of `compose_hybrid_rhs` integrated with `Tsit5`.
This is the same entry `hybrid_data_residual` uses. Not a new solver.
"""
function hybrid_residual_sciml_solve(model, p, term, rate_fn, u0, tspan, times, data)
    rhs = compose_hybrid_rhs(model, p, term, rate_fn)
    prob = SciMLBase.ODEProblem(rhs, Float64.(u0), tspan)
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return Inf
    return array_data_rmse(Array(sol), data)
end

"""
    hybrid_residual_model_solve(model, p, u0, tspan, times, data)

`SciMLBase.ODEProblem(model, u0, tspan, p)` residual versus observations.
Identity `rate_fn` must agree with this solve at noise 0.
"""
function hybrid_residual_model_solve(model::UDEModel, p, u0, tspan, times, data)
    prob = SciMLBase.ODEProblem(model, Float64.(u0), tspan, p)
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return Inf
    return array_data_rmse(Array(sol), data)
end

"""
    hybrid_residual_predict_ude(model, p, u0, tspan, times, data)

`predict_ude` residual versus the same observations.
"""
function hybrid_residual_predict_ude(model::UDEModel, p, u0, tspan, times, data)
    pred = predict_ude(p, Float64.(u0), tspan, times, model)
    return array_data_rmse(pred, data)
end

function hybrid_identity_term(model::UDEModel)
    terms = neural_destruction_terms(model)
    length(terms) == 1 || return nothing
    return only(terms)
end

function hybrid_residual_bundle(model::UDEModel, p, u0; tspan = (0.0, 0.8),
        n_points::Int = 16, rng_seed::Integer = 7, noise_σ::Float64 = 0.0)
    term = hybrid_identity_term(model)
    term === nothing && return nothing
    times, clean, noisy, _ = generate_from_compiled_model(
        model, p, MersenneTwister(rng_seed);
        u0 = Float64.(u0), tspan = tspan, n_points = n_points, noise_σ = noise_σ)
    data = noise_σ == 0 ? clean : noisy
    rate = neural_identity_rate(model, p, term)
    hybrid = hybrid_data_residual(
        model, p, term, rate, Float64.(u0), tspan, times, data)
    sciml = hybrid_residual_sciml_solve(
        model, p, term, rate, u0, tspan, times, data)
    model_solve = hybrid_residual_model_solve(model, p, u0, tspan, times, data)
    pred = hybrid_residual_predict_ude(model, p, u0, tspan, times, data)
    return (;
        term, times, clean, noisy, data, rate,
        hybrid, sciml, model_solve, pred,
        n_points = length(times),
        nstates = size(data, 1))
end

# -- Agreement rows -----------------------------------------------------------

"""
    residual_solver_agreement_row(model, p, u0; kwargs...)

`hybrid_data_residual` matches `SciMLBase.solve` of the composed RHS.
At noise 0 it also matches the compiled-model solve and `predict_ude`.
`compile_network` stays at zero.
"""
function residual_solver_agreement_row(model::UDEModel, p, u0; kwargs...)
    bundle = hybrid_residual_bundle(model, p, u0; kwargs...)
    bundle === nothing && return (; holds = false, reason = :not_single)
    n = with_compile_network_counter() do counter
        hybrid_residual_sciml_solve(
            model, p, bundle.term, bundle.rate,
            u0, (first(bundle.times), last(bundle.times)),
            bundle.times, bundle.data)
        counter[]
    end
    hybrid_sciml = bundle.hybrid ≈ bundle.sciml ||
                   abs(bundle.hybrid - bundle.sciml) < 1e-12
    identity_small = bundle.hybrid < 1e-6
    model_small = bundle.model_solve < 1e-6
    pred_small = bundle.pred < 1e-6
    return (;
        hybrid = bundle.hybrid,
        sciml = bundle.sciml,
        model_solve = bundle.model_solve,
        pred = bundle.pred,
        compiles = n,
        hybrid_sciml,
        identity_small,
        model_small,
        pred_small,
        n_points = bundle.n_points,
        holds = n == 0 && hybrid_sciml && identity_small &&
                model_small && pred_small &&
                isfinite(bundle.hybrid) && isfinite(bundle.sciml))
end

function residual_solver_vs_zero_rate_row(model::UDEModel, p, u0)
    term = hybrid_identity_term(model)
    term === nothing && return (; holds = false, reason = :not_single)
    traj = hybrid_generate(model, p, u0; n_points = 12)
    identity = hybrid_residual_sciml_solve(
        model, p, term, neural_identity_rate(model, p, term),
        traj.u0, traj.tspan, traj.times, traj.data)
    zeroed = hybrid_residual_sciml_solve(
        model, p, term, zero_rate,
        traj.u0, traj.tspan, traj.times, traj.data)
    helper = hybrid_data_residual(
        model, p, term, neural_identity_rate(model, p, term),
        traj.u0, traj.tspan, traj.times, traj.data)
    return (;
        identity, zeroed, helper,
        holds = identity < 1e-6 && zeroed > identity &&
                helper ≈ identity && isfinite(zeroed))
end

function residual_solver_mask_row(model::UDEModel, p, u0)
    term = hybrid_identity_term(model)
    term === nothing && return (; holds = false, reason = :not_single)
    traj = hybrid_generate(model, p, u0; n_points = 12)
    rate = neural_identity_rate(model, p, term)
    mask = trues(size(traj.data))
    mask[1, 2:end] .= false
    masked = hybrid_data_residual(
        model, p, term, rate, traj.u0, traj.tspan, traj.times, traj.data;
        mask = mask)
    empty_mask = hybrid_data_residual(
        model, p, term, rate, traj.u0, traj.tspan, traj.times, traj.data;
        mask = falses(size(traj.data)))
    return (;
        masked, empty_mask,
        holds = isfinite(masked) && masked < 1e-6 && empty_mask == Inf)
end

function residual_solver_irregular_row(model::UDEModel, p, u0)
    term = hybrid_identity_term(model)
    term === nothing && return (; holds = false, reason = :not_single)
    saveat = [0.0, 0.04, 0.17, 0.33, 0.52, 0.80]
    tspan = (0.0, 0.80)
    pred = predict_ude(p, Float64.(u0), tspan, saveat, model)
    hybrid = hybrid_data_residual(
        model, p, term, neural_identity_rate(model, p, term),
        Float64.(u0), tspan, saveat, pred)
    sciml = hybrid_residual_sciml_solve(
        model, p, term, neural_identity_rate(model, p, term),
        u0, tspan, saveat, pred)
    return (;
        hybrid, sciml, n_points = length(saveat),
        holds = hybrid < 1e-6 && sciml < 1e-6 && hybrid ≈ sciml)
end

function residual_solver_session_row(model::UDEModel, p, u0)
    term = hybrid_identity_term(model)
    term === nothing && return (; holds = false, reason = :not_single)
    tspan = (0.0, 0.8)
    times = collect(range(first(tspan), last(tspan); length = 12))
    session = training_solve_session(model, Float64.(u0), tspan, p)
    remade = predict_ude_session(session, p, Float64.(u0), tspan, times)
    hybrid = hybrid_data_residual(
        model, p, term, neural_identity_rate(model, p, term),
        Float64.(u0), tspan, times, remade)
    sciml = hybrid_residual_sciml_solve(
        model, p, term, neural_identity_rate(model, p, term),
        u0, tspan, times, remade)
    n = with_compile_network_counter() do counter
        predict_ude_session(session, p, Float64.(u0), tspan, times)
        counter[]
    end
    return (;
        hybrid, sciml, compiles = n, remake_count = session.remake_count,
        holds = hybrid < 1e-6 && sciml < 1e-6 && n == 0 &&
                session.remake_count ≥ 2)
end

# -- Noise checks ------------------------------------------------------------

"""
    noise0_vs_noisy_residual_row(model, p, u0; noise_σ=0.05)

Identity residual versus clean noise-0 data is ~0. The same rate versus
noisy observations is larger. This is a data residual, not an F1 paint.
"""
function noise0_vs_noisy_residual_row(model::UDEModel, p, u0;
        noise_σ::Float64 = 0.05, n_points::Int = 16, tspan = (0.0, 0.8),
        rng_seed::Integer = 211)
    term = hybrid_identity_term(model)
    term === nothing && return (; holds = false, reason = :not_single)
    times, clean, noisy, _ = generate_from_compiled_model(
        model, p, MersenneTwister(rng_seed);
        u0 = Float64.(u0), tspan = tspan, n_points = n_points, noise_σ = noise_σ)
    rate = neural_identity_rate(model, p, term)
    vs_clean = hybrid_data_residual(
        model, p, term, rate, Float64.(u0), tspan, times, clean)
    vs_noisy = hybrid_data_residual(
        model, p, term, rate, Float64.(u0), tspan, times, noisy)
    sciml_clean = hybrid_residual_sciml_solve(
        model, p, term, rate, u0, tspan, times, clean)
    sciml_noisy = hybrid_residual_sciml_solve(
        model, p, term, rate, u0, tspan, times, noisy)
    pred_clean = hybrid_residual_predict_ude(model, p, u0, tspan, times, clean)
    return (;
        vs_clean, vs_noisy, sciml_clean, sciml_noisy, pred_clean, noise_σ,
        noisy_worse = vs_noisy > vs_clean,
        clean_small = vs_clean < 1e-6,
        paints_f1 = false,
        holds = vs_clean < 1e-6 && vs_noisy > vs_clean &&
                sciml_clean < 1e-6 && sciml_noisy > sciml_clean &&
                pred_clean < 1e-6 && isfinite(vs_noisy) &&
                !occursin("0.99", string(vs_noisy)))
end

function noise_grid_residual_row(model::UDEModel, p, u0;
        sigmas = (0.0, 0.01, 0.05, 0.10))
    term = hybrid_identity_term(model)
    term === nothing && return (; holds = false, reason = :not_single)
    rows = NamedTuple[]
    for σ in sigmas
        times, clean, noisy, _ = generate_from_compiled_model(
            model, p, MersenneTwister(223);
            u0 = Float64.(u0), tspan = (0.0, 0.8), n_points = 14, noise_σ = σ)
        data = σ == 0 ? clean : noisy
        residual = hybrid_data_residual(
            model, p, term, neural_identity_rate(model, p, term),
            Float64.(u0), (0.0, 0.8), times, data)
        push!(rows, (; noise_σ = σ, residual, finite = isfinite(residual)))
    end
    increasing = rows[1].residual < rows[end].residual
    zero_small = rows[1].residual < 1e-6 && rows[1].noise_σ == 0.0
    return (;
        rows, increasing, zero_small,
        holds = length(rows) == length(sigmas) && increasing && zero_small &&
                all(r -> r.finite, rows) &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50)
end

function noise_does_not_paint_f1_row()
    built = hybrid_linear_unknown_model(227)
    row = noise0_vs_noisy_residual_row(built.model, built.packed, [0.30, 0.25])
    return (;
        row...,
        skeleton = RECOVERY_THRESHOLDS.support_f1_ude,
        clean_gate = RECOVERY_THRESHOLDS.support_f1_clean,
        holds = row.holds &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50 &&
                RECOVERY_THRESHOLDS.support_f1_clean == 0.99 &&
                RECOVERY_THRESHOLDS.support_f1_ude <
                RECOVERY_THRESHOLDS.support_f1_clean)
end

# -- Smoke versus protocol residual -------------------------------------------

"""
    smoke_vs_protocol_residual_row()

Smoke generate is 1 IC / 8 points. Protocol generate is 9 ICs / 50 points
at seed 103. Neither path trains. Smoke residual is not the protocol
residual. Protocol ICs are not dropped.
"""
function smoke_vs_protocol_residual_row()
    smoke_fp = unique_claim_fingerprint(; smoke = true)
    protocol_fp = unique_claim_fingerprint()
    truth = hybrid_known_hill_truth(229)
    smoke_set = unique_claim_experiment_set(
        MersenneTwister(103), truth.net; smoke = true,
        truth_params = truth.truth_params)
    protocol_set = unique_claim_experiment_set(
        MersenneTwister(103), truth.net; smoke = false,
        truth_params = truth.truth_params)
    built = hybrid_linear_unknown_model(229)
    term = only(neural_destruction_terms(built.model))
    rate = neural_identity_rate(built.model, built.packed, term)
    smoke_exp = first(smoke_set.experiments)
    smoke_residual = hybrid_data_residual(
        built.model, built.packed, term, rate,
        smoke_exp.u0, (first(smoke_exp.times), last(smoke_exp.times)),
        smoke_exp.times, smoke_exp.observations)
    protocol_residuals = Float64[]
    for exp in protocol_set.experiments
        push!(protocol_residuals,
            hybrid_data_residual(
                built.model, built.packed, term, rate,
                exp.u0, (first(exp.times), last(exp.times)),
                exp.times, exp.observations))
    end
    return (;
        smoke_ics = length(smoke_set),
        protocol_ics = length(protocol_set),
        smoke_points = size(smoke_exp.observations, 2),
        protocol_points = size(first(protocol_set.experiments).observations, 2),
        smoke_residual,
        protocol_residuals,
        smoke_fp_ics = smoke_fp.n_ics,
        protocol_fp_ics = protocol_fp.n_ics,
        smoke_fp_points = smoke_fp.n_points,
        protocol_fp_points = protocol_fp.n_points,
        compiled_once = experiment_set_is_compiled_once(smoke_set) &&
                        experiment_set_is_compiled_once(protocol_set),
        holds = length(smoke_set) == 1 && length(protocol_set) == 9 &&
                size(smoke_exp.observations, 2) == 8 &&
                size(first(protocol_set.experiments).observations, 2) == 50 &&
                smoke_fp.n_ics == 1 && protocol_fp.n_ics == 9 &&
                smoke_fp.n_points == 8 && protocol_fp.n_points == 50 &&
                smoke_fp.n_ics != protocol_fp.n_ics &&
                isfinite(smoke_residual) &&
                length(protocol_residuals) == 9 &&
                all(isfinite, protocol_residuals) &&
                experiment_set_is_compiled_once(smoke_set) &&
                experiment_set_is_compiled_once(protocol_set))
end

function smoke_identity_on_self_row()
    built = hybrid_linear_unknown_model(233)
    fp = unique_claim_fingerprint(; smoke = true)
    term = only(neural_destruction_terms(built.model))
    u0 = [0.30, 0.25]
    tspan = (0.0, 0.8)
    times, clean, _, _ = generate_from_compiled_model(
        built.model, built.packed, MersenneTwister(233);
        u0 = u0, tspan = tspan, n_points = fp.n_points, noise_σ = 0.0)
    residual = hybrid_data_residual(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        u0, tspan, times, clean)
    sciml = hybrid_residual_sciml_solve(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        u0, tspan, times, clean)
    return (;
        n_points = length(times),
        residual, sciml,
        smoke_points = fp.n_points,
        holds = length(times) == fp.n_points && residual < 1e-6 &&
                sciml < 1e-6 && fp.n_ics == 1)
end

function protocol_fingerprint_not_dropped_row()
    fp = unique_claim_fingerprint()
    ics = unique_claim_protocol_ics()
    return (;
        n_ics = fp.n_ics,
        n_points = fp.n_points,
        seed = fp.seed,
        n_table = length(ics),
        holds = fp.n_ics == 9 && fp.n_points == 50 && fp.seed == 103 &&
                length(ics) == 9 && !fp.smoke &&
                unique_claim_is_protocol())
end

# -- Failed compose paths -----------------------------------------------------

"""
    failed_compose_linear_term_row()

`compose_hybrid_rhs` requires a `NeuralDestructionTerm`. A linear
destruction term throws. `validate_network` stays open.
"""
function failed_compose_linear_term_row()
    net = build_linear_test_network()
    rng = MersenneTwister(239)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    linear = [t for t in model.compiled.destruction_terms if t isa LinearDestructionTerm]
    threw = false
    try
        compose_hybrid_rhs(model, packed, first(linear), zero_rate)
    catch
        threw = true
    end
    return (;
        n_linear = length(linear),
        n_neural = length(neural_destruction_terms(model)),
        threw,
        validate_open = validate_network(net) === net,
        holds = !isempty(linear) && isempty(neural_destruction_terms(model)) &&
                threw && validate_network(net) === net)
end

function failed_compose_empty_terms_row()
    net = build_mm_recovery_network(; known = true)
    rng = MersenneTwister(241)
    model, _ = build_ude_model(rng, net)
    terms = neural_destruction_terms(model)
    threw = false
    try
        only(terms)
    catch
        threw = true
    end
    return (;
        n_terms = length(terms),
        holes = count_unknown_destructions(net),
        threw,
        validate_open = validate_network(net) === net,
        holds = isempty(terms) && threw &&
                count_unknown_destructions(net) == 0 &&
                validate_network(net) === net)
end

function failed_compose_dual_only_row()
    net = build_dual_unknown_network()
    rng = MersenneTwister(251)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_cb = 0.9, k_c = 0.5), p0.nn)
    terms = neural_destruction_terms(model)
    only_threw = false
    try
        only(terms)
    catch
        only_threw = true
    end
    u0 = [0.22, 0.18, 0.16]
    matches = Bool[]
    for term in terms
        rhs = compose_hybrid_rhs(
            model, packed, term, neural_identity_rate(model, packed, term))
        push!(matches, ude_system(u0, packed, 0.0, model) ≈ rhs(u0, packed, 0.0))
    end
    return (;
        n_terms = length(terms),
        only_threw,
        matches,
        admits = unique_claim_recovery_admits(net),
        validate_open = validate_network(net) === net,
        holds = length(terms) == 2 && only_threw && all(matches) &&
                unique_claim_recovery_admits(net) == false &&
                validate_network(net) === net)
end

function failed_compose_export_row()
    failed = _failed_discovery(
        ArgumentError("insufficient finite trajectory samples"),
        DiscoveryConfig(backend = ExplicitSTLSQ());
        prefix = "Raw-data discovery failed")
    threw = false
    message = ""
    try
        export_rhs(failed)
    catch error
        threw = error isa ArgumentError
        message = error isa ArgumentError ? error.msg : string(error)
    end
    return (;
        retcode = failed.retcode,
        threw, message,
        holds = failed.retcode === InsufficientSamples && threw &&
                occursin("failed discovery", message) &&
                !occursin("0.99", message))
end

function hybrid_residual_failed_solve_row()
    built = hybrid_linear_unknown_model(257)
    term = only(neural_destruction_terms(built.model))
    exploding = hybrid_residual_sciml_solve(
        built.model, built.packed, term, constant_rate(1e6),
        [50.0, 50.0], (0.0, 50.0), collect(range(0.0, 50.0; length = 8)),
        ones(2, 8))
    helper = hybrid_data_residual(
        built.model, built.packed, term, constant_rate(1e6),
        [50.0, 50.0], (0.0, 50.0), collect(range(0.0, 50.0; length = 8)),
        ones(2, 8))
    return (;
        exploding, helper,
        holds = (exploding == Inf || !isfinite(exploding) || exploding > 1.0) &&
                (helper == Inf || !isfinite(helper) || helper > 1.0) &&
                !occursin("0.99", string(exploding)))
end

function hybrid_residual_shape_guard_row()
    built = hybrid_linear_unknown_model(263)
    term = only(neural_destruction_terms(built.model))
    traj = hybrid_generate(built.model, built.packed, [0.30, 0.25]; n_points = 8)
    rate = neural_identity_rate(built.model, built.packed, term)
    ok = hybrid_residual_sciml_solve(
        built.model, built.packed, term, rate,
        traj.u0, traj.tspan, traj.times, traj.data)
    bad_states = hybrid_residual_sciml_solve(
        built.model, built.packed, term, rate,
        traj.u0, traj.tspan, traj.times, ones(3, length(traj.times)))
    bad_helper = hybrid_data_residual(
        built.model, built.packed, term, rate,
        traj.u0, traj.tspan, traj.times, ones(3, length(traj.times)))
    return (;
        ok, bad_states, bad_helper,
        holds = ok < 1e-6 && bad_states == Inf && bad_helper == Inf)
end

function failed_compose_wrong_rate_row()
    built = hybrid_linear_unknown_model(269)
    term = only(neural_destruction_terms(built.model))
    u0 = [0.30, 0.25]
    identity = compose_hybrid_rhs(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term))
    shifted = compose_hybrid_rhs(
        built.model, built.packed, term, constant_rate(2.5))
    a = ude_system(u0, built.packed, 0.0, built.model)
    b = identity(u0, built.packed, 0.0)
    c = shifted(u0, built.packed, 0.0)
    return (;
        match_identity = a ≈ b,
        match_shifted = a ≈ c,
        holds = a ≈ b && !(a ≈ c) && all(isfinite, a) && all(isfinite, c))
end

function failed_compose_empty_export_row()
    empty = DiscoveryResult(
        true, "ok", nothing, nothing, nothing, ImplicitCandidate[],
        RunMetadata(), DiscoverySuccess)
    threw = false
    try
        export_rhs(empty)
    catch error
        threw = error isa ArgumentError
    end
    return (; threw, holds = threw)
end

# -- Fixture residual-vs-solver paths -----------------------------------------

function hill_residual_solver_path()
    built = hybrid_linear_unknown_model(271)
    agree = residual_solver_agreement_row(built.model, built.packed, [0.30, 0.25])
    zeroed = residual_solver_vs_zero_rate_row(built.model, built.packed, [0.30, 0.25])
    mask = residual_solver_mask_row(built.model, built.packed, [0.30, 0.25])
    irregular = residual_solver_irregular_row(built.model, built.packed, [0.30, 0.25])
    return (;
        agree, zeroed, mask, irregular,
        n_heads = neural_head_count(built.model),
        holds = agree.holds && zeroed.holds && mask.holds && irregular.holds &&
                neural_head_count(built.model) == 1)
end

function mm_residual_solver_path()
    net = build_mm_recovery_network(; known = false)
    rng = MersenneTwister(277)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), p0.nn)
    agree = residual_solver_agreement_row(model, packed, [0.30, 0.25])
    return (;
        agree,
        holes = count_unknown_destructions(net),
        holds = agree.holds && count_unknown_destructions(net) == 1)
end

function two_regulator_residual_solver_path()
    net = build_two_regulator_unknown_network()
    rng = MersenneTwister(281)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_es = 0.8, k_i = 0.5, k_e = 0.4), p0.nn)
    agree = residual_solver_agreement_row(
        model, packed, [0.25, 0.20, 0.15]; n_points = 12)
    return (;
        agree,
        n_regs = length(only(neural_destruction_terms(model)).regulators),
        holds = agree.holds &&
                length(only(neural_destruction_terms(model)).regulators) == 2)
end

function six_state_residual_solver_path()
    net = build_six_state_unknown_network(; known = false)
    rng = MersenneTwister(283)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    packed = pack_parameters(phys, p0.nn)
    u0 = [0.22, 0.18, 0.16, 0.14, 0.12, 0.10]
    agree = residual_solver_agreement_row(model, packed, u0; n_points = 10)
    return (;
        agree,
        nstates = model.compiled.nstates,
        holds = agree.holds && model.compiled.nstates == 6)
end

function default_example_residual_solver_path()
    net = DEFAULT_EXAMPLE_NETWORK
    rng = MersenneTwister(293)
    model, p0 = build_ude_model(rng, net)
    agree = residual_solver_agreement_row(model, p0, [0.20, 0.10]; n_points = 12)
    return (;
        agree,
        holes = count_unknown_destructions(net),
        dense = neural_index_is_dense(model),
        holds = agree.holds && count_unknown_destructions(net) == 1 &&
                neural_index_is_dense(model))
end

function competitive_residual_solver_path()
    net = build_competitive_test_network(; known = false)
    rng = MersenneTwister(307)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_in = 0.9, k_s = 0.8, k_i = 0.5), p0.nn)
    agree = residual_solver_agreement_row(
        model, packed, [0.25, 0.45, 0.20]; n_points = 12)
    return (;
        agree,
        holes = count_unknown_destructions(net),
        holds = agree.holds)
end

function three_state_residual_solver_path()
    net = build_three_state_unknown_network()
    rng = MersenneTwister(311)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    packed = pack_parameters(phys, p0.nn)
    n = model.compiled.nstates
    u0 = fill(0.20, n)
    agree = residual_solver_agreement_row(model, packed, u0; n_points = 10)
    return (;
        agree,
        nstates = n,
        holes = count_unknown_destructions(net),
        holds = agree.holds && count_unknown_destructions(net) == 1)
end

function remapped_residual_solver_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(313)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters(remapped_two_regulator_phys_truth(), p0.nn)
    terms = neural_destruction_terms(model)
    u0 = remapped_two_regulator_state()
    tspan = (0.0, 0.6)
    times = collect(range(0.0, 0.6; length = 10))
    pred = predict_ude(packed, Float64.(u0), tspan, times, model)
    rows = NamedTuple[]
    n = with_compile_network_counter() do counter
        for term in terms
            rate = neural_identity_rate(model, packed, term)
            hybrid = hybrid_data_residual(
                model, packed, term, rate, Float64.(u0), tspan, times, pred)
            sciml = hybrid_residual_sciml_solve(
                model, packed, term, rate, u0, tspan, times, pred)
            push!(rows,
                (;
                    nn_index = term.nn_index,
                    hybrid, sciml,
                    match = hybrid < 1e-6 && sciml < 1e-6 && hybrid ≈ sciml))
        end
        counter[]
    end
    return (;
        compiles = n,
        n_terms = length(terms),
        rows,
        dense = neural_index_is_dense(model),
        holds = n == 0 && length(terms) == 2 &&
                neural_index_is_dense(model) && all(r -> r.match, rows))
end

function skipped_duplicate_residual_solver_row()
    net = build_skipped_duplicate_unknown_network()
    rng = MersenneTwister(317)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_b = 0.5, k_c = 0.4), p0.nn)
    terms = neural_destruction_terms(model)
    u0 = [0.2, 0.3, 0.4]
    tspan = (0.0, 0.6)
    times = collect(range(0.0, 0.6; length = 8))
    pred = predict_ude(packed, u0, tspan, times, model)
    matches = Bool[]
    for term in terms
        hybrid = hybrid_data_residual(
            model, packed, term, neural_identity_rate(model, packed, term),
            u0, tspan, times, pred)
        push!(matches, hybrid < 1e-6)
    end
    return (;
        n_terms = length(terms),
        dense = neural_index_is_dense(model),
        matches,
        holds = length(terms) == 2 && neural_index_is_dense(model) &&
                all(matches))
end

function skipped_middle_residual_solver_row()
    net = build_skipped_middle_unknown_network()
    rng = MersenneTwister(331)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    packed = pack_parameters(phys, p0.nn)
    terms = neural_destruction_terms(model)
    u0 = [0.22, 0.18, 0.16, 0.14]
    tspan = (0.0, 0.6)
    times = collect(range(0.0, 0.6; length = 8))
    pred = predict_ude(packed, u0, tspan, times, model)
    matches = [hybrid_data_residual(
                   model, packed, term, neural_identity_rate(model, packed, term),
                   u0, tspan, times, pred) < 1e-6 for term in terms]
    return (;
        n_terms = length(terms),
        holes = count_unknown_destructions(net),
        dense = neural_index_is_dense(model),
        matches,
        holds = length(terms) ≥ 2 && all(matches) &&
                neural_index_is_dense(model) &&
                validate_network(net) === net)
end

function multi_ic_residual_solver_row()
    built = hybrid_linear_unknown_model(337)
    term = only(neural_destruction_terms(built.model))
    rate = neural_identity_rate(built.model, built.packed, term)
    ics = [[0.30, 0.25], [0.22, 0.18], [0.40, 0.20]]
    residuals = Float64[]
    scimls = Float64[]
    n = with_compile_network_counter() do counter
        for u0 in ics
            traj = hybrid_generate(built.model, built.packed, u0; n_points = 12)
            push!(residuals,
                hybrid_data_residual(
                    built.model, built.packed, term, rate,
                    traj.u0, traj.tspan, traj.times, traj.data))
            push!(scimls,
                hybrid_residual_sciml_solve(
                    built.model, built.packed, term, rate,
                    traj.u0, traj.tspan, traj.times, traj.data))
        end
        counter[]
    end
    return (;
        compiles = n,
        residuals, scimls,
        n_ics = length(ics),
        holds = n == 0 && length(residuals) == 3 &&
                all(<(1e-6), residuals) && all(<(1e-6), scimls) &&
                all(isfinite, residuals))
end

function hill_known_generate_unknown_solver_row()
    truth = hybrid_known_hill_truth(347)
    set = generate_experiment_set_from_compiled_model(
        truth.truth, MersenneTwister(347);
        initial_conditions = [[0.30, 0.25]],
        tspan = (0.0, 0.8), n_points = 16, noise_σ = 0.0)
    built = hybrid_linear_unknown_model(347)
    exp = first(set.experiments)
    term = only(neural_destruction_terms(built.model))
    residual = hybrid_data_residual(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        exp.u0, (first(exp.times), last(exp.times)),
        exp.times, exp.observations)
    sciml = hybrid_residual_sciml_solve(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        exp.u0, (first(exp.times), last(exp.times)),
        exp.times, exp.observations)
    return (;
        compiled_once = experiment_set_is_compiled_once(set),
        residual, sciml,
        finite = isfinite(residual) && isfinite(sciml),
        holes_train = count_unknown_destructions(built.net),
        holes_truth = count_unknown_destructions(truth.net),
        holds = experiment_set_is_compiled_once(set) &&
                isfinite(residual) && isfinite(sciml) &&
                residual ≈ sciml &&
                count_unknown_destructions(built.net) == 1 &&
                count_unknown_destructions(truth.net) == 0)
end

function session_residual_solver_path()
    built = hybrid_linear_unknown_model(349)
    row = residual_solver_session_row(built.model, built.packed, [0.30, 0.25])
    return row
end

function mm_known_no_residual_row()
    net = build_mm_recovery_network(; known = true)
    rng = MersenneTwister(353)
    model, _ = build_ude_model(rng, net)
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(neural_destruction_terms(model)),
        validate_open = validate_network(net) === net,
        holds = count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                validate_network(net) === net)
end

function repressilator_no_residual_row()
    net = build_repressilator_network()
    rng = MersenneTwister(359)
    model, _ = build_ude_model(rng, net)
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(neural_destruction_terms(model)),
        nstates = model.compiled.nstates,
        holds = count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                model.compiled.nstates == 3)
end

function kinetic_known_no_residual_row()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(367)
    model, _ = build_ude_model(rng, net)
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(neural_destruction_terms(model)),
        validate_open = validate_network(net) === net,
        holds = count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                validate_network(net) === net)
end

function linear_zero_hole_residual_row()
    net = build_linear_test_network()
    rng = MersenneTwister(373)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    terms = neural_destruction_terms(model)
    tspan = (0.0, 0.6)
    times = collect(range(0.0, 0.6; length = 8))
    u0 = [0.2, 0.1]
    model_solve = hybrid_residual_model_solve(model, packed, u0, tspan, times,
        predict_ude(packed, u0, tspan, times, model))
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(terms),
        model_solve,
        validate_open = validate_network(net) === net,
        holds = isempty(terms) && model_solve < 1e-6 &&
                count_unknown_destructions(net) == 0 &&
                validate_network(net) === net)
end

# -- Typed row / catalog ------------------------------------------------------

struct HybridResidualRow
    name::Symbol
    n_terms::Int
    hybrid::Float64
    sciml::Float64
    compiles::Int
    holds::Bool
end

function hybrid_residual_row(name::Symbol, model, packed, u0)
    terms = neural_destruction_terms(model)
    isempty(terms) && return HybridResidualRow(name, 0, NaN, NaN, 0, false)
    term = first(terms)
    tspan = (0.0, 0.6)
    times = collect(range(0.0, 0.6; length = 8))
    n = with_compile_network_counter() do counter
        pred = predict_ude(packed, Float64.(u0), tspan, times, model)
        hybrid = hybrid_data_residual(
            model, packed, term, neural_identity_rate(model, packed, term),
            Float64.(u0), tspan, times, pred)
        sciml = hybrid_residual_sciml_solve(
            model, packed, term, neural_identity_rate(model, packed, term),
            u0, tspan, times, pred)
        (counter[], hybrid, sciml)
    end
    return HybridResidualRow(name, length(terms), n[2], n[3], n[1],
        n[1] == 0 && isfinite(n[2]) && n[2] < 1e-6 && n[2] ≈ n[3])
end

function hybrid_residual_row_namedtuple(row::HybridResidualRow)
    return (;
        name = row.name,
        n_terms = row.n_terms,
        hybrid = row.hybrid,
        sciml = row.sciml,
        compiles = row.compiles,
        holds = row.holds)
end

function hybrid_residual_typed_matrix()
    hill = hybrid_linear_unknown_model(379)
    hill_row = hybrid_residual_row(:hill, hill.model, hill.packed, [0.30, 0.25])
    mm_net = build_mm_recovery_network(; known = false)
    mm_model, mm_p0 = build_ude_model(MersenneTwister(383), mm_net)
    mm_packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), mm_p0.nn)
    mm_row = hybrid_residual_row(:mm, mm_model, mm_packed, [0.30, 0.25])
    return (;
        hill = hybrid_residual_row_namedtuple(hill_row),
        mm = hybrid_residual_row_namedtuple(mm_row),
        holds = hill_row.holds && mm_row.holds)
end

# -- Source locks -------------------------------------------------------------

function hybrid_data_residual_uses_sciml_solve_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function hybrid_data_residual", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("compose_hybrid_rhs", body) &&
           occursin("SciMLBase.ODEProblem", body) &&
           occursin("Tsit5()", body) &&
           occursin("sensealg = nothing", body) &&
           occursin("sqrt(mean(abs2", body)
end

function hybrid_residual_sciml_solve_source_holds()
    src = read(hybrid_residual_source_path(), String)
    start = findfirst(
        "function hybrid_residual_sciml_solve(model, p, term, rate_fn, u0, tspan, times, data)",
        src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("compose_hybrid_rhs", body) &&
           occursin("SciMLBase.ODEProblem", body) &&
           occursin("Tsit5()", body) &&
           !occursin("Rodas5", body)
end

function hybrid_residual_model_solve_source_holds()
    src = read(hybrid_residual_source_path(), String)
    start = findfirst(
        "function hybrid_residual_model_solve(model::UDEModel, p, u0, tspan, times, data)",
        src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("SciMLBase.ODEProblem(model", body) &&
           occursin("Tsit5()", body)
end

function predict_ude_uses_odeproblem_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Training.jl"), String)
    start = findfirst("function predict_ude(p, u0, tspan, saveat, nn, st;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("SciMLBase.ODEProblem", body) &&
           occursin("solver_config.algorithm", body)
end

# -- Matrices -----------------------------------------------------------------

function hybrid_residual_identity_matrix()
    hill = hill_residual_solver_path()
    mm = mm_residual_solver_path()
    two = two_regulator_residual_solver_path()
    six = six_state_residual_solver_path()
    default = default_example_residual_solver_path()
    comp = competitive_residual_solver_path()
    three = three_state_residual_solver_path()
    return (;
        hill, mm, two, six, default, comp, three,
        holds = hill.holds && mm.holds && two.holds && six.holds &&
                default.holds && comp.holds && three.holds)
end

function hybrid_residual_honesty_matrix()
    linear = failed_compose_linear_term_row()
    empty = failed_compose_empty_terms_row()
    dual = failed_compose_dual_only_row()
    export_failed = failed_compose_export_row()
    empty_export = failed_compose_empty_export_row()
    exploding = hybrid_residual_failed_solve_row()
    shape = hybrid_residual_shape_guard_row()
    wrong = failed_compose_wrong_rate_row()
    remap = remapped_residual_solver_row()
    skipped = skipped_duplicate_residual_solver_row()
    middle = skipped_middle_residual_solver_row()
    multi = multi_ic_residual_solver_row()
    known = hill_known_generate_unknown_solver_row()
    session = session_residual_solver_path()
    mm_known = mm_known_no_residual_row()
    repress = repressilator_no_residual_row()
    kinetic = kinetic_known_no_residual_row()
    zero = linear_zero_hole_residual_row()
    noise = noise_does_not_paint_f1_row()
    grid = begin
        built = hybrid_linear_unknown_model(389)
        noise_grid_residual_row(built.model, built.packed, [0.30, 0.25])
    end
    smoke = smoke_vs_protocol_residual_row()
    smoke_self = smoke_identity_on_self_row()
    protocol = protocol_fingerprint_not_dropped_row()
    typed = hybrid_residual_typed_matrix()
    return (;
        linear, empty, dual, export_failed, empty_export, exploding, shape,
        wrong, remap, skipped, middle, multi, known, session, mm_known,
        repress, kinetic, zero, noise, grid, smoke, smoke_self, protocol, typed,
        holds = linear.holds && empty.holds && dual.holds &&
                export_failed.holds && empty_export.holds && exploding.holds &&
                shape.holds && wrong.holds && remap.holds && skipped.holds &&
                middle.holds && multi.holds && known.holds && session.holds &&
                mm_known.holds && repress.holds && kinetic.holds && zero.holds &&
                noise.holds && grid.holds && smoke.holds && smoke_self.holds &&
                protocol.holds && typed.holds)
end

function hybrid_residual_fixture_matrix()
    identity = hybrid_residual_identity_matrix()
    honesty = hybrid_residual_honesty_matrix()
    return (;
        identity, honesty,
        holds = identity.holds && honesty.holds)
end

function hybrid_residual_fixture_names()
    return (
        :hill_solver, :mm_solver, :two_regulator, :six_state,
        :default_example, :competitive, :three_state, :linear_term,
        :empty_terms, :dual_only, :failed_export, :empty_export,
        :exploding, :shape, :wrong_rate, :remapped, :skipped_duplicate,
        :skipped_middle, :multi_ic, :known_generate, :session,
        :mm_known, :repressilator, :kinetic, :linear_zero, :noise,
        :noise_grid, :smoke_protocol, :smoke_self, :protocol_fp)
end

function format_hybrid_residual_index()
    io = IOBuffer()
    println(io, "| fixture | role |")
    println(io, "|---|---|")
    println(io, "| hill_solver | hybrid residual matches SciMLBase.solve |")
    println(io, "| mm_solver | MM unknown residual versus solver |")
    println(io, "| two_regulator | D(S,I) residual versus solver |")
    println(io, "| six_state | six-state residual versus solver |")
    println(io, "| default_example | p53/Mdm2 residual versus solver |")
    println(io, "| competitive | competitive unknown residual versus solver |")
    println(io, "| three_state | three-state residual versus solver |")
    println(io, "| linear_term | compose rejects a LinearDestructionTerm |")
    println(io, "| empty_terms | known MM has no neural compose term |")
    println(io, "| dual_only | only() throws; each head still composes |")
    println(io, "| failed_export | export_rhs rejects InsufficientSamples |")
    println(io, "| empty_export | export_rhs rejects empty candidates |")
    println(io, "| exploding | a huge rate is Inf-or-large, not 0.99 F1 |")
    println(io, "| shape | mismatched observation width is Inf |")
    println(io, "| wrong_rate | a constant rate does not recover ude_system |")
    println(io, "| remapped | each remapped head residual matches solve |")
    println(io, "| skipped_duplicate | two dense heads residual-match |")
    println(io, "| skipped_middle | remapped 1:n heads residual-match |")
    println(io, "| multi_ic | three-IC identity residual versus solver |")
    println(io, "| known_generate | known Hill generate, unknown residual |")
    println(io, "| session | TrainingSolveSession matches residual |")
    println(io, "| mm_known | known MM has no residual term |")
    println(io, "| repressilator | known three-state has no residual term |")
    println(io, "| kinetic | known kinetic network has no residual term |")
    println(io, "| linear_zero | 0-hole model solve residual is ~0 |")
    println(io, "| noise | noise-0 residual ~0; noisy residual is larger |")
    println(io, "| noise_grid | residual grows with σ; F1 floor stays 0.50 |")
    println(io, "| smoke_protocol | 1 IC / 8 points is not 9 ICs / 50 points |")
    println(io, "| smoke_self | smoke-width identity residual versus self |")
    println(io, "| protocol_fp | protocol fingerprint keeps 9 ICs / 50 points |")
    return String(take!(io))
end

function hybrid_residual_index_holds()
    text = format_hybrid_residual_index()
    names = hybrid_residual_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("SciMLBase.solve", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Source checks ----------------------------------------------------------
