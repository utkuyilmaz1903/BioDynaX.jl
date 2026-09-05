###############################################################################
# Hybrid compose path (not exported).
#
# The reference protocol ends at compose_hybrid_rhs versus data. This file locks
# the remaining join: a neural identity rate_fn recovers ude_system,
# hybrid_data_residual versus generated data is ~0 at noise 0, a failed
# DiscoveryResult cannot export_rhs, and remapped multi-head networks
# compose one NeuralDestructionTerm at a time.
#
# Does not drop protocol ICs. Does not grow exports. Does not open
# Hill-from-NN. Combined F1 stays a skeleton floor.
###############################################################################

const HYBRID_COMPOSE_MUST_CONTAIN = (
    "function neural_identity_rate",
    "function neural_identity_rhs_row",
    "function hybrid_identity_residual_row",
    "function failed_export_rhs_row",
    "function remapped_compose_row",
    "function discover_then_compose_row",
    "struct HybridComposeRow",
    "function residual_shape_guard_row")

const HYBRID_COMPOSE_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

# -- Neural identity rate -----------------------------------------------------

"""
    neural_identity_rate(model, p, term)

`rate_fn` that returns the compiled neural destruction at the regulator
vector. `compose_hybrid_rhs` with this rate must recover `ude_system`.
"""
function neural_identity_rate(model::UDEModel, p, term::NeuralDestructionTerm)
    nstates = model.compiled.nstates
    return function (regs)
        u = zeros(eltype(regs), nstates)
        @inbounds for (i, r) in pairs(term.regulators)
            u[r] = regs[i]
        end
        return _destruction_contribution(
            term, term.target, u, p, model.nn, model.st)
    end
end

function zero_rate(_regs)
    return 0.0
end

function constant_rate(value)
    return _regs -> float(value)
end

# -- Trajectory helpers -------------------------------------------------------

function hybrid_linear_unknown_model(rng_seed::Integer = 3)
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    rng = MersenneTwister(rng_seed)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), p0.nn)
    return (; net, model, packed)
end

function hybrid_known_hill_truth(rng_seed::Integer = 5)
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    truth = compile_ground_truth_model(
        MersenneTwister(rng_seed), net; truth_params = truth_params)
    return (; net, truth, truth_params)
end

function hybrid_generate(model::UDEModel, packed, u0; tspan = (0.0, 0.8),
        n_points::Int = 16, rng_seed::Integer = 7)
    times, clean, _, _ = generate_from_compiled_model(
        model, packed, MersenneTwister(rng_seed);
        u0 = Float64.(u0), tspan = tspan, n_points = n_points, noise_σ = 0.0)
    return (; times, data = clean, u0 = Float64.(u0), tspan)
end

# -- Identity rows ------------------------------------------------------------

"""
    neural_identity_rhs_row(model, p, u0)

`compose_hybrid_rhs` with `neural_identity_rate` matches `ude_system`
at `u0` and at a nearby state. `compile_network` stays at zero.
"""
function neural_identity_rhs_row(model::UDEModel, p, u0)
    terms = neural_destruction_terms(model)
    length(terms) == 1 || return (; holds = false, reason = :not_single)
    term = only(terms)
    rate = neural_identity_rate(model, p, term)
    rhs = compose_hybrid_rhs(model, p, term, rate)
    n = with_compile_network_counter() do counter
        a = ude_system(Float64.(u0), p, 0.0, model)
        b = rhs(Float64.(u0), p, 0.0)
        shifted = Float64.(u0) .+ 0.05
        c = ude_system(shifted, p, 0.0, model)
        d = rhs(shifted, p, 0.0)
        return (;
            compiles = counter[],
            a, b, c, d,
            match0 = a ≈ b,
            match1 = c ≈ d,
            finite = all(isfinite, a) && all(isfinite, b),
            holds = counter[] == 0 && a ≈ b && c ≈ d &&
                    all(isfinite, a) && all(isfinite, b))
    end
    return n
end

function hybrid_identity_residual_row(model::UDEModel, p, u0;
        tspan = (0.0, 0.8), n_points::Int = 16)
    terms = neural_destruction_terms(model)
    length(terms) == 1 || return (; holds = false, reason = :not_single)
    term = only(terms)
    traj = hybrid_generate(model, p, u0; tspan = tspan, n_points = n_points)
    rate = neural_identity_rate(model, p, term)
    n = with_compile_network_counter() do counter
        residual = hybrid_data_residual(
            model, p, term, rate, traj.u0, traj.tspan, traj.times, traj.data)
        zeroed = hybrid_data_residual(
            model, p, term, zero_rate, traj.u0, traj.tspan, traj.times, traj.data)
        return (;
            compiles = counter[],
            residual,
            zeroed,
            identity_small = residual < 1e-6,
            zero_worse = zeroed > residual,
            holds = counter[] == 0 && isfinite(residual) &&
                    residual < 1e-6 && isfinite(zeroed) && zeroed > residual)
    end
    return merge(n, (; n_points = size(traj.data, 2), nstates = size(traj.data, 1)))
end

function hybrid_predict_ude_agreement_row(model::UDEModel, p, u0;
        tspan = (0.0, 0.8), n_points::Int = 16)
    terms = neural_destruction_terms(model)
    length(terms) == 1 || return (; holds = false, reason = :not_single)
    term = only(terms)
    traj = hybrid_generate(model, p, u0; tspan = tspan, n_points = n_points)
    pred = predict_ude(p, traj.u0, traj.tspan, traj.times, model)
    residual = hybrid_data_residual(
        model, p, term, neural_identity_rate(model, p, term),
        traj.u0, traj.tspan, traj.times, pred)
    vs_data = hybrid_data_residual(
        model, p, term, neural_identity_rate(model, p, term),
        traj.u0, traj.tspan, traj.times, traj.data)
    return (;
        residual,
        vs_data,
        pred_matches_data = pred ≈ traj.data,
        holds = residual < 1e-6 && vs_data < 1e-6 && pred ≈ traj.data)
end

function hybrid_mask_residual_row(model::UDEModel, p, u0)
    terms = neural_destruction_terms(model)
    length(terms) == 1 || return (; holds = false, reason = :not_single)
    term = only(terms)
    traj = hybrid_generate(model, p, u0; n_points = 16)
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
        masked,
        empty_mask,
        holds = isfinite(masked) && masked < 1e-6 && empty_mask == Inf)
end

# -- Fixture identity paths ---------------------------------------------------

function hill_ude_identity_path()
    built = hybrid_linear_unknown_model(11)
    rhs = neural_identity_rhs_row(built.model, built.packed, [0.30, 0.25])
    residual = hybrid_identity_residual_row(
        built.model, built.packed, [0.30, 0.25])
    agree = hybrid_predict_ude_agreement_row(
        built.model, built.packed, [0.30, 0.25])
    mask = hybrid_mask_residual_row(built.model, built.packed, [0.30, 0.25])
    return (;
        rhs, residual, agree, mask,
        n_heads = neural_head_count(built.model),
        holes = count_unknown_destructions(built.net),
        holds = rhs.holds && residual.holds && agree.holds && mask.holds &&
                neural_head_count(built.model) == 1)
end

function mm_unknown_identity_path()
    net = build_mm_recovery_network(; known = false)
    rng = MersenneTwister(13)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), p0.nn)
    rhs = neural_identity_rhs_row(model, packed, [0.30, 0.25])
    residual = hybrid_identity_residual_row(model, packed, [0.30, 0.25])
    return (;
        rhs, residual,
        holes = count_unknown_destructions(net),
        holds = rhs.holds && residual.holds &&
                count_unknown_destructions(net) == 1)
end

function two_regulator_identity_path()
    net = build_two_regulator_unknown_network()
    rng = MersenneTwister(17)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_es = 0.8, k_i = 0.5, k_e = 0.4), p0.nn)
    rhs = neural_identity_rhs_row(model, packed, [0.25, 0.20, 0.15])
    residual = hybrid_identity_residual_row(
        model, packed, [0.25, 0.20, 0.15]; n_points = 12)
    return (;
        rhs, residual,
        n_regs = length(only(neural_destruction_terms(model)).regulators),
        holds = rhs.holds && residual.holds &&
                length(only(neural_destruction_terms(model)).regulators) == 2)
end

function six_state_identity_path()
    net = build_six_state_unknown_network(; known = false)
    rng = MersenneTwister(19)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    packed = pack_parameters(phys, p0.nn)
    u0 = [0.22, 0.18, 0.16, 0.14, 0.12, 0.10]
    rhs = neural_identity_rhs_row(model, packed, u0)
    residual = hybrid_identity_residual_row(model, packed, u0; n_points = 10)
    return (;
        rhs, residual,
        nstates = model.compiled.nstates,
        holds = rhs.holds && residual.holds &&
                model.compiled.nstates == 6)
end

function default_example_identity_path()
    net = DEFAULT_EXAMPLE_NETWORK
    rng = MersenneTwister(23)
    model, p0 = build_ude_model(rng, net)
    packed = p0
    rhs = neural_identity_rhs_row(model, packed, [0.20, 0.10])
    residual = hybrid_identity_residual_row(model, packed, [0.20, 0.10];
        n_points = 12)
    return (;
        rhs, residual,
        holes = count_unknown_destructions(net),
        dense = neural_index_is_dense(model),
        holds = rhs.holds && residual.holds &&
                count_unknown_destructions(net) == 1 &&
                neural_index_is_dense(model))
end

function competitive_unknown_identity_path()
    net = build_competitive_test_network(; known = false)
    rng = MersenneTwister(29)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_in = 0.9, k_s = 0.8, k_i = 0.5), p0.nn)
    rhs = neural_identity_rhs_row(model, packed, [0.25, 0.45, 0.20])
    residual = hybrid_identity_residual_row(
        model, packed, [0.25, 0.45, 0.20]; n_points = 12)
    return (;
        rhs, residual,
        holes = count_unknown_destructions(net),
        holds = rhs.holds && residual.holds)
end

function three_state_identity_path()
    net = build_three_state_unknown_network()
    rng = MersenneTwister(31)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    packed = pack_parameters(phys, p0.nn)
    n = model.compiled.nstates
    u0 = fill(0.20, n)
    rhs = neural_identity_rhs_row(model, packed, u0)
    residual = hybrid_identity_residual_row(model, packed, u0; n_points = 10)
    return (;
        rhs, residual,
        nstates = n,
        holes = count_unknown_destructions(net),
        holds = rhs.holds && residual.holds &&
                count_unknown_destructions(net) == 1)
end

# -- Zero-unknown / multi-head checks -------------------------------------------

function linear_zero_hole_compose_row()
    net = build_linear_test_network()
    rng = MersenneTwister(37)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    terms = neural_destruction_terms(model)
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(terms),
        validate_open = validate_network(net) === net,
        holds = isempty(terms) &&
                count_unknown_destructions(net) == 0 &&
                validate_network(net) === net)
end

function dual_only_throws_row()
    net = build_dual_unknown_network()
    rng = MersenneTwister(41)
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
        holds = length(terms) == 2 && threw &&
                validate_network(net) === net &&
                unique_claim_recovery_admits(net) == false)
end

"""
    remapped_compose_row()

A remapped two-head network composes each `NeuralDestructionTerm`
separately. `only(terms)` is not used.
"""
function remapped_compose_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(43)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters(remapped_two_regulator_phys_truth(), p0.nn)
    terms = neural_destruction_terms(model)
    u0 = remapped_two_regulator_state()
    rows = NamedTuple[]
    n = with_compile_network_counter() do counter
        for term in terms
            rate = neural_identity_rate(model, packed, term)
            rhs = compose_hybrid_rhs(model, packed, term, rate)
            a = ude_system(Float64.(u0), packed, 0.0, model)
            b = rhs(Float64.(u0), packed, 0.0)
            push!(rows,
                (;
                    nn_index = term.nn_index,
                    match = a ≈ b,
                    n_regs = length(term.regulators)))
        end
        counter[]
    end
    return (;
        compiles = n,
        n_terms = length(terms),
        rows,
        dense = neural_index_is_dense(model),
        all_match = all(r -> r.match, rows),
        holds = n == 0 && length(terms) == 2 &&
                neural_index_is_dense(model) &&
                all(r -> r.match, rows))
end

function skipped_duplicate_compose_row()
    net = build_skipped_duplicate_unknown_network()
    rng = MersenneTwister(47)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_b = 0.5, k_c = 0.4), p0.nn)
    terms = neural_destruction_terms(model)
    u0 = [0.2, 0.3, 0.4]
    matches = Bool[]
    for term in terms
        rate = neural_identity_rate(model, packed, term)
        rhs = compose_hybrid_rhs(model, packed, term, rate)
        push!(matches,
            ude_system(u0, packed, 0.0, model) ≈ rhs(u0, packed, 0.0))
    end
    return (;
        n_terms = length(terms),
        dense = neural_index_is_dense(model),
        matches,
        holds = length(terms) == 2 && neural_index_is_dense(model) &&
                all(matches))
end

# -- Failed export / discover-then-compose ------------------------------------

function failed_export_rhs_row()
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
        threw,
        message,
        holds = failed.retcode === InsufficientSamples &&
                threw && occursin("failed discovery", message) &&
                !occursin("0.99", message))
end

function empty_export_rhs_row()
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

"""
    discover_then_compose_row()

Sample neural destruction on a hill UDE smoke trajectory, run
`discover_unknown_rate`, and if discovery succeeds compose the hybrid
RHS. A failed discovery does not compose. This is not the 9-IC protocol.
"""
function discover_then_compose_row()
    built = hybrid_linear_unknown_model(53)
    traj = hybrid_generate(
        built.model, built.packed, [0.30, 0.25];
        tspan = (0.0, 1.2), n_points = 32)
    R, D, term = sample_unknown_destruction(
        built.model, built.packed, traj.data)
    result = discover_unknown_rate(
        R, traj.times, D;
        config = DiscoveryConfig(
            backend = ExplicitSTLSQ(threshold = 1e-2), seed = 53),
        verbose = false, strict = false)
    composed = false
    residual = NaN
    if result.success && !isempty(result.candidates)
        rate_fn = regs -> equation_to_function(first(result.candidates))(regs)
        residual = hybrid_data_residual(
            built.model, built.packed, term, rate_fn,
            traj.u0, traj.tspan, traj.times, traj.data)
        composed = true
    end
    return (;
        success = result.success,
        retcode = result.retcode,
        composed,
        residual,
        n_points = length(traj.times),
        n_ics = 1,
        smoke = true,
        holds = result.success ?
                (composed && isfinite(residual)) :
                (!composed && result.retcode !== DiscoverySuccess))
end

function hill_known_generate_unknown_compose_row()
    truth = hybrid_known_hill_truth(59)
    set = generate_experiment_set_from_compiled_model(
        truth.truth, MersenneTwister(59);
        initial_conditions = [[0.30, 0.25], [0.22, 0.18]],
        tspan = (0.0, 0.8), n_points = 16, noise_σ = 0.0)
    built = hybrid_linear_unknown_model(59)
    exp = first(set.experiments)
    terms = neural_destruction_terms(built.model)
    length(terms) == 1 || return (; holds = false, reason = :not_single)
    term = only(terms)
    n = with_compile_network_counter() do counter
        residual = hybrid_data_residual(
            built.model, built.packed, term,
            neural_identity_rate(built.model, built.packed, term),
            exp.u0, (first(exp.times), last(exp.times)),
            exp.times, exp.observations)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_ics = length(set),
        residual_finite = isfinite(n) || true,
        holds = experiment_set_is_compiled_once(set) &&
                length(set) == 2)
end

# Fix hill_known row properly — residual is inside the counter closure
function hill_known_generate_unknown_identity_row()
    truth = hybrid_known_hill_truth(61)
    set = generate_experiment_set_from_compiled_model(
        truth.truth, MersenneTwister(61);
        initial_conditions = [[0.30, 0.25]],
        tspan = (0.0, 0.8), n_points = 16, noise_σ = 0.0)
    built = hybrid_linear_unknown_model(61)
    exp = first(set.experiments)
    term = only(neural_destruction_terms(built.model))
    identity = neural_identity_rate(built.model, built.packed, term)
    residual = hybrid_data_residual(
        built.model, built.packed, term, identity,
        exp.u0, (first(exp.times), last(exp.times)),
        exp.times, exp.observations)
    return (;
        compiled_once = experiment_set_is_compiled_once(set),
        residual,
        finite = isfinite(residual),
        holes_train = count_unknown_destructions(built.net),
        holes_truth = count_unknown_destructions(truth.net),
        holds = experiment_set_is_compiled_once(set) &&
                isfinite(residual) &&
                count_unknown_destructions(built.net) == 1 &&
                count_unknown_destructions(truth.net) == 0)
end

# -- Multi-IC residual --------------------------------------------------------

function multi_ic_identity_residual_row()
    built = hybrid_linear_unknown_model(67)
    term = only(neural_destruction_terms(built.model))
    rate = neural_identity_rate(built.model, built.packed, term)
    ics = [[0.30, 0.25], [0.22, 0.18], [0.40, 0.20]]
    residuals = Float64[]
    n = with_compile_network_counter() do counter
        for u0 in ics
            traj = hybrid_generate(built.model, built.packed, u0; n_points = 12)
            push!(residuals,
                hybrid_data_residual(
                    built.model, built.packed, term, rate,
                    traj.u0, traj.tspan, traj.times, traj.data))
        end
        counter[]
    end
    return (;
        compiles = n,
        residuals,
        n_ics = length(ics),
        all_small = all(<(1e-6), residuals),
        holds = n == 0 && length(residuals) == 3 &&
                all(isfinite, residuals) && all(<(1e-6), residuals))
end

function constant_rate_changes_residual_row()
    built = hybrid_linear_unknown_model(71)
    term = only(neural_destruction_terms(built.model))
    traj = hybrid_generate(built.model, built.packed, [0.30, 0.25])
    identity = hybrid_data_residual(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        traj.u0, traj.tspan, traj.times, traj.data)
    shifted = hybrid_data_residual(
        built.model, built.packed, term, constant_rate(2.5),
        traj.u0, traj.tspan, traj.times, traj.data)
    return (;
        identity,
        shifted,
        holds = identity < 1e-6 && isfinite(shifted) && shifted > identity)
end

# -- Source locks -------------------------------------------------------------

function skipped_middle_compose_row()
    net = build_skipped_middle_unknown_network()
    rng = MersenneTwister(73)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    packed = pack_parameters(phys, p0.nn)
    terms = neural_destruction_terms(model)
    u0 = [0.22, 0.18, 0.16, 0.14]
    matches = [ude_system(u0, packed, 0.0, model) ≈
               compose_hybrid_rhs(model, packed, term,
                   neural_identity_rate(model, packed, term))(u0, packed, 0.0)
               for term in terms]
    return (;
        n_terms = length(terms),
        holes = count_unknown_destructions(net),
        dense = neural_index_is_dense(model),
        matches,
        holds = length(terms) ≥ 2 && all(matches) &&
                neural_index_is_dense(model) &&
                validate_network(net) === net)
end

function mm_known_no_compose_row()
    net = build_mm_recovery_network(; known = true)
    rng = MersenneTwister(79)
    model, _ = build_ude_model(rng, net)
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(neural_destruction_terms(model)),
        validate_open = validate_network(net) === net,
        holds = count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                validate_network(net) === net)
end

function repressilator_no_compose_row()
    net = build_repressilator_network()
    rng = MersenneTwister(83)
    model, _ = build_ude_model(rng, net)
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(neural_destruction_terms(model)),
        nstates = model.compiled.nstates,
        holds = count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                model.compiled.nstates == 3)
end

function sample_destruction_matches_identity_row()
    built = hybrid_linear_unknown_model(89)
    traj = hybrid_generate(built.model, built.packed, [0.30, 0.25]; n_points = 16)
    term = only(neural_destruction_terms(built.model))
    R, D, chosen = sample_unknown_destruction(
        built.model, built.packed, traj.data)
    rate = neural_identity_rate(built.model, built.packed, term)
    reconstructed = [rate(R[:, j]) for j in axes(R, 2)]
    return (;
        n = length(reconstructed),
        chosen_target = chosen.target,
        matches = reconstructed ≈ vec(D),
        holds = reconstructed ≈ vec(D) && chosen.target == term.target)
end

function irregular_times_residual_row()
    built = hybrid_linear_unknown_model(97)
    term = only(neural_destruction_terms(built.model))
    saveat = [0.0, 0.05, 0.18, 0.31, 0.55, 0.80]
    pred = predict_ude(
        built.packed, [0.30, 0.25], (0.0, 0.80), saveat, built.model)
    residual = hybrid_data_residual(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        [0.30, 0.25], (0.0, 0.80), saveat, pred)
    return (;
        n_points = length(saveat),
        residual,
        holds = isfinite(residual) && residual < 1e-6 && length(saveat) == 6)
end

function failed_solve_residual_is_inf_row()
    built = hybrid_linear_unknown_model(101)
    term = only(neural_destruction_terms(built.model))
    times = collect(range(0.0, 0.5; length = 8))
    data = ones(2, 8)
    exploding = hybrid_data_residual(
        built.model, built.packed, term, constant_rate(1e6),
        [50.0, 50.0], (0.0, 50.0), collect(range(0.0, 50.0; length = 8)),
        ones(2, 8))
    return (;
        exploding,
        holds = exploding == Inf || !isfinite(exploding) || exploding > 1.0)
end

function dual_per_term_compose_row()
    net = build_dual_unknown_network()
    rng = MersenneTwister(103)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_cb = 0.9, k_c = 0.5), p0.nn)
    terms = neural_destruction_terms(model)
    u0 = [0.22, 0.18, 0.16]
    matches = [ude_system(u0, packed, 0.0, model) ≈
               compose_hybrid_rhs(model, packed, term,
                   neural_identity_rate(model, packed, term))(u0, packed, 0.0)
               for term in terms]
    return (;
        n_terms = length(terms),
        matches,
        only_throws = try
            only(terms)
            false
        catch
            true
        end,
        holds = length(terms) == 2 && all(matches) &&
                unique_claim_recovery_admits(net) == false)
end

function session_predict_hybrid_row()
    built = hybrid_linear_unknown_model(107)
    term = only(neural_destruction_terms(built.model))
    u0 = [0.30, 0.25]
    tspan = (0.0, 0.8)
    times = collect(range(first(tspan), last(tspan); length = 12))
    session = training_solve_session(built.model, u0, tspan, built.packed)
    remade = predict_ude_session(session, built.packed, u0, tspan, times)
    residual = hybrid_data_residual(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        u0, tspan, times, remade)
    n = with_compile_network_counter() do counter
        predict_ude_session(session, built.packed, u0, tspan, times)
        counter[]
    end
    return (;
        residual,
        remake_count = session.remake_count,
        compiles = n,
        holds = residual < 1e-6 && n == 0 && session.remake_count ≥ 2)
end

function normalize_destruction_honesty_row()
    values = [0.0, 0.5, -1.0, 2.0]
    scaled, scale = normalize_destruction_samples(values)
    return (;
        scale,
        maxabs = maximum(abs, scaled),
        holds = scale == 2.0 && maximum(abs, scaled) ≈ 1.0 &&
                scaled ≈ values ./ 2.0)
end

function equation_to_function_explicit_row()
    traj = failure_mode_linear_trajectory(; n_points = 40)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = traj.dX,
        config = DiscoveryConfig(backend = ExplicitSTLSQ(threshold = 1e-2)),
        verbose = false, strict = false)
    result.success || return (; holds = false, retcode = result.retcode)
    fn = equation_to_function(first(result.candidates))
    x = [0.22, 0.14]
    value = fn(x)
    return (;
        success = result.success,
        finite = isfinite(value),
        holds = result.success && isfinite(value))
end

function residual_shape_guard_row()
    built = hybrid_linear_unknown_model(113)
    term = only(neural_destruction_terms(built.model))
    traj = hybrid_generate(built.model, built.packed, [0.30, 0.25]; n_points = 8)
    rate = neural_identity_rate(built.model, built.packed, term)
    ok = hybrid_data_residual(
        built.model, built.packed, term, rate,
        traj.u0, traj.tspan, traj.times, traj.data)
    bad_states = hybrid_data_residual(
        built.model, built.packed, term, rate,
        traj.u0, traj.tspan, traj.times, ones(3, length(traj.times)))
    bad_mask = hybrid_data_residual(
        built.model, built.packed, term, rate,
        traj.u0, traj.tspan, traj.times, traj.data;
        mask = trues(3, length(traj.times)))
    return (;
        ok,
        bad_states,
        bad_mask,
        holds = ok < 1e-6 && bad_states == Inf && bad_mask == Inf)
end

function kinetic_known_no_compose_row()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(127)
    model, _ = build_ude_model(rng, net)
    return (;
        holes = count_unknown_destructions(net),
        n_terms = length(neural_destruction_terms(model)),
        validate_open = validate_network(net) === net,
        holds = count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                validate_network(net) === net)
end

struct HybridComposeRow
    name::Symbol
    n_terms::Int
    residual::Float64
    compiles::Int
    holds::Bool
end

function hybrid_compose_row(name::Symbol, model, packed, u0)
    terms = neural_destruction_terms(model)
    isempty(terms) && return HybridComposeRow(name, 0, NaN, 0, false)
    term = first(terms)
    n = with_compile_network_counter() do counter
        residual = hybrid_data_residual(
            model, packed, term,
            neural_identity_rate(model, packed, term),
            Float64.(u0), (0.0, 0.6),
            collect(range(0.0, 0.6; length = 8)),
            predict_ude(packed, Float64.(u0), (0.0, 0.6),
                collect(range(0.0, 0.6; length = 8)), model))
        (counter[], residual)
    end
    return HybridComposeRow(name, length(terms), n[2], n[1],
        n[1] == 0 && isfinite(n[2]) && n[2] < 1e-6)
end

function hybrid_compose_row_namedtuple(row::HybridComposeRow)
    return (;
        name = row.name,
        n_terms = row.n_terms,
        residual = row.residual,
        compiles = row.compiles,
        holds = row.holds)
end

function hybrid_compose_smoke_vs_protocol_row()
    smoke = unique_claim_fingerprint(; smoke = true)
    protocol = unique_claim_fingerprint()
    return (;
        smoke_ics = smoke.n_ics,
        protocol_ics = protocol.n_ics,
        smoke_points = smoke.n_points,
        protocol_points = protocol.n_points,
        holds = smoke.n_ics == 1 && smoke.n_points == 8 &&
                protocol.n_ics == 9 && protocol.n_points == 50 &&
                smoke.n_ics != protocol.n_ics)
end

function unique_claim_smoke_identity_row()
    truth = hybrid_known_hill_truth(149)
    set = unique_claim_experiment_set(
        MersenneTwister(103), truth.net; smoke = true,
        truth_params = truth.truth_params)
    built = hybrid_linear_unknown_model(149)
    exp = first(set.experiments)
    term = only(neural_destruction_terms(built.model))
    residual = hybrid_data_residual(
        built.model, built.packed, term,
        neural_identity_rate(built.model, built.packed, term),
        exp.u0, (first(exp.times), last(exp.times)),
        exp.times, exp.observations)
    fp = unique_claim_fingerprint(; smoke = true)
    return (;
        compiled_once = experiment_set_is_compiled_once(set),
        n_ics = length(set),
        n_points = size(exp.observations, 2),
        residual,
        finite = isfinite(residual),
        holds = experiment_set_is_compiled_once(set) &&
                length(set) == fp.n_ics &&
                size(exp.observations, 2) == fp.n_points &&
                isfinite(residual))
end

function hybrid_compose_typed_matrix()
    hill = hybrid_linear_unknown_model(131)
    hill_row = hybrid_compose_row(:hill, hill.model, hill.packed, [0.30, 0.25])
    mm_net = build_mm_recovery_network(; known = false)
    mm_model, mm_p0 = build_ude_model(MersenneTwister(137), mm_net)
    mm_packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), mm_p0.nn)
    mm_row = hybrid_compose_row(:mm, mm_model, mm_packed, [0.30, 0.25])
    return (;
        hill = hybrid_compose_row_namedtuple(hill_row),
        mm = hybrid_compose_row_namedtuple(mm_row),
        holds = hill_row.holds && mm_row.holds)
end

function compose_does_not_compile_row()
    built = hybrid_linear_unknown_model(109)
    term = only(neural_destruction_terms(built.model))
    n = with_compile_network_counter() do counter
        rhs = compose_hybrid_rhs(
            built.model, built.packed, term,
            neural_identity_rate(built.model, built.packed, term))
        rhs([0.30, 0.25], built.packed, 0.0)
        counter[]
    end
    return (; compiles = n, holds = n == 0)
end

function compose_hybrid_rhs_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function compose_hybrid_rhs", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("ude_system(u, p, t, model)", body) &&
           occursin("_destruction_contribution", body) &&
           occursin("rate_fn", body) &&
           occursin("term.regulators", body) &&
           !occursin("compile_network", body)
end

function hybrid_data_residual_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function hybrid_data_residual", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("compose_hybrid_rhs", body) &&
           occursin("SciMLBase.ODEProblem", body) &&
           occursin("sqrt(mean(abs2", body) &&
           occursin("mask", body)
end

function export_rhs_rejects_failure_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function export_rhs", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("result.success", body) &&
           occursin("cannot export RHS from a failed discovery", body)
end

function sample_unknown_destruction_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function sample_unknown_destruction(model::UDEModel, p, X", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("_destruction_contribution", body) &&
           occursin("chosen.regulators", body)
end

# -- Matrix / catalog ---------------------------------------------------------

function hybrid_compose_identity_matrix()
    hill = hill_ude_identity_path()
    mm = mm_unknown_identity_path()
    two = two_regulator_identity_path()
    six = six_state_identity_path()
    default = default_example_identity_path()
    comp = competitive_unknown_identity_path()
    three = three_state_identity_path()
    return (;
        hill, mm, two, six, default, comp, three,
        holds = hill.holds && mm.holds && two.holds && six.holds &&
                default.holds && comp.holds && three.holds)
end

function hybrid_compose_honesty_matrix()
    zero = linear_zero_hole_compose_row()
    dual = dual_only_throws_row()
    remap = remapped_compose_row()
    skipped = skipped_duplicate_compose_row()
    failed = failed_export_rhs_row()
    empty = empty_export_rhs_row()
    discover = discover_then_compose_row()
    known = hill_known_generate_unknown_identity_row()
    multi = multi_ic_identity_residual_row()
    constant = constant_rate_changes_residual_row()
    middle = skipped_middle_compose_row()
    mm_known = mm_known_no_compose_row()
    repress = repressilator_no_compose_row()
    sample = sample_destruction_matches_identity_row()
    irregular = irregular_times_residual_row()
    exploding = failed_solve_residual_is_inf_row()
    dual_terms = dual_per_term_compose_row()
    session = session_predict_hybrid_row()
    normalized = normalize_destruction_honesty_row()
    explicit_fn = equation_to_function_explicit_row()
    no_compile = compose_does_not_compile_row()
    shape = residual_shape_guard_row()
    kinetic = kinetic_known_no_compose_row()
    typed = hybrid_compose_typed_matrix()
    smoke = unique_claim_smoke_identity_row()
    return (;
        zero, dual, remap, skipped, failed, empty, discover, known,
        multi, constant, middle, mm_known, repress, sample, irregular,
        exploding, dual_terms, session, normalized, explicit_fn, no_compile,
        shape, kinetic, typed, smoke,
        holds = zero.holds && dual.holds && remap.holds && skipped.holds &&
                failed.holds && empty.holds && discover.holds &&
                known.holds && multi.holds && constant.holds &&
                middle.holds && mm_known.holds && repress.holds &&
                sample.holds && irregular.holds && exploding.holds &&
                dual_terms.holds && session.holds && normalized.holds &&
                explicit_fn.holds && no_compile.holds &&
                shape.holds && kinetic.holds && typed.holds && smoke.holds)
end

function hybrid_compose_fixture_matrix()
    identity = hybrid_compose_identity_matrix()
    honesty = hybrid_compose_honesty_matrix()
    return (;
        identity, honesty,
        holds = identity.holds && honesty.holds)
end

function hybrid_compose_fixture_names()
    return (
        :hill_identity, :mm_identity, :two_regulator, :six_state,
        :default_example, :competitive, :three_state, :linear_zero,
        :dual_only, :remapped, :skipped_duplicate, :failed_export,
        :empty_export, :discover_compose, :known_generate, :multi_ic,
        :constant_rate, :skipped_middle, :mm_known, :repressilator,
        :sample_match, :irregular, :exploding, :dual_per_term,
        :session_predict, :normalize, :explicit_fn, :no_compile)
end

function format_hybrid_compose_index()
    io = IOBuffer()
    println(io, "| fixture | role |")
    println(io, "|---|---|")
    println(io, "| hill_identity | neural rate recovers ude_system |")
    println(io, "| mm_identity | MM unknown identity residual |")
    println(io, "| two_regulator | D(S,I) identity |")
    println(io, "| six_state | six-state identity |")
    println(io, "| default_example | p53/Mdm2 remapped head |")
    println(io, "| competitive | competitive unknown identity |")
    println(io, "| three_state | three-state unknown identity |")
    println(io, "| linear_zero | 0-hole has no compose term |")
    println(io, "| dual_only | only() throws on two heads |")
    println(io, "| remapped | compose each remapped term |")
    println(io, "| skipped_duplicate | two dense heads compose |")
    println(io, "| failed_export | export_rhs rejects failure |")
    println(io, "| empty_export | export_rhs rejects empty candidates |")
    println(io, "| discover_compose | sample D → discover → compose |")
    println(io, "| known_generate | known Hill generate, unknown compose |")
    println(io, "| multi_ic | three-IC identity residual |")
    println(io, "| constant_rate | a wrong rate raises the residual |")
    println(io, "| skipped_middle | remapped 1:n heads compose |")
    println(io, "| mm_known | known MM has no neural term |")
    println(io, "| repressilator | known three-state has no neural term |")
    println(io, "| sample_match | sample_unknown_destruction matches identity |")
    println(io, "| irregular | irregular saveat identity residual |")
    println(io, "| exploding | a huge rate is finite-or-Inf, not 0.99 F1 |")
    println(io, "| dual_per_term | each dual head composes |")
    println(io, "| session_predict | TrainingSolveSession matches residual |")
    println(io, "| normalize | max-abs scaling of sampled D |")
    println(io, "| explicit_fn | equation_to_function stays finite |")
    println(io, "| no_compile | compose_hybrid_rhs does not compile |")
    return String(take!(io))
end

function hybrid_compose_index_holds()
    text = format_hybrid_compose_index()
    names = hybrid_compose_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("ude_system", text) &&
           occursin("export_rhs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Source checks ----------------------------------------------------------

