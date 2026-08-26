###############################################################################
# Identifiability product rows (not exported).
#
# UniqueClaim already names coefficients_are_biological_constants from
# unidentifiable_edge. This file locks the remaining join:
# production_destruction_tradeoff → identifiability_product →
# UniqueClaimProtocolRow → format_protocol_result, including collinearity
# print and the coefficients boolean on live fixtures.
#
# Does not change production_destruction_tradeoff return keys.
# Does not drop protocol ICs. Does not grow exports. Does not open
# Hill-from-NN. Combined F1 stays a skeleton floor.
###############################################################################

const IDENTIFIABILITY_PRODUCT_MUST_CONTAIN = (
    "function live_production_destruction_tradeoff",
    "function identifiability_product_row",
    "function coefficients_are_biological_constants_row",
    "function format_protocol_collinearity_row",
    "function join_tradeoff_protocol_row",
    "struct IdentifiabilityProductRow",
    "function collinearity_warning_row",
    "function smoke_vs_protocol_print_row")

const IDENTIFIABILITY_PRODUCT_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function identifiability_product_locked_sentences()
    return (;
        join = "production_destruction_tradeoff joins UniqueClaimProtocolRow through identifiability_product.",
        coefficients = "coefficients_are_biological_constants is false exactly when unidentifiable_edge is true.",
        collinearity = "format_protocol_result prints collinearity only when the cosine is finite.",
        practical = "The tradeoff is a practical Fisher/Jacobian cosine, not StructuralIdentifiability.jl.")
end

identifiability_product_contract() = identifiability_product_locked_sentences().join

function identifiability_product_source_path()
    joinpath(pkgdir(BioDynaX), "src", "IdentifiabilityProduct.jl")
end

function identifiability_product_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "identifiability-product.md")
end

function identifiability_product_test_path()
    joinpath(pkgdir(BioDynaX), "test", "test_identifiability_product.jl")
end

function identifiability_jl_source_path()
    joinpath(pkgdir(BioDynaX), "src", "Identifiability.jl")
end

# -- Live tradeoff helpers ----------------------------------------------------

function identifiability_short_trajectory(model::UDEModel, p, u0;
        tspan = (0.0, 1.2), n_points::Int = 16)
    times = collect(range(first(tspan), last(tspan); length = n_points))
    data = predict_ude(p, Float64.(u0), tspan, times, model)
    return (; times, data, u0 = Float64.(u0), tspan)
end

"""
    live_production_destruction_tradeoff(model, p, u0; kwargs...)

Run `production_destruction_tradeoff` on a short `predict_ude` trajectory.
Does not train. Does not drop protocol ICs.
"""
function live_production_destruction_tradeoff(model::UDEModel, p, u0;
        tspan = (0.0, 1.2), n_points::Int = 16,
        production_param::Symbol = :k_prod,
        term = nothing, kwargs...)
    traj = identifiability_short_trajectory(
        model, p, u0; tspan = tspan, n_points = n_points)
    return production_destruction_tradeoff(
        model, p, traj.data, traj.times, traj.u0, traj.tspan;
        production_param = production_param, term = term, kwargs...)
end

function synthetic_ude_from_tradeoff(tradeoff;
        residual = 0.01, recall = 1.0, f1 = 0.57,
        extras = String[], unknown_holes::Integer = 1)
    ude = (;
        data_residual = residual,
        support_recall = recall,
        support_f1 = f1,
        extras = extras,
        identifiability = tradeoff)
    result = build_protocol_result(ude; unknown_holes = unknown_holes)
    return merge(ude, (; protocol_result = result, locked_kpis = locked_ude_kpis(ude)))
end

function ident_phys_from_schema(model::UDEModel, nn; fill::Float64 = 0.8)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> fill, length(schema.phys_names)))
    return pack_parameters(phys, nn)
end

# -- Coefficient / product rows ----------------------------------------------

"""
    coefficients_are_biological_constants_row()

Boolean matrix for `coefficients_are_biological_constants`. Missing
ident still returns true; `unique_claim_identifiability_holds` is false.
"""
function coefficients_are_biological_constants_row()
    edge_true = (; unidentifiable_edge = true, production_param = :k_prod,
        collinearity = 0.997)
    edge_false = (; unidentifiable_edge = false, production_param = :k_prod,
        collinearity = 0.11)
    missing_flag = (; production_param = :k_prod)
    product_true = identifiability_product(edge_true)
    product_false = identifiability_product(edge_false)
    product_missing = identifiability_product(missing_flag)
    product_nothing = identifiability_product(nothing; unknown_holes = 0)
    return (;
        true_coeff = coefficients_are_biological_constants(edge_true),
        false_coeff = coefficients_are_biological_constants(edge_false),
        nothing_coeff = coefficients_are_biological_constants(nothing),
        missing_coeff = coefficients_are_biological_constants(missing_flag),
        product_true_coeff = product_true.coefficients_are_biological_constants,
        product_false_coeff = product_false.coefficients_are_biological_constants,
        product_missing_edge = product_missing.unidentifiable_edge,
        product_nothing_holes = product_nothing.unknown_holes,
        holds = coefficients_are_biological_constants(edge_true) == false &&
                coefficients_are_biological_constants(edge_false) == true &&
                coefficients_are_biological_constants(nothing) == true &&
                coefficients_are_biological_constants(missing_flag) == true &&
                product_true.coefficients_are_biological_constants == false &&
                product_false.coefficients_are_biological_constants == true &&
                product_missing.unidentifiable_edge == false &&
                product_nothing.unknown_holes == 0 &&
                unique_claim_identifiability_holds(edge_true) &&
                unique_claim_identifiability_holds(edge_false) == false &&
                unique_claim_identifiability_holds(nothing) == false)
end

function coefficients_follow_live_edge_row(tradeoff)
    product = identifiability_product(tradeoff)
    coeff = coefficients_are_biological_constants(tradeoff)
    return (;
        edge = tradeoff.unidentifiable_edge,
        coeff,
        product_coeff = product.coefficients_are_biological_constants,
        matches = coeff == !tradeoff.unidentifiable_edge &&
                  product.coefficients_are_biological_constants == coeff,
        holds = coeff == !tradeoff.unidentifiable_edge &&
                product.coefficients_are_biological_constants == coeff &&
                product.practical_not_structural)
end

# -- Format / collinearity print ----------------------------------------------

"""
    format_protocol_collinearity_row()

`format_protocol_result` prints `collinearity` only when the cosine is
finite. NaN and a missing field stay silent. Coefficients stay `!edge`.
"""
function format_protocol_collinearity_row()
    with_col = format_protocol_result((;
        unidentifiable_edge = true,
        collinearity = 0.997,
        production_param = :k_prod);
        residual = 0.01, support_recall = 1.0, support_f1 = 0.57,
        extras = String[], seed = 103, n_ics = 9, n_points = 50)
    without = format_protocol_result((; unidentifiable_edge = true);
        residual = 0.01, support_recall = 1.0, support_f1 = 0.57,
        extras = String[], seed = 103, n_ics = 9, n_points = 50)
    nan_col = format_protocol_result((;
        unidentifiable_edge = true, collinearity = NaN);
        residual = 0.01, support_recall = 1.0, support_f1 = 0.57)
    identified = format_protocol_result((;
        unidentifiable_edge = false, collinearity = 0.11,
        production_param = :k_rs);
        residual = 0.01, support_recall = 1.0, support_f1 = 0.57)
    return (;
        with_col, without, nan_col, identified,
        prints_finite = occursin("collinearity: 0.997", with_col),
        silent_missing = !occursin("collinearity:", without),
        silent_nan = !occursin("collinearity:", nan_col),
        identified_coeff = occursin(
            "coefficients_are_biological_constants: true", identified),
        unidentified_coeff = occursin(
            "coefficients_are_biological_constants: false", with_col),
        production = occursin("production_param: k_prod", with_col) &&
                     occursin("production_param: k_rs", identified),
        order = format_protocol_result_field_order_holds(with_col) &&
                format_protocol_result_field_order_holds(without),
        no_f1_paint = !occursin("support_f1_ude = 0.99", with_col),
        holds = occursin("collinearity: 0.997", with_col) &&
                !occursin("collinearity:", without) &&
                !occursin("collinearity:", nan_col) &&
                occursin("coefficients_are_biological_constants: false", with_col) &&
                occursin("coefficients_are_biological_constants: true", identified) &&
                occursin("production_param: k_prod", with_col) &&
                occursin("production_param: k_rs", identified) &&
                format_protocol_result_field_order_holds(with_col) &&
                !occursin("0.99 F1", with_col))
end

function collinearity_warning_row()
    high = (;
        production_param = :k_prod,
        collinearity = 0.997,
        unidentifiable_edge = true)
    low = (;
        production_param = :k_prod,
        collinearity = 0.11,
        unidentifiable_edge = false)
    nan_rep = (;
        production_param = :k_prod,
        collinearity = NaN,
        unidentifiable_edge = false)
    high_txt = format_production_destruction_warning(high)
    low_txt = format_production_destruction_warning(low)
    nan_txt = format_production_destruction_warning(nan_rep)
    return (;
        high_txt, low_txt, nan_txt,
        high_warns = occursin("collinear", high_txt) && occursin("0.997", high_txt),
        low_below = occursin("below threshold", low_txt),
        nan_na = occursin("NA", nan_txt),
        not_structural = occursin("not structural", high_txt),
        holds = occursin("collinear", high_txt) && occursin("0.997", high_txt) &&
                occursin("below threshold", low_txt) && occursin("NA", nan_txt) &&
                occursin("not structural", high_txt) &&
                !occursin("0.99 F1", high_txt))
end

function format_protocol_sections_row()
    ident = (;
        unidentifiable_edge = true,
        collinearity = 0.991,
        production_param = :k_prod)
    sections = format_protocol_sections(ident;
        residual = 0.02, support_recall = 1.0, support_f1 = 0.57,
        extras = String[], seed = 103, n_ics = 9, n_points = 50,
        adam_iters = 100, bfgs_iters = 50)
    return (;
        order = sections.order_holds,
        ident_has_coeff = occursin(
            "coefficients_are_biological_constants: false",
            sections.identifiability),
        ident_has_col = occursin("collinearity: 0.991", sections.identifiability),
        fit_has_residual = occursin("hybrid_data_residual:", sections.fit),
        disc_has_f1 = occursin("support_f1:", sections.discovery),
        repro_has_seed = occursin("seed: 103", sections.reproduction),
        repro_has_ics = occursin("n_ics: 9", sections.reproduction),
        holds = sections.order_holds &&
                occursin("coefficients_are_biological_constants: false",
                    sections.identifiability) &&
                occursin("collinearity: 0.991", sections.identifiability) &&
                occursin("hybrid_data_residual:", sections.fit) &&
                occursin("support_f1:", sections.discovery) &&
                occursin("seed: 103", sections.reproduction) &&
                occursin("n_ics: 9", sections.reproduction) &&
                !occursin("support_f1_ude = 0.99", sections.text))
end

function smoke_vs_protocol_print_row()
    ident = (; unidentifiable_edge = true, collinearity = 0.99,
        production_param = :k_prod)
    protocol_fp = unique_claim_fingerprint()
    smoke_fp = unique_claim_fingerprint(; smoke = true)
    protocol_txt = format_protocol_result(ident, protocol_fp;
        residual = 0.01, support_recall = 1.0, support_f1 = 0.57)
    smoke_txt = format_protocol_result(ident, smoke_fp;
        residual = 0.01, support_recall = 1.0, support_f1 = 0.57)
    return (;
        protocol_ics = occursin("n_ics: 9", protocol_txt),
        protocol_points = occursin("n_points: 50", protocol_txt),
        protocol_seed = occursin("seed: 103", protocol_txt),
        protocol_kind = occursin("protocol_kind: protocol", protocol_txt),
        smoke_ics = occursin("n_ics: 1", smoke_txt),
        smoke_points = occursin("n_points: 8", smoke_txt),
        smoke_kind = occursin("protocol_kind: smoke", smoke_txt),
        smoke_flag = occursin("smoke: true", smoke_txt),
        same_coeff = occursin(
            "coefficients_are_biological_constants: false", protocol_txt) &&
            occursin(
            "coefficients_are_biological_constants: false", smoke_txt),
        holds = occursin("n_ics: 9", protocol_txt) &&
                occursin("n_points: 50", protocol_txt) &&
                occursin("seed: 103", protocol_txt) &&
                occursin("n_ics: 1", smoke_txt) &&
                occursin("n_points: 8", smoke_txt) &&
                occursin("protocol_kind: smoke", smoke_txt) &&
                smoke_fp.n_ics != protocol_fp.n_ics &&
                unique_claim_fingerprint_is_protocol(protocol_fp) &&
                unique_claim_fingerprint_is_smoke(smoke_fp))
end

# -- Typed join ---------------------------------------------------------------

struct IdentifiabilityProductRow
    name::Symbol
    unknown_holes::Int
    unidentifiable_edge::Bool
    coefficients_are_biological_constants::Bool
    collinearity::Float64
    production_param::Symbol
    extras_label::String
    protocol_kind::Symbol
    holds::Bool
end

function identifiability_product_row_namedtuple(row::IdentifiabilityProductRow)
    return (;
        name = row.name,
        unknown_holes = row.unknown_holes,
        unidentifiable_edge = row.unidentifiable_edge,
        coefficients_are_biological_constants =
            row.coefficients_are_biological_constants,
        collinearity = row.collinearity,
        production_param = row.production_param,
        extras_label = row.extras_label,
        protocol_kind = row.protocol_kind,
        holds = row.holds)
end

"""
    join_tradeoff_protocol_row(name, tradeoff; kwargs...)

Join a live or synthetic tradeoff to `identifiability_product` and a
`UniqueClaimProtocolRow`. Combined F1 is stored and is not a failure
symbol.
"""
function join_tradeoff_protocol_row(name::Symbol, tradeoff;
        residual = 0.01, recall = 1.0, f1 = 0.57,
        extras = String[], unknown_holes::Integer = 1,
        fingerprint::UniqueClaimFingerprint = unique_claim_fingerprint())
    product = identifiability_product(tradeoff; unknown_holes = unknown_holes)
    ude = synthetic_ude_from_tradeoff(tradeoff;
        residual = residual, recall = recall, f1 = f1,
        extras = extras, unknown_holes = unknown_holes)
    row = unique_claim_protocol_row(ude; fingerprint = fingerprint)
    coeff = coefficients_are_biological_constants(tradeoff)
    warning = format_production_destruction_warning(tradeoff)
    text_has_coeff = occursin(
        "coefficients_are_biological_constants: $(coeff)", row.text)
    extras_ok = extras_print_is_hardcoded_attempt(row.extras_label) == false
    f1_not_failure = :support_f1 ∉ row.kpi_failures
    order = format_protocol_result_field_order_holds(row.text)
    holds = product.coefficients_are_biological_constants == coeff &&
            coeff == !tradeoff.unidentifiable_edge &&
            row.protocol_result.coefficients_are_biological_constants == coeff &&
            text_has_coeff && extras_ok && f1_not_failure && order &&
            row.protocol_result.canonical_hill_from_nn === false &&
            row.protocol_result.claim === :recall_plus_data_residual &&
            !occursin("support_f1_ude = 0.99", row.text)
    typed = IdentifiabilityProductRow(
        name,
        product.unknown_holes,
        tradeoff.unidentifiable_edge === true,
        coeff,
        Float64(tradeoff.collinearity),
        product.production_param,
        row.extras_label,
        fingerprint.kind,
        holds)
    return (;
        typed,
        product,
        protocol = unique_claim_protocol_row_namedtuple(row),
        warning,
        text = row.text,
        kpi_failures = row.kpi_failures,
        holds)
end

function identifiability_product_row(name::Symbol, model, p, u0; kwargs...)
    tradeoff = live_production_destruction_tradeoff(model, p, u0)
    return join_tradeoff_protocol_row(name, tradeoff; kwargs...)
end

# -- Fixture tradeoff paths ---------------------------------------------------

function hill_known_tradeoff_path()
    truth = hybrid_known_hill_truth(409)
    u0 = [0.30, 0.25]
    trade = live_production_destruction_tradeoff(
        truth.truth.model, truth.truth.parameters, u0;
        production_param = :k_prod)
    join = join_tradeoff_protocol_row(:hill_known, trade; unknown_holes = 0)
    coeff = coefficients_follow_live_edge_row(trade)
    return (;
        trade, join, coeff,
        holes = count_unknown_destructions(truth.net),
        neural = isempty(neural_destruction_terms(truth.truth.model)),
        collinearity_nan = !isfinite(trade.collinearity),
        validate_open = validate_network(truth.net) === truth.net,
        holds = join.holds && coeff.holds &&
                count_unknown_destructions(truth.net) == 0 &&
                isempty(neural_destruction_terms(truth.truth.model)) &&
                !isfinite(trade.collinearity) &&
                validate_network(truth.net) === truth.net)
end

function hill_unknown_tradeoff_path()
    built = hybrid_linear_unknown_model(419)
    trade = live_production_destruction_tradeoff(
        built.model, built.packed, [0.30, 0.25];
        production_param = :k_prod)
    join = join_tradeoff_protocol_row(:hill_unknown, trade; unknown_holes = 1)
    coeff = coefficients_follow_live_edge_row(trade)
    return (;
        trade, join, coeff,
        holes = count_unknown_destructions(built.net),
        neural = length(neural_destruction_terms(built.model)),
        collinearity_finite = isfinite(trade.collinearity),
        holds = join.holds && coeff.holds &&
                count_unknown_destructions(built.net) == 1 &&
                length(neural_destruction_terms(built.model)) == 1 &&
                isfinite(trade.collinearity) &&
                trade.production_param === :k_prod)
end

function mm_unknown_tradeoff_path()
    net = build_mm_recovery_network(; known = false)
    rng = MersenneTwister(421)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), p0.nn)
    trade = live_production_destruction_tradeoff(
        model, packed, [0.30, 0.25]; production_param = :k_prod)
    join = join_tradeoff_protocol_row(:mm_unknown, trade)
    return (;
        trade, join,
        holes = count_unknown_destructions(net),
        collinearity_finite = isfinite(trade.collinearity),
        holds = join.holds && count_unknown_destructions(net) == 1 &&
                isfinite(trade.collinearity))
end

function mm_known_tradeoff_path()
    net = build_mm_recovery_network(; known = true)
    rng = MersenneTwister(431)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters(
        (k_prod = 0.9, vmax = 1.4, km = 0.45, k_rs = 1.0, k_r = 0.6), p0.nn)
    trade = live_production_destruction_tradeoff(
        model, packed, [0.30, 0.25]; production_param = :k_prod)
    join = join_tradeoff_protocol_row(:mm_known, trade; unknown_holes = 0)
    return (;
        trade, join,
        holes = count_unknown_destructions(net),
        neural = isempty(neural_destruction_terms(model)),
        collinearity_nan = !isfinite(trade.collinearity),
        validate_open = validate_network(net) === net,
        holds = join.holds && count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                !isfinite(trade.collinearity) &&
                validate_network(net) === net)
end

function linear_zero_hole_tradeoff_path()
    net = build_linear_test_network()
    rng = MersenneTwister(433)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    trade = live_production_destruction_tradeoff(
        model, packed, [0.20, 0.10]; production_param = :k_ba)
    join = join_tradeoff_protocol_row(:linear_zero, trade;
        unknown_holes = 0, fingerprint = unique_claim_fingerprint(; smoke = true))
    return (;
        trade, join,
        holes = count_unknown_destructions(net),
        production = trade.production_param,
        collinearity_nan = !isfinite(trade.collinearity),
        smoke = join.typed.protocol_kind === :smoke,
        validate_open = validate_network(net) === net,
        holds = join.holds && count_unknown_destructions(net) == 0 &&
                !isfinite(trade.collinearity) &&
                trade.production_param === :k_ba &&
                join.typed.protocol_kind === :smoke &&
                validate_network(net) === net)
end

function two_regulator_tradeoff_path()
    net = build_two_regulator_unknown_network()
    rng = MersenneTwister(439)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_es = 0.8, k_i = 0.5, k_e = 0.4), p0.nn)
    names = parameter_schema(model).phys_names
    production = :k_es in names ? :k_es : first(names)
    trade = live_production_destruction_tradeoff(
        model, packed, [0.25, 0.20, 0.15];
        production_param = production, n_points = 12)
    join = join_tradeoff_protocol_row(:two_regulator, trade)
    return (;
        trade, join,
        n_regs = length(only(neural_destruction_terms(model)).regulators),
        holds = join.holds &&
                length(only(neural_destruction_terms(model)).regulators) == 2)
end

function six_state_tradeoff_path()
    net = build_six_state_unknown_network(; known = false)
    rng = MersenneTwister(443)
    model, p0 = build_ude_model(rng, net)
    packed = ident_phys_from_schema(model, p0.nn)
    names = parameter_schema(model).phys_names
    production = first(names)
    u0 = [0.22, 0.18, 0.16, 0.14, 0.12, 0.10]
    trade = live_production_destruction_tradeoff(
        model, packed, u0; production_param = production, n_points = 10)
    join = join_tradeoff_protocol_row(:six_state, trade)
    return (;
        trade, join,
        nstates = model.compiled.nstates,
        holds = join.holds && model.compiled.nstates == 6)
end

function default_example_tradeoff_path()
    net = DEFAULT_EXAMPLE_NETWORK
    rng = MersenneTwister(449)
    model, p0 = build_ude_model(rng, net)
    names = parameter_schema(model).phys_names
    production = first(names)
    trade = live_production_destruction_tradeoff(
        model, p0, [0.20, 0.10]; production_param = production, n_points = 12)
    join = join_tradeoff_protocol_row(:default_example, trade)
    return (;
        trade, join,
        holes = count_unknown_destructions(net),
        dense = neural_index_is_dense(model),
        holds = join.holds && count_unknown_destructions(net) == 1 &&
                neural_index_is_dense(model))
end

function remapped_tradeoff_path()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(457)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters(remapped_two_regulator_phys_truth(), p0.nn)
    terms = neural_destruction_terms(model)
    u0 = remapped_two_regulator_state()
    names = parameter_schema(model).phys_names
    production = first(names)
    rows = NamedTuple[]
    for term in terms
        trade = live_production_destruction_tradeoff(
            model, packed, u0; production_param = production,
            term = term, n_points = 10)
        join = join_tradeoff_protocol_row(
            Symbol(:remap_, term.nn_index), trade; unknown_holes = 2)
        push!(rows, (;
            nn_index = term.nn_index,
            collinearity = trade.collinearity,
            join_holds = join.holds,
            coeff = coefficients_are_biological_constants(trade)))
    end
    return (;
        n_terms = length(terms),
        rows,
        dense = neural_index_is_dense(model),
        admits = unique_claim_recovery_admits(net),
        holds = length(terms) == 2 && neural_index_is_dense(model) &&
                all(r -> r.join_holds, rows) &&
                unique_claim_recovery_admits(net) == false)
end

function dual_tradeoff_path()
    net = build_dual_unknown_network()
    rng = MersenneTwister(461)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_cb = 0.9, k_c = 0.5), p0.nn)
    terms = neural_destruction_terms(model)
    u0 = [0.22, 0.18, 0.16]
    names = parameter_schema(model).phys_names
    production = first(names)
    first_trade = live_production_destruction_tradeoff(
        model, packed, u0; production_param = production,
        term = first(terms), n_points = 10)
    join = join_tradeoff_protocol_row(:dual, first_trade; unknown_holes = 2)
    only_threw = false
    try
        only(terms)
    catch
        only_threw = true
    end
    return (;
        n_terms = length(terms),
        join,
        only_threw,
        admits = unique_claim_recovery_admits(net),
        validate_open = validate_network(net) === net,
        holds = join.holds && length(terms) == 2 && only_threw &&
                unique_claim_recovery_admits(net) == false &&
                validate_network(net) === net)
end

function competitive_unknown_tradeoff_path()
    net = build_competitive_test_network(; known = false)
    rng = MersenneTwister(463)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_in = 0.9, k_s = 0.8, k_i = 0.5), p0.nn)
    names = parameter_schema(model).phys_names
    production = first(names)
    trade = live_production_destruction_tradeoff(
        model, packed, [0.25, 0.45, 0.20];
        production_param = production, n_points = 12)
    join = join_tradeoff_protocol_row(:competitive, trade)
    return (;
        trade, join,
        holes = count_unknown_destructions(net),
        holds = join.holds)
end

function three_state_tradeoff_path()
    net = build_three_state_unknown_network()
    rng = MersenneTwister(467)
    model, p0 = build_ude_model(rng, net)
    packed = ident_phys_from_schema(model, p0.nn)
    names = parameter_schema(model).phys_names
    production = first(names)
    n = model.compiled.nstates
    trade = live_production_destruction_tradeoff(
        model, packed, fill(0.20, n);
        production_param = production, n_points = 10)
    join = join_tradeoff_protocol_row(:three_state, trade)
    return (;
        trade, join,
        nstates = n,
        holes = count_unknown_destructions(net),
        holds = join.holds && count_unknown_destructions(net) == 1)
end

function skipped_duplicate_tradeoff_path()
    net = build_skipped_duplicate_unknown_network()
    rng = MersenneTwister(479)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_b = 0.5, k_c = 0.4), p0.nn)
    terms = neural_destruction_terms(model)
    names = parameter_schema(model).phys_names
    production = first(names)
    trade = live_production_destruction_tradeoff(
        model, packed, [0.2, 0.3, 0.4];
        production_param = production, term = first(terms), n_points = 8)
    join = join_tradeoff_protocol_row(:skipped_duplicate, trade;
        unknown_holes = length(terms))
    return (;
        n_terms = length(terms),
        dense = neural_index_is_dense(model),
        join,
        holds = join.holds && length(terms) == 2 &&
                neural_index_is_dense(model))
end

function repressilator_tradeoff_path()
    net = build_repressilator_network()
    rng = MersenneTwister(487)
    model, p0 = build_ude_model(rng, net)
    packed = ident_phys_from_schema(model, p0.nn)
    names = parameter_schema(model).phys_names
    trade = live_production_destruction_tradeoff(
        model, packed, fill(0.20, 3);
        production_param = first(names), n_points = 10)
    join = join_tradeoff_protocol_row(:repressilator, trade; unknown_holes = 0)
    return (;
        trade, join,
        holes = count_unknown_destructions(net),
        neural = isempty(neural_destruction_terms(model)),
        holds = join.holds && count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)))
end

# -- Report / verbose / missing production ------------------------------------

function report_verbose_tradeoff_row()
    built = hybrid_linear_unknown_model(491)
    traj = identifiability_short_trajectory(
        built.model, built.packed, [0.30, 0.25])
    io = IOBuffer()
    report = redirect_stdout(io) do
        report_production_destruction_tradeoff(
            built.model, built.packed, traj.data, traj.times,
            traj.u0, traj.tspan; verbose = true)
    end
    printed = String(take!(io))
    silent = redirect_stdout(devnull) do
        report_production_destruction_tradeoff(
            built.model, built.packed, traj.data, traj.times,
            traj.u0, traj.tspan; verbose = false)
    end
    warning = format_production_destruction_warning(report)
    return (;
        report, silent, printed, warning,
        printed_has_warning = occursin("collinear", printed) ||
            occursin("collinearity", printed) ||
            occursin("Practical", printed),
        same_edge = report.unidentifiable_edge == silent.unidentifiable_edge,
        holds = isfinite(report.collinearity) &&
                report.unidentifiable_edge == silent.unidentifiable_edge &&
                (occursin("collinear", printed) ||
                 occursin("collinearity", printed) ||
                 occursin("Practical", printed)) &&
                !occursin("0.99 F1", printed))
end

function missing_production_param_row()
    built = hybrid_linear_unknown_model(499)
    trade = live_production_destruction_tradeoff(
        built.model, built.packed, [0.30, 0.25];
        production_param = :not_a_parameter)
    return (;
        production = trade.production_param,
        correlation = trade.production_correlation,
        collinearity = trade.collinearity,
        holds = trade.production_param === :not_a_parameter &&
                isnan(trade.production_correlation))
end

function frozen_k_prod_raw_unchanged_row()
    built = hybrid_linear_unknown_model(503)
    raw = built.packed.phys.k_prod
    trade = live_production_destruction_tradeoff(
        built.model, built.packed, [0.30, 0.25])
    return (;
        raw,
        same = built.packed.phys.k_prod ≈ raw,
        trade_param = trade.production_param,
        holds = built.packed.phys.k_prod ≈ raw &&
                trade.production_param === :k_prod &&
                isfinite(trade.collinearity))
end

function compile_free_tradeoff_row()
    built = hybrid_linear_unknown_model(509)
    n = with_compile_network_counter() do counter
        live_production_destruction_tradeoff(
            built.model, built.packed, [0.30, 0.25])
        counter[]
    end
    return (; compiles = n, holds = n == 0)
end

function extras_not_invented_on_join_row()
    trade = (;
        production_param = :k_prod,
        condition_number = 10.0,
        production_correlation = 0.2,
        collinearity = 0.2,
        unidentifiable_edge = false,
        fisher = nothing)
    join = join_tradeoff_protocol_row(:synthetic_false, trade;
        extras = String[])
    na = join_tradeoff_protocol_row(:synthetic_na, trade;
        extras = nothing)
    leftover = join_tradeoff_protocol_row(:synthetic_live, trade;
        extras = ["1", "r"], f1 = 0.57)
    return (;
        join, na, leftover,
        none = join.typed.extras_label == "(none)",
        na_label = na.typed.extras_label == "NA",
        live = leftover.typed.extras_label == "1, r",
        no_attempt = extras_print_is_hardcoded_attempt(leftover.typed.extras_label) == false,
        holds = join.holds && na.holds && leftover.holds &&
                join.typed.extras_label == "(none)" &&
                na.typed.extras_label == "NA" &&
                leftover.typed.extras_label == "1, r" &&
                extras_print_is_hardcoded_attempt(leftover.typed.extras_label) == false)
end

function kpi_f1_not_a_failure_on_join_row()
    trade = (;
        production_param = :k_prod,
        condition_number = 1e7,
        production_correlation = 0.99,
        collinearity = 0.99,
        unidentifiable_edge = true,
        fisher = nothing)
    join = join_tradeoff_protocol_row(:synthetic_true, trade;
        residual = 0.01, recall = 1.0, f1 = 0.57)
    low_f1 = join_tradeoff_protocol_row(:synthetic_low_f1, trade;
        residual = 0.01, recall = 1.0, f1 = 0.10)
    return (;
        join, low_f1,
        failures = join.kpi_failures,
        low_failures = low_f1.kpi_failures,
        holds = join.holds && low_f1.holds &&
                :support_f1 ∉ join.kpi_failures &&
                :support_f1 ∉ low_f1.kpi_failures &&
                isempty(join.kpi_failures) &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50)
end

function protocol_row_rejects_hill_from_nn_row()
    trade = (;
        production_param = :k_prod,
        condition_number = 10.0,
        production_correlation = 0.2,
        collinearity = 0.2,
        unidentifiable_edge = true,
        fisher = nothing)
    ude = synthetic_ude_from_tradeoff(trade)
    opened = merge(ude.protocol_result, (; canonical_hill_from_nn = true))
    threw = false
    try
        assert_protocol_result_fields(opened)
    catch
        threw = true
    end
    return (;
        threw,
        closed = ude.protocol_result.canonical_hill_from_nn === false,
        holds = threw && ude.protocol_result.canonical_hill_from_nn === false)
end

# -- Source locks -------------------------------------------------------------

function production_destruction_tradeoff_source_holds()
    src = read(identifiability_jl_source_path(), String)
    start = findfirst(
        "function production_destruction_tradeoff(", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\n\"\"\"", rest, 40)
    body = nxt === nothing ? rest[1:min(lastindex(rest), 2500)] :
        rest[1:(first(nxt) - 1)]
    return occursin("assess_identifiability", body) &&
           occursin("collinearity", body) &&
           occursin("unidentifiable_edge", body) &&
           occursin("_destruction_contribution", body) &&
           !occursin("StructuralIdentifiability", body)
end

function format_production_destruction_warning_source_holds()
    src = read(identifiability_jl_source_path(), String)
    start = findfirst("function format_production_destruction_warning", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\n\"\"\"", rest, 20)
    body = nxt === nothing ? rest[1:min(lastindex(rest), 1200)] :
        rest[1:(first(nxt) - 1)]
    return occursin("not structural", body) &&
           occursin("collinear", body)
end

function format_protocol_result_collinearity_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function format_protocol_result(ident;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\n\"\"\"", rest, 40)
    body = nxt === nothing ? rest[1:min(lastindex(rest), 2500)] :
        rest[1:(first(nxt) - 1)]
    return occursin("coefficients_are_biological_constants", body) &&
           occursin("collinearity", body) &&
           occursin("isfinite(ident.collinearity)", body) &&
           occursin("canonical_hill_from_nn: false", body)
end

function coefficients_are_biological_constants_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "UniqueClaim.jl"), String)
    start = findfirst(
        "function coefficients_are_biological_constants(ident)", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("unidentifiable_edge", body) &&
           occursin("!ident.unidentifiable_edge", body)
end

# -- Matrices / catalog -------------------------------------------------------

function identifiability_product_fixture_matrix()
    coeff = coefficients_are_biological_constants_row()
    collinearity = format_protocol_collinearity_row()
    warning = collinearity_warning_row()
    sections = format_protocol_sections_row()
    smoke = smoke_vs_protocol_print_row()
    extras = extras_not_invented_on_join_row()
    kpi = kpi_f1_not_a_failure_on_join_row()
    hill_closed = protocol_row_rejects_hill_from_nn_row()
    missing = missing_production_param_row()
    frozen = frozen_k_prod_raw_unchanged_row()
    compile_free = compile_free_tradeoff_row()
    verbose = report_verbose_tradeoff_row()
    hill_known = hill_known_tradeoff_path()
    hill_unknown = hill_unknown_tradeoff_path()
    mm_unknown = mm_unknown_tradeoff_path()
    mm_known = mm_known_tradeoff_path()
    linear = linear_zero_hole_tradeoff_path()
    two = two_regulator_tradeoff_path()
    six = six_state_tradeoff_path()
    default = default_example_tradeoff_path()
    remap = remapped_tradeoff_path()
    dual = dual_tradeoff_path()
    comp = competitive_unknown_tradeoff_path()
    three = three_state_tradeoff_path()
    skipped = skipped_duplicate_tradeoff_path()
    repress = repressilator_tradeoff_path()
    middle = skipped_middle_tradeoff_path()
    kinetic = kinetic_known_tradeoff_path()
    cond = condition_threshold_row()
    matched = format_matches_joined_protocol_row()
    return (;
        coeff, collinearity, warning, sections, smoke, extras, kpi,
        hill_closed, missing, frozen, compile_free, verbose,
        hill_known, hill_unknown, mm_unknown, mm_known, linear, two,
        six, default, remap, dual, comp, three, skipped, repress,
        middle, kinetic, cond, matched,
        holds = coeff.holds && collinearity.holds && warning.holds &&
                sections.holds && smoke.holds && extras.holds && kpi.holds &&
                hill_closed.holds && missing.holds && frozen.holds &&
                compile_free.holds && verbose.holds && hill_known.holds &&
                hill_unknown.holds && mm_unknown.holds && mm_known.holds &&
                linear.holds && two.holds && six.holds && default.holds &&
                remap.holds && dual.holds && comp.holds && three.holds &&
                skipped.holds && repress.holds && middle.holds &&
                kinetic.holds && cond.holds && matched.holds)
end

function skipped_middle_tradeoff_path()
    net = build_skipped_middle_unknown_network()
    rng = MersenneTwister(523)
    model, p0 = build_ude_model(rng, net)
    packed = ident_phys_from_schema(model, p0.nn)
    terms = neural_destruction_terms(model)
    names = parameter_schema(model).phys_names
    production = first(names)
    u0 = [0.22, 0.18, 0.16, 0.14]
    trade = live_production_destruction_tradeoff(
        model, packed, u0; production_param = production,
        term = first(terms), n_points = 8)
    join = join_tradeoff_protocol_row(:skipped_middle, trade;
        unknown_holes = length(terms))
    return (;
        n_terms = length(terms),
        dense = neural_index_is_dense(model),
        join,
        validate_open = validate_network(net) === net,
        holds = join.holds && length(terms) ≥ 2 &&
                neural_index_is_dense(model) &&
                validate_network(net) === net)
end

function kinetic_known_tradeoff_path()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(529)
    model, p0 = build_ude_model(rng, net)
    packed = ident_phys_from_schema(model, p0.nn)
    names = parameter_schema(model).phys_names
    n = model.compiled.nstates
    trade = live_production_destruction_tradeoff(
        model, packed, fill(0.20, n);
        production_param = first(names), n_points = 10)
    join = join_tradeoff_protocol_row(:kinetic, trade; unknown_holes = 0)
    return (;
        trade, join,
        holes = count_unknown_destructions(net),
        neural = isempty(neural_destruction_terms(model)),
        validate_open = validate_network(net) === net,
        holds = join.holds && count_unknown_destructions(net) == 0 &&
                isempty(neural_destruction_terms(model)) &&
                validate_network(net) === net)
end

function condition_threshold_row()
    high = (;
        production_param = :k_prod,
        condition_number = 1e7,
        production_correlation = 0.2,
        collinearity = 0.11,
        unidentifiable_edge = true,
        fisher = nothing)
    low = (;
        production_param = :k_prod,
        condition_number = 10.0,
        production_correlation = 0.2,
        collinearity = 0.11,
        unidentifiable_edge = false,
        fisher = nothing)
    high_join = join_tradeoff_protocol_row(:high_cond, high)
    low_join = join_tradeoff_protocol_row(:low_cond, low)
    return (;
        high_join, low_join,
        high_coeff = high_join.typed.coefficients_are_biological_constants,
        low_coeff = low_join.typed.coefficients_are_biological_constants,
        holds = high_join.holds && low_join.holds &&
                high_join.typed.coefficients_are_biological_constants == false &&
                low_join.typed.coefficients_are_biological_constants == true)
end

function format_matches_joined_protocol_row()
    built = hybrid_linear_unknown_model(541)
    trade = live_production_destruction_tradeoff(
        built.model, built.packed, [0.30, 0.25])
    ude = synthetic_ude_from_tradeoff(trade)
    row = unique_claim_protocol_row(ude)
    matched = try
        assert_format_matches_protocol_result(row.protocol_result, row.text)
        true
    catch
        false
    end
    asserted = try
        assert_unique_claim_protocol_row(row)
        true
    catch
        false
    end
    return (;
        matched, asserted,
        kind = row.fingerprint.kind,
        n_ics = row.fingerprint.n_ics,
        holds = matched && asserted &&
                row.fingerprint.n_ics == 9 &&
                row.fingerprint.n_points == 50 &&
                row.protocol_result.canonical_hill_from_nn === false)
end

function identifiability_product_typed_matrix()
    built = hybrid_linear_unknown_model(521)
    live = identifiability_product_row(
        :hill_live, built.model, built.packed, [0.30, 0.25])
    known = hill_known_tradeoff_path()
    return (;
        live = identifiability_product_row_namedtuple(live.typed),
        known = identifiability_product_row_namedtuple(known.join.typed),
        holds = live.holds && known.holds &&
                live.typed.unknown_holes == 1 &&
                known.join.typed.unknown_holes == 0)
end

function identifiability_product_fixture_names()
    return (
        :coefficients, :collinearity_print, :warning, :sections,
        :smoke_protocol, :extras, :kpi_f1, :hill_closed,
        :missing_param, :frozen_raw, :compile_free, :verbose,
        :hill_known, :hill_unknown, :mm_unknown, :mm_known,
        :linear_zero, :two_regulator, :six_state, :default_example,
        :remapped, :dual, :competitive, :three_state,
        :skipped_duplicate, :repressilator, :skipped_middle,
        :kinetic, :condition, :format_match)
end

function format_identifiability_product_index()
    io = IOBuffer()
    println(io, "| fixture | role |")
    println(io, "|---|---|")
    println(io, "| coefficients | boolean follows unidentifiable_edge |")
    println(io, "| collinearity_print | finite cosine prints; NaN stays silent |")
    println(io, "| warning | practical collinearity warning, not structural |")
    println(io, "| sections | IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION |")
    println(io, "| smoke_protocol | 1 IC / 8 points is not 9 ICs / 50 points |")
    println(io, "| extras | NA / (none) / live leftovers; no F1-attempt paint |")
    println(io, "| kpi_f1 | combined F1 is not a KPI failure symbol |")
    println(io, "| hill_closed | canonical_hill_from_nn stays false |")
    println(io, "| missing_param | unknown production_param leaves correlation NaN |")
    println(io, "| frozen_raw | live tradeoff does not mutate packed phys |")
    println(io, "| compile_free | tradeoff does not call compile_network |")
    println(io, "| verbose | report_production_destruction_tradeoff prints |")
    println(io, "| hill_known | 0-hole Hill has NaN D-scale collinearity |")
    println(io, "| hill_unknown | unknown Hill has a finite D-scale cosine |")
    println(io, "| mm_unknown | unknown MM tradeoff joins the protocol row |")
    println(io, "| mm_known | known MM has no neural D-scale cosine |")
    println(io, "| linear_zero | 0-hole linear uses smoke fingerprint |")
    println(io, "| two_regulator | D(S,I) tradeoff |")
    println(io, "| six_state | six-state tradeoff |")
    println(io, "| default_example | p53/Mdm2 remapped head tradeoff |")
    println(io, "| remapped | each remapped head gets its own tradeoff |")
    println(io, "| dual | two-hole network does not admit unique-claim |")
    println(io, "| competitive | competitive unknown tradeoff |")
    println(io, "| three_state | three-state unknown tradeoff |")
    println(io, "| skipped_duplicate | dense two-head tradeoff |")
    println(io, "| repressilator | known three-state has no neural cosine |")
    println(io, "| skipped_middle | remapped 1:n heads tradeoff |")
    println(io, "| kinetic | known kinetic network has no neural cosine |")
    println(io, "| condition | Fisher condition follows the coefficients boolean |")
    println(io, "| format_match | joined protocol row matches format_protocol_result |")
    return String(take!(io))
end

function identifiability_product_index_holds()
    text = format_identifiability_product_index()
    names = identifiability_product_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("unidentifiable_edge", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Docs / contract ----------------------------------------------------------

function identifiability_product_source_holds()
    src = read(identifiability_product_source_path(), String)
    docs = isfile(identifiability_product_docs_path()) ?
        read(identifiability_product_docs_path(), String) : ""
    impl = read(identifiability_jl_source_path(), String)
    return all(occursin(needle, src) for needle in IDENTIFIABILITY_PRODUCT_MUST_CONTAIN) &&
           !occursin("support_f1_ude = 0.99", impl) &&
           !occursin("support_f1_ude = 0.99", docs) &&
           !occursin("function validate_network", docs)
end

function identifiability_product_source_violations()
    src = read(identifiability_product_source_path(), String)
    docs = isfile(identifiability_product_docs_path()) ?
        read(identifiability_product_docs_path(), String) : ""
    missing = [s for s in IDENTIFIABILITY_PRODUCT_MUST_CONTAIN if !occursin(s, src)]
    forbidden = String[]
    occursin("support_f1_ude = 0.99", docs) &&
        push!(forbidden, "docs: support_f1_ude = 0.99")
    occursin("function validate_network", docs) &&
        push!(forbidden, "docs: function validate_network")
    return (; missing, forbidden)
end

function identifiability_product_docs_hold()
    path = identifiability_product_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(identifiability_product_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("identifiability-product.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function identifiability_product_landing_docs_hold()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    sentences = identifiability_product_locked_sentences()
    return occursin("identifiability-product", howto) &&
           occursin("production_destruction_tradeoff", howto) &&
           occursin(sentences.join, sciml)
end

function identifiability_product_example_source_holds()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    docs = read(identifiability_product_docs_path(), String)
    return occursin("coefficients_are_biological_constants", howto) &&
           occursin("UniqueClaimProtocolRow", docs) &&
           occursin("collinearity", docs) &&
           occursin("1 IC / 8 points", docs)
end

function identifiability_product_docs_mention_helpers()
    path = identifiability_product_docs_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("live_production_destruction_tradeoff", text) &&
           occursin("join_tradeoff_protocol_row", text) &&
           occursin("format_protocol_collinearity_row", text) &&
           occursin("coefficients_are_biological_constants_row", text)
end

function identifiability_product_test_file_holds()
    path = identifiability_product_test_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("identifiability_product_contract_holds", text) &&
           occursin("public_export_list_holds", text) &&
           occursin("RECOVERY_THRESHOLDS.support_f1_ude == 0.50", text)
end

function identifiability_product_module_include_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "BioDynaX.jl"), String)
    tests = read(joinpath(pkgdir(BioDynaX), "test", "runtests.jl"), String)
    return occursin("include(\"IdentifiabilityProduct.jl\")", src) &&
           occursin("test_identifiability_product.jl", tests)
end

function unique_claim_product_blocks_hold_on_join()
    trade = (;
        production_param = :k_prod,
        condition_number = 12.0,
        production_correlation = 0.2,
        collinearity = 0.2,
        unidentifiable_edge = true,
        fisher = nothing)
    join = join_tradeoff_protocol_row(:blocks, trade)
    blocks = unique_claim_product_blocks()
    pos = protocol_block_positions(join.text)
    starts = [range === nothing ? nothing : first(range) for range in values(pos)]
    return (;
        blocks,
        starts,
        order = protocol_block_order_holds(join.text),
        holds = join.holds && blocks == UNIQUE_CLAIM_PRODUCT_BLOCKS &&
                protocol_block_order_holds(join.text) &&
                starts[1] < starts[2] < starts[3] < starts[4])
end

function recovery_thresholds_untouched_row()
    lock = recovery_thresholds_lock()
    return (;
        ude = RECOVERY_THRESHOLDS.support_f1_ude,
        clean = RECOVERY_THRESHOLDS.support_f1_clean,
        residual = RECOVERY_THRESHOLDS.data_residual,
        recall = RECOVERY_THRESHOLDS.support_recall,
        holds = RECOVERY_THRESHOLDS == lock &&
                lock.support_f1_ude == 0.50 &&
                lock.support_f1_clean == 0.99 &&
                lock.data_residual == 0.30 &&
                lock.support_recall == 0.99)
end

function identifiability_product_contract_holds()
    return identifiability_product_source_holds() &&
           production_destruction_tradeoff_source_holds() &&
           format_production_destruction_warning_source_holds() &&
           format_protocol_result_collinearity_source_holds() &&
           coefficients_are_biological_constants_source_holds() &&
           identifiability_product_docs_hold() &&
           identifiability_product_landing_docs_hold() &&
           identifiability_product_example_source_holds() &&
           identifiability_product_docs_mention_helpers() &&
           identifiability_product_index_holds() &&
           identifiability_product_test_file_holds() &&
           identifiability_product_module_include_holds() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source() &&
           unique_claim_product_blocks_hold_on_join().holds &&
           recovery_thresholds_untouched_row().holds
end
