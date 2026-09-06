###############################################################################
# Denominator / domain safety (not exported).
#
# denominator_violation_count was ImplicitCandidate-only. This file locks
# train / validation / orthant-domain split counts, ExplicitCandidate = 0,
# failed discovery = typemax, and the UDE extras path that still walks
# the domain grid. Combined F1 stays a skeleton floor.
#
# Does not drop protocol ICs. Does not grow exports. Does not open
# Hill-from-NN. Does not put a single-unknown-term check into validate_network.
###############################################################################

const DENOMINATOR_DOMAIN_MUST_CONTAIN = (
    "function denominator_split_counts",
    "function ude_extras_denominator_row",
    "function synthetic_safe_implicit_candidate",
    "function synthetic_unsafe_implicit_candidate",
    "function extras_denominator_live_row",
    "struct DenominatorDomainRow",
    "function domain_grid_nonneg_row",
    "function explicit_candidate_zero_violations_row")

const DENOMINATOR_DOMAIN_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function recovery_jl_source_path_for_denominator()
    joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
end

function discovery_jl_source_path_for_denominator()
    joinpath(pkgdir(BioDynaX), "src", "Discovery.jl")
end

# -- Synthetic candidates -----------------------------------------------------

function unit_rate_basis_spec()
    num = [
        MonomialTerm(Int[], Int[], "1"),
        MonomialTerm([1], [1], "x[1]"),
        MonomialTerm([1], [2], "x[1]^2")
    ]
    return LocalBasisSpec(1, [1], num, num[2:end])
end

function two_state_basis_spec()
    num = [
        MonomialTerm(Int[], Int[], "1"),
        MonomialTerm([1], [1], "x[1]"),
        MonomialTerm([2], [1], "x[2]"),
        MonomialTerm([1], [2], "x[1]^2"),
        MonomialTerm([2], [2], "x[2]^2")
    ]
    return LocalBasisSpec(1, [1, 2], num, num[2:end])
end

"""Safe implicit candidate: denominator is identically 1."""
function synthetic_safe_implicit_candidate()
    spec = unit_rate_basis_spec()
    return ImplicitCandidate(
        1, spec,
        Float64[0.0, 1.0, 0.0],
        Float64[0.0, 0.0],
        ones(Float64, 5),
        0.0, 1.0)
end

"""Unsafe implicit candidate: denominator = 1 - 2 r, negative for r > 0.5."""
function synthetic_unsafe_implicit_candidate()
    spec = unit_rate_basis_spec()
    return ImplicitCandidate(
        1, spec,
        Float64[0.0, 1.0, 0.0],
        Float64[-2.0, 0.0],
        ones(Float64, 5),
        0.0, -1.0)
end

"""Near-zero implicit candidate: denominator = 1 - r, singular at r = 1."""
function synthetic_near_zero_implicit_candidate()
    spec = unit_rate_basis_spec()
    return ImplicitCandidate(
        1, spec,
        Float64[0.0, 1.0, 0.0],
        Float64[-1.0, 0.0],
        ones(Float64, 5),
        0.0, 0.0)
end

function synthetic_explicit_candidate()
    spec = unit_rate_basis_spec()
    return ExplicitCandidate(1, spec, Float64[0.0, 1.0, 0.0], 0.0)
end

function synthetic_two_state_unsafe_candidate()
    spec = two_state_basis_spec()
    den = zeros(Float64, length(spec.denominator))
    den[1] = -3.0
    return ImplicitCandidate(
        1, spec,
        zeros(Float64, length(spec.numerator)),
        den,
        ones(Float64, length(spec.numerator) + length(spec.denominator)),
        0.0, -1.0)
end

function regulator_grid(n::Int = 40; lo = 0.1, hi = 2.0)
    return reshape(collect(range(lo, hi; length = n)), 1, :)
end

function two_state_grid(n::Int = 40; seed::Integer = 7)
    rng = MersenneTwister(seed)
    return 0.15 .+ 1.6 .* rand(rng, 2, n)
end

function split_train_val(X::AbstractMatrix; fraction = 0.2)
    n = size(X, 2)
    n_val = n ≤ 2 ? 0 : clamp(round(Int, fraction * n), 1, n - 1)
    train = n_val == 0 ? X : X[:, 1:(n - n_val)]
    val = n_val == 0 ? X : X[:, (n - n_val + 1):n]
    return train, val
end

# -- Core rows ----------------------------------------------------------------

function explicit_candidate_zero_violations_row()
    cand = synthetic_explicit_candidate()
    X = regulator_grid(40)
    domain = _denominator_domain_grid(X; n = 16)
    train, val = split_train_val(X)
    split = denominator_split_counts(cand, train, val, domain)
    raw = denominator_violation_count(cand, X)
    return (;
        raw,
        split,
        holds = raw == 0 && split.total == 0 && !split.any)
end

function missing_candidate_typemax_row()
    X = regulator_grid(20)
    raw = denominator_violation_count(nothing, X)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 8)
    split = denominator_split_counts(nothing, train, val, domain)
    return (;
        raw,
        train = split.train,
        val = split.val,
        domain = split.domain,
        holds = raw == typemax(Int) &&
                split.train == typemax(Int) &&
                split.val == typemax(Int) &&
                split.domain == typemax(Int) &&
                split.any)
end

function safe_split_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(40)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 16)
    split = denominator_split_counts(cand, train, val, domain)
    raw = denominator_violation_count(cand, X)
    return (;
        raw,
        split,
        holds = raw == 0 && split.total == 0 && !split.any &&
                split.train == 0 && split.val == 0 && split.domain == 0)
end

function unsafe_split_row()
    cand = synthetic_unsafe_implicit_candidate()
    X = regulator_grid(40)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 16)
    split = denominator_split_counts(cand, train, val, domain)
    raw = denominator_violation_count(cand, X)
    return (;
        raw,
        split,
        holds = raw > 0 && split.any && split.train > 0)
end

function near_zero_split_row()
    cand = synthetic_near_zero_implicit_candidate()
    X = reshape(collect(range(0.2, 1.5; length = 40)), 1, :)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 24)
    split = denominator_split_counts(cand, train, val, domain)
    raw = denominator_violation_count(cand, X)
    return (;
        raw,
        split,
        holds = raw > 0 && split.any)
end

function two_state_unsafe_split_row()
    cand = synthetic_two_state_unsafe_candidate()
    X = two_state_grid(48)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 16)
    split = denominator_split_counts(cand, train, val, domain)
    return (;
        split,
        nstates = size(X, 1),
        holds = split.any && size(X, 1) == 2)
end

function extras_denominator_live_row()
    cand = synthetic_unsafe_implicit_candidate()
    truth = hill_rate_support(2)
    extras = discovered_support_extras(
        cand, truth.numerator, truth.denominator)
    X = regulator_grid(40)
    row = ude_extras_denominator_row(cand, X; extras = extras)
    return (;
        extras,
        extras_label = row.extras_label,
        extras_live = row.extras_live,
        hardcoded = row.hardcoded,
        any = row.any,
        holds = row.holds && row.extras_live && row.any &&
                row.hardcoded == false &&
                extras_print_is_hardcoded_attempt(row.extras_label) == false)
end

function extras_empty_still_splits_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(36)
    row = ude_extras_denominator_row(cand, X; extras = String[])
    return (;
        extras_label = row.extras_label,
        extras_live = row.extras_live,
        n_domain = row.n_domain,
        holds = row.holds && row.extras_label == "(none)" &&
                row.extras_live == false && row.n_domain == 32 &&
                row.total == 0)
end

function extras_nothing_is_na_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(24)
    row = ude_extras_denominator_row(cand, X; extras = nothing)
    return (;
        extras_label = row.extras_label,
        holds = row.holds && row.extras_label == "NA" &&
                row.extras_live == false &&
                extras_print_is_hardcoded_attempt(row.extras_label) == false)
end

function extras_hardcoded_attempt_rejected_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(24)
    label = extras_print_label("1, r remain after the UDE F1 attempt")
    live = extras_print_label(("1", "r"))
    return (;
        label,
        live,
        attempt = extras_print_is_hardcoded_attempt(label),
        live_ok = extras_print_is_hardcoded_attempt(live) == false,
        holds = extras_print_is_hardcoded_attempt(label) &&
                extras_print_is_hardcoded_attempt(live) == false)
end

# -- Domain grid checks ------------------------------------------------------

function domain_grid_nonneg_row()
    X = [0.2 0.4 0.8; 0.1 0.3 0.5]
    grid = _denominator_domain_grid(X; n = 32, seed = 42)
    return (;
        size = size(grid),
        nonneg = all(≥(0), grid),
        holds = size(grid) == (2, 32) && all(≥(0), grid))
end

function domain_grid_disabled_row()
    X = regulator_grid(10)
    grid = _denominator_domain_grid(X; n = 0)
    return (;
        rows = size(grid, 1),
        cols = size(grid, 2),
        holds = size(grid, 1) == 1 && size(grid, 2) == 0)
end

function domain_grid_clips_negative_pad_row()
    X = reshape(Float64[0.0, 0.01, 0.02], 1, :)
    grid = _denominator_domain_grid(X; n = 16, seed = 3)
    return (;
        lo = minimum(grid),
        holds = all(≥(0), grid) && size(grid) == (1, 16))
end

function domain_grid_spans_observed_row()
    X = reshape(collect(range(0.4, 1.6; length = 20)), 1, :)
    grid = _denominator_domain_grid(X; n = 20, seed = 9)
    return (;
        gmin = minimum(grid),
        gmax = maximum(grid),
        xmin = minimum(X),
        xmax = maximum(X),
        holds = minimum(grid) ≥ 0 &&
                minimum(grid) ≤ minimum(X) &&
                maximum(grid) ≥ maximum(X))
end

function domain_grid_seed_reproducible_row()
    X = two_state_grid(12; seed = 1)
    a = _denominator_domain_grid(X; n = 10, seed = 11)
    b = _denominator_domain_grid(X; n = 10, seed = 11)
    c = _denominator_domain_grid(X; n = 10, seed = 12)
    return (;
        same = a == b,
        different = a != c,
        holds = a == b && a != c)
end

function implicit_safety_rejects_unsafe_row()
    cand = synthetic_unsafe_implicit_candidate()
    X = regulator_grid(30)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 12)
    threw = try
        _check_denominator_safety(
            cand.specification,
            cand.numerator_coefficients,
            cand.denominator_coefficients,
            train, val, domain, 1e-8)
        false
    catch err
        err isa DomainError
    end
    return (; threw, holds = threw)
end

function implicit_safety_accepts_safe_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(30)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 12)
    minimum_value = _check_denominator_safety(
        cand.specification,
        cand.numerator_coefficients,
        cand.denominator_coefficients,
        train, val, domain, 1e-8)
    return (;
        minimum_value,
        holds = minimum_value ≥ 1e-8)
end

# -- Typed row ----------------------------------------------------------------

struct DenominatorDomainRow
    name::Symbol
    train::Int
    val::Int
    domain::Int
    extras_live::Bool
    holds::Bool
end

function denominator_domain_row(name::Symbol, candidate, X;
        extras = String[], domain_samples::Int = 16)
    row = ude_extras_denominator_row(
        candidate, X; extras = extras, domain_samples = domain_samples)
    typed = DenominatorDomainRow(
        name, row.train, row.val, row.domain, row.extras_live, row.holds)
    return (; row, typed, holds = row.holds && typed.holds)
end

function denominator_domain_row_namedtuple(row::DenominatorDomainRow)
    return (;
        name = row.name,
        train = row.train,
        val = row.val,
        domain = row.domain,
        extras_live = row.extras_live,
        holds = row.holds)
end

# -- Fixture rows (compile-free grids on real libraries) ----------------------

function fixture_safe_library_row(name::Symbol, network::BiologicalNetwork;
        target::Int = 1, n::Int = 36)
    spec = local_basis(network, target; degree = 2,
        include_interactions = false, scope = :graph)
    cand = ImplicitCandidate(
        target, spec,
        zeros(Float64, length(spec.numerator)),
        zeros(Float64, length(spec.denominator)),
        ones(Float64, length(spec.numerator) + length(spec.denominator)),
        0.0, 1.0)
    nstates = length(state_nodes(network))
    rng = MersenneTwister(hash((name, target)))
    X = 0.2 .+ 1.4 .* rand(rng, nstates, n)
    packed = denominator_domain_row(name, cand, X)
    return (;
        packed,
        nstates,
        n_terms = candidate_count(spec),
        validate_open = validate_network(network) === network,
        holds = packed.holds && packed.row.total == 0 &&
                validate_network(network) === network)
end

function hill_unknown_denominator_row()
    return fixture_safe_library_row(
        :hill, build_hill_recovery_network(;
            known = false, hill_order = 2))
end

function hill_known_denominator_row()
    return fixture_safe_library_row(
        :hill_known, build_hill_recovery_network(;
            known = true, hill_order = 2))
end

function mm_unknown_denominator_row()
    return fixture_safe_library_row(:mm, build_mm_recovery_network(;
        known = false))
end

function mm_known_denominator_row()
    return fixture_safe_library_row(:mm_known, build_mm_recovery_network(;
        known = true))
end

function two_regulator_denominator_row()
    return fixture_safe_library_row(:two, build_two_regulator_unknown_network())
end

function three_state_denominator_row()
    return fixture_safe_library_row(:three, build_three_state_unknown_network())
end

function wrong_graph_denominator_row()
    return fixture_safe_library_row(:wrong, build_wrong_graph_unknown_network())
end

function six_state_denominator_row()
    return fixture_safe_library_row(:six, build_six_state_unknown_network())
end

function six_state_wrong_denominator_row()
    return fixture_safe_library_row(
        :six_wrong, build_six_state_wrong_graph_network())
end

function default_example_denominator_row()
    return fixture_safe_library_row(:default, DEFAULT_EXAMPLE_NETWORK)
end

function remapped_denominator_row()
    net = build_remapped_two_regulator_network()
    n = length(state_nodes(net))
    rows = [fixture_safe_library_row(Symbol(:remap_, t), net; target = t)
            for t in 1:n]
    return (;
        n = length(rows),
        holes = count_unknown_destructions(net),
        holds = all(r -> r.holds, rows) &&
                count_unknown_destructions(net) == 2 &&
                reference_protocol_recovery_admits(net) == false)
end

function dual_denominator_row()
    net = build_dual_unknown_network()
    packed = fixture_safe_library_row(:dual, net)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        holds = packed.holds && count_unknown_destructions(net) == 2 &&
                reference_protocol_recovery_admits(net) == false)
end

function linear_zero_denominator_row()
    return fixture_safe_library_row(:linear, build_linear_test_network())
end

function competitive_denominator_row()
    return fixture_safe_library_row(
        :competitive, build_competitive_test_network(; known = false))
end

function skipped_duplicate_denominator_row()
    return fixture_safe_library_row(
        :skipped, build_skipped_duplicate_unknown_network())
end

function skipped_middle_denominator_row()
    net = build_skipped_middle_unknown_network()
    packed = fixture_safe_library_row(:middle, net)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        holds = packed.holds && count_unknown_destructions(net) ≥ 2)
end

function repressilator_denominator_row()
    return fixture_safe_library_row(:repress, build_repressilator_network())
end

function kinetic_denominator_row()
    return fixture_safe_library_row(
        :kinetic, build_kinetic_generalization_network())
end

function ablation_denominator_row()
    return fixture_safe_library_row(:ablation, build_rate_ablation_network())
end

function three_state_no_distractor_denominator_row()
    return fixture_safe_library_row(
        :three_nodist, build_three_state_unknown_network(;
            with_distractor = false))
end

# -- UDE extras path on live discovery (1-IC smoke, not protocol) -------------

function extras_on_hill_truth_discovery_row()
    r = collect(range(0.1, 2.0; length = 80))
    D = reshape(hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2), 1, :)
    R = reshape(r, 1, :)
    times = collect(range(0.0, 1.0; length = 80))
    discovery = discover_unknown_rate(
        R, times, D;
        config = reference_protocol_discovery_config(),
        verbose = false, strict = false)
    if !discovery.success || isempty(discovery.candidates)
        return (;
            success = discovery.success,
            holds = discovery.success == false)
    end
    cand = discovery.candidates[1]
    extras = discovered_support_extras(
        cand, hill_rate_support(2).numerator, hill_rate_support(2).denominator)
    row = ude_extras_denominator_row(cand, R; extras = extras)
    return (;
        success = true,
        extras,
        extras_live = row.extras_live,
        any = row.any,
        total = row.total,
        n_ics = 1,
        holds = row.holds && row.n_domain == 32 &&
                extras_print_is_hardcoded_attempt(row.extras_label) == false)
end

function extras_on_mm_truth_discovery_row()
    r = collect(range(0.1, 2.0; length = 80))
    D = reshape(mm_rate_truth(r; vmax = 1.4, km = 0.45), 1, :)
    R = reshape(r, 1, :)
    times = collect(range(0.0, 1.0; length = 80))
    discovery = discover_unknown_rate(
        R, times, D;
        config = reference_protocol_discovery_config(),
        verbose = false, strict = false)
    if !discovery.success || isempty(discovery.candidates)
        return (;
            success = discovery.success,
            holds = discovery.success == false)
    end
    cand = discovery.candidates[1]
    extras = discovered_support_extras(
        cand, mm_rate_support().numerator, mm_rate_support().denominator)
    row = ude_extras_denominator_row(cand, R; extras = extras)
    return (;
        success = true,
        extras,
        total = row.total,
        holds = row.holds && extras_print_is_hardcoded_attempt(row.extras_label) == false)
end

function smoke_vs_protocol_denominator_row()
    smoke = reference_protocol_fingerprint(; smoke = true)
    proto = reference_protocol_fingerprint()
    return (;
        smoke_ics = smoke.n_ics,
        proto_ics = proto.n_ics,
        proto_points = proto.n_points,
        proto_seed = proto.seed,
        holds = smoke.n_ics == 1 && proto.n_ics == 9 &&
                proto.n_points == 50 && proto.seed == 103 &&
                !proto.smoke && smoke.smoke)
end

# -- Source locks -------------------------------------------------------------

# -- Matrices / catalog -------------------------------------------------------

function denominator_domain_fixture_names()
    return (
        :explicit_zero, :missing_typemax, :safe_split, :unsafe_split,
        :near_zero, :two_state_unsafe, :extras_live, :extras_empty,
        :extras_na, :hardcoded, :grid_nonneg, :grid_disabled,
        :grid_clip, :grid_span, :grid_seed, :safety_unsafe,
        :safety_safe, :hill, :hill_known, :mm, :mm_known, :two,
        :three, :wrong, :six, :six_wrong, :default, :remap, :dual,
        :linear, :competitive, :skipped, :middle, :repress, :kinetic,
        :ablation, :three_nodist, :smoke_protocol)
end

function denominator_domain_typed_matrix()
    safe = denominator_domain_row(
        :safe, synthetic_safe_implicit_candidate(), regulator_grid(24))
    unsafe = denominator_domain_row(
        :unsafe, synthetic_unsafe_implicit_candidate(), regulator_grid(24);
        extras = ["1"])
    explicit = denominator_domain_row(
        :explicit, synthetic_explicit_candidate(), regulator_grid(24))
    return (;
        safe = denominator_domain_row_namedtuple(safe.typed),
        unsafe = denominator_domain_row_namedtuple(unsafe.typed),
        explicit = denominator_domain_row_namedtuple(explicit.typed),
        holds = safe.holds && unsafe.holds && explicit.holds &&
                safe.typed.train == 0 && unsafe.typed.train > 0 &&
                explicit.typed.train == 0 && unsafe.typed.extras_live)
end

function format_denominator_domain_index()
    io = IOBuffer()
    println(io, "| row | meaning |")
    println(io, "|---|---|")
    println(io, "| explicit_zero | ExplicitCandidate violations = 0 |")
    println(io, "| missing_typemax | nothing records typemax |")
    println(io, "| safe_split | identically-1 denominator is clean |")
    println(io, "| unsafe_split | 1 - 2r is singular on the grid |")
    println(io, "| near_zero | 1 - r crosses the floor |")
    println(io, "| two_state_unsafe | two-row domain grid still splits |")
    println(io, "| extras_live | leftover monomials still walk the grid |")
    println(io, "| extras_empty | (none) still allocates the domain grid |")
    println(io, "| extras_na | unscored extras print NA |")
    println(io, "| hardcoded | F1-attempt leftover string stays rejected |")
    println(io, "| grid_nonneg | orthant grid stays non-negative |")
    println(io, "| grid_disabled | domain_samples = 0 is empty |")
    println(io, "| grid_clip | pad does not enter the negative orthant |")
    println(io, "| grid_span | grid covers observed bounds |")
    println(io, "| grid_seed | same seed reproduces the grid |")
    println(io, "| safety_unsafe | _check_denominator_safety throws |")
    println(io, "| safety_safe | identically-1 denominator is admitted |")
    println(io, "| hill | unknown Hill library, zero den coeffs |")
    println(io, "| hill_known | known Hill 0-hole library |")
    println(io, "| mm | unknown MM library |")
    println(io, "| mm_known | known MM 0-hole library |")
    println(io, "| two | two-regulator reaction-only library |")
    println(io, "| three | 3-state graph library |")
    println(io, "| wrong | wrong-graph library |")
    println(io, "| six | 6-state graph library |")
    println(io, "| six_wrong | 6-state wrong-graph library |")
    println(io, "| default | p53/Mdm2 INPUT + two states |")
    println(io, "| remap | remapped 2-hole per-target libraries |")
    println(io, "| dual | dual unknown does not admit |")
    println(io, "| linear | 0-hole linear library |")
    println(io, "| competitive | competitive unknown library |")
    println(io, "| skipped | skipped-duplicate library |")
    println(io, "| middle | remapped 1:n skipped-middle library |")
    println(io, "| repress | repressilator 0-hole library |")
    println(io, "| kinetic | known kinetic 0-hole library |")
    println(io, "| ablation | rate-ablation library |")
    println(io, "| three_nodist | 3-state without Z |")
    println(io, "| smoke_protocol | 1-IC smoke is not 9 ICs / 50 points |")
    return String(take!(io))
end

function suite_section_denominator_catalog()
    rows = NamedTuple[]
    for section in recovery_suite_sections()
        net = recovery_suite_section_network(section)
        n_dyn = length(state_nodes(net))
        for target in 1:n_dyn
            packed = fixture_safe_library_row(
                Symbol(section, :_, target), net; target = target, n = 24)
            push!(rows, (;
                section,
                target,
                n_terms = packed.n_terms,
                holds = packed.holds))
        end
    end
    return (;
        n = length(rows),
        rows,
        holds = !isempty(rows) && all(r -> r.holds, rows) &&
                length(unique(r.section for r in rows)) ==
                length(recovery_suite_sections()))
end

function format_suite_denominator_catalog()
    catalog = suite_section_denominator_catalog()
    io = IOBuffer()
    println(io, "| section | target | n_terms |")
    println(io, "|---|---|---|")
    for row in catalog.rows
        println(io, "| ", row.section, " | ", row.target, " | ",
            row.n_terms, " |")
    end
    return String(take!(io))
end

function suite_denominator_catalog_holds()
    catalog = suite_section_denominator_catalog()
    text = format_suite_denominator_catalog()
    return catalog.holds &&
           occursin("three_state", text) &&
           occursin("wrong_graph", text) &&
           count(==('|'), text) ≥ 40 &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Source checks ----------------------------------------------------------

function reference_protocol_not_faster_by_dropping_ics_denominator_row()
    fp = reference_protocol_fingerprint()
    ics = reference_protocol_protocol_ics()
    return (;
        n_ics = fp.n_ics,
        n_table = length(ics),
        holds = fp.n_ics == 9 && length(ics) == 9 &&
                fp.n_points == 50 && fp.seed == 103 && !fp.smoke)
end

function empty_domain_split_is_train_val_only_row()
    cand = synthetic_unsafe_implicit_candidate()
    X = regulator_grid(30)
    train, val = split_train_val(X)
    empty = _denominator_domain_grid(X; n = 0)
    split = denominator_split_counts(cand, train, val, empty)
    return (;
        domain = split.domain,
        train = split.train,
        holds = split.domain == 0 && split.train > 0)
end

function discovery_config_domain_samples_row()
    cfg = reference_protocol_discovery_config()
    return (;
        n = cfg.backend.domain_samples,
        floor = cfg.backend.denominator_floor,
        backend = nameof(typeof(cfg.backend)),
        holds = cfg.backend isa ImplicitSINDyPI &&
                cfg.backend.domain_samples == 32 &&
                cfg.backend.denominator_floor == 1e-8)
end

function default_backend_domain_samples_row()
    backend = ImplicitSINDyPI()
    return (;
        n = backend.domain_samples,
        holds = backend.domain_samples == 256 &&
                backend.denominator_floor == 1e-8)
end

function floor_sensitivity_row()
    cand = synthetic_near_zero_implicit_candidate()
    X = reshape(Float64[0.999999, 1.0, 1.000001], 1, :)
    loose = denominator_violation_count(cand, X; floor = 1e-3)
    tight = denominator_violation_count(cand, X; floor = 1e-12)
    return (;
        loose, tight,
        holds = loose ≥ tight && loose ≥ 1)
end

function split_matches_sum_row()
    cand = synthetic_unsafe_implicit_candidate()
    X = regulator_grid(40)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 16)
    split = denominator_split_counts(cand, train, val, domain)
    return (;
        split,
        holds = split.total == split.train + split.val + split.domain &&
                split.any == (split.total > 0))
end

function extra_candidates_keep_denominator_row()
    net = build_rate_ablation_network()
    r = collect(range(0.1, 2.0; length = 40))
    X = permutedims(hcat(r, r .^ 2))
    y = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    spec0 = local_basis(net, 1; degree = 2, include_interactions = false,
        scope = :graph, X = X, derivative = y, extra_candidates = 0)
    spec1 = local_basis(net, 1; degree = 2, include_interactions = false,
        scope = :graph, X = X, derivative = y, extra_candidates = 1)
    cand0 = ImplicitCandidate(
        1, spec0,
        zeros(Float64, length(spec0.numerator)),
        zeros(Float64, length(spec0.denominator)),
        ones(Float64, length(spec0.numerator) + length(spec0.denominator)),
        0.0, 1.0)
    cand1 = ImplicitCandidate(
        1, spec1,
        zeros(Float64, length(spec1.numerator)),
        zeros(Float64, length(spec1.denominator)),
        ones(Float64, length(spec1.numerator) + length(spec1.denominator)),
        0.0, 1.0)
    n0 = denominator_violation_count(cand0, X)
    n1 = denominator_violation_count(cand1, X)
    return (;
        n0, n1,
        wider = candidate_count(spec1) ≥ candidate_count(spec0),
        holds = n0 == 0 && n1 == 0 &&
                candidate_count(spec1) ≥ candidate_count(spec0))
end

function six_state_per_target_denominator_row()
    net = build_six_state_unknown_network()
    n = length(state_nodes(net))
    rows = [fixture_safe_library_row(Symbol(:six_, t), net; target = t, n = 20)
            for t in 1:n]
    return (;
        n,
        holds = n == 6 && all(r -> r.holds, rows))
end

function default_per_target_denominator_row()
    net = DEFAULT_EXAMPLE_NETWORK
    n = length(state_nodes(net))
    rows = [fixture_safe_library_row(Symbol(:def_, t), net; target = t)
            for t in 1:n]
    return (;
        n,
        holds = n == 2 && all(r -> r.holds, rows))
end

function remapped_per_target_denominator_row()
    net = build_remapped_two_regulator_network()
    n = length(state_nodes(net))
    rows = [fixture_safe_library_row(Symbol(:rm_, t), net; target = t, n = 20)
            for t in 1:n]
    return (;
        n,
        holes = count_unknown_destructions(net),
        holds = all(r -> r.holds, rows) &&
                count_unknown_destructions(net) == 2)
end

function format_split_markdown(split)
    io = IOBuffer()
    println(io, "| slice | violations |")
    println(io, "|---|---|")
    println(io, "| train | ", split.train, " |")
    println(io, "| val | ", split.val, " |")
    println(io, "| domain | ", split.domain, " |")
    println(io, "| total | ", split.total, " |")
    return String(take!(io))
end

function format_split_markdown_holds()
    cand = synthetic_unsafe_implicit_candidate()
    X = regulator_grid(24)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 8)
    split = denominator_split_counts(cand, train, val, domain)
    text = format_split_markdown(split)
    return occursin("train", text) &&
           occursin("domain", text) &&
           occursin(string(split.total), text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function combined_f1_not_a_denominator_kpi_row()
    return (;
        fields = REFERENCE_PROTOCOL_KPI_FIELDS,
        floor = RECOVERY_THRESHOLDS.support_f1_ude,
        holds = :support_f1 ∉ REFERENCE_PROTOCOL_KPI_FIELDS &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50 &&
                RECOVERY_THRESHOLDS.support_f1_clean == 0.99)
end

function single_sample_split_row()
    cand = synthetic_safe_implicit_candidate()
    X = reshape(Float64[0.4], 1, :)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 4)
    split = denominator_split_counts(cand, train, val, domain)
    return (;
        n_train = size(train, 2),
        n_val = size(val, 2),
        split,
        holds = split.total == 0 && size(train, 2) == 1)
end

function all_zero_state_grid_row()
    cand = synthetic_safe_implicit_candidate()
    X = zeros(Float64, 1, 12)
    raw = denominator_violation_count(cand, X)
    domain = _denominator_domain_grid(X; n = 8)
    return (;
        raw,
        domain_nonneg = all(≥(0), domain),
        holds = raw == 0 && all(≥(0), domain))
end
