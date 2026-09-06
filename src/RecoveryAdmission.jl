###############################################################################
# Recovery-suite admission and protocol row (not exported).
#
# Reference-protocol suite sections admit a network only when exactly one unknown
# D(z) is present. validate_network stays a topology/metadata checker.
# ReferenceProtocolRow joins ReferenceProtocolFingerprint, protocol_result,
# extras_print_label, and named KPI failures.
###############################################################################

"""Kind of each `run_recovery_suite` section. `:reference_protocol` requires exactly one unknown term."""
const RECOVERY_SUITE_SECTION_KINDS = (
    linear = :known_kinetics,
    mm = :known_kinetics,
    hill = :known_kinetics,
    competitive = :known_kinetics,
    ude_discovery = :reference_protocol,
    mm_unknown = :reference_protocol,
    ablation = :analytical,
    three_state = :graph_prior,
    wrong_graph = :graph_prior,
    six_state = :graph_prior,
    six_state_wrong_graph = :graph_prior,
    identifiability = :identifiability,
    ident_interventions = :reference_protocol,
    partial_obs = :reference_protocol,
    competitive_unknown = :analytical,
    literature = :literature)

const RECOVERY_SUITE_REFERENCE_PROTOCOL_SECTIONS = (
    :ude_discovery, :mm_unknown, :ident_interventions, :partial_obs)

const RECOVERY_SUITE_KNOWN_KINETICS_SECTIONS = (
    :linear, :mm, :hill, :competitive)

"""Expected unknown-`D(z)` count of each suite fixture. `:ablation` does not compile."""
const RECOVERY_SUITE_EXPECTED_HOLES = (
    linear = 0,
    mm = 0,
    hill = 0,
    competitive = 0,
    ude_discovery = 1,
    mm_unknown = 1,
    ablation = nothing,
    three_state = 1,
    wrong_graph = 1,
    six_state = 1,
    six_state_wrong_graph = 1,
    identifiability = 0,
    ident_interventions = 1,
    partial_obs = 1,
    competitive_unknown = 1,
    literature = 0)

"""Hole policy. Only `:exactly_one` rejects 0/2 holes before training."""
const RECOVERY_SUITE_HOLE_POLICY = (
    linear = :open,
    mm = :open,
    hill = :open,
    competitive = :open,
    ude_discovery = :exactly_one,
    mm_unknown = :exactly_one,
    ablation = :library_fixture,
    three_state = :open,
    wrong_graph = :open,
    six_state = :open,
    six_state_wrong_graph = :open,
    identifiability = :open,
    ident_interventions = :exactly_one,
    partial_obs = :exactly_one,
    competitive_unknown = :open,
    literature = :open)

function recovery_suite_sections()
    return Tuple(keys(RECOVERY_SUITE_SECTION_KINDS))
end

function recovery_suite_expected_holes(section::Symbol)
    holes = RECOVERY_SUITE_EXPECTED_HOLES
    hasproperty(holes, section) || throw(ArgumentError(
        "unknown recovery suite section $section"))
    return getproperty(holes, section)
end

function recovery_suite_hole_policy(section::Symbol)
    policies = RECOVERY_SUITE_HOLE_POLICY
    hasproperty(policies, section) || throw(ArgumentError(
        "unknown recovery suite section $section"))
    return getproperty(policies, section)
end

function recovery_suite_section_compiles(section::Symbol)
    recovery_suite_hole_policy(section) !== :library_fixture
end

function recovery_suite_admits_hole_count(section::Symbol, n::Integer)
    policy = recovery_suite_hole_policy(section)
    policy === :exactly_one && return n == 1
    policy === :library_fixture && return false
    return true
end

function recovery_suite_section_kinds()
    return RECOVERY_SUITE_SECTION_KINDS
end

function recovery_suite_reference_protocol_sections()
    return RECOVERY_SUITE_REFERENCE_PROTOCOL_SECTIONS
end

function recovery_suite_section_kind(section::Symbol)
    kinds = RECOVERY_SUITE_SECTION_KINDS
    hasproperty(kinds, section) || throw(ArgumentError(
        "unknown recovery suite section $section"))
    return getproperty(kinds, section)
end

function recovery_suite_section_requires_single_hole(section::Symbol)
    recovery_suite_section_kind(section) === :reference_protocol
end

"""Compiled fixture the suite uses for `section`, when one exists."""
function recovery_suite_section_network(section::Symbol)
    section === :linear && return build_linear_test_network()
    section === :mm && return build_mm_test_network()
    section === :hill &&
        return build_hill_recovery_network(; known = true, hill_order = 2)
    section === :competitive && return build_competitive_test_network()
    section === :ude_discovery &&
        return build_hill_recovery_network(; known = false, hill_order = 2)
    section === :mm_unknown && return build_mm_recovery_network(; known = false)
    section === :identifiability &&
        return build_hill_recovery_network(; known = true, hill_order = 2)
    section === :ident_interventions &&
        return build_hill_recovery_network(; known = false, hill_order = 2)
    section === :partial_obs &&
        return build_hill_recovery_network(; known = false, hill_order = 2)
    section === :competitive_unknown &&
        return build_competitive_test_network(; known = false)
    section === :literature && return build_repressilator_network(; hill_order = 2)
    section === :ablation && return build_rate_ablation_network()
    section === :three_state && return build_three_state_unknown_network()
    section === :wrong_graph && return build_wrong_graph_unknown_network()
    section === :six_state && return build_six_state_unknown_network()
    section === :six_state_wrong_graph && return build_six_state_wrong_graph_network()
    throw(ArgumentError("section $section has no compiled network fixture"))
end

"""
    admit_recovery_suite_network(section, network = recovery_suite_section_network(section))

`validate_network` always runs. Reference-protocol sections then require exactly
one unknown `D(z)` via `assert_reference_protocol_recovery_network`. Does not
train a UDE.
"""
function admit_recovery_suite_network(section::Symbol,
        network::BiologicalNetwork = recovery_suite_section_network(section))
    validate_network(network)
    if recovery_suite_section_requires_single_hole(section)
        assert_reference_protocol_recovery_network(network)
    end
    return network
end

function recovery_suite_admission_row(section::Symbol,
        network::BiologicalNetwork = recovery_suite_section_network(section))
    kind = recovery_suite_section_kind(section)
    policy = recovery_suite_hole_policy(section)
    compiles = recovery_suite_section_compiles(section)
    n = if compiles
        count_unknown_destructions(network)
    else
        nothing
    end
    validate_open = try
        validate_network(network) === network &&
            (compiles ? reference_protocol_compiler_stays_open(network) : true)
    catch
        false
    end
    requires_hole = policy === :exactly_one
    admitted = try
        admit_recovery_suite_network(section, network)
        true
    catch
        false
    end
    return (;
        section,
        kind,
        policy,
        unknown_holes = n,
        expected_holes = recovery_suite_expected_holes(section),
        requires_single_hole = requires_hole,
        compiles,
        validate_open,
        admitted,
        recovery_admits = n == 1,
        single_hole_in_validate_network = validate_network_stays_open_source() == false)
end

function recovery_suite_rejects_zero_and_dual_holes(section::Symbol = :ude_discovery)
    zero = build_zero_unknown_linear_network()
    two = build_dual_unknown_network()
    one = recovery_suite_section_network(section)
    zero_row = recovery_suite_admission_row(section, zero)
    two_row = recovery_suite_admission_row(section, two)
    one_row = recovery_suite_admission_row(section, one)
    return (;
        section,
        zero = zero_row,
        two = two_row,
        one = one_row,
        holds = zero_row.validate_open && two_row.validate_open &&
                one_row.validate_open &&
                zero_row.admitted == false &&
                two_row.admitted == false &&
                one_row.admitted &&
                zero_row.single_hole_in_validate_network == false)
end

function recovery_suite_known_kinetics_admit_zero_holes()
    rows = [recovery_suite_admission_row(section)
            for section in RECOVERY_SUITE_KNOWN_KINETICS_SECTIONS]
    return (;
        rows,
        holds = all(
            row -> row.admitted && row.validate_open &&
                       row.requires_single_hole == false,
            rows))
end

"""
    recovery_suite_admission_matrix()

One row per `run_recovery_suite` section: kind, hole policy, fixture
hole count, validate-open, admission. Does not train a UDE.
"""
function recovery_suite_admission_matrix()
    rows = [recovery_suite_admission_row(section)
            for section in recovery_suite_sections()]
    fixture_holds = all(rows) do row
        row.unknown_holes == row.expected_holes &&
            row.single_hole_in_validate_network == false &&
            (row.compiles ? row.validate_open : row.policy === :library_fixture) &&
            (row.policy === :exactly_one ? row.admitted == (row.unknown_holes == 1) :
             row.policy === :library_fixture ? row.admitted :
             row.admitted)
    end
    return (;
        rows,
        n_sections = length(rows),
        reference_protocol = count(row -> row.kind === :reference_protocol, rows),
        holds = fixture_holds && length(rows) == length(RECOVERY_SUITE_SECTION_KINDS))
end

"""
    recovery_suite_zero_dual_matrix()

0-hole and 2-hole probes on every section. Reference-protocol sections reject
both without training. Open sections still admit. `validate_network`
stays open on both probes. Ablation is a library fixture and is not
probed with a compiled dual unknown.
"""
function recovery_suite_zero_dual_matrix()
    zero = build_zero_unknown_linear_network()
    two = build_dual_unknown_network()
    rows = []
    for section in recovery_suite_sections()
        policy = recovery_suite_hole_policy(section)
        if policy === :library_fixture
            fixture = recovery_suite_admission_row(section)
            push!(rows,
                (;
                    section,
                    policy,
                    zero = nothing,
                    two = nothing,
                    fixture,
                    holds = fixture.admitted && fixture.compiles == false &&
                                fixture.single_hole_in_validate_network == false))
            continue
        end
        zero_row = recovery_suite_admission_row(section, zero)
        two_row = recovery_suite_admission_row(section, two)
        expect_admit = policy !== :exactly_one
        push!(rows,
            (;
                section,
                policy,
                zero = zero_row,
                two = two_row,
                fixture = recovery_suite_admission_row(section),
                holds = zero_row.validate_open && two_row.validate_open &&
                            zero_row.admitted == expect_admit &&
                            two_row.admitted == expect_admit &&
                            zero_row.unknown_holes == 0 &&
                            two_row.unknown_holes == 2 &&
                            zero_row.single_hole_in_validate_network == false))
    end
    return (;
        rows,
        n_sections = length(rows),
        holds = all(row -> row.holds, rows))
end

function recovery_suite_open_sections_admit_zero_and_dual()
    matrix = recovery_suite_zero_dual_matrix()
    open_rows = [row for row in matrix.rows if row.policy === :open]
    return (;
        rows = open_rows,
        holds = !isempty(open_rows) && all(row -> row.holds, open_rows))
end

function recovery_suite_reference_protocol_sections_reject_zero_and_dual()
    matrix = recovery_suite_zero_dual_matrix()
    claim_rows = [row for row in matrix.rows if row.policy === :exactly_one]
    return (;
        rows = claim_rows,
        holds = length(claim_rows) == length(RECOVERY_SUITE_REFERENCE_PROTOCOL_SECTIONS) &&
                all(
            row -> row.holds && row.zero.admitted == false &&
                       row.two.admitted == false,
            claim_rows))
end

function recovery_suite_uses_admission_helper()
    path = joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
    src = read(path, String)
    return occursin("admit_recovery_suite_network(:ude_discovery)", src) &&
           occursin("admit_recovery_suite_network(:mm_unknown)", src) &&
           occursin("admit_recovery_suite_network(:ident_interventions)", src) &&
           occursin("admit_recovery_suite_network(:partial_obs)", src) &&
           occursin("function run_recovery_suite", src)
end

# -- Named KPI failure symbols ------------------------------------------------

function reference_protocol_kpi_failure_symbols()
    return REFERENCE_PROTOCOL_KPI_FIELDS
end

function reference_protocol_kpi_failure_symbols_hold(failures)
    allowed = reference_protocol_kpi_failure_symbols()
    return all(sym -> sym in allowed, failures) &&
           !(:support_f1 in failures) &&
           !(:canonical_hill_from_nn in failures)
end

function format_reference_protocol_kpi_failures(failures)
    isempty(failures) && return "(none)"
    return join(string.(failures), ", ")
end

function reference_protocol_kpi_failure_message(failures)
    isempty(failures) && return "reference-protocol KPIs pass"
    return "reference-protocol KPIs failed: $(format_reference_protocol_kpi_failures(failures))"
end

function named_kpi_failure_row(;
        unidentifiable_edge = true,
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57)
    kpis = locked_ude_kpis((;
        data_residual,
        support_recall,
        support_f1,
        identifiability = (; unidentifiable_edge)))
    failures = reference_protocol_kpi_failures(kpis)
    return (;
        kpis,
        failures,
        symbols_hold = reference_protocol_kpi_failure_symbols_hold(failures),
        message = reference_protocol_kpi_failure_message(failures),
        label = format_reference_protocol_kpi_failures(failures),
        kpis_hold = reference_protocol_kpis_hold(kpis))
end

# -- Protocol row (fingerprint + result + extras + KPI names) -----------------

"""
    ReferenceProtocolRow

Typed recovery print row. Joins `ReferenceProtocolFingerprint`,
`protocol_result`, live extras label, and named KPI failures.
Not exported. Combined F1 is stored but is not a failure symbol.
"""
struct ReferenceProtocolRow
    fingerprint::ReferenceProtocolFingerprint
    protocol_result::NamedTuple
    kpis::NamedTuple
    kpi_failures::Vector{Symbol}
    extras_label::String
    text::String
end

function reference_protocol_protocol_row(ude;
        fingerprint::ReferenceProtocolFingerprint = reference_protocol_fingerprint(),
        equations = nothing)
    result = hasproperty(ude, :protocol_result) && ude.protocol_result !== nothing ?
             ude.protocol_result : build_protocol_result(ude)
    kpis = hasproperty(ude, :locked_kpis) && ude.locked_kpis !== nothing ?
           ude.locked_kpis : locked_ude_kpis(ude)
    failures = reference_protocol_kpi_failures(kpis)
    text = format_recovery_protocol(ude, fingerprint; equations = equations)
    return ReferenceProtocolRow(
        fingerprint,
        result,
        kpis,
        collect(Symbol, failures),
        extras_print_label(result.extras),
        text)
end

function reference_protocol_protocol_row_namedtuple(row::ReferenceProtocolRow)
    return (;
        kind = row.fingerprint.kind,
        n_ics = row.fingerprint.n_ics,
        n_points = row.fingerprint.n_points,
        is_protocol = reference_protocol_fingerprint_is_protocol(row.fingerprint),
        unknown_holes = row.protocol_result.unknown_holes,
        unidentifiable_edge = row.protocol_result.unidentifiable_edge,
        coefficients_are_biological_constants = row.protocol_result.coefficients_are_biological_constants,
        extras_label = row.extras_label,
        kpi_failures = Tuple(row.kpi_failures),
        claim = row.protocol_result.claim,
        canonical_hill_from_nn = row.protocol_result.canonical_hill_from_nn)
end

function assert_reference_protocol_protocol_row(row::ReferenceProtocolRow)
    reference_protocol_fingerprint_holds(row.fingerprint) || throw(ErrorException(
        "ReferenceProtocolRow fingerprint is not a locked protocol or smoke object"))
    assert_protocol_result_fields(row.protocol_result)
    assert_format_matches_protocol_result(row.protocol_result, row.text)
    extras_print_label(row.protocol_result.extras) == row.extras_label ||
        throw(ErrorException("printed extras label does not match protocol_result"))
    reference_protocol_kpi_failure_symbols_hold(row.kpi_failures) ||
        throw(ErrorException(
            "KPI failures must be named from REFERENCE_PROTOCOL_KPI_FIELDS; got $(row.kpi_failures)"))
    Set(row.kpi_failures) == Set(reference_protocol_kpi_failures(row.kpis)) ||
        throw(ErrorException("stored KPI failures do not match reference_protocol_kpi_failures"))
    extras_print_is_hardcoded_attempt(row.extras_label) && throw(ErrorException(
        "ReferenceProtocolRow must not invent UDE F1-attempt extras"))
    occursin("hybrid_data_residual:", row.text) || throw(ErrorException(
        "protocol row text must print hybrid_data_residual"))
    return row
end

"""Hard-recovery row: protocol fingerprint and empty named KPI failures."""
function assert_reference_protocol_protocol_row_holds(row::ReferenceProtocolRow)
    assert_reference_protocol_protocol_row(row)
    reference_protocol_fingerprint_is_protocol(row.fingerprint) || throw(ErrorException(
        "hard recovery row must use the protocol fingerprint, not smoke"))
    isempty(row.kpi_failures) || throw(ErrorException(
        reference_protocol_kpi_failure_message(row.kpi_failures)))
    row.protocol_result.canonical_hill_from_nn === false || throw(ErrorException(
        "canonical_hill_from_nn must stay false"))
    reference_protocol_f1_reaches_analytical_threshold(row.protocol_result.support_f1) &&
        throw(ErrorException(
            "UDE support_f1 must stay below support_f1_clean; got $(row.protocol_result.support_f1)"))
    return row
end

function reference_protocol_protocol_row_from_fields(;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        unidentifiable_edge = true,
        unknown_holes = 1,
        smoke::Bool = false,
        equations = "D(z) = vmax * r^2 / (K^2 + r^2)")
    ident = (; unidentifiable_edge, production_param = :k_prod)
    ude = (;
        data_residual,
        support_recall,
        support_f1,
        extras,
        identifiability = ident,
        protocol_result = build_protocol_result(
            (;
                data_residual,
                support_recall,
                support_f1,
                extras,
                identifiability = ident);
            unknown_holes),
        locked_kpis = locked_ude_kpis((;
            data_residual,
            support_recall,
            support_f1,
            extras,
            identifiability = ident)))
    return reference_protocol_protocol_row(ude;
        fingerprint = reference_protocol_fingerprint(; smoke),
        equations = equations)
end

function recovery_hard_named_kpi_spec()
    return (;
        symbols = reference_protocol_kpi_failure_symbols(),
        names = (:unidentifiable_edge, :data_residual, :support_recall),
        f1_is_not_a_symbol = !(:support_f1 in REFERENCE_PROTOCOL_KPI_FIELDS),
        empty_label = format_reference_protocol_kpi_failures(Symbol[]),
        miss_edge = format_reference_protocol_kpi_failures(Symbol[:unidentifiable_edge]),
        miss_all = format_reference_protocol_kpi_failures(collect(REFERENCE_PROTOCOL_KPI_FIELDS)))
end
