###############################################################################
# Recovery-suite admission and protocol row (not exported).
#
# Unique-claim suite sections admit a network only when exactly one unknown
# D(z) is present. validate_network stays a topology/metadata checker.
# UniqueClaimProtocolRow joins UniqueClaimFingerprint, protocol_result,
# extras_print_label, and named KPI failures.
###############################################################################

"""Kind of each `run_recovery_suite` section. `:unique_claim` requires one hole."""
const RECOVERY_SUITE_SECTION_KINDS = (
    linear = :known_kinetics,
    mm = :known_kinetics,
    hill = :known_kinetics,
    competitive = :known_kinetics,
    ude_discovery = :unique_claim,
    mm_unknown = :unique_claim,
    ablation = :analytical,
    three_state = :graph_prior,
    wrong_graph = :graph_prior,
    six_state = :graph_prior,
    six_state_wrong_graph = :graph_prior,
    identifiability = :identifiability,
    ident_interventions = :unique_claim,
    partial_obs = :unique_claim,
    competitive_unknown = :analytical,
    literature = :literature)

const RECOVERY_SUITE_UNIQUE_CLAIM_SECTIONS = (
    :ude_discovery, :mm_unknown, :ident_interventions, :partial_obs)

const RECOVERY_SUITE_KNOWN_KINETICS_SECTIONS = (
    :linear, :mm, :hill, :competitive)

function recovery_suite_section_kinds()
    return RECOVERY_SUITE_SECTION_KINDS
end

function recovery_suite_unique_claim_sections()
    return RECOVERY_SUITE_UNIQUE_CLAIM_SECTIONS
end

function recovery_suite_section_kind(section::Symbol)
    kinds = RECOVERY_SUITE_SECTION_KINDS
    hasproperty(kinds, section) || throw(ArgumentError(
        "unknown recovery suite section $section"))
    return getproperty(kinds, section)
end

function recovery_suite_section_requires_single_hole(section::Symbol)
    recovery_suite_section_kind(section) === :unique_claim
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

`validate_network` always runs. Unique-claim sections then require exactly
one unknown `D(z)` via `assert_unique_claim_recovery_network`. Does not
train a UDE.
"""
function admit_recovery_suite_network(section::Symbol,
        network::BiologicalNetwork = recovery_suite_section_network(section))
    validate_network(network)
    if recovery_suite_section_requires_single_hole(section)
        assert_unique_claim_recovery_network(network)
    end
    return network
end

function recovery_suite_admission_row(section::Symbol,
        network::BiologicalNetwork = recovery_suite_section_network(section))
    kind = recovery_suite_section_kind(section)
    n = count_unknown_destructions(network)
    validate_open = unique_claim_compiler_stays_open(network)
    requires_hole = kind === :unique_claim
    admitted = try
        admit_recovery_suite_network(section, network)
        true
    catch
        false
    end
    return (;
        section,
        kind,
        unknown_holes = n,
        requires_single_hole = requires_hole,
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

function recovery_suite_uses_admission_helper()
    path = joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
    src = read(path, String)
    return occursin("admit_recovery_suite_network(:ude_discovery)", src) &&
           occursin("admit_recovery_suite_network(:mm_unknown)", src) &&
           occursin("admit_recovery_suite_network(:ident_interventions)", src) &&
           occursin("admit_recovery_suite_network(:partial_obs)", src) &&
           occursin("function run_recovery_suite", src)
end

function recovery_suite_admission_source_violations()
    path = joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
    src = read(path, String)
    required = (
        "admit_recovery_suite_network(:ude_discovery)",
        "admit_recovery_suite_network(:mm_unknown)",
        "admit_recovery_suite_network(:ident_interventions)",
        "admit_recovery_suite_network(:partial_obs)",
        "only_unknown_destruction")
    forbidden = (
        "validate_network(ude_net); count_unknown_destructions",
        "if unknown_holes != 1; return validate_network")
    missing = [s for s in required if !occursin(s, src)]
    hits = [s for s in forbidden if occursin(s, src)]
    return (; missing, forbidden = hits)
end

# -- Named KPI failure symbols ------------------------------------------------

function unique_claim_kpi_failure_symbols()
    return UNIQUE_CLAIM_KPI_FIELDS
end

function unique_claim_kpi_failure_symbols_hold(failures)
    allowed = unique_claim_kpi_failure_symbols()
    return all(sym -> sym in allowed, failures) &&
           !(:support_f1 in failures) &&
           !(:canonical_hill_from_nn in failures)
end

function format_unique_claim_kpi_failures(failures)
    isempty(failures) && return "(none)"
    return join(string.(failures), ", ")
end

function unique_claim_kpi_failure_message(failures)
    isempty(failures) && return "unique-claim KPIs hold"
    return "unique-claim KPIs failed: $(format_unique_claim_kpi_failures(failures))"
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
    failures = unique_claim_kpi_failures(kpis)
    return (;
        kpis,
        failures,
        symbols_hold = unique_claim_kpi_failure_symbols_hold(failures),
        message = unique_claim_kpi_failure_message(failures),
        label = format_unique_claim_kpi_failures(failures),
        hold = unique_claim_kpis_hold(kpis))
end

# -- Protocol row (fingerprint + result + extras + KPI names) -----------------

"""
    UniqueClaimProtocolRow

Typed recovery print row. Joins `UniqueClaimFingerprint`,
`protocol_result`, live extras label, and named KPI failures.
Not exported. Combined F1 is stored but is not a failure symbol.
"""
struct UniqueClaimProtocolRow
    fingerprint::UniqueClaimFingerprint
    protocol_result::NamedTuple
    kpis::NamedTuple
    kpi_failures::Vector{Symbol}
    extras_label::String
    text::String
end

function unique_claim_protocol_row(ude;
        fingerprint::UniqueClaimFingerprint = unique_claim_fingerprint(),
        equations = nothing)
    result = hasproperty(ude, :protocol_result) ? ude.protocol_result :
             build_protocol_result(ude)
    kpis = hasproperty(ude, :locked_kpis) ? ude.locked_kpis : locked_ude_kpis(ude)
    failures = unique_claim_kpi_failures(kpis)
    text = format_recovery_protocol(ude, fingerprint; equations = equations)
    return UniqueClaimProtocolRow(
        fingerprint,
        result,
        kpis,
        collect(Symbol, failures),
        extras_print_label(result.extras),
        text)
end

function unique_claim_protocol_row_namedtuple(row::UniqueClaimProtocolRow)
    return (;
        kind = row.fingerprint.kind,
        n_ics = row.fingerprint.n_ics,
        n_points = row.fingerprint.n_points,
        is_protocol = unique_claim_fingerprint_is_protocol(row.fingerprint),
        unknown_holes = row.protocol_result.unknown_holes,
        unidentifiable_edge = row.protocol_result.unidentifiable_edge,
        coefficients_are_biological_constants = row.protocol_result.coefficients_are_biological_constants,
        extras_label = row.extras_label,
        kpi_failures = Tuple(row.kpi_failures),
        claim = row.protocol_result.claim,
        canonical_hill_from_nn = row.protocol_result.canonical_hill_from_nn)
end

function assert_unique_claim_protocol_row(row::UniqueClaimProtocolRow)
    unique_claim_fingerprint_holds(row.fingerprint) || throw(ErrorException(
        "UniqueClaimProtocolRow fingerprint is not a locked protocol or smoke object"))
    assert_protocol_result_fields(row.protocol_result)
    assert_format_matches_protocol_result(row.protocol_result, row.text)
    extras_print_label(row.protocol_result.extras) == row.extras_label ||
        throw(ErrorException("printed extras label does not match protocol_result"))
    unique_claim_kpi_failure_symbols_hold(row.kpi_failures) ||
        throw(ErrorException(
            "KPI failures must be named from UNIQUE_CLAIM_KPI_FIELDS; got $(row.kpi_failures)"))
    Set(row.kpi_failures) == Set(unique_claim_kpi_failures(row.kpis)) ||
        throw(ErrorException("stored KPI failures do not match unique_claim_kpi_failures"))
    extras_print_is_hardcoded_attempt(row.extras_label) && throw(ErrorException(
        "UniqueClaimProtocolRow must not invent UDE F1-attempt extras"))
    occursin("hybrid_data_residual:", row.text) || throw(ErrorException(
        "protocol row text must print hybrid_data_residual"))
    return row
end

"""Hard-recovery row: protocol fingerprint and empty named KPI failures."""
function assert_unique_claim_protocol_row_holds(row::UniqueClaimProtocolRow)
    assert_unique_claim_protocol_row(row)
    unique_claim_fingerprint_is_protocol(row.fingerprint) || throw(ErrorException(
        "hard recovery row must use the protocol fingerprint, not smoke"))
    isempty(row.kpi_failures) || throw(ErrorException(
        unique_claim_kpi_failure_message(row.kpi_failures)))
    row.protocol_result.canonical_hill_from_nn === false || throw(ErrorException(
        "canonical_hill_from_nn must stay false"))
    unique_claim_f1_reaches_analytical_gate(row.protocol_result.support_f1) &&
        throw(ErrorException(
            "UDE support_f1 must stay below support_f1_clean; got $(row.protocol_result.support_f1)"))
    return row
end

function unique_claim_protocol_row_from_fields(;
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
    return unique_claim_protocol_row(ude;
        fingerprint = unique_claim_fingerprint(; smoke),
        equations = equations)
end

function recovery_hard_named_kpi_contract()
    return (;
        symbols = unique_claim_kpi_failure_symbols(),
        names = (:unidentifiable_edge, :data_residual, :support_recall),
        f1_is_not_a_symbol = !(:support_f1 in UNIQUE_CLAIM_KPI_FIELDS),
        empty_label = format_unique_claim_kpi_failures(Symbol[]),
        miss_edge = format_unique_claim_kpi_failures(Symbol[:unidentifiable_edge]),
        miss_all = format_unique_claim_kpi_failures(collect(UNIQUE_CLAIM_KPI_FIELDS)))
end
