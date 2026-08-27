###############################################################################
# Failure-mode instrument (not exported).
#
# DiscoveryRetcode must name every silent-looking miss. 0/2-hole networks
# still pass validate_network. KPI failure symbols stay the three claim
# gates. extras print NA / (none) / live leftovers. Combined F1 is never
# painted as 0.99 on a failed row.
#
# Does not drop protocol ICs. Does not grow exports. Does not put a
# single-hole gate into validate_network.
###############################################################################

const FAILURE_MODE_MUST_CONTAIN = (
    "function discovery_retcode_catalog",
    "function discovery_retcode_mapper_row",
    "function insufficient_samples_row",
    "function empty_support_row",
    "function hole_validate_row",
    "function kpi_failure_grid",
    "function extras_print_catalog_row",
    "function failed_discovery_does_not_paint_f1")

const FAILURE_MODE_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

const DISCOVERY_RETCODE_SYMBOLS = (
    :DiscoverySuccess,
    :InsufficientSamples,
    :DenominatorUnsafe,
    :EmptySupport,
    :SingularLibrary,
    :DiscoveryFailed)

function failure_mode_locked_sentences()
    return (;
        retcode = "DiscoveryRetcode names InsufficientSamples, DenominatorUnsafe, EmptySupport, SingularLibrary, DiscoveryFailed, and DiscoverySuccess.",
        validate = "validate_network stays a topology checker; 0-hole and 2-hole networks still validate.",
        kpi = "KPI failure symbols are unidentifiable_edge, data_residual, and support_recall; combined F1 is never a failure symbol.",
        extras = "extras print NA for missing, (none) for an empty collection, and the live leftovers otherwise.")
end

failure_mode_contract() = failure_mode_locked_sentences().validate

function failure_mode_source_path()
    joinpath(pkgdir(BioDynaX), "src", "FailureModes.jl")
end

function failure_mode_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "failure-modes.md")
end

# -- Retcode catalog ----------------------------------------------------------

"""
    discovery_retcode_catalog()

Every `DiscoveryRetcode` instance, in enum order. Adding a retcode
without updating this catalog is a contract break.
"""
function discovery_retcode_catalog()
    instances = (
        DiscoverySuccess,
        InsufficientSamples,
        DenominatorUnsafe,
        EmptySupport,
        SingularLibrary,
        DiscoveryFailed)
    names = Tuple(Symbol(r) for r in instances)
    return (;
        instances,
        names,
        n = length(instances),
        holds = names == DISCOVERY_RETCODE_SYMBOLS &&
                length(unique(instances)) == 6 &&
                DiscoverySuccess isa DiscoveryRetcode)
end

"""
    discovery_retcode_mapper_row()

`_discovery_retcode` maps DomainError → DenominatorUnsafe, singular
linear algebra → SingularLibrary, insufficient / empty-support
ArgumentError → those two retcodes, and everything else →
DiscoveryFailed.
"""
function discovery_retcode_mapper_row()
    insufficient = _discovery_retcode(
        ArgumentError("insufficient finite trajectory samples"))
    empty = _discovery_retcode(
        ArgumentError("empty support: no terms survived thresholding"))
    den = _discovery_retcode(
        DomainError(-1.0, "discovered denominator is singular"))
    singular = _discovery_retcode(LinearAlgebra.SingularException(1))
    other = _discovery_retcode(ErrorException("unexpected discovery miss"))
    unsupported = _discovery_retcode(
        ArgumentError("unsupported discovery backend DummyBackend"))
    return (;
        insufficient,
        empty,
        den,
        singular,
        other,
        unsupported,
        holds = insufficient === InsufficientSamples &&
                empty === EmptySupport &&
                den === DenominatorUnsafe &&
                singular === SingularLibrary &&
                other === DiscoveryFailed &&
                unsupported === DiscoveryFailed)
end

function discovery_retcode_mapper_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function _discovery_retcode", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("DenominatorUnsafe", body) &&
           occursin("SingularLibrary", body) &&
           occursin("InsufficientSamples", body) &&
           occursin("EmptySupport", body) &&
           occursin("DiscoveryFailed", body) &&
           occursin("insufficient", body) &&
           occursin("empty support", body)
end

function discovery_sample_floor_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function _run_discovery", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("size(X, 2) ≥ 20", body) &&
           occursin("insufficient finite trajectory samples", body) &&
           occursin("empty support: no terms survived thresholding", body)
end

function discovery_n_samples_entry_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function discover_equations(p_trained, nn, st;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("n_samples ≥ 20", body) &&
           occursin("n_samples must be at least 20", body)
end

# -- Trajectory probe ---------------------------------------------------------

"""
    failure_mode_linear_trajectory(; n_points, tspan, u0)

Compile the linear fixture once and generate a noise-0 trajectory.
Used by retcode rows. Does not train a UDE.
"""
function failure_mode_linear_trajectory(;
        n_points::Int = 40,
        tspan = (0.0, 1.5),
        u0 = [0.22, 0.14],
        rng_seed::Integer = 3)
    net = build_linear_test_network()
    rng = MersenneTwister(rng_seed)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    times, clean, _, _ = generate_from_compiled_model(
        model, packed, MersenneTwister(rng_seed);
        u0 = Float64.(u0), tspan = tspan, n_points = n_points, noise_σ = 0.0)
    dX = estimate_derivatives(clean, times)
    return (;
        net, model, packed, times, X = clean, dX,
        n_points = size(clean, 2),
        nstates = size(clean, 1))
end

function failure_mode_hill_trajectory(;
        n_points::Int = 40,
        tspan = (0.0, 1.5),
        u0 = [0.30, 0.25],
        known::Bool = true)
    net = build_hill_recovery_network(; known = known, hill_order = 2)
    rng = MersenneTwister(5)
    model, p0 = build_ude_model(rng, net)
    packed = known ?
             pack_parameters(
        (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6), p0.nn) :
             pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), p0.nn)
    times, clean, _, _ = generate_from_compiled_model(
        model, packed, MersenneTwister(5);
        u0 = Float64.(u0), tspan = tspan, n_points = n_points, noise_σ = 0.0)
    dX = estimate_derivatives(clean, times)
    return (; net, model, packed, times, X = clean, dX, n_points = size(clean, 2))
end

function _explicit_config(; threshold = 1e-2, seed = 42)
    DiscoveryConfig(backend = ExplicitSTLSQ(threshold = threshold), seed = seed)
end

function _implicit_config(; threshold = 1e-2, seed = 42, bootstrap = 4)
    DiscoveryConfig(
        backend = ImplicitSINDyPI(
            threshold = threshold, bootstrap_samples = bootstrap,
            domain_samples = 16, chunk_size = 32),
        seed = seed)
end

# -- Live retcode rows --------------------------------------------------------

"""
    insufficient_samples_row()

Raw-data `discover_equations` with 12 columns returns
`InsufficientSamples` when `strict=false`. It does not throw. The
sample floor is 20.
"""
function insufficient_samples_row()
    traj = failure_mode_linear_trajectory(; n_points = 12)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = traj.dX,
        config = _explicit_config(),
        verbose = false, strict = false)
    threw = false
    try
        discover_equations(
            traj.X, traj.times, traj.net;
            derivatives = traj.dX,
            config = _explicit_config(),
            verbose = false, strict = true)
    catch
        threw = true
    end
    return (;
        n_points = traj.n_points,
        success = result.success,
        retcode = result.retcode,
        threw_strict = threw,
        holds = traj.n_points == 12 &&
                result.success == false &&
                result.retcode === InsufficientSamples &&
                threw)
end

"""
    insufficient_samples_boundary_row()

19 columns fail. 20 columns are admitted to `_run_discovery` (they may
still fail later for another honest reason).
"""
function insufficient_samples_boundary_row()
    short = failure_mode_linear_trajectory(; n_points = 19)
    long = failure_mode_linear_trajectory(; n_points = 20)
    short_result = discover_equations(
        short.X, short.times, short.net;
        derivatives = short.dX,
        config = _explicit_config(),
        verbose = false, strict = false)
    long_result = discover_equations(
        long.X, long.times, long.net;
        derivatives = long.dX,
        config = _explicit_config(),
        verbose = false, strict = false)
    return (;
        short_n = short.n_points,
        long_n = long.n_points,
        short_retcode = short_result.retcode,
        long_is_insufficient = long_result.retcode === InsufficientSamples,
        holds = short.n_points == 19 &&
                long.n_points == 20 &&
                short_result.retcode === InsufficientSamples &&
                long_result.retcode !== InsufficientSamples)
end

"""
    n_samples_entry_throws_row()

The `discover_equations(p, nn, st; n_samples)` entry throws at 19 even
when `strict=false`. That is a different path from the raw-data result.
"""
function n_samples_entry_throws_row()
    traj = failure_mode_linear_trajectory(; n_points = 24)
    threw = false
    message = ""
    try
        discover_equations(
            traj.packed, traj.model.nn, traj.model.st;
            model = traj.model, network = traj.net,
            u0 = [0.22, 0.14], tspan = (0.0, 1.0),
            n_samples = 19, verbose = false, strict = false)
    catch error
        threw = error isa ArgumentError
        message = error isa ArgumentError ? error.msg : string(error)
    end
    return (;
        threw,
        message,
        holds = threw && occursin("at least 20", message))
end

"""
    empty_support_row()

A huge explicit threshold wipes every term. The raw-data entry returns
`EmptySupport` when `strict=false`.
"""
function empty_support_row()
    traj = failure_mode_linear_trajectory(; n_points = 32)
    dX = zero(traj.X)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = dX,
        config = _explicit_config(; threshold = 1e-2),
        verbose = false, strict = false)
    return (;
        n_points = traj.n_points,
        success = result.success,
        retcode = result.retcode,
        n_candidates = length(result.candidates),
        holds = result.success == false &&
                result.retcode === EmptySupport &&
                traj.n_points ≥ 20)
end

"""
    explicit_success_row()

Linear known dynamics with a modest explicit threshold recover a
support. This is not the unique-claim Hill path.
"""
function explicit_success_row()
    traj = failure_mode_linear_trajectory(; n_points = 40)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = traj.dX,
        config = _explicit_config(; threshold = 1e-2),
        verbose = false, strict = false)
    return (;
        n_points = traj.n_points,
        success = result.success,
        retcode = result.retcode,
        n_candidates = length(result.candidates),
        holds = result.success &&
                result.retcode === DiscoverySuccess &&
                !isempty(result.candidates))
end

"""
    implicit_insufficient_row()

Implicit SINDy-PI uses the same 20-sample floor. 10 columns fail before
a bootstrap Gram is built.
"""
function implicit_insufficient_row()
    traj = failure_mode_linear_trajectory(; n_points = 10)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = traj.dX,
        config = _implicit_config(),
        verbose = false, strict = false)
    return (;
        n_points = traj.n_points,
        retcode = result.retcode,
        holds = result.success == false &&
                result.retcode === InsufficientSamples)
end

function discover_unknown_rate_insufficient_row()
    times = collect(range(0.0, 0.5; length = 8))
    R = [0.2 .+ 0.01 .* times'; 0.15 .+ 0.01 .* times']
    D = reshape(0.3 .+ 0.02 .* times, 1, :)
    result = discover_unknown_rate(
        R, times, D;
        network = build_hill_recovery_network(; known = false, hill_order = 2),
        config = rate_discovery_config(),
        verbose = false, strict = false)
    return (;
        n_points = length(times),
        success = result.success,
        retcode = result.retcode,
        holds = result.success == false &&
                result.retcode === InsufficientSamples)
end

function failed_discovery_result_row()
    result = _failed_discovery(
        ErrorException("synthetic miss"),
        _explicit_config();
        prefix = "Discovery failed")
    return (;
        success = result.success,
        retcode = result.retcode,
        n_candidates = length(result.candidates),
        paints_f1 = occursin("0.99", result.message) ||
                    occursin("support_f1", result.message),
        holds = result.success == false &&
                result.retcode === DiscoveryFailed &&
                isempty(result.candidates) &&
                !occursin("0.99", result.message))
end

function failed_discovery_does_not_paint_f1()
    rows = (
        failed_discovery_result_row(),
        insufficient_samples_row(),
        empty_support_row())
    return all(
        row -> row.holds &&
            !occursin("support_f1_ude = 0.99",
                hasproperty(row, :retcode) ? string(row.retcode) : ""),
        rows) &&
           all(row -> row.success == false || row.retcode === DiscoverySuccess,
        rows)
end

# -- 0/2-hole versus validate_network -----------------------------------------

"""
    hole_validate_spec(name, builder, expected_holes, recovery_admits)

One network in the hole-versus-validate catalog.
"""
struct HoleValidateSpec
    name::Symbol
    builder::Function
    expected_holes::Int
    recovery_admits::Bool
    unique_claim_section::Bool
end

function hole_validate_specs()
    return (
        HoleValidateSpec(:linear_zero, build_linear_test_network, 0, false, false),
        HoleValidateSpec(:zero_alias, build_zero_unknown_linear_network, 0, false, false),
        HoleValidateSpec(
            :hill_known, () -> build_hill_recovery_network(; known = true, hill_order = 2),
            0, false, false),
        HoleValidateSpec(:hill_unknown,
            () -> build_hill_recovery_network(; known = false, hill_order = 2),
            1, true, true),
        HoleValidateSpec(
            :mm_known, () -> build_mm_recovery_network(; known = true), 0, false, false),
        HoleValidateSpec(
            :mm_unknown, () -> build_mm_recovery_network(; known = false), 1, true, true),
        HoleValidateSpec(:dual, build_dual_unknown_network, 2, false, true),
        HoleValidateSpec(
            :two_regulator, build_two_regulator_unknown_network, 1, true, false),
        HoleValidateSpec(
            :six_state, () -> build_six_state_unknown_network(; known = false),
            1, true, false),
        HoleValidateSpec(
            :skipped_duplicate, build_skipped_duplicate_unknown_network, 2, false, false),
        HoleValidateSpec(
            :skipped_middle, build_skipped_middle_unknown_network, 3, false, false),
        HoleValidateSpec(:competitive_known,
            () -> build_competitive_test_network(; known = true), 0, false, false),
        HoleValidateSpec(:competitive_unknown,
            () -> build_competitive_test_network(; known = false), 1, true, false),
        HoleValidateSpec(:repressilator, build_repressilator_network, 0, false, false),
        HoleValidateSpec(:default_example, () -> DEFAULT_EXAMPLE_NETWORK, 1, true, false),
        HoleValidateSpec(:mm_test, build_mm_test_network, 0, false, false),
        HoleValidateSpec(
            :three_state, () -> build_three_state_unknown_network(), 1, true, false),
        HoleValidateSpec(
            :kinetic_known, build_kinetic_generalization_network, 0, false, false))
end

"""
    hole_validate_row(spec)

`validate_network` returns the same network for 0, 1, 2, and 3 holes.
`unique_claim_recovery_admits` is true only for a single unknown
destruction. `assert_single_unknown_destruction` throws unless holes==1.
"""
function hole_validate_row(spec::HoleValidateSpec)
    net = spec.builder()
    holes = count_unknown_destructions(net)
    validated = validate_network(net)
    admits = unique_claim_recovery_admits(net)
    model, _ = build_ude_model(MersenneTwister(0), net)
    threw = false
    try
        assert_single_unknown_destruction(model)
    catch
        threw = true
    end
    return (;
        name = spec.name,
        holes,
        expected = spec.expected_holes,
        validate_open = validated === net,
        admits,
        expected_admits = spec.recovery_admits,
        assert_throws = threw,
        expected_throw = spec.expected_holes != 1,
        holds = holes == spec.expected_holes &&
                validated === net &&
                admits == spec.recovery_admits &&
                threw == (spec.expected_holes != 1))
end

function hole_validate_matrix()
    specs = hole_validate_specs()
    rows = [hole_validate_row(spec) for spec in specs]
    n_zero = count(r -> r.holes == 0, rows)
    n_one = count(r -> r.holes == 1, rows)
    n_multi = count(r -> r.holes ≥ 2, rows)
    return (;
        n = length(rows),
        rows,
        n_zero,
        n_one,
        n_multi,
        all_validate_open = all(r -> r.validate_open, rows),
        holds = all(r -> r.holds, rows) &&
                n_zero ≥ 4 && n_one ≥ 4 && n_multi ≥ 2 &&
                all(r -> r.validate_open, rows))
end

function hole_validate_zero_and_dual_row()
    zero = hole_validate_row(HoleValidateSpec(
        :linear_zero, build_linear_test_network, 0, false, false))
    dual = hole_validate_row(HoleValidateSpec(
        :dual, build_dual_unknown_network, 2, false, true))
    single = hole_validate_row(HoleValidateSpec(
        :hill_unknown,
        () -> build_hill_recovery_network(; known = false, hill_order = 2),
        1, true, true))
    return (;
        zero, dual, single,
        holds = zero.holds && dual.holds && single.holds &&
                zero.validate_open && dual.validate_open &&
                single.validate_open &&
                zero.admits == false && dual.admits == false &&
                single.admits)
end

"""
    discovery_on_zero_hole_row()

Discovery on a 0-hole linear network is allowed. `validate_network`
does not block it. This is not unique-claim recovery.
"""
function discovery_on_zero_hole_row()
    traj = failure_mode_linear_trajectory(; n_points = 32)
    holes = count_unknown_destructions(traj.net)
    validated = validate_network(traj.net)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = traj.dX,
        config = _explicit_config(),
        verbose = false, strict = false)
    return (;
        holes,
        validate_open = validated === traj.net,
        success = result.success,
        retcode = result.retcode,
        recovery_admits = unique_claim_recovery_admits(traj.net),
        holds = holes == 0 &&
                validated === traj.net &&
                unique_claim_recovery_admits(traj.net) == false &&
                result.retcode !== nothing)
end

function discovery_on_dual_hole_row()
    net = build_dual_unknown_network()
    rng = MersenneTwister(11)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    packed = pack_parameters(phys, p0.nn)
    times, clean, _, _ = generate_from_compiled_model(
        model, packed, MersenneTwister(11);
        u0 = [0.22, 0.18, 0.16], tspan = (0.0, 0.8), n_points = 32,
        noise_σ = 0.0)
    dX = estimate_derivatives(clean, times)
    result = discover_equations(
        clean, times, net;
        derivatives = dX,
        config = _explicit_config(),
        verbose = false, strict = false)
    return (;
        holes = count_unknown_destructions(net),
        validate_open = validate_network(net) === net,
        recovery_admits = unique_claim_recovery_admits(net),
        retcode = result.retcode,
        holds = count_unknown_destructions(net) == 2 &&
                validate_network(net) === net &&
                unique_claim_recovery_admits(net) == false)
end

function validate_network_body_has_no_hole_gate()
    return validate_network_stays_open_source()
end

# -- KPI failure grid ---------------------------------------------------------

"""
    kpi_probe_row(; unidentifiable_edge, data_residual, support_recall, support_f1)

One synthetic KPI row. Combined F1 is stored and must not appear in
`unique_claim_kpi_failures`.
"""
function kpi_probe_row(;
        unidentifiable_edge::Bool,
        data_residual::Float64,
        support_recall::Float64,
        support_f1::Float64)
    kpis = (;
        unidentifiable_edge,
        data_residual,
        support_recall,
        support_f1)
    failures = unique_claim_kpi_failures(kpis)
    expected = Symbol[]
    unidentifiable_edge === true || push!(expected, :unidentifiable_edge)
    unique_claim_residual_holds(data_residual) || push!(expected, :data_residual)
    unique_claim_recall_holds(support_recall) || push!(expected, :support_recall)
    return (;
        unidentifiable_edge,
        data_residual,
        support_recall,
        support_f1,
        failures = Tuple(failures),
        expected = Tuple(expected),
        symbols_hold = unique_claim_kpi_failure_symbols_hold(failures),
        f1_absent = !(:support_f1 in failures) &&
                    !(:canonical_hill_from_nn in failures),
        message = unique_claim_kpi_failure_message(failures),
        label = format_unique_claim_kpi_failures(failures),
        hold = unique_claim_kpis_hold(kpis),
        holds = failures == expected &&
                unique_claim_kpi_failure_symbols_hold(failures) &&
                !(:support_f1 in failures))
end

function kpi_failure_grid()
    edges = (true, false)
    residuals = (0.003, 0.31, 0.80)
    recalls = (1.0, 0.99, 0.50)
    f1s = (0.10, 0.50, 0.57, 0.99)
    rows = NamedTuple[]
    for edge in edges, residual in residuals, recall in recalls, f1 in f1s
        push!(rows,
            kpi_probe_row(
                unidentifiable_edge = edge,
                data_residual = residual,
                support_recall = recall,
                support_f1 = f1))
    end
    n_fail = count(r -> !isempty(r.failures), rows)
    n_hold = count(r -> r.hold, rows)
    painted = count(r -> :support_f1 in r.failures, rows)
    return (;
        n = length(rows),
        n_fail,
        n_hold,
        painted,
        rows,
        holds = all(r -> r.holds, rows) &&
                painted == 0 &&
                n_fail ≥ 1 && n_hold ≥ 1 &&
                length(rows) == 2 * 3 * 3 * 4)
end

function kpi_failure_named_examples()
    pass = kpi_probe_row(
        unidentifiable_edge = true, data_residual = 0.003,
        support_recall = 1.0, support_f1 = 0.57)
    residual = kpi_probe_row(
        unidentifiable_edge = true, data_residual = 0.80,
        support_recall = 1.0, support_f1 = 0.99)
    recall = kpi_probe_row(
        unidentifiable_edge = true, data_residual = 0.003,
        support_recall = 0.40, support_f1 = 0.99)
    ident = kpi_probe_row(
        unidentifiable_edge = false, data_residual = 0.003,
        support_recall = 1.0, support_f1 = 0.10)
    all_fail = kpi_probe_row(
        unidentifiable_edge = false, data_residual = 0.80,
        support_recall = 0.40, support_f1 = 0.99)
    return (;
        pass, residual, recall, ident, all_fail,
        holds = pass.holds && pass.hold &&
                residual.holds && residual.failures == (:data_residual,) &&
                recall.holds && recall.failures == (:support_recall,) &&
                ident.holds && ident.failures == (:unidentifiable_edge,) &&
                all_fail.holds && length(all_fail.failures) == 3 &&
                residual.f1_absent && recall.f1_absent &&
                all_fail.f1_absent)
end

function kpi_f1_never_failure_symbol_row()
    high = kpi_probe_row(
        unidentifiable_edge = true, data_residual = 0.003,
        support_recall = 1.0, support_f1 = 0.99)
    low = kpi_probe_row(
        unidentifiable_edge = true, data_residual = 0.003,
        support_recall = 1.0, support_f1 = 0.10)
    return (;
        high, low,
        high_hold = high.hold,
        low_hold = low.hold,
        holds = high.holds && low.holds && high.hold && low.hold &&
                high.failures == () && low.failures == () &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50 &&
                RECOVERY_THRESHOLDS.support_f1_clean == 0.99)
end

# -- extras NA / (none) / live ------------------------------------------------

"""
    extras_print_case(name, extras, expected, hardcoded)

One extras printer case.
"""
struct ExtrasPrintCase
    name::Symbol
    extras::Any
    expected::String
    hardcoded::Bool
end

function extras_print_cases()
    return (
        ExtrasPrintCase(:missing, nothing, "NA", false),
        ExtrasPrintCase(:empty_tuple, (), "(none)", false),
        ExtrasPrintCase(:empty_string_vector, String[], "(none)", false),
        ExtrasPrintCase(:empty_any_vector, Any[], "(none)", false),
        ExtrasPrintCase(:live_pair, ("1", "r"), "1, r", false),
        ExtrasPrintCase(:live_vector, ["s", "s^2"], "s, s^2", false),
        ExtrasPrintCase(:live_single, ("r",), "r", false),
        ExtrasPrintCase(:string_passthrough, "1, r", "1, r", false),
        ExtrasPrintCase(:hardcoded_attempt,
            "1, r remain after the UDE F1 attempt",
            "1, r remain after the UDE F1 attempt", true),
        ExtrasPrintCase(:na_string, "NA", "NA", false))
end

function extras_print_catalog_row()
    cases = extras_print_cases()
    rows = NamedTuple[]
    for case in cases
        label = extras_print_label(case.extras)
        hardcoded = extras_print_is_hardcoded_attempt(label)
        push!(rows,
            (;
                name = case.name,
                label,
                expected = case.expected,
                hardcoded,
                expected_hardcoded = case.hardcoded,
                holds = label == case.expected && hardcoded == case.hardcoded))
    end
    return (;
        n = length(rows),
        rows,
        has_na = any(r -> r.label == "NA", rows),
        has_none = any(r -> r.label == "(none)", rows),
        has_live = any(r -> r.label == "1, r", rows),
        has_hardcoded = any(r -> r.hardcoded, rows),
        holds = all(r -> r.holds, rows) &&
                any(r -> r.label == "NA", rows) &&
                any(r -> r.label == "(none)", rows) &&
                any(r -> r.label == "1, r", rows) &&
                any(r -> r.hardcoded, rows))
end

function extras_empty_vs_na_row()
    missing = extras_print_label(nothing)
    empty = extras_print_label(String[])
    live = extras_print_label(("1", "r"))
    return (;
        missing,
        empty,
        live,
        distinct = missing != empty && empty != live && missing != live,
        holds = missing == "NA" && empty == "(none)" && live == "1, r" &&
                missing != empty)
end

function extras_hardcoded_attempt_row()
    honest = extras_print_label(("1", "r"))
    attempt = extras_print_label("1, r remain after the UDE F1 attempt")
    return (;
        honest,
        attempt,
        honest_hardcoded = extras_print_is_hardcoded_attempt(honest),
        attempt_hardcoded = extras_print_is_hardcoded_attempt(attempt),
        holds = honest == "1, r" &&
                extras_print_is_hardcoded_attempt(honest) == false &&
                extras_print_is_hardcoded_attempt(attempt))
end

function kpi_threshold_boundary_row()
    residual_pass = kpi_probe_row(
        unidentifiable_edge = true,
        data_residual = RECOVERY_THRESHOLDS.data_residual,
        support_recall = 1.0, support_f1 = 0.57)
    residual_fail = kpi_probe_row(
        unidentifiable_edge = true,
        data_residual = RECOVERY_THRESHOLDS.data_residual + 1e-9,
        support_recall = 1.0, support_f1 = 0.57)
    recall_pass = kpi_probe_row(
        unidentifiable_edge = true,
        data_residual = 0.003,
        support_recall = RECOVERY_THRESHOLDS.support_recall,
        support_f1 = 0.57)
    recall_fail = kpi_probe_row(
        unidentifiable_edge = true,
        data_residual = 0.003,
        support_recall = RECOVERY_THRESHOLDS.support_recall - 1e-9,
        support_f1 = 0.57)
    return (;
        residual_pass, residual_fail, recall_pass, recall_fail,
        holds = residual_pass.hold && !residual_fail.hold &&
                recall_pass.hold && !recall_fail.hold &&
                residual_fail.failures == (:data_residual,) &&
                recall_fail.failures == (:support_recall,) &&
                residual_pass.f1_absent && residual_fail.f1_absent)
end

function hill_insufficient_samples_row()
    traj = failure_mode_hill_trajectory(; n_points = 12, known = true)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = traj.dX,
        config = _explicit_config(),
        verbose = false, strict = false)
    return (;
        n_points = traj.n_points,
        holes = count_unknown_destructions(traj.net),
        retcode = result.retcode,
        holds = traj.n_points == 12 &&
                count_unknown_destructions(traj.net) == 0 &&
                result.retcode === InsufficientSamples)
end

function hill_known_explicit_success_row()
    traj = failure_mode_hill_trajectory(; n_points = 40, known = true)
    result = discover_equations(
        traj.X, traj.times, traj.net;
        derivatives = traj.dX,
        config = _explicit_config(; threshold = 1e-2),
        verbose = false, strict = false)
    return (;
        n_points = traj.n_points,
        holes = count_unknown_destructions(traj.net),
        success = result.success,
        retcode = result.retcode,
        validate_open = validate_network(traj.net) === traj.net,
        holds = count_unknown_destructions(traj.net) == 0 &&
                validate_network(traj.net) === traj.net &&
                result.success &&
                result.retcode === DiscoverySuccess)
end

function mm_insufficient_samples_row()
    net = build_mm_recovery_network(; known = true)
    rng = MersenneTwister(17)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters(
        (k_prod = 0.9, vmax = 1.6, km = 0.45, k_rs = 1.0, k_r = 0.6), p0.nn)
    times, clean, _, _ = generate_from_compiled_model(
        model, packed, MersenneTwister(17);
        u0 = [0.30, 0.25], tspan = (0.0, 0.8), n_points = 11, noise_σ = 0.0)
    dX = estimate_derivatives(clean, times)
    result = discover_equations(
        clean, times, net;
        derivatives = dX,
        config = _explicit_config(),
        verbose = false, strict = false)
    return (;
        n_points = size(clean, 2),
        holes = count_unknown_destructions(net),
        retcode = result.retcode,
        holds = size(clean, 2) == 11 &&
                count_unknown_destructions(net) == 0 &&
                result.retcode === InsufficientSamples)
end

function select_discovery_all_fail_row()
    traj = failure_mode_linear_trajectory(; n_points = 24)
    result = select_discovery_config(
        traj.packed, traj.model;
        thresholds = (1e-2, 1e-3),
        n_samples = 12,
        u0 = [0.22, 0.14],
        tspan = (0.0, 0.8),
        config = _explicit_config(),
        verbose = false)
    return (;
        success = result.success,
        retcode = result.retcode,
        paints_f1 = occursin("0.99", result.message),
        holds = result.success == false &&
                result.retcode === DiscoveryFailed &&
                !occursin("0.99", result.message))
end

function discovery_retcode_exports_hold()
    exported = (
        :DiscoveryRetcode, :DiscoverySuccess, :InsufficientSamples,
        :DenominatorUnsafe, :EmptySupport, :SingularLibrary, :DiscoveryFailed)
    return all(sym -> sym in LOCKED_PUBLIC_EXPORTS, exported) &&
           public_export_list_holds()
end

function extras_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Recovery.jl"), String)
    start = findfirst("function _format_protocol_extras", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("NA", body) &&
           occursin("(none)", body) &&
           occursin("isempty(extras)", body)
end

# -- Failed protocol print honesty --------------------------------------------

function failed_protocol_print_row()
    ident = (;
        unidentifiable_edge = false,
        production_param = :k_prod)
    text = format_protocol_result(
        ident;
        residual = 0.80,
        equations = nothing,
        extras = nothing,
        support_f1 = 0.10,
        support_recall = 0.40,
        unknown_holes = 0,
        seed = 103,
        n_ics = 9,
        n_points = 50,
        protocol_kind = :protocol,
        smoke = false)
    return (;
        text,
        has_na = occursin("NA", text),
        has_none_eq = occursin("(none)", text),
        paints_ude_f1 = occursin("support_f1_ude = 0.99", text),
        holes_zero = occursin("0", text),
        holds = occursin("NA", text) &&
                !occursin("support_f1_ude = 0.99", text) &&
                !occursin("HTTP 200", text))
end

function failed_protocol_two_hole_print_row()
    ident = (;
        unidentifiable_edge = false,
        production_param = :k_prod)
    text = format_protocol_result(
        ident;
        residual = 0.80,
        equations = nothing,
        extras = String[],
        support_f1 = 0.10,
        support_recall = 0.40,
        unknown_holes = 2,
        smoke = true)
    return (;
        text,
        extras_none = occursin("(none)", text),
        paints_ude_f1 = occursin("support_f1_ude = 0.99", text),
        holds = occursin("(none)", text) &&
                !occursin("support_f1_ude = 0.99", text))
end

# -- Discovery retcode matrix -------------------------------------------------

function discovery_retcode_live_matrix()
    mapper = discovery_retcode_mapper_row()
    catalog = discovery_retcode_catalog()
    insufficient = insufficient_samples_row()
    boundary = insufficient_samples_boundary_row()
    entry = n_samples_entry_throws_row()
    empty = empty_support_row()
    success = explicit_success_row()
    implicit = implicit_insufficient_row()
    rate = discover_unknown_rate_insufficient_row()
    failed = failed_discovery_result_row()
    return (;
        mapper, catalog, insufficient, boundary, entry, empty, success,
        implicit, rate, failed,
        holds = mapper.holds && catalog.holds && insufficient.holds &&
                boundary.holds && entry.holds && empty.holds &&
                success.holds && implicit.holds && rate.holds &&
                failed.holds)
end

function failure_mode_extended_matrix()
    holes = hole_validate_matrix()
    zero_dual = hole_validate_zero_and_dual_row()
    zero_disc = discovery_on_zero_hole_row()
    dual_disc = discovery_on_dual_hole_row()
    kpi = kpi_failure_grid()
    examples = kpi_failure_named_examples()
    f1 = kpi_f1_never_failure_symbol_row()
    extras = extras_print_catalog_row()
    empty_na = extras_empty_vs_na_row()
    hardcoded = extras_hardcoded_attempt_row()
    print_row = failed_protocol_print_row()
    two_print = failed_protocol_two_hole_print_row()
    boundary = kpi_threshold_boundary_row()
    hill_short = hill_insufficient_samples_row()
    hill_ok = hill_known_explicit_success_row()
    mm_short = mm_insufficient_samples_row()
    select_fail = select_discovery_all_fail_row()
    exports = discovery_retcode_exports_hold()
    return (;
        holes, zero_dual, zero_disc, dual_disc, kpi, examples, f1,
        extras, empty_na, hardcoded, print_row, two_print, boundary,
        hill_short, hill_ok, mm_short, select_fail, exports,
        holds = holes.holds && zero_dual.holds && zero_disc.holds &&
                dual_disc.holds && kpi.holds && examples.holds &&
                f1.holds && extras.holds && empty_na.holds &&
                hardcoded.holds && print_row.holds && two_print.holds &&
                boundary.holds && hill_short.holds && hill_ok.holds &&
                mm_short.holds && select_fail.holds && exports)
end

function failure_mode_fixture_matrix()
    live = discovery_retcode_live_matrix()
    extra = failure_mode_extended_matrix()
    return (;
        live, extra,
        holds = live.holds && extra.holds)
end

# -- Catalog index ------------------------------------------------------------

function failure_mode_fixture_names()
    return (
        :catalog, :mapper, :insufficient, :boundary, :n_samples_entry,
        :empty_support, :explicit_success, :implicit_insufficient,
        :rate_insufficient, :failed_result, :hole_matrix, :zero_dual,
        :zero_discovery, :dual_discovery, :kpi_grid, :kpi_examples,
        :f1_never_symbol, :extras_catalog, :extras_empty_na,
        :extras_hardcoded, :failed_print, :two_hole_print,
        :kpi_boundary, :hill_insufficient, :hill_success,
        :mm_insufficient, :select_all_fail, :retcode_exports)
end

function failure_mode_row_index()
    names = failure_mode_fixture_names()
    return (;
        n = length(names),
        names,
        unique = length(unique(names)) == length(names),
        holds = length(unique(names)) == length(names) && length(names) ≥ 20)
end

function format_failure_mode_index()
    io = IOBuffer()
    println(io, "| fixture | retcode / gate |")
    println(io, "|---|---|")
    println(io, "| catalog | six DiscoveryRetcode values |")
    println(io, "| mapper | error class to retcode |")
    println(io, "| insufficient | 12 samples → InsufficientSamples |")
    println(io, "| boundary | 19 fail / 20 admitted |")
    println(io, "| n_samples_entry | nn/st entry throws at 19 |")
    println(io, "| empty_support | zero derivatives → EmptySupport |")
    println(io, "| explicit_success | linear ExplicitSTLSQ |")
    println(io, "| implicit_insufficient | implicit 10 samples |")
    println(io, "| rate_insufficient | discover_unknown_rate 8 pts |")
    println(io, "| failed_result | DiscoveryFailed, no 0.99 F1 |")
    println(io, "| hole_matrix | validate open on 0/1/2/3 holes |")
    println(io, "| zero_dual | recovery rejects; validate open |")
    println(io, "| zero_discovery | discovery allowed on 0-hole |")
    println(io, "| dual_discovery | discovery allowed on 2-hole |")
    println(io, "| kpi_grid | 72 synthetic KPI combinations |")
    println(io, "| kpi_examples | named one-gate failures |")
    println(io, "| f1_never_symbol | 0.10 and 0.99 F1 are not gates |")
    println(io, "| extras_catalog | NA / (none) / live / attempt |")
    println(io, "| extras_empty_na | nothing ≠ empty collection |")
    println(io, "| extras_hardcoded | F1-attempt leftover string |")
    println(io, "| failed_print | protocol print does not paint 0.99 |")
    println(io, "| two_hole_print | extras (none) on a 2-hole row |")
    println(io, "| kpi_boundary | residual 0.30 and recall 0.99 edges |")
    println(io, "| hill_insufficient | known Hill 12 samples |")
    println(io, "| hill_success | known Hill ExplicitSTLSQ |")
    println(io, "| mm_insufficient | known MM 11 samples |")
    println(io, "| select_all_fail | 12-sample sweep → DiscoveryFailed |")
    println(io, "| retcode_exports | six retcodes stay in LOCKED_PUBLIC_EXPORTS |")
    return String(take!(io))
end

function failure_mode_index_holds()
    text = format_failure_mode_index()
    index = failure_mode_row_index()
    return index.holds &&
           occursin("InsufficientSamples", text) &&
           occursin("validate open", text) &&
           occursin("0.99 F1", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Formatter / test-file lock -----------------------------------------------

function failure_mode_test_path()
    joinpath(pkgdir(BioDynaX), "test", "test_failure_modes.jl")
end

function failure_mode_test_file_holds()
    path = failure_mode_test_path()
    isfile(path) || return false
    text = read(path, String)
    return !occursin('\t', text) &&
           !occursin("support_f1_ude = 0.99", text) &&
           !occursin("function validate_network", text) &&
           occursin("@testset", text) &&
           occursin("discovery_retcode_catalog", text) &&
           occursin("hole_validate_matrix", text) &&
           occursin("kpi_failure_grid", text) &&
           occursin("extras_print_catalog_row", text)
end

function failure_mode_formatter_lock_holds()
    return julia_formatter_toml_holds() && failure_mode_test_file_holds()
end

# -- Source / docs contracts --------------------------------------------------

function failure_mode_source_holds()
    src = read(failure_mode_source_path(), String)
    docs = isfile(failure_mode_docs_path()) ?
           read(failure_mode_docs_path(), String) : ""
    impl = read(discovery_jl_source_path(), String)
    return all(occursin(needle, src) for needle in FAILURE_MODE_MUST_CONTAIN) &&
           !occursin("support_f1_ude = 0.99", impl) &&
           !occursin("support_f1_ude = 0.99", docs) &&
           !occursin("function validate_network", docs)
end

function failure_mode_source_violations()
    src = read(failure_mode_source_path(), String)
    docs = isfile(failure_mode_docs_path()) ?
           read(failure_mode_docs_path(), String) : ""
    missing = [s for s in FAILURE_MODE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = String[]
    occursin("support_f1_ude = 0.99", docs) &&
        push!(forbidden, "docs: support_f1_ude = 0.99")
    occursin("function validate_network", docs) &&
        push!(forbidden, "docs: function validate_network")
    return (; missing, forbidden)
end

function failure_mode_docs_hold()
    path = failure_mode_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(failure_mode_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("failure-modes.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function failure_mode_landing_docs_hold()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    architecture = read(
        joinpath(pkgdir(BioDynaX), "docs", "src", "architecture.md"), String)
    sentences = failure_mode_locked_sentences()
    return occursin("failure-modes", howto) &&
           occursin("DiscoveryRetcode", howto) &&
           occursin(sentences.validate, architecture)
end

function failure_mode_docs_mention_helpers()
    path = failure_mode_docs_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("discovery_retcode_catalog", text) &&
           occursin("insufficient_samples_row", text) &&
           occursin("hole_validate_matrix", text) &&
           occursin("kpi_failure_grid", text) &&
           occursin("extras_print_catalog_row", text)
end

function failure_mode_contract_holds()
    return failure_mode_source_holds() &&
           discovery_retcode_mapper_source_holds() &&
           discovery_sample_floor_source_holds() &&
           discovery_n_samples_entry_source_holds() &&
           extras_source_holds() &&
           validate_network_stays_open_source() &&
           failure_mode_docs_hold() &&
           failure_mode_landing_docs_hold() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           failure_mode_formatter_lock_holds()
end
