###############################################################################
# Reference-protocol report helpers (not exported).
#
# Print and check order is IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION.
# This file does not change RECOVERY_THRESHOLDS, the public export list, or
# validate_network. Canonical Hill from a trained NN is not supported.
###############################################################################

"""Product-block labels in the locked stdout order."""
const UNIQUE_CLAIM_PRODUCT_BLOCKS = (
    :IDENTIFIABILITY, :FIT, :DISCOVERY, :REPRODUCTION)

"""Field order of `build_protocol_result`. Changing it is a product break."""
const PROTOCOL_RESULT_FIELDS = (
    :unknown_holes,
    :unidentifiable_edge,
    :coefficients_are_biological_constants,
    :data_residual,
    :support_recall,
    :support_f1,
    :extras,
    :canonical_hill_from_nn,
    :claim)

"""KPI names that decide `unique_claim_kpis_hold`. Combined F1 is not here."""
const UNIQUE_CLAIM_KPI_FIELDS = (
    :unidentifiable_edge, :data_residual, :support_recall)

"""Public `names(BioDynaX)` lock, including the module name. Not an export."""
const LOCKED_PUBLIC_EXPORTS = (
    :ACTIVATION,
    :AbstractADPolicy,
    :AbstractConstraintStrategy,
    :AugmentedLagrangianConfig,
    :BiologicalNetwork,
    :COMPETITIVE,
    :CUSTOM_KINETIC,
    :CompetitiveMetadata,
    :CustomKineticMetadata,
    :DenominatorUnsafe,
    :DiscoveryConfig,
    :DiscoveryFailed,
    :DiscoveryResult,
    :DiscoveryRetcode,
    :DiscoverySuccess,
    :EdgeKind,
    :EdgeSpec,
    :EmptyMetadata,
    :EmptySupport,
    :Experiment,
    :ExperimentSet,
    :ExplicitCandidate,
    :ExplicitSTLSQ,
    :HILL,
    :HillMetadata,
    :HorizonCurriculum,
    :INHIBITION,
    :INPUT,
    :ImplicitCandidate,
    :ImplicitSINDyPI,
    :InputDriveMetadata,
    :InsufficientSamples,
    :KineticFamily,
    :KineticMetadata,
    :LATENT,
    :LinearDecayMetadata,
    :MASS_ACTION,
    :MassActionMetadata,
    :MetadataLike,
    :NeuralDestructionTerm,
    :NodeKind,
    :NodeSpec,
    :ParameterSchema,
    :ProductionAD,
    :RECOVERY_THRESHOLDS,
    :ReactionSpec,
    :SATURATION,
    :STATE,
    :SaturationMetadata,
    :SingularLibrary,
    :SolverConfig,
    :StabilitySelection,
    :StructuralPositivity,
    :TrainingConfig,
    :TrainingResult,
    :TrainingRetcode,
    :UDEModel,
    :UNKNOWN_NN,
    :UnknownTermResult,
    :ZygoteAD,
    :allocate_cache,
    :auto_sensealg,
    :build_ude_function,
    :build_ude_model,
    :candidate_parents,
    :compile_mechanism,
    :compose_hybrid_rhs,
    :default_solver_config,
    :discover_equations,
    :discover_unknown_rate,
    :discover_unknown_term,
    :equation_to_function,
    :equation_to_latex,
    :estimate_derivatives,
    :experiment_from_csv,
    :export_rhs,
    :format_stability_selection,
    :generate_experiment_set,
    :hybrid_data_residual,
    :local_basis,
    :pack_parameters,
    :parameter_schema,
    :positive_parameter,
    :predict_ude,
    :report,
    :sample_unknown_destruction,
    :stability_selection_report,
    :state_nodes,
    :train_experiments,
    :train_ude,
    :ude_rhs!,
    :ude_system,
    :validate_network,
    :write_experiment_csv
)

"""Reference-example source strings the protocol fingerprint requires."""
const UNIQUE_CLAIM_EXAMPLE_MUST_CONTAIN = (
    "UNIQUE_CLAIM_PROTOCOL.seed",
    "UNIQUE_CLAIM_PROTOCOL.adam_iterations",
    "UNIQUE_CLAIM_PROTOCOL.bfgs_iterations",
    "unique_claim_protocol_ics",
    "unique_claim_protocol_n_points",
    "unique_claim_experiment_set",
    "unique_claim_discovery_config",
    "unique_claim_discovery_extras",
    "unique_claim_fingerprint",
    "production_destruction_tradeoff",
    "assert_single_unknown_destruction",
    "format_protocol_result",
    "assert_unique_claim_residual",
    "sample_unknown_destruction_grid",
    "_regulator_grid",
    "ReactionSpec",
    "HillMetadata")

"""Reference-example source strings that would reopen an unsupported claim or fixture."""
const UNIQUE_CLAIM_EXAMPLE_MUST_NOT_CONTAIN = (
    "build_hill_recovery_network",
    "Note:",
    "HTTP 200",
    "TagBot ran")

"""
    locked_public_names()

`names(BioDynaX)` lock: module name plus `LOCKED_PUBLIC_EXPORTS`.
"""
locked_public_names() = (:BioDynaX, LOCKED_PUBLIC_EXPORTS...)

"""True when the public export list is exactly the freeze-plus-reference-example set."""
public_export_list_holds() = issetequal(names(BioDynaX), collect(locked_public_names()))

"""Numeric copy of `RECOVERY_THRESHOLDS` used to detect a silent loosen."""
recovery_thresholds_lock() = (
    nn_correlation = 0.90,
    nn_rate_rmse = 0.12,
    support_f1_clean = 0.99,
    support_f1_ude = 0.50,
    support_f1_noisy = 0.50,
    support_recall = 0.99,
    discovered_rate_rmse = 0.20,
    data_residual = 0.30)

recovery_thresholds_hold() = RECOVERY_THRESHOLDS == recovery_thresholds_lock()

"""SciML formatter keys this repository actually commits."""
julia_formatter_lock() = (
    style = "sciml",
    margin = 92,
    indent = 4,
    whitespace_in_kwargs = true,
    remove_extra_newlines = true)

julia_formatter_toml_path() = joinpath(pkgdir(BioDynaX), ".JuliaFormatter.toml")

function julia_formatter_toml_holds()
    text = read(julia_formatter_toml_path(), String)
    lock = julia_formatter_lock()
    return occursin("style = \"$(lock.style)\"", text) &&
           occursin("margin = $(lock.margin)", text) &&
           occursin("indent = $(lock.indent)", text) &&
           occursin("whitespace_in_kwargs = $(lock.whitespace_in_kwargs)", text) &&
           occursin("remove_extra_newlines = $(lock.remove_extra_newlines)", text)
end

function unique_claim_example_path()
    joinpath(pkgdir(BioDynaX), "examples", "unknown_inhibition.jl")
end

function unique_claim_example_source_violations()
    src = read(unique_claim_example_path(), String)
    missing = [s for s in UNIQUE_CLAIM_EXAMPLE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in UNIQUE_CLAIM_EXAMPLE_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

"""
    validate_network_stays_open_source() -> Bool

`validate_network` must stay a topology/metadata checker. The reference-protocol
single-unknown-term workflow lives in `assert_single_unknown_destruction`.
"""
function validate_network_stays_open_source()
    path = joinpath(pkgdir(BioDynaX), "src", "Network.jl")
    src = read(path, String)
    start = findfirst("function validate_network", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    forbidden = ("reference-protocol", "unique_claim", "unknown_holes", "single-hole",
        "assert_single_unknown_destruction", "NeuralDestructionTerm")
    return !any(occursin(needle, body) for needle in forbidden)
end

# -- Protocol fingerprint -----------------------------------------------------

"""`:protocol` for the 9-IC job; `:smoke` for the 1-IC compile check."""
unique_claim_protocol_kind(; smoke::Bool = false) = smoke ? :smoke : :protocol

function unique_claim_protocol_n_ics(; smoke::Bool = false)
    smoke ? UNIQUE_CLAIM_PROTOCOL.smoke_n_ics : UNIQUE_CLAIM_PROTOCOL.n_ics
end

function unique_claim_protocol_n_points(; smoke::Bool = false)
    smoke ? UNIQUE_CLAIM_PROTOCOL.smoke_n_points : UNIQUE_CLAIM_PROTOCOL.n_points
end

"""
    unique_claim_protocol_ics(; smoke=false)

Initial conditions for the reference-protocol path. Smoke returns the first
`smoke_n_ics` row(s). The full table is the recovery protocol.
"""
function unique_claim_protocol_ics(; smoke::Bool = false)
    ics = _unknown_edge_ics()
    n = unique_claim_protocol_n_ics(; smoke)
    n ≤ length(ics) || throw(ArgumentError(
        "reference-protocol IC request $n exceeds table $(length(ics))"))
    return smoke ? ics[1:n] : ics
end

"""
    unique_claim_is_protocol(; kwargs...) -> Bool

True only for the locked reproduction fingerprint: not smoke, seed 103,
9 ICs, 50 points, adam 100 / bfgs 50. A missing kwarg uses the protocol
default, so `unique_claim_is_protocol()` is true and
`unique_claim_is_protocol(; smoke=true)` is false.
"""
function unique_claim_is_protocol(;
        smoke::Bool = false,
        seed = UNIQUE_CLAIM_PROTOCOL.seed,
        n_ics = UNIQUE_CLAIM_PROTOCOL.n_ics,
        n_points = UNIQUE_CLAIM_PROTOCOL.n_points,
        adam_iterations = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
        bfgs_iterations = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations)
    proto = UNIQUE_CLAIM_PROTOCOL
    smoke && return false
    return seed == proto.seed &&
           n_ics == proto.n_ics &&
           n_points == proto.n_points &&
           adam_iterations == proto.adam_iterations &&
           bfgs_iterations == proto.bfgs_iterations
end

"""
    unique_claim_reproduction(; smoke=false, overrides...)

REPRODUCTION-block NamedTuple. Smoke zeros BFGS and drops discovery
bootstrap/seed because the compile check is not the protocol.
"""
function unique_claim_reproduction(;
        smoke::Bool = false,
        seed = UNIQUE_CLAIM_PROTOCOL.seed,
        n_ics = unique_claim_protocol_n_ics(; smoke),
        n_points = unique_claim_protocol_n_points(; smoke),
        adam_iters = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
        bfgs_iters = smoke ? 0 : UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
        bootstrap = smoke ? nothing : UNIQUE_CLAIM_PROTOCOL.bootstrap,
        discovery_seed = smoke ? nothing : UNIQUE_CLAIM_PROTOCOL.discovery_seed)
    kind = unique_claim_protocol_kind(; smoke)
    return (;
        seed,
        n_ics,
        n_points,
        adam_iters,
        bfgs_iters,
        bootstrap,
        discovery_seed,
        protocol_kind = kind,
        smoke,
        is_protocol = unique_claim_is_protocol(;
            smoke, seed, n_ics, n_points,
            adam_iterations = adam_iters,
            bfgs_iterations = bfgs_iters))
end

"""
    unique_claim_training_budget(; smoke=false)

Adam/BFGS/IC/point budget the reference-example example and recovery job share.
Smoke keeps the protocol Adam count only when the caller passes it; the
default smoke budget used by CI is still 1 IC / 8 points / BFGS 0.
"""
function unique_claim_training_budget(; smoke::Bool = false)
    proto = UNIQUE_CLAIM_PROTOCOL
    repro = unique_claim_reproduction(; smoke)
    return (;
        seed = proto.seed,
        adam_iterations = proto.adam_iterations,
        bfgs_iterations = repro.bfgs_iters,
        n_ics = repro.n_ics,
        n_points = repro.n_points,
        tspan = proto.tspan,
        observation_noise = proto.observation_noise,
        bootstrap = repro.bootstrap,
        discovery_seed = repro.discovery_seed,
        smoke,
        protocol_kind = repro.protocol_kind,
        is_protocol = repro.is_protocol)
end

# -- Identifiability product --------------------------------------------------

"""
    coefficients_are_biological_constants(ident) -> Bool

`false` when `unidentifiable_edge` is true. Missing ident treats the edge
as absent (`false`) so coefficients would *look* identified — that row
must still fail `unique_claim_identifiability_holds`.
"""
function coefficients_are_biological_constants(ident)
    ident === nothing && return true
    hasproperty(ident, :unidentifiable_edge) || return true
    return !ident.unidentifiable_edge
end

function unique_claim_identifiability_holds(ident)
    ident === nothing && return false
    hasproperty(ident, :unidentifiable_edge) || return false
    return ident.unidentifiable_edge === true
end

function assert_unique_claim_identifiability(ident)
    unique_claim_identifiability_holds(ident) || throw(ErrorException(
        "reference protocol requires unidentifiable_edge == true"))
    return ident
end

"""
    identifiability_product(ident; unknown_holes=1)

IDENTIFIABILITY-block NamedTuple. Does not mutate
`production_destruction_tradeoff` return keys.
"""
function identifiability_product(ident; unknown_holes::Integer = 1)
    edge = ident !== nothing && hasproperty(ident, :unidentifiable_edge) ?
           ident.unidentifiable_edge : false
    production = ident !== nothing && hasproperty(ident, :production_param) ?
                 ident.production_param : :k_prod
    collinearity = ident !== nothing && hasproperty(ident, :collinearity) ?
                   ident.collinearity : NaN
    return (;
        unknown_holes,
        unidentifiable_edge = edge === true,
        coefficients_are_biological_constants = !edge,
        production_param = production,
        collinearity,
        practical_not_structural = true)
end

# -- Fit checks ----------------------------------------------------------------

unique_claim_residual_holds(residual) = residual ≤ RECOVERY_THRESHOLDS.data_residual

unique_claim_recall_holds(recall) = recall ≥ RECOVERY_THRESHOLDS.support_recall

function assert_unique_claim_recall(recall)
    unique_claim_recall_holds(recall) || throw(ErrorException(
        "true-monomial recall $(recall) is below RECOVERY_THRESHOLDS.support_recall"))
    return recall
end

unique_claim_f1_meets_skeleton_floor(f1) = f1 ≥ RECOVERY_THRESHOLDS.support_f1_ude

unique_claim_f1_reaches_analytical_gate(f1) = f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean

"""
    unique_claim_kpi_failures(kpis) -> Vector{Symbol}

Failed acceptance checks, in product order. Combined F1 is never appended.
"""
function unique_claim_kpi_failures(kpis)
    failures = Symbol[]
    kpis.unidentifiable_edge === true ||
        push!(failures, :unidentifiable_edge)
    unique_claim_residual_holds(kpis.data_residual) ||
        push!(failures, :data_residual)
    unique_claim_recall_holds(kpis.support_recall) ||
        push!(failures, :support_recall)
    return failures
end

function assert_unique_claim_kpis(kpis)
    failures = unique_claim_kpi_failures(kpis)
    isempty(failures) || throw(ErrorException(
        "reference-protocol KPIs failed: $(join(failures, ", "))"))
    return kpis
end

# -- Discovery extras ---------------------------------------------------------

"""
    unique_claim_truth_support(; family=:hill, order=2)

True implicit support used to label leftover monomials. Hill-from-NN is
not opened by calling this.
"""
function unique_claim_truth_support(; family::Symbol = :hill, order::Int = 2)
    family === :hill && return hill_rate_support(order)
    family === :mm && return mm_rate_support()
    throw(ArgumentError(
        "reference-protocol extras support family must be :hill or :mm; got $family"))
end

function unique_claim_discovery_extras(candidate;
        family::Symbol = :hill, order::Int = 2, atol::Real = 1e-8)
    truth = unique_claim_truth_support(; family, order)
    return discovered_support_extras(
        candidate, truth.numerator, truth.denominator; atol = atol)
end

function unique_claim_discovery_extras(discovery::DiscoveryResult; kwargs...)
    discovery.success && !isempty(discovery.candidates) || return String[]
    return unique_claim_discovery_extras(discovery.candidates[1]; kwargs...)
end

"""
    unique_claim_f1_attempt_verdict(; extras, reaches_clean) -> Symbol

Verdict for `benchmark/ude_f1_attempt.jl`. Reaching the analytical
clean threshold on a surrogate does not reopen Hill-from-NN by itself.
"""
function unique_claim_f1_attempt_verdict(; extras, reaches_clean::Bool)
    reaches_clean && return :reopen_only_after_protocol_holds_clean
    !isempty(extras) && return :extras_remain_claim_stays_recall_plus_residual
    return :no_extras_but_clean_gate_not_reached
end

# -- Single-hole instrument (compiler stays open) -----------------------------

count_unknown_destructions(model::UDEModel) = length(neural_destruction_terms(model))

function count_unknown_destructions(network::BiologicalNetwork)
    compiled = compile_mechanism(network)
    return count(term -> term isa NeuralDestructionTerm, compiled.destruction_terms)
end

# -- Protocol result / stdout -------------------------------------------------

function protocol_result_field_order()
    return PROTOCOL_RESULT_FIELDS
end

function unique_claim_product_blocks()
    return UNIQUE_CLAIM_PRODUCT_BLOCKS
end

function assert_protocol_result_fields(result)
    got = Tuple(keys(result))
    got == PROTOCOL_RESULT_FIELDS || throw(ErrorException(
        "protocol_result field order must be $PROTOCOL_RESULT_FIELDS; got $got"))
    result.canonical_hill_from_nn === false || throw(ErrorException(
        "canonical_hill_from_nn must stay false"))
    result.claim === :recall_plus_data_residual || throw(ErrorException(
        "protocol_result.claim must be :recall_plus_data_residual"))
    return result
end

function protocol_block_positions(text::AbstractString)
    blocks = UNIQUE_CLAIM_PRODUCT_BLOCKS
    found = ntuple(i -> findfirst(string(blocks[i]), text), length(blocks))
    return NamedTuple{blocks}(found)
end

function protocol_block_order_holds(text::AbstractString)
    pos = protocol_block_positions(text)
    starts = [range === nothing ? nothing : first(range) for range in values(pos)]
    any(s -> s === nothing, starts) && return false
    return starts[1] < starts[2] < starts[3] < starts[4]
end

function _protocol_section_between(text, start_label, stop_label)
    ia = findfirst(start_label, text)
    ia === nothing && return ""
    ib = findfirst(stop_label, text)
    stop = ib === nothing ? lastindex(text) : prevind(text, first(ib))
    return String(text[first(ia):stop])
end

"""
    format_protocol_sections(ident; kwargs...)

Split `format_protocol_result` into the four report sections so tests can
fail a single section without matching the whole string.
"""
function format_protocol_sections(ident; kwargs...)
    text = format_protocol_result(ident; kwargs...)
    ident_txt = _protocol_section_between(text, "IDENTIFIABILITY", "FIT")
    fit_txt = _protocol_section_between(text, "FIT", "DISCOVERY")
    disc_txt = _protocol_section_between(text, "DISCOVERY", "REPRODUCTION")
    repro_at = findfirst("REPRODUCTION", text)
    repro_txt = repro_at === nothing ? "" : String(text[first(repro_at):end])
    return (;
        identifiability = ident_txt,
        fit = fit_txt,
        discovery = disc_txt,
        reproduction = repro_txt,
        order_holds = protocol_block_order_holds(text),
        text)
end

"""
    format_recovery_protocol(ude; kwargs...)

Print a recovery-suite row with the same report section order as the
reference-example example. Uses live extras from `protocol_result` when present.
"""
function format_recovery_protocol(ude;
        seed = UNIQUE_CLAIM_PROTOCOL.seed,
        n_ics = UNIQUE_CLAIM_PROTOCOL.n_ics,
        n_points = UNIQUE_CLAIM_PROTOCOL.n_points,
        adam_iters = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
        bfgs_iters = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
        bootstrap = UNIQUE_CLAIM_PROTOCOL.bootstrap,
        discovery_seed = UNIQUE_CLAIM_PROTOCOL.discovery_seed,
        smoke::Bool = false,
        equations = nothing)
    ident = hasproperty(ude, :identifiability) && ude.identifiability !== nothing ?
            ude.identifiability : (; unidentifiable_edge = false)
    result = hasproperty(ude, :protocol_result) && ude.protocol_result !== nothing ?
             ude.protocol_result : build_protocol_result(ude)
    extras = hasproperty(result, :extras) ? result.extras : nothing
    eqs = if equations !== nothing
        equations
    elseif hasproperty(ude, :discovery) && ude.discovery !== nothing &&
           hasproperty(ude.discovery, :equations)
        ude.discovery.equations
    else
        nothing
    end
    return format_protocol_result(ident;
        residual = result.data_residual,
        equations = eqs,
        extras = extras,
        support_f1 = result.support_f1,
        support_recall = result.support_recall,
        unknown_holes = result.unknown_holes,
        seed,
        n_ics,
        n_points,
        adam_iters,
        bfgs_iters,
        bootstrap,
        discovery_seed,
        smoke)
end

# -- Typed protocol vs smoke fingerprint --------------------------------------

"""
    UniqueClaimFingerprint

Typed reproduction object. `:protocol` is seed 103 / 9 ICs / 50 points /
Adam 100 / BFGS 50. `:smoke` is 1 IC / 8 points / BFGS 0 and is not the
protocol. Not exported.
"""
struct UniqueClaimFingerprint
    kind::Symbol
    seed::Int
    n_ics::Int
    n_points::Int
    adam_iterations::Int
    bfgs_iterations::Int
    tspan::Tuple{Float64, Float64}
    bootstrap::Union{Int, Nothing}
    discovery_seed::Union{Int, Nothing}
    observation_noise::Float64
    smoke::Bool
end

function unique_claim_fingerprint(; smoke::Bool = false)
    proto = UNIQUE_CLAIM_PROTOCOL
    if smoke
        return UniqueClaimFingerprint(
            :smoke,
            proto.seed,
            proto.smoke_n_ics,
            proto.smoke_n_points,
            proto.adam_iterations,
            0,
            proto.tspan,
            nothing,
            nothing,
            proto.observation_noise,
            true)
    end
    return UniqueClaimFingerprint(
        :protocol,
        proto.seed,
        proto.n_ics,
        proto.n_points,
        proto.adam_iterations,
        proto.bfgs_iterations,
        proto.tspan,
        proto.bootstrap,
        proto.discovery_seed,
        proto.observation_noise,
        false)
end

function unique_claim_fingerprint(kind::Symbol)
    kind === :smoke ? unique_claim_fingerprint(; smoke = true) :
    kind === :protocol ? unique_claim_fingerprint() :
    throw(ArgumentError("fingerprint kind must be :protocol or :smoke; got $kind"))
end

function unique_claim_fingerprint_is_protocol(fp::UniqueClaimFingerprint)
    proto = UNIQUE_CLAIM_PROTOCOL
    return !fp.smoke &&
           fp.kind === :protocol &&
           fp.seed == proto.seed &&
           fp.n_ics == proto.n_ics &&
           fp.n_points == proto.n_points &&
           fp.adam_iterations == proto.adam_iterations &&
           fp.bfgs_iterations == proto.bfgs_iterations &&
           fp.tspan == proto.tspan &&
           fp.bootstrap == proto.bootstrap &&
           fp.discovery_seed == proto.discovery_seed &&
           fp.observation_noise == proto.observation_noise
end

function unique_claim_fingerprint_is_smoke(fp::UniqueClaimFingerprint)
    proto = UNIQUE_CLAIM_PROTOCOL
    return fp.smoke &&
           fp.kind === :smoke &&
           fp.n_ics == proto.smoke_n_ics &&
           fp.n_points == proto.smoke_n_points &&
           fp.bfgs_iterations == 0 &&
           fp.bootstrap === nothing &&
           fp.discovery_seed === nothing &&
           !unique_claim_fingerprint_is_protocol(fp)
end

function unique_claim_fingerprint_holds(fp::UniqueClaimFingerprint)
    fp.smoke && return unique_claim_fingerprint_is_smoke(fp)
    return unique_claim_fingerprint_is_protocol(fp)
end

function unique_claim_fingerprint_namedtuple(fp::UniqueClaimFingerprint)
    return (;
        kind = fp.kind,
        seed = fp.seed,
        n_ics = fp.n_ics,
        n_points = fp.n_points,
        adam_iterations = fp.adam_iterations,
        bfgs_iterations = fp.bfgs_iterations,
        tspan = fp.tspan,
        bootstrap = fp.bootstrap,
        discovery_seed = fp.discovery_seed,
        observation_noise = fp.observation_noise,
        smoke = fp.smoke,
        is_protocol = unique_claim_fingerprint_is_protocol(fp))
end

function format_unique_claim_fingerprint(fp::UniqueClaimFingerprint)
    io = IOBuffer()
    println(io, "fingerprint_kind: ", fp.kind)
    println(io, "  seed: ", fp.seed)
    println(io, "  n_ics: ", fp.n_ics)
    println(io, "  n_points: ", fp.n_points)
    println(io, "  adam_iterations: ", fp.adam_iterations)
    println(io, "  bfgs_iterations: ", fp.bfgs_iterations)
    println(io, "  tspan: ", fp.tspan)
    println(io, "  bootstrap: ", _format_protocol_value(fp.bootstrap))
    println(io, "  discovery_seed: ", _format_protocol_value(fp.discovery_seed))
    println(io, "  observation_noise: ", fp.observation_noise)
    println(io, "  smoke: ", fp.smoke)
    println(io, "  is_protocol: ", unique_claim_fingerprint_is_protocol(fp))
    return String(take!(io))
end

function format_protocol_result(ident, fingerprint::UniqueClaimFingerprint; kwargs...)
    return format_protocol_result(ident;
        seed = fingerprint.seed,
        n_ics = fingerprint.n_ics,
        n_points = fingerprint.n_points,
        adam_iters = fingerprint.adam_iterations,
        bfgs_iters = fingerprint.bfgs_iterations,
        bootstrap = fingerprint.bootstrap,
        discovery_seed = fingerprint.discovery_seed,
        protocol_kind = fingerprint.kind,
        smoke = fingerprint.smoke,
        kwargs...)
end

function format_recovery_protocol(ude, fingerprint::UniqueClaimFingerprint; kwargs...)
    return format_recovery_protocol(ude;
        seed = fingerprint.seed,
        n_ics = fingerprint.n_ics,
        n_points = fingerprint.n_points,
        adam_iters = fingerprint.adam_iterations,
        bfgs_iters = fingerprint.bfgs_iterations,
        bootstrap = fingerprint.bootstrap,
        discovery_seed = fingerprint.discovery_seed,
        smoke = fingerprint.smoke,
        kwargs...)
end

function unique_claim_reproduction(fp::UniqueClaimFingerprint)
    return unique_claim_reproduction(;
        smoke = fp.smoke,
        seed = fp.seed,
        n_ics = fp.n_ics,
        n_points = fp.n_points,
        adam_iters = fp.adam_iterations,
        bfgs_iters = fp.bfgs_iterations,
        bootstrap = fp.bootstrap,
        discovery_seed = fp.discovery_seed)
end

# -- protocol_result fields vs stdout print order -----------------------------

"""Printed labels inside each report section, in stdout order."""
const PROTOCOL_PRINT_FIELDS = (
    IDENTIFIABILITY = (
        :unknown_holes, :unidentifiable_edge, :coefficients_are_biological_constants),
    FIT = (
        :hybrid_data_residual, :support_recall),
    DISCOVERY = (
        :equations, :support_f1, :extras, :canonical_hill_from_nn, :claim),
    REPRODUCTION = (
        :seed, :n_ics, :n_points, :adam_iters, :bfgs_iters, :bootstrap,
        :discovery_seed, :protocol_kind, :smoke))

"""Maps `PROTOCOL_RESULT_FIELDS` to the string `format_protocol_result` prints."""
const PROTOCOL_RESULT_PRINT_LABELS = (
    unknown_holes = "unknown_holes",
    unidentifiable_edge = "unidentifiable_edge",
    coefficients_are_biological_constants = "coefficients_are_biological_constants",
    data_residual = "hybrid_data_residual",
    support_recall = "support_recall",
    support_f1 = "support_f1",
    extras = "extras",
    canonical_hill_from_nn = "canonical_hill_from_nn",
    claim = "acceptance_criteria")

function protocol_print_fields()
    return PROTOCOL_PRINT_FIELDS
end

function protocol_result_print_labels()
    return PROTOCOL_RESULT_PRINT_LABELS
end

function protocol_result_field_to_print_label(field::Symbol)
    labels = PROTOCOL_RESULT_PRINT_LABELS
    hasproperty(labels, field) || throw(ArgumentError(
        "protocol_result field $field is not in PROTOCOL_RESULT_FIELDS"))
    return getproperty(labels, field)
end

function protocol_result_print_order()
    labels = Symbol[]
    for block in UNIQUE_CLAIM_PRODUCT_BLOCKS
        append!(labels, getproperty(PROTOCOL_PRINT_FIELDS, block))
    end
    return Tuple(labels)
end

"""
    extras_print_label(extras) -> String

Formatting of the DISCOVERY extras line. `nothing` is unscored (`NA`).
An empty collection is `(none)`. Live leftovers are joined. The F1
attempt leftover pair is never invented here.
"""
extras_print_label(extras) = _format_protocol_extras(extras)

function extras_print_is_hardcoded_attempt(label::AbstractString)
    return occursin("1, r remain after the UDE F1 attempt", label)
end

function format_protocol_result_field_order_holds(text::AbstractString)
    protocol_block_order_holds(text) || return false
    labels = [string(protocol_result_field_to_print_label(field), ":")
              for field in PROTOCOL_RESULT_FIELDS]
    starts = Int[]
    for label in labels
        range = findfirst(label, text)
        range === nothing && return false
        push!(starts, first(range))
    end
    return issorted(starts)
end

function format_protocol_print_labels_hold(text::AbstractString)
    for label in protocol_result_print_order()
        printed = hasproperty(PROTOCOL_RESULT_PRINT_LABELS, label) ?
                  getproperty(PROTOCOL_RESULT_PRINT_LABELS, label) : string(label)
        occursin(string(printed, ":"), text) || return false
    end
    return extras_print_is_hardcoded_attempt(text) == false
end

function assert_format_matches_protocol_result(result, text::AbstractString)
    format_protocol_result_field_order_holds(text) || throw(ErrorException(
        "format_protocol_result print order must follow PROTOCOL_RESULT_FIELDS"))
    occursin("unknown_holes: $(result.unknown_holes)", text) ||
        throw(ErrorException("printed unknown_holes does not match protocol_result"))
    occursin("unidentifiable_edge: $(result.unidentifiable_edge)", text) ||
        throw(ErrorException("printed unidentifiable_edge does not match protocol_result"))
    occursin(
        "coefficients_are_biological_constants: $(result.coefficients_are_biological_constants)",
        text) || throw(ErrorException(
        "printed coefficients_are_biological_constants does not match protocol_result"))
    occursin(
        "hybrid_data_residual: $(_format_protocol_value(result.data_residual))", text) ||
        throw(ErrorException("printed hybrid_data_residual does not match data_residual"))
    occursin("support_recall: $(_format_protocol_value(result.support_recall))", text) ||
        throw(ErrorException("printed support_recall does not match protocol_result"))
    occursin("support_f1: $(_format_protocol_value(result.support_f1))", text) ||
        throw(ErrorException("printed support_f1 does not match protocol_result"))
    extras_line = extras_print_label(result.extras)
    occursin("extras: $(extras_line)", text) || throw(ErrorException(
        "printed extras do not match protocol_result"))
    occursin("canonical_hill_from_nn: false", text) || throw(ErrorException(
        "printed canonical_hill_from_nn must stay false"))
    occursin("acceptance_criteria: $(_acceptance_criteria_label())", text) ||
        throw(ErrorException("printed acceptance criteria do not match the thresholds"))
    extras_print_is_hardcoded_attempt(text) && throw(ErrorException(
        "format_protocol_result must not invent UDE F1-attempt extras"))
    return text
end

# -- Recovery-path hole admission (compiler stays open) -----------------------

unique_claim_recovery_admits(model::UDEModel) = count_unknown_destructions(model) == 1

function unique_claim_recovery_admits(network::BiologicalNetwork)
    count_unknown_destructions(network) == 1
end

function unique_claim_compiler_stays_open(network::BiologicalNetwork)
    validate_network(network) === network || return false
    compiled = compile_mechanism(network)
    return compiled isa CompiledMechanism && neural_index_is_dense(compiled)
end

"""
    assert_unique_claim_recovery_network(network) -> network

`validate_network` is called and stays a topology checker. The reference-protocol
recovery instrument then requires exactly one unknown `D(z)`.
"""
function assert_unique_claim_recovery_network(network::BiologicalNetwork)
    validate_network(network)
    n = count_unknown_destructions(network)
    n == 1 || throw(ErrorException(
        "reference-protocol recovery requires exactly one unknown destruction D(z); got $n"))
    return network
end

function unique_claim_recovery_admission(network::BiologicalNetwork)
    n = count_unknown_destructions(network)
    return (;
        unknown_holes = n,
        validate_open = unique_claim_compiler_stays_open(network),
        recovery_admits = n == 1,
        single_hole_in_validate_network = validate_network_stays_open_source() == false)
end

function recovery_suite_uses_single_hole_instrument()
    path = joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
    src = read(path, String)
    return occursin("only_unknown_destruction", src) &&
           occursin("admit_recovery_suite_network", src) &&
           occursin("function run_recovery_suite", src)
end

# -- F1 attempt probe (not the protocol) --------------------------------------

"""Description of `benchmark/ude_f1_attempt.jl`. Not a reproduction fingerprint."""
const UNIQUE_CLAIM_F1_ATTEMPT = (
    is_protocol = false,
    trains_ude = false,
    n_ics = 0,
    uses_protocol_ics = false,
    library = :same_monomial,
    new_atoms = false,
    support_f1_ude = 0.50,
    support_f1_clean = 0.99,
    script = "benchmark/ude_f1_attempt.jl")

const UNIQUE_CLAIM_F1_ATTEMPT_MUST_CONTAIN = (
    "same library",
    "No new atoms",
    "unique_claim_discovery_extras",
    "unique_claim_f1_attempt_verdict",
    "support_f1_clean",
    "support_f1_ude",
    "not the recovery protocol",
    "UNIQUE_CLAIM_F1_ATTEMPT")

const UNIQUE_CLAIM_F1_ATTEMPT_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "canonical Hill from a trained NN is open",
    "unique_claim_protocol_ics()",
    "train_ude",
    "train_experiments")

function unique_claim_f1_attempt_path()
    joinpath(pkgdir(BioDynaX), UNIQUE_CLAIM_F1_ATTEMPT.script)
end

function unique_claim_f1_attempt_source_violations()
    src = read(unique_claim_f1_attempt_path(), String)
    missing = [s for s in UNIQUE_CLAIM_F1_ATTEMPT_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in UNIQUE_CLAIM_F1_ATTEMPT_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

function unique_claim_f1_attempt_row(; extras, f1)
    reaches_clean = unique_claim_f1_reaches_analytical_gate(f1)
    meets_skeleton = unique_claim_f1_meets_skeleton_floor(f1)
    return (;
        extras,
        f1,
        reaches_clean,
        meets_skeleton,
        is_protocol = false,
        verdict = unique_claim_f1_attempt_verdict(; extras, reaches_clean))
end

function unique_claim_f1_attempt_contract()
    lock = recovery_thresholds_lock()
    return (;
        UNIQUE_CLAIM_F1_ATTEMPT...,
        support_f1_ude = lock.support_f1_ude,
        support_f1_clean = lock.support_f1_clean,
        is_protocol = false)
end

function unique_claim_f1_attempt_holds()
    contract = unique_claim_f1_attempt_contract()
    violations = unique_claim_f1_attempt_source_violations()
    return contract.is_protocol == false &&
           contract.trains_ude == false &&
           contract.n_ics == 0 &&
           contract.uses_protocol_ics == false &&
           contract.new_atoms == false &&
           contract.support_f1_ude == RECOVERY_THRESHOLDS.support_f1_ude &&
           contract.support_f1_clean == RECOVERY_THRESHOLDS.support_f1_clean &&
           isempty(violations.missing) &&
           isempty(violations.forbidden)
end
