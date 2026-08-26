###############################################################################
# Unique-claim product contract (not exported).
#
# Print and gate order is IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION.
# This file does not change RECOVERY_THRESHOLDS, the public export list, or
# validate_network. Canonical Hill from a trained NN stays closed.
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
    :StructuralPositivity,
    :TrainingConfig,
    :TrainingResult,
    :TrainingRetcode,
    :UDEModel,
    :UNKNOWN_NN,
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
    :equation_to_function,
    :equation_to_latex,
    :estimate_derivatives,
    :experiment_from_csv,
    :export_rhs,
    :generate_experiment_set,
    :hybrid_data_residual,
    :local_basis,
    :pack_parameters,
    :parameter_schema,
    :positive_parameter,
    :predict_ude,
    :sample_unknown_destruction,
    :state_nodes,
    :train_experiments,
    :train_ude,
    :ude_rhs!,
    :ude_system,
    :validate_network,
    :write_experiment_csv)

"""Phrases that must not appear on user-facing landing docs."""
const UNIQUE_CLAIM_DOCS_FORBIDDEN_PHRASES = (
    "HTTP 200",
    "TagBot ran",
    "]add BioDynaX")

"""Golden-path source strings the protocol fingerprint requires."""
const UNIQUE_CLAIM_EXAMPLE_MUST_CONTAIN = (
    "UNIQUE_CLAIM_PROTOCOL.seed",
    "UNIQUE_CLAIM_PROTOCOL.adam_iterations",
    "UNIQUE_CLAIM_PROTOCOL.bfgs_iterations",
    "unique_claim_protocol_ics",
    "unique_claim_protocol_n_points",
    "unique_claim_discovery_config",
    "unique_claim_discovery_extras",
    "production_destruction_tradeoff",
    "assert_single_unknown_destruction",
    "format_protocol_result",
    "assert_unique_claim_residual",
    "sample_unknown_destruction_grid",
    "_regulator_grid",
    "ReactionSpec",
    "HillMetadata")

"""Golden-path source strings that would reopen a closed claim or fixture."""
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

"""True when the public export list is exactly the freeze-plus-golden-path set."""
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

function unique_claim_user_doc_paths()
    root = pkgdir(BioDynaX)
    return (
        joinpath(root, "README.md"),
        joinpath(root, "docs", "src", "index.md"),
        joinpath(root, "docs", "src", "tutorial.md"),
        joinpath(root, "docs", "src", "howto.md"),
        joinpath(root, "docs", "src", "unique-claim.md"))
end

"""Hits of ops-lab phrases on landing docs. Empty is the honest state."""
function unique_claim_docs_forbidden_hits()
    hits = String[]
    for path in unique_claim_user_doc_paths()
        isfile(path) || continue
        text = read(path, String)
        for phrase in UNIQUE_CLAIM_DOCS_FORBIDDEN_PHRASES
            occursin(phrase, text) &&
                push!(hits, string(basename(path), ": ", phrase))
        end
    end
    return hits
end

function unique_claim_example_source_violations()
    src = read(unique_claim_example_path(), String)
    missing = [s for s in UNIQUE_CLAIM_EXAMPLE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in UNIQUE_CLAIM_EXAMPLE_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

"""
    validate_network_stays_open_source() -> Bool

`validate_network` must stay a topology/metadata checker. The unique-claim
single-hole instrument lives in `assert_single_unknown_destruction`.
"""
function validate_network_stays_open_source()
    path = joinpath(pkgdir(BioDynaX), "src", "Network.jl")
    src = read(path, String)
    start = findfirst("function validate_network", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    forbidden = ("unique-claim", "unique_claim", "unknown_holes", "single-hole",
        "assert_single_unknown_destruction", "NeuralDestructionTerm")
    return !any(occursin(needle, body) for needle in forbidden)
end

"""Locked English sentences the product page must keep honest."""
unique_claim_locked_sentences() = (;
    claim = "Unique claim: recall, hybrid residual versus data, unidentifiable_edge.",
    f1 = "Combined support F1 is a skeleton floor (0.50), not the UDE claim.",
    hill = "Canonical Hill from a trained neural rate is closed.",
    coefficients = "Coefficients are not biological constants when the edge is unidentifiable.",
    smoke = "BIODYNAX_SMOKE=1 (1 IC / 8 points) is not the seed-103 / 9-IC protocol.",
    preview = "Research preview. Not v1.0. Not in General.")

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

Initial conditions for the unique-claim path. Smoke returns the first
`smoke_n_ics` row(s). The full table is the recovery protocol.
"""
function unique_claim_protocol_ics(; smoke::Bool = false)
    ics = _unknown_edge_ics()
    n = unique_claim_protocol_n_ics(; smoke)
    n ≤ length(ics) || throw(ArgumentError(
        "unique-claim IC request $n exceeds table $(length(ics))"))
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

Adam/BFGS/IC/point budget the golden-path example and recovery job share.
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
        "unique-claim protocol requires unidentifiable_edge == true"))
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

# -- Fit gates ----------------------------------------------------------------

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

Failed claim gates, in product order. Combined F1 is never appended.
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
        "unique-claim KPIs failed: $(join(failures, ", "))"))
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
        "unique-claim extras support family must be :hill or :mm; got $family"))
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

Honesty verdict for `benchmark/ude_f1_attempt.jl`. Reaching the analytical
clean gate on a surrogate does not reopen Hill-from-NN by itself.
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

Split `format_protocol_result` into the four product blocks so tests can
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

Print a recovery-suite row with the same product block order as the
golden-path example. Uses live extras from `protocol_result` when present.
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
    ident = hasproperty(ude, :identifiability) ? ude.identifiability :
            (; unidentifiable_edge = false)
    result = hasproperty(ude, :protocol_result) ? ude.protocol_result :
             build_protocol_result(ude)
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
