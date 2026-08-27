###############################################################################
# Docs executable path (not exported).
#
# One page joins the H–L surfaces (hybrid residual, identifiability
# product, graph-local library, denominator domain, parameter schema
# pack) with executable helpers. Landing leftover scanners walk
# tutorial / howto / sciml / architecture. This file does not restate
# A–G pages (DiscoveryWorkspace, TrainingReuse, SciMLSolveSurface,
# RecoverySuiteSkip, ExperimentCheckpoint, FailureModes, HybridCompose).
#
# Does not grow exports. Does not drop protocol ICs. Does not open
# Hill-from-NN. Combined F1 stays a skeleton floor.
###############################################################################

const DOCS_EXECUTABLE_MUST_CONTAIN = (
    "function docs_executable_join_row",
    "function leftover_contradiction_hits",
    "function tutorial_mentions_hl_row",
    "function howto_links_hl_row",
    "function sciml_carries_hl_sentences_row",
    "struct DocsExecutableRow",
    "function live_hl_kinetic_join_row",
    "function no_restated_ag_pages_row")

const DOCS_EXECUTABLE_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

const DOCS_EXECUTABLE_HL_PAGES = (
    "hybrid-residual.md",
    "identifiability-product.md",
    "graph-local-library.md",
    "denominator-domain.md",
    "parameter-schema-pack.md")

const DOCS_EXECUTABLE_AG_PAGES = (
    "discovery-streaming.md",
    "training-reuse.md",
    "sciml-solve-surface.md",
    "recovery-suite-skip.md",
    "experiment-checkpoint.md",
    "failure-modes.md",
    "hybrid-compose.md")

const DOCS_EXECUTABLE_LEFTOVER_PHRASES = (
    "k_custom is absent",
    "k_custom is missing from parameter_schema",
    "CustomDestructionTerm is missing from parameter_schema",
    "denominator_violation_count is ImplicitCandidate-only",
    "local_basis target is a node id",
    "support_f1_ude = 0.99",
    "]add BioDynaX",
    "HTTP 200",
    "TagBot ran")

function docs_executable_locked_sentences()
    return (;
        join = "The executable docs path joins hybrid residual, identifiability product, graph-local library, denominator domain, and parameter schema pack.",
        leftover = "tutorial, howto, and sciml must not restate closed H–L holes as current facts.",
        protocol = "Smoke (1 IC / 8 points) is not the seed-103 / 9-IC protocol.",
        ag = "This page does not restate the A–G workspaces.")
end

docs_executable_contract() = docs_executable_locked_sentences().join

function docs_executable_source_path()
    joinpath(pkgdir(BioDynaX), "src", "DocsExecutable.jl")
end

function docs_executable_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "docs-executable.md")
end

function docs_executable_test_path()
    joinpath(pkgdir(BioDynaX), "test", "test_docs_executable.jl")
end

function docs_executable_landing_paths()
    root = joinpath(pkgdir(BioDynaX), "docs", "src")
    return (
        joinpath(root, "tutorial.md"),
        joinpath(root, "howto.md"),
        joinpath(root, "sciml.md"),
        joinpath(root, "architecture.md"))
end

# -- Join H–L contracts -------------------------------------------------------

function docs_hl_contract_strings()
    return (
        hybrid_residual_contract(),
        identifiability_product_contract(),
        graph_local_library_contract(),
        denominator_domain_contract(),
        parameter_schema_pack_contract())
end

function docs_executable_join_row()
    residual = hybrid_residual_contract()
    ident = identifiability_product_contract()
    library = graph_local_library_contract()
    denom = denominator_domain_contract()
    schema = parameter_schema_pack_contract()
    sentences = docs_hl_contract_strings()
    return (;
        residual, ident, library, denom, schema,
        n = length(sentences),
        holds = residual == hybrid_residual_locked_sentences().solve &&
                ident == identifiability_product_locked_sentences().join &&
                library == graph_local_library_locked_sentences().prior &&
                denom == denominator_domain_locked_sentences().split &&
                schema == parameter_schema_pack_locked_sentences().custom &&
                length(unique(sentences)) == 5)
end

struct DocsExecutableRow
    name::Symbol
    n_surfaces::Int
    leftover::Int
    holds::Bool
end

function docs_executable_row(name::Symbol)
    join = docs_executable_join_row()
    leftovers = leftover_contradiction_hits()
    typed = DocsExecutableRow(name, join.n, length(leftovers),
        join.holds && isempty(leftovers))
    return (; join, leftovers, typed, holds = typed.holds)
end

function docs_executable_row_namedtuple(row::DocsExecutableRow)
    return (;
        name = row.name,
        n_surfaces = row.n_surfaces,
        leftover = row.leftover,
        holds = row.holds)
end

# -- Leftover scanners --------------------------------------------------------

function leftover_contradiction_hits()
    hits = String[]
    for path in docs_executable_landing_paths()
        isfile(path) || continue
        text = read(path, String)
        for phrase in DOCS_EXECUTABLE_LEFTOVER_PHRASES
            occursin(phrase, text) || continue
            push!(hits, string(basename(path), ": ", phrase))
        end
    end
    return hits
end

function leftover_contradiction_row()
    hits = leftover_contradiction_hits()
    return (;
        n = length(hits),
        hits,
        holds = isempty(hits))
end

function landing_file_exists_row()
    paths = docs_executable_landing_paths()
    return (;
        n = length(paths),
        holds = all(isfile, paths) && length(paths) == 4)
end

function tutorial_mentions_hl_row()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"), String)
    sentences = docs_executable_locked_sentences()
    return (;
        join = occursin(sentences.join, text),
        residual = occursin("hybrid-residual", text),
        ident = occursin("identifiability-product", text),
        library = occursin("graph-local-library", text),
        denom = occursin("denominator-domain", text),
        schema = occursin("parameter-schema-pack", text),
        holds = occursin(sentences.join, text) &&
                occursin("hybrid-residual", text) &&
                occursin("identifiability-product", text) &&
                occursin("graph-local-library", text) &&
                occursin("denominator-domain", text) &&
                occursin("parameter-schema-pack", text) &&
                occursin("9", text) && occursin("103", text))
end

function howto_links_hl_row()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    return (;
        pages = count(p -> occursin(p, text), DOCS_EXECUTABLE_HL_PAGES),
        docs_exec = occursin("docs-executable", text),
        holds = all(p -> occursin(p, text), DOCS_EXECUTABLE_HL_PAGES) &&
                occursin("docs-executable", text) &&
                occursin("unpack_parameters", text) &&
                occursin("denominator_split_counts", text))
end

function sciml_carries_hl_sentences_row()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    needed = docs_hl_contract_strings()
    return (;
        n = count(s -> occursin(s, text), needed),
        holds = all(s -> occursin(s, text), needed) &&
                occursin(docs_executable_locked_sentences().join, text))
end

function architecture_carries_hl_links_row()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "architecture.md"), String)
    return (;
        pages = count(p -> occursin(p, text), DOCS_EXECUTABLE_HL_PAGES),
        holds = all(p -> occursin(p, text), DOCS_EXECUTABLE_HL_PAGES) &&
                occursin("docs-executable", text))
end

function no_restated_ag_pages_row()
    path = docs_executable_docs_path()
    isfile(path) || return (; holds = false)
    text = read(path, String)
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    restated = String[]
    for page in DOCS_EXECUTABLE_AG_PAGES
        occursin("This page restates $page", text) && push!(restated, page)
    end
    return (;
        restated,
        make_keeps_ag = all(p -> occursin(p, make), DOCS_EXECUTABLE_AG_PAGES),
        holds = isempty(restated) &&
                occursin("does not restate the A–G workspaces", text) &&
                all(p -> occursin(p, make), DOCS_EXECUTABLE_AG_PAGES))
end

function unique_claim_paths_include_hl_row()
    joined = join(unique_claim_user_doc_paths(), " ")
    return (;
        n = count(p -> occursin(p, joined), DOCS_EXECUTABLE_HL_PAGES),
        exec = occursin("docs-executable.md", joined),
        holds = all(p -> occursin(p, joined), DOCS_EXECUTABLE_HL_PAGES) &&
                occursin("docs-executable.md", joined))
end

function make_lists_hl_and_exec_row()
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    return (;
        holds = all(p -> occursin(p, make), DOCS_EXECUTABLE_HL_PAGES) &&
                occursin("docs-executable.md", make))
end

# -- Live executable joins (cheap, not the protocol) --------------------------

function live_hl_kinetic_join_row()
    kinetic = kinetic_custom_in_schema_row()
    den = safe_split_row()
    lib = hill_unknown_library_row()
    extras = extras_empty_still_splits_row()
    return (;
        kinetic_custom = kinetic.has_custom,
        den_clean = den.holds,
        lib = lib.holds,
        extras = extras.holds,
        holds = kinetic.holds && kinetic.has_custom &&
                den.holds && lib.holds && extras.holds)
end

function live_hl_remapped_join_row()
    pack = remapped_pack_unpack_row()
    den = remapped_denominator_row()
    lib = remapped_library_row()
    return (;
        nn_heads = pack.nn_heads,
        admits = unique_claim_recovery_admits(
            build_remapped_two_regulator_network()),
        holds = pack.holds && den.holds && lib.holds &&
                pack.nn_heads == 2 &&
                unique_claim_recovery_admits(
                    build_remapped_two_regulator_network()) == false)
end

function live_hl_linear_join_row()
    pack = pack_unpack_linear_row()
    den = linear_zero_denominator_row()
    schema = linear_schema_names_are_mass_action_row()
    return (;
        holds = pack.holds && den.holds && schema.holds)
end

function live_hl_default_join_row()
    schema = default_example_schema_row()
    den = default_example_denominator_row()
    lib = default_example_library_row()
    return (;
        holds = schema.holds && den.holds && lib.holds)
end

function live_hl_smoke_protocol_row()
    smoke = unique_claim_fingerprint(; smoke = true)
    proto = unique_claim_fingerprint()
    return (;
        smoke_ics = smoke.n_ics,
        proto_ics = proto.n_ics,
        holds = smoke.n_ics == 1 && proto.n_ics == 9 &&
                proto.n_points == 50 && proto.seed == 103 &&
                !proto.smoke &&
                docs_executable_locked_sentences().protocol ==
                "Smoke (1 IC / 8 points) is not the seed-103 / 9-IC protocol.")
end

function live_hl_thresholds_row()
    return (;
        ude = RECOVERY_THRESHOLDS.support_f1_ude,
        clean = RECOVERY_THRESHOLDS.support_f1_clean,
        holds = RECOVERY_THRESHOLDS.support_f1_ude == 0.50 &&
                RECOVERY_THRESHOLDS.support_f1_clean == 0.99 &&
                :support_f1 ∉ UNIQUE_CLAIM_KPI_FIELDS)
end

function live_hl_exports_row()
    return (;
        holds = !(:docs_executable_join_row in names(BioDynaX)) &&
                !(:DocsExecutableRow in names(BioDynaX)) &&
                !(:leftover_contradiction_hits in names(BioDynaX)) &&
                :pack_parameters in names(BioDynaX) &&
                :hybrid_data_residual in names(BioDynaX) &&
                public_export_list_holds())
end

# -- Page / source locks ------------------------------------------------------

function docs_executable_source_holds()
    src = read(docs_executable_source_path(), String)
    docs = isfile(docs_executable_docs_path()) ?
           read(docs_executable_docs_path(), String) : ""
    return all(occursin(needle, src) for needle in DOCS_EXECUTABLE_MUST_CONTAIN) &&
           !occursin("support_f1_ude = 0.99", docs) &&
           !occursin("function validate_network", docs)
end

function docs_executable_source_violations()
    src = read(docs_executable_source_path(), String)
    docs = isfile(docs_executable_docs_path()) ?
           read(docs_executable_docs_path(), String) : ""
    missing = [s for s in DOCS_EXECUTABLE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = String[]
    occursin("support_f1_ude = 0.99", docs) &&
        push!(forbidden, "docs: support_f1_ude = 0.99")
    occursin("function validate_network", docs) &&
        push!(forbidden, "docs: function validate_network")
    return (; missing, forbidden)
end

function docs_executable_docs_hold()
    path = docs_executable_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(docs_executable_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("docs-executable.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function docs_executable_landing_docs_hold()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    sentences = docs_executable_locked_sentences()
    return occursin("docs-executable", howto) &&
           occursin("docs_executable_join_row", howto) &&
           occursin(sentences.join, sciml)
end

function docs_executable_example_source_holds()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    docs = read(docs_executable_docs_path(), String)
    tutorial = read(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"), String)
    return occursin("docs_executable_join_row", howto) &&
           occursin("leftover_contradiction_hits", docs) &&
           occursin("H–L", docs) &&
           occursin("docs-executable", tutorial)
end

function docs_executable_docs_mention_helpers()
    path = docs_executable_docs_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("docs_executable_join_row", text) &&
           occursin("leftover_contradiction_hits", text) &&
           occursin("live_hl_kinetic_join_row", text) &&
           occursin("no_restated_ag_pages_row", text)
end

function docs_executable_test_file_holds()
    path = docs_executable_test_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("docs_executable_contract_holds", text) &&
           occursin("public_export_list_holds", text) &&
           occursin("RECOVERY_THRESHOLDS.support_f1_ude == 0.50", text)
end

function docs_executable_module_include_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "BioDynaX.jl"), String)
    tests = read(joinpath(pkgdir(BioDynaX), "test", "runtests.jl"), String)
    return occursin("include(\"DocsExecutable.jl\")", src) &&
           occursin("test_docs_executable.jl", tests)
end

function recovery_thresholds_untouched_docs_row()
    lock = recovery_thresholds_lock()
    return (;
        holds = RECOVERY_THRESHOLDS == lock &&
                lock.support_f1_ude == 0.50 &&
                lock.support_f1_clean == 0.99)
end

function unique_claim_not_faster_docs_row()
    fp = unique_claim_fingerprint()
    return (;
        n_ics = fp.n_ics,
        holds = fp.n_ics == 9 && fp.n_points == 50 &&
                fp.seed == 103 && !fp.smoke)
end

function hill_from_nn_closed_docs_row()
    return (;
        holds = :canonical_hill_from_nn in PROTOCOL_RESULT_FIELDS &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50)
end

function validate_open_docs_row()
    linear = build_linear_test_network()
    dual = build_dual_unknown_network()
    return (;
        holds = validate_network_stays_open_source() &&
                validate_network(linear) === linear &&
                validate_network(dual) === dual)
end

# -- Catalog ------------------------------------------------------------------

function docs_executable_fixture_names()
    return (
        :join, :leftover, :landing_exists, :tutorial, :howto, :sciml,
        :architecture, :no_ag, :paths, :make, :kinetic, :remap,
        :linear, :default, :smoke, :thresholds, :exports)
end

function docs_executable_fixture_matrix()
    join = docs_executable_join_row()
    leftover = leftover_contradiction_row()
    exists = landing_file_exists_row()
    tutorial = tutorial_mentions_hl_row()
    howto = howto_links_hl_row()
    sciml = sciml_carries_hl_sentences_row()
    architecture = architecture_carries_hl_links_row()
    no_ag = no_restated_ag_pages_row()
    paths = unique_claim_paths_include_hl_row()
    make = make_lists_hl_and_exec_row()
    kinetic = live_hl_kinetic_join_row()
    remap = live_hl_remapped_join_row()
    linear = live_hl_linear_join_row()
    default = live_hl_default_join_row()
    smoke = live_hl_smoke_protocol_row()
    thresholds = live_hl_thresholds_row()
    exports = live_hl_exports_row()
    return (;
        join, leftover, exists, tutorial, howto, sciml, architecture,
        no_ag, paths, make, kinetic, remap, linear, default, smoke,
        thresholds, exports,
        holds = join.holds && leftover.holds && exists.holds &&
                tutorial.holds && howto.holds && sciml.holds &&
                architecture.holds && no_ag.holds && paths.holds &&
                make.holds && kinetic.holds && remap.holds &&
                linear.holds && default.holds && smoke.holds &&
                thresholds.holds && exports.holds)
end

function docs_executable_typed_matrix()
    row = docs_executable_row(:join)
    return (;
        named = docs_executable_row_namedtuple(row.typed),
        holds = row.holds && row.typed.n_surfaces == 5 &&
                row.typed.leftover == 0)
end

function format_docs_executable_index()
    io = IOBuffer()
    println(io, "| row | meaning |")
    println(io, "|---|---|")
    println(io, "| join | H–L contract sentences stay distinct |")
    println(io, "| leftover | landing pages drop closed-hole phrases |")
    println(io, "| landing_exists | tutorial / howto / sciml / architecture |")
    println(io, "| tutorial | tutorial names the five H–L pages |")
    println(io, "| howto | howto links the five H–L pages |")
    println(io, "| sciml | sciml carries the five H–L sentences |")
    println(io, "| architecture | architecture links H–L and this page |")
    println(io, "| no_ag | this page does not restate A–G |")
    println(io, "| paths | unique_claim_user_doc_paths includes H–L |")
    println(io, "| make | make.jl lists H–L and docs-executable |")
    println(io, "| kinetic | live :k_custom + safe denominator + Hill library |")
    println(io, "| remap | remapped pack / denominator / library |")
    println(io, "| linear | linear pack / denominator / schema names |")
    println(io, "| default | p53/Mdm2 schema / denominator / library |")
    println(io, "| smoke | 1 IC is not 9 ICs / 50 points |")
    println(io, "| thresholds | UDE F1 floor stays 0.50 |")
    println(io, "| exports | helpers stay unexported |")
    return String(take!(io))
end

function docs_executable_index_holds()
    text = format_docs_executable_index()
    names = docs_executable_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("H–L", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function leftover_hits_on(path::AbstractString)
    isfile(path) || return String[]
    text = read(path, String)
    return [phrase for phrase in DOCS_EXECUTABLE_LEFTOVER_PHRASES
            if occursin(phrase, text)]
end

function leftover_tutorial_row()
    hits = leftover_hits_on(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"))
    return (; n = length(hits), hits, holds = isempty(hits))
end

function leftover_howto_row()
    hits = leftover_hits_on(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"))
    return (; n = length(hits), hits, holds = isempty(hits))
end

function leftover_sciml_row()
    hits = leftover_hits_on(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"))
    return (; n = length(hits), hits, holds = isempty(hits))
end

function leftover_architecture_row()
    hits = leftover_hits_on(joinpath(pkgdir(BioDynaX), "docs", "src",
        "architecture.md"))
    return (; n = length(hits), hits, holds = isempty(hits))
end

function leftover_hl_pages_row()
    root = joinpath(pkgdir(BioDynaX), "docs", "src")
    hits = String[]
    for page in DOCS_EXECUTABLE_HL_PAGES
        path = joinpath(root, page)
        isfile(path) || (push!(hits, "missing $page"); continue)
        text = read(path, String)
        occursin("support_f1_ude = 0.99", text) &&
            push!(hits, "$page: support_f1_ude = 0.99")
        occursin("]add BioDynaX", text) &&
            push!(hits, "$page: ]add BioDynaX")
        occursin("HTTP 200", text) && push!(hits, "$page: HTTP 200")
        occursin("TagBot ran", text) && push!(hits, "$page: TagBot ran")
    end
    return (; n = length(hits), hits, holds = isempty(hits))
end

function live_hl_hill_join_row()
    lib = hill_unknown_library_row()
    den = hill_unknown_denominator_row()
    schema = hill_unknown_schema_row()
    return (; holds = lib.holds && den.holds && schema.holds)
end

function live_hl_mm_join_row()
    lib = mm_unknown_library_row()
    den = mm_unknown_denominator_row()
    schema = mm_unknown_schema_row()
    return (; holds = lib.holds && den.holds && schema.holds)
end

function live_hl_three_join_row()
    lib = three_state_library_row()
    den = three_state_denominator_row()
    schema = three_state_schema_row()
    return (; holds = lib.holds && den.holds && schema.holds)
end

function live_hl_dual_join_row()
    lib = dual_library_row()
    den = dual_denominator_row()
    schema = dual_schema_matches_heads_row()
    return (;
        admits = lib.admits,
        holds = lib.holds && den.holds && schema.holds &&
                lib.admits == false)
end

function live_hl_wrong_graph_join_row()
    lib = wrong_graph_library_row()
    den = wrong_graph_denominator_row()
    schema = wrong_graph_schema_row()
    return (; holds = lib.holds && den.holds && schema.holds)
end

function live_hl_repressilator_join_row()
    den = repressilator_denominator_row()
    schema = repressilator_schema_row()
    return (; holds = den.holds && schema.holds)
end

function live_hl_extras_join_row()
    live = extras_denominator_live_row()
    empty = extras_empty_still_splits_row()
    na = extras_nothing_is_na_row()
    hard = extras_hardcoded_attempt_rejected_row()
    return (;
        holds = live.holds && empty.holds && na.holds && hard.holds)
end

function live_hl_frozen_join_row()
    z = frozen_phys_zero_gradient_row()
    r = frozen_phys_restore_row()
    c = frozen_phys_config_copy_row()
    return (; holds = z.holds && r.holds && c.holds)
end

function live_hl_parent_gate_join_row()
    none = nothing_candidate_gate_row()
    scope = default_scope_is_graph_row()
    return (; holds = none.holds && scope.holds)
end

function executable_snippet_join()
    return docs_executable_join_row().holds &&
           leftover_contradiction_row().holds &&
           live_hl_kinetic_join_row().holds
end

function executable_snippet_protocol()
    fp = unique_claim_fingerprint()
    return fp.n_ics == 9 && fp.n_points == 50 && fp.seed == 103 && !fp.smoke
end

function executable_snippet_thresholds()
    return RECOVERY_THRESHOLDS.support_f1_ude == 0.50 &&
           RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

function executable_snippets_row()
    return (;
        join = executable_snippet_join(),
        protocol = executable_snippet_protocol(),
        thresholds = executable_snippet_thresholds(),
        holds = executable_snippet_join() &&
                executable_snippet_protocol() &&
                executable_snippet_thresholds())
end

function format_leftover_catalog()
    io = IOBuffer()
    println(io, "| page | leftover hits |")
    println(io, "|---|---|")
    for (name, row) in (
        :tutorial => leftover_tutorial_row(),
        :howto => leftover_howto_row(),
        :sciml => leftover_sciml_row(),
        :architecture => leftover_architecture_row())
        println(io, "| ", name, " | ", row.n, " |")
    end
    return String(take!(io))
end

function leftover_catalog_holds()
    text = format_leftover_catalog()
    return leftover_tutorial_row().holds &&
           leftover_howto_row().holds &&
           leftover_sciml_row().holds &&
           leftover_architecture_row().holds &&
           leftover_hl_pages_row().holds &&
           occursin("tutorial", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function hl_pages_exist_row()
    root = joinpath(pkgdir(BioDynaX), "docs", "src")
    return (;
        holds = all(p -> isfile(joinpath(root, p)), DOCS_EXECUTABLE_HL_PAGES) &&
                all(p -> isfile(joinpath(root, p)), DOCS_EXECUTABLE_AG_PAGES))
end

function docs_executable_fixture_matrix_extended()
    hill = live_hl_hill_join_row()
    mm = live_hl_mm_join_row()
    three = live_hl_three_join_row()
    dual = live_hl_dual_join_row()
    wrong = live_hl_wrong_graph_join_row()
    repress = live_hl_repressilator_join_row()
    extras = live_hl_extras_join_row()
    frozen = live_hl_frozen_join_row()
    gates = live_hl_parent_gate_join_row()
    snippets = executable_snippets_row()
    return (;
        hill, mm, three, dual, wrong, repress, extras, frozen, gates,
        snippets,
        holds = hill.holds && mm.holds && three.holds && dual.holds &&
                wrong.holds && repress.holds && extras.holds &&
                frozen.holds && gates.holds && snippets.holds)
end

function hl_page_locked_sentences()
    return (
        "hybrid-residual.md" => values(hybrid_residual_locked_sentences()),
        "identifiability-product.md" => values(identifiability_product_locked_sentences()),
        "graph-local-library.md" => values(graph_local_library_locked_sentences()),
        "denominator-domain.md" => values(denominator_domain_locked_sentences()),
        "parameter-schema-pack.md" => values(parameter_schema_pack_locked_sentences()))
end

function hl_own_page_sentences_row()
    root = joinpath(pkgdir(BioDynaX), "docs", "src")
    missing = String[]
    for (page, sentences) in hl_page_locked_sentences()
        text = read(joinpath(root, page), String)
        for sentence in sentences
            occursin(sentence, text) ||
                push!(missing, string(page, ": ", sentence))
        end
    end
    return (;
        n_missing = length(missing),
        missing,
        holds = isempty(missing))
end

function ag_pages_still_listed_row()
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    paths = join(unique_claim_user_doc_paths(), " ")
    return (;
        make = all(p -> occursin(p, make), DOCS_EXECUTABLE_AG_PAGES),
        paths = all(p -> occursin(p, paths), DOCS_EXECUTABLE_AG_PAGES),
        holds = all(p -> occursin(p, make), DOCS_EXECUTABLE_AG_PAGES) &&
                all(p -> occursin(p, paths), DOCS_EXECUTABLE_AG_PAGES))
end

function unique_claim_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "unique-claim.md")
    hits = leftover_hits_on(path)
    text = read(path, String)
    return (;
        n = length(hits),
        hits,
        has_protocol = occursin("103", text) && occursin("9", text),
        holds = isempty(hits) && occursin("103", text))
end

function experimental_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "experimental.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function benchmarks_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "benchmarks.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function stability_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "stability.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function compiled_path_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "compiled-path.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function hl_helper_names_on_own_pages_row()
    root = joinpath(pkgdir(BioDynaX), "docs", "src")
    checks = (
        ("hybrid-residual.md", "hybrid_data_residual"),
        ("identifiability-product.md", "identifiability_product"),
        ("graph-local-library.md", "local_has_true_parent_gate"),
        ("denominator-domain.md", "denominator_split_counts"),
        ("parameter-schema-pack.md", "unpack_parameters"))
    missing = String[]
    for (page, needle) in checks
        text = read(joinpath(root, page), String)
        occursin(needle, text) || push!(missing, string(page, " ", needle))
    end
    return (; missing, holds = isempty(missing))
end

function format_hl_sentence_catalog()
    io = IOBuffer()
    println(io, "| page | locked sentences |")
    println(io, "|---|---|")
    for (page, sentences) in hl_page_locked_sentences()
        println(io, "| ", page, " | ", length(collect(sentences)), " |")
    end
    return String(take!(io))
end

function hl_sentence_catalog_holds()
    text = format_hl_sentence_catalog()
    return hl_own_page_sentences_row().holds &&
           occursin("hybrid-residual.md", text) &&
           occursin("parameter-schema-pack.md", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function extra_user_doc_leftover_row()
    extra = (
        leftover_tutorial_row(),
        leftover_howto_row(),
        leftover_sciml_row(),
        leftover_architecture_row(),
        unique_claim_page_leftover_row(),
        experimental_page_leftover_row(),
        benchmarks_page_leftover_row(),
        stability_page_leftover_row(),
        compiled_path_page_leftover_row())
    return (;
        n = length(extra),
        holds = all(r -> r.holds, extra))
end

function live_hl_six_join_row()
    lib = six_state_library_row()
    den = six_state_denominator_row()
    schema = six_state_schema_row()
    return (; holds = lib.holds && den.holds && schema.holds)
end

function live_hl_competitive_join_row()
    lib = competitive_library_row()
    den = competitive_denominator_row()
    schema = competitive_schema_row()
    return (; holds = lib.holds && den.holds && schema.holds)
end

function live_hl_skipped_join_row()
    lib = skipped_duplicate_library_row()
    den = skipped_duplicate_denominator_row()
    schema = skipped_duplicate_schema_row()
    return (; holds = lib.holds && den.holds && schema.holds)
end

function live_hl_known_hill_join_row()
    den = hill_known_denominator_row()
    schema = hill_known_schema_row()
    return (; holds = den.holds && schema.holds)
end

function live_hl_known_mm_join_row()
    den = mm_known_denominator_row()
    schema = mm_known_schema_row()
    return (; holds = den.holds && schema.holds)
end

function public_export_untouched_docs_row()
    return (;
        holds = !(:docs_executable_join_row in names(BioDynaX)) &&
                !(:leftover_contradiction_hits in names(BioDynaX)) &&
                public_export_list_holds())
end

function user_doc_inventory_row(path::AbstractString)
    text = isfile(path) ? read(path, String) : ""
    leftovers = leftover_hits_on(path)
    return (;
        name = basename(path),
        exists = isfile(path),
        nbytes = sizeof(text),
        leftover = length(leftovers),
        has_103 = occursin("103", text),
        has_f1_paint = occursin("support_f1_ude = 0.99", text),
        holds = isfile(path) && isempty(leftovers) &&
                !occursin("support_f1_ude = 0.99", text))
end

function user_doc_inventory()
    rows = [user_doc_inventory_row(path)
            for path in unique_claim_user_doc_paths()]
    return (;
        n = length(rows),
        rows,
        holds = !isempty(rows) && all(r -> r.holds, rows) &&
                any(r -> r.name == "docs-executable.md", rows) &&
                any(r -> r.name == "parameter-schema-pack.md", rows))
end

function format_user_doc_inventory()
    catalog = user_doc_inventory()
    io = IOBuffer()
    println(io, "| page | bytes | leftover | seed 103 |")
    println(io, "|---|---|---|---|")
    for row in catalog.rows
        println(io, "| ", row.name, " | ", row.nbytes, " | ",
            row.leftover, " | ", row.has_103, " |")
    end
    return String(take!(io))
end

function user_doc_inventory_holds()
    catalog = user_doc_inventory()
    text = format_user_doc_inventory()
    return catalog.holds &&
           occursin("docs-executable.md", text) &&
           occursin("tutorial.md", text) &&
           count(==('|'), text) ≥ 40 &&
           !occursin("support_f1_ude = 0.99", text)
end

function required_tutorial_tokens_row()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"), String)
    tokens = (
        "seed 103", "9 initial conditions", "hybrid_data_residual",
        "compose_hybrid_rhs", "docs-executable", "hybrid-residual",
        "identifiability-product", "graph-local-library",
        "denominator-domain", "parameter-schema-pack",
        "skeleton floor 0.50", "canonical_hill_from_nn")
    missing = [t for t in tokens if !occursin(t, text)]
    return (; missing, holds = isempty(missing))
end

function required_howto_tokens_row()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    tokens = (
        "docs_executable_join_row", "unpack_parameters",
        "denominator_split_counts", "local_has_true_parent_gate",
        "coefficients_are_biological_constants", "hybrid_data_residual")
    missing = [t for t in tokens if !occursin(t, text)]
    return (; missing, holds = isempty(missing))
end

function required_sciml_tokens_row()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    missing = [s for s in docs_hl_contract_strings() if !occursin(s, text)]
    return (;
        missing,
        join = occursin(docs_executable_locked_sentences().join, text),
        holds = isempty(missing) &&
                occursin(docs_executable_locked_sentences().join, text))
end

function executable_contract_lengths_row()
    lengths = Int[length(s) for s in docs_hl_contract_strings()]
    return (;
        lengths,
        min = minimum(lengths),
        holds = length(lengths) == 5 &&
                minimum(lengths) ≥ 40 &&
                length(unique(docs_hl_contract_strings())) == 5)
end

function docs_executable_page_not_ag_row()
    text = read(docs_executable_docs_path(), String)
    return (;
        holds = occursin("does not restate the A–G workspaces", text) &&
                !occursin("This page restates discovery-streaming.md", text) &&
                !occursin("This page restates training-reuse.md", text) &&
                !occursin("This page restates hybrid-compose.md", text))
end

function live_hl_ident_join_row()
    coeff = coefficients_are_biological_constants_row()
    extras = extras_not_invented_on_join_row()
    kpi = kpi_f1_not_a_failure_on_join_row()
    return (; holds = coeff.holds && extras.holds && kpi.holds)
end

function live_hl_residual_smoke_row()
    smoke = smoke_vs_protocol_residual_row()
    return (;
        holds = smoke.holds && smoke.protocol_ics == 9)
end

function format_docs_executable_report()
    join = docs_executable_join_row()
    leftover = leftover_contradiction_row()
    inventory = user_doc_inventory()
    io = IOBuffer()
    println(io, "# Docs executable report")
    println(io, "")
    println(io, "H–L contract strings: ", join.n)
    println(io, "Landing leftover hits: ", leftover.n)
    println(io, "User-doc inventory rows: ", inventory.n)
    println(io, "")
    println(io, format_docs_executable_index())
    println(io, "")
    println(io, format_leftover_catalog())
    println(io, "")
    println(io, format_user_doc_inventory())
    println(io, "")
    println(io, format_hl_sentence_catalog())
    return String(take!(io))
end

function docs_executable_report_holds()
    text = format_docs_executable_report()
    return occursin("H–L contract strings: 5", text) &&
           occursin("Landing leftover hits: 0", text) &&
           occursin("docs-executable.md", text) &&
           occursin("hybrid-residual.md", text) &&
           !occursin("support_f1_ude = 0.99", text) &&
           sizeof(text) > 400
end

function leftover_phrase_catalog()
    io = IOBuffer()
    println(io, "| phrase | landing hits |")
    println(io, "|---|---|")
    for phrase in DOCS_EXECUTABLE_LEFTOVER_PHRASES
        n = 0
        for path in docs_executable_landing_paths()
            isfile(path) || continue
            occursin(phrase, read(path, String)) && (n += 1)
        end
        println(io, "| `", phrase, "` | ", n, " |")
    end
    return String(take!(io))
end

function leftover_phrase_catalog_holds()
    text = leftover_phrase_catalog()
    return occursin("k_custom is absent", text) &&
           occursin("| 0 |", text) &&
           !occursin("| 1 |", text) &&
           !occursin("| 2 |", text)
end

function ag_page_inventory_row()
    root = joinpath(pkgdir(BioDynaX), "docs", "src")
    rows = NamedTuple[]
    for page in DOCS_EXECUTABLE_AG_PAGES
        path = joinpath(root, page)
        text = isfile(path) ? read(path, String) : ""
        push!(rows,
            (;
                page,
                exists = isfile(path),
                leftover = length(leftover_hits_on(path)),
                holds = isfile(path) && !occursin("support_f1_ude = 0.99", text)))
    end
    return (;
        n = length(rows),
        rows,
        holds = length(rows) == 7 && all(r -> r.holds, rows))
end

function hl_page_inventory_row()
    root = joinpath(pkgdir(BioDynaX), "docs", "src")
    rows = NamedTuple[]
    for page in DOCS_EXECUTABLE_HL_PAGES
        path = joinpath(root, page)
        text = isfile(path) ? read(path, String) : ""
        push!(rows,
            (;
                page,
                exists = isfile(path),
                leftover = length(leftover_hits_on(path)),
                holds = isfile(path) && !occursin("support_f1_ude = 0.99", text)))
    end
    return (;
        n = length(rows),
        rows,
        holds = length(rows) == 5 && all(r -> r.holds, rows))
end

function executable_join_is_not_protocol_row()
    join = docs_executable_join_row()
    fp = unique_claim_fingerprint()
    return (;
        n_surfaces = join.n,
        n_ics = fp.n_ics,
        holds = join.holds && fp.n_ics == 9 && !fp.smoke &&
                join.n == 5)
end

function readme_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "README.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function index_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "index.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function api_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "api.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function metadata_page_leftover_row()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "metadata.md")
    hits = leftover_hits_on(path)
    return (; n = length(hits), hits, holds = isempty(hits))
end

function changelog_does_not_paint_f1_row()
    text = read(joinpath(pkgdir(BioDynaX), "CHANGELOG.md"), String)
    return (;
        holds = !occursin("support_f1_ude = 0.99", text) &&
                occursin("Docs executable path", text))
end

function news_does_not_paint_f1_row()
    text = read(joinpath(pkgdir(BioDynaX), "NEWS.md"), String)
    return (;
        holds = !occursin("support_f1_ude = 0.99", text) &&
                occursin("Docs executable path", text))
end

function hl_surface_callables()
    return (
        :hybrid_residual => hybrid_residual_contract,
        :identifiability => identifiability_product_contract,
        :graph_local => graph_local_library_contract,
        :denominator => denominator_domain_contract,
        :schema_pack => parameter_schema_pack_contract)
end

function hl_surface_callables_row()
    pairs = hl_surface_callables()
    texts = String[string(name, ": ", f()) for (name, f) in pairs]
    return (;
        n = length(texts),
        texts,
        holds = length(texts) == 5 &&
                length(unique(texts)) == 5 &&
                all(t -> !occursin("support_f1_ude = 0.99", t), texts))
end

function format_hl_surface_callables()
    io = IOBuffer()
    println(io, "| surface | contract |")
    println(io, "|---|---|")
    for (name, f) in hl_surface_callables()
        println(io, "| ", name, " | ", f(), " |")
    end
    return String(take!(io))
end

function hl_surface_callables_hold()
    text = format_hl_surface_callables()
    return hl_surface_callables_row().holds &&
           occursin("hybrid_residual", text) &&
           occursin("schema_pack", text) &&
           occursin("k_custom", text)
end

function docs_executable_not_faster_protocol_row()
    smoke = unique_claim_fingerprint(; smoke = true)
    proto = unique_claim_fingerprint()
    ics = unique_claim_protocol_ics()
    return (;
        smoke_ics = smoke.n_ics,
        proto_ics = proto.n_ics,
        table = length(ics),
        holds = smoke.n_ics == 1 && proto.n_ics == 9 &&
                length(ics) == 9 && proto.n_points == 50 &&
                proto.seed == 103 && !proto.smoke)
end

function combined_f1_not_docs_kpi_row()
    return (;
        holds = :support_f1 ∉ UNIQUE_CLAIM_KPI_FIELDS &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50 &&
                RECOVERY_THRESHOLDS.support_f1_clean == 0.99)
end

function format_join_sentence_list()
    io = IOBuffer()
    println(io, "H–L locked contracts:")
    for sentence in docs_hl_contract_strings()
        println(io, "- ", sentence)
    end
    println(io, "Protocol: seed 103 / 9 ICs / 50 points.")
    println(io, "Smoke: 1 IC / 8 points.")
    return String(take!(io))
end

function join_sentence_list_holds()
    text = format_join_sentence_list()
    return all(s -> occursin(s, text), docs_hl_contract_strings()) &&
           occursin("9 ICs", text) &&
           occursin("1 IC", text) &&
           !occursin("HTTP 200", text)
end

function docs_executable_contract_holds()
    return docs_executable_source_holds() &&
           docs_executable_docs_hold() &&
           docs_executable_landing_docs_hold() &&
           docs_executable_example_source_holds() &&
           docs_executable_docs_mention_helpers() &&
           docs_executable_index_holds() &&
           docs_executable_test_file_holds() &&
           docs_executable_module_include_holds() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source() &&
           recovery_thresholds_untouched_docs_row().holds &&
           unique_claim_not_faster_docs_row().holds &&
           hill_from_nn_closed_docs_row().holds &&
           validate_open_docs_row().holds &&
           docs_executable_typed_matrix().holds &&
           leftover_contradiction_row().holds &&
           leftover_catalog_holds() &&
           hl_pages_exist_row().holds &&
           docs_executable_fixture_matrix_extended().holds &&
           docs_executable_fixture_matrix().holds &&
           hl_own_page_sentences_row().holds &&
           ag_pages_still_listed_row().holds &&
           hl_helper_names_on_own_pages_row().holds &&
           hl_sentence_catalog_holds() &&
           extra_user_doc_leftover_row().holds &&
           live_hl_six_join_row().holds &&
           live_hl_competitive_join_row().holds &&
           live_hl_skipped_join_row().holds &&
           live_hl_known_hill_join_row().holds &&
           live_hl_known_mm_join_row().holds &&
           public_export_untouched_docs_row().holds &&
           user_doc_inventory_holds() &&
           required_tutorial_tokens_row().holds &&
           required_howto_tokens_row().holds &&
           required_sciml_tokens_row().holds &&
           executable_contract_lengths_row().holds &&
           docs_executable_page_not_ag_row().holds &&
           live_hl_ident_join_row().holds &&
           live_hl_residual_smoke_row().holds &&
           docs_executable_report_holds() &&
           leftover_phrase_catalog_holds() &&
           ag_page_inventory_row().holds &&
           hl_page_inventory_row().holds &&
           executable_join_is_not_protocol_row().holds &&
           readme_leftover_row().holds &&
           index_page_leftover_row().holds &&
           api_page_leftover_row().holds &&
           metadata_page_leftover_row().holds &&
           changelog_does_not_paint_f1_row().holds &&
           news_does_not_paint_f1_row().holds &&
           hl_surface_callables_hold() &&
           docs_executable_not_faster_protocol_row().holds &&
           combined_f1_not_docs_kpi_row().holds &&
           docs_executable_join_row().n == 5 &&
           isempty(leftover_contradiction_hits()) &&
           executable_snippet_protocol() &&
           executable_snippet_thresholds() &&
           landing_file_exists_row().holds &&
           make_lists_hl_and_exec_row().holds &&
           unique_claim_paths_include_hl_row().holds &&
           join_sentence_list_holds()
end
