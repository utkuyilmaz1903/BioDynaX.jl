###############################################################################
# Claim-scope honesty (not exported).
#
# Howto CSV is a synthetic fixture. Licensed experimental series are absent.
# Partial-observation UDE training on masked states is not claimed.
# Graph priors lock library membership / parent booleans, not an F1 win
# versus DataDrivenSparse. Multi-seed UDE is a report. Unknown topology
# and general CRN are out of scope. Thresholds and exports are unchanged.
###############################################################################

const CLAIM_SCOPE_HOWTO_SYNTHETIC = "That committed CSV is a synthetic fixture"

const CLAIM_SCOPE_HOWTO_NOT_LICENSED = "not a licensed experimental series"

const CLAIM_SCOPE_ABSENCE = "That absence is the result"

const CLAIM_SCOPE_DDS_UNRESOLVED = "could not be resolved"

const CLAIM_SCOPE_DDS_NOT_WIN = "not a skip-as-win"

const CLAIM_SCOPE_OUT_OF_SCOPE = (
    "unknown topology",
    "general CRN",
    "single noisy CSV")

function claim_scope_honesty_locked_sentences()
    return (;
        csv = "That committed CSV is a synthetic fixture, not a licensed experimental series.",
        absence = "No licensed experimental time series in this repository matches the unique-claim protocol.",
        result = "That absence is the result.",
        partial = "UDE training on missing states is not claimed.",
        graph = "The locked prior is library membership of the distractor z, not a F1 gap after Occam.",
        dds = "DataDrivenSparse could not be resolved against this preview.",
        seeds = "The red gate remains single-seed 103/104; recovery_seeds.jl --ude is a report, not a gate.",
        scope = "Unique-claim requires a known graph and one hole; unknown topology, a single noisy CSV, and a general CRN solver are not claimed.")
end

function howto_csv_fixture_row()
    sentences = claim_scope_honesty_locked_sentences()
    csv = joinpath(pkgdir(BioDynaX), "examples", "data",
        "unknown_inhibition.csv")
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    benches = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "benchmarks.md"), String)
    experimental = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "experimental.md"), String)
    header = isfile(csv) ? readline(csv) : ""
    wetlab = occursin("wet-lab recovery", lowercase(howto)) ||
             occursin("licensed experimental CSV recovered", lowercase(howto))
    return (;
        csv_exists = isfile(csv),
        header,
        synthetic = occursin(CLAIM_SCOPE_HOWTO_SYNTHETIC, howto),
        not_licensed = occursin(CLAIM_SCOPE_HOWTO_NOT_LICENSED, howto),
        sentence = occursin(sentences.csv, howto),
        absence = occursin(sentences.absence, benches),
        result = occursin(CLAIM_SCOPE_ABSENCE, benches),
        no_wetlab = !wetlab && !occursin("wet-lab recovery", lowercase(benches)),
        no_experimental_claim = !occursin("recovered from wet-lab", lowercase(experimental)),
        holds = isfile(csv) && header == "t,S,R" &&
                occursin(sentences.csv, howto) &&
                occursin(sentences.absence, benches) &&
                occursin(CLAIM_SCOPE_ABSENCE, benches) &&
                !wetlab)
end

"""
    assert_partial_obs_does_not_claim_ude_mask_train(row)

Fail closed if a partial-observation row claims a full UDE fit on
masked states. Not exported.
"""
function assert_partial_obs_does_not_claim_ude_mask_train(row)
    hasproperty(row, :ude_mask_train_claimed) || throw(ErrorException(
        "partial_obs row must name ude_mask_train_claimed"))
    row.ude_mask_train_claimed === false || throw(ErrorException(
        "missing-state UDE training is not claimed; ude_mask_train_claimed must stay false"))
    return row
end

function partial_obs_ude_fit_claim_source_holds()
    recovery = read(joinpath(pkgdir(BioDynaX), "src", "Recovery.jl"), String)
    skip = read(joinpath(pkgdir(BioDynaX), "src", "RecoverySuiteSkip.jl"),
        String)
    benches = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "benchmarks.md"), String)
    sentences = claim_scope_honesty_locked_sentences()
    return occursin("ude_mask_train_claimed = false", recovery) &&
           occursin("partial_obs_does_not_train_unknown_edge_source", skip) &&
           occursin(sentences.partial, benches)
end

function graph_prior_boolean_lock()
    return (;
        three_state = (:local_has_true_parent, :local_false_parent),
        six_state = (:local_has_true_parent, :local_false_parent,
            :Z_in_local_library),
        kpi_is_f1 = false,
        beats_datadriven = false)
end

function graph_prior_docs_hold()
    sentences = claim_scope_honesty_locked_sentences()
    benches = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "benchmarks.md"), String)
    scope = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "out-of-scope.md"), String)
    return occursin("local_has_true_parent", benches) &&
           occursin("Z_in_local_library", benches) &&
           occursin(CLAIM_SCOPE_DDS_UNRESOLVED, benches) &&
           occursin(CLAIM_SCOPE_DDS_NOT_WIN, benches) &&
           occursin("not “we beat them”", benches) &&
           occursin(sentences.dds, benches) &&
           occursin(sentences.graph, scope) &&
           !occursin("we beat DataDrivenSparse", benches)
end

function recovery_seeds_ude_is_report_row()
    sentences = claim_scope_honesty_locked_sentences()
    script = joinpath(pkgdir(BioDynaX), "benchmark", "recovery_seeds.jl")
    src = isfile(script) ? read(script, String) : ""
    ci = read(joinpath(pkgdir(BioDynaX), ".github", "workflows", "ci.yml"),
        String)
    contributing = read(joinpath(pkgdir(BioDynaX), "CONTRIBUTING.md"), String)
    scope = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "out-of-scope.md"), String)
    return (;
        script_exists = isfile(script),
        disclaimer = occursin("not a CI job", src),
        ude_flag = occursin("--ude", src),
        red_gate_script = occursin("CI stays on seeds 103/104", src),
        contributing_gate = occursin("seed 103 / 104", contributing),
        ci_runs_ude = occursin("recovery_seeds.jl --ude", ci),
        sentence = occursin(sentences.seeds, scope),
        holds = isfile(script) && occursin("not a CI job", src) &&
                occursin("--ude", src) &&
                occursin("CI stays on seeds 103/104", src) &&
                occursin("The red gate stays seed 103 / 104", contributing) &&
                !occursin("recovery_seeds.jl --ude", ci) &&
                occursin(sentences.seeds, scope))
end

function unknown_topology_out_of_scope_row()
    sentences = claim_scope_honesty_locked_sentences()
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "out-of-scope.md")
    text = isfile(path) ? read(path, String) : ""
    unique_page = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "unique-claim.md"), String)
    missing = [phrase for phrase in CLAIM_SCOPE_OUT_OF_SCOPE
               if !occursin(phrase, text)]
    return (;
        page_exists = isfile(path),
        sentence = occursin(sentences.scope, text),
        unique_not_crn = occursin("general CRN", unique_page) ||
                         occursin("general CRN solver", unique_page),
        missing,
        holds = isfile(path) && isempty(missing) &&
                occursin(sentences.scope, text) &&
                occursin("2–20", text))
end

function claim_scope_honesty_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "out-of-scope.md")
end

function claim_scope_honesty_contract_holds()
    return howto_csv_fixture_row().holds &&
           partial_obs_ude_fit_claim_source_holds() &&
           graph_prior_docs_hold() &&
           recovery_seeds_ude_is_report_row().holds &&
           unknown_topology_out_of_scope_row().holds &&
           recovery_thresholds_hold() &&
           public_export_list_holds() &&
           validate_network_stays_open_source()
end
