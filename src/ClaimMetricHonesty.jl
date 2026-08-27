###############################################################################
# Claim-metric honesty (not exported).
#
# Same-library UDE extras do not reach support_f1_clean. unidentifiable_edge
# is the Fisher/Jacobian cosine or condition-number flag.
# MM unknown gates NN RMSE and residual, not Hill recall 0.99.
# Does not change RECOVERY_THRESHOLDS, the public export list, or
# validate_network. Canonical Hill from a trained NN stays closed.
###############################################################################

const CLAIM_METRIC_HONESTY_NOT_STRUCTURAL = "not StructuralIdentifiability.jl"

const CLAIM_METRIC_HONESTY_EXTRACTED_HILL = "cannot claim we extracted Hill"

const CLAIM_METRIC_HONESTY_EXTRACTED_HILL_FORBIDDEN = (
    "we extracted Hill",
    "extracted the Hill",
    "canonical Hill from a trained NN is open")

const CLAIM_METRIC_HONESTY_MM_SENTENCE = "MM unknown gates NN RMSE and hybrid residual; Hill recall 0.99 is not applied."

function claim_metric_honesty_locked_sentences()
    return (;
        extracted_hill = "The same-library UDE extras probe cannot claim we extracted Hill.",
        fisher = "unidentifiable_edge is the Fisher/Jacobian cosine or condition-number flag, not StructuralIdentifiability.jl.",
        coefficients = "coefficients_are_biological_constants is !unidentifiable_edge.",
        mm = CLAIM_METRIC_HONESTY_MM_SENTENCE,
        floor = "support_f1_ude stays 0.50; support_f1_clean stays 0.99.")
end

function ude_f1_attempt_live_row()
    r = collect(range(0.1, 2.0; length = 180))
    hill = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    nn_like = hill .+ 0.04 .+ 0.04 .* r
    times = collect(range(0.0, 1.0; length = length(r)))
    result = discover_unknown_rate(
        reshape(r, 1, :), times, reshape(vec(nn_like), 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = 103),
        verbose = false, strict = false)
    truth = hill_rate_support(2)
    extras = result.success ?
             unique_claim_discovery_extras(result.candidates[1]) : String[]
    f1 = if result.success
        support_f1(result.candidates[1], truth.numerator,
            truth.denominator).combined.f1
    else
        0.0
    end
    reaches_clean = unique_claim_f1_reaches_analytical_gate(f1)
    verdict = unique_claim_f1_attempt_verdict(; extras, reaches_clean)
    return (;
        success = result.success,
        extras,
        f1,
        reaches_clean,
        meets_skeleton = unique_claim_f1_meets_skeleton_floor(f1),
        extras_one = "1" in extras,
        extras_r = "r" in extras,
        extracted_hill = false,
        verdict,
        floor = RECOVERY_THRESHOLDS.support_f1_ude,
        clean_gate = RECOVERY_THRESHOLDS.support_f1_clean,
        holds = result.success && ("1" in extras) && ("r" in extras) &&
                !reaches_clean &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50 &&
                RECOVERY_THRESHOLDS.support_f1_clean == 0.99 &&
                verdict === :extras_remain_claim_stays_recall_plus_residual)
end

function _unqualified_phrase_hit(text::AbstractString, phrase::AbstractString;
        allowed_prefix::AbstractString = "cannot claim ")
    for r in findall(phrase, text)
        start = first(r)
        start == firstindex(text) && return true
        prefix = text[firstindex(text):(start - 1)]
        endswith(prefix, allowed_prefix) && continue
        return true
    end
    return false
end

function extracted_hill_forbidden_hits()
    hits = String[]
    for path in unique_claim_user_doc_paths()
        isfile(path) || continue
        text = read(path, String)
        for phrase in CLAIM_METRIC_HONESTY_EXTRACTED_HILL_FORBIDDEN
            if phrase == "we extracted Hill"
                _unqualified_phrase_hit(text, phrase) &&
                    push!(hits, string(basename(path), ": ", phrase))
            else
                occursin(phrase, text) &&
                    push!(hits, string(basename(path), ": ", phrase))
            end
        end
    end
    return hits
end

function extracted_hill_docs_hold()
    sentences = claim_metric_honesty_locked_sentences()
    unique_page = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "unique-claim.md"), String)
    benches = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "benchmarks.md"), String)
    metric_page = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "claim-metric-honesty.md"), String)
    return occursin(sentences.extracted_hill, unique_page) &&
           occursin(sentences.extracted_hill, metric_page) &&
           occursin(CLAIM_METRIC_HONESTY_EXTRACTED_HILL, unique_page) &&
           occursin(sentences.floor, metric_page) &&
           occursin("0.50", benches) &&
           isempty(extracted_hill_forbidden_hits())
end

function printed_protocol_not_structural_holds()
    ident = (;
        unidentifiable_edge = true,
        production_param = :k_prod,
        collinearity = 0.997)
    text = format_protocol_result(ident;
        residual = 0.003, support_recall = 1.0, support_f1 = 0.57,
        extras = ("1", "r"), unknown_holes = 1, seed = 103, n_ics = 9,
        n_points = 50, smoke = false)
    return occursin("practical Fisher/Jacobian; not StructuralIdentifiability.jl",
               text) &&
           occursin("coefficients_are_biological_constants: false", text) &&
           occursin(CLAIM_METRIC_HONESTY_NOT_STRUCTURAL, text) &&
           !occursin("StructuralIdentifiability.jl proves", text)
end

function identifiability_docs_not_structural_hold()
    sentences = claim_metric_honesty_locked_sentences()
    unique_page = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "unique-claim.md"), String)
    experimental = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "experimental.md"), String)
    metric_page = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "claim-metric-honesty.md"), String)
    ident_src = read(joinpath(pkgdir(BioDynaX), "src", "Identifiability.jl"),
        String)
    return occursin(sentences.fisher, unique_page) &&
           occursin(sentences.fisher, metric_page) &&
           occursin(sentences.coefficients, metric_page) &&
           occursin(CLAIM_METRIC_HONESTY_NOT_STRUCTURAL, experimental) &&
           occursin("unidentifiable_edge_from_fisher", ident_src) &&
           occursin("This is not structural identifiability", ident_src)
end

function unidentifiable_edge_formula_row()
    cond_only = unidentifiable_edge_from_fisher(;
        condition_number = 1.0e7, collinearity = 0.10)
    cosine_only = unidentifiable_edge_from_fisher(;
        condition_number = 10.0, collinearity = 0.96)
    neither = unidentifiable_edge_from_fisher(;
        condition_number = 10.0, collinearity = 0.50)
    nan_pair = unidentifiable_edge_from_fisher(;
        condition_number = NaN, collinearity = NaN)
    return (;
        cond_only,
        cosine_only,
        neither,
        nan_pair,
        coeff_when_edge = !cond_only,
        holds = cond_only && cosine_only && !neither && !nan_pair &&
                coefficients_are_biological_constants((;
                    unidentifiable_edge = true)) == false &&
                coefficients_are_biological_constants((;
                    unidentifiable_edge = false)) == true)
end

function mm_unknown_claim_gates()
    lock = recovery_thresholds_lock()
    return (;
        nn_rate_rmse = lock.nn_rate_rmse,
        data_residual = lock.data_residual,
        hill_recall = lock.support_recall,
        hill_f1_clean = lock.support_f1_clean,
        applies_hill_recall = false,
        applies_hill_f1_clean = false,
        family = :mm,
        measured_recall = 0.5,
        measured_f1 = 0.33)
end

"""
    mm_unknown_claim_holds(mm) -> Bool

NN RMSE and hybrid residual only. Hill recall 0.99 / clean F1 0.99 are
not MM gates. Not exported.
"""
function mm_unknown_claim_holds(mm)
    gates = mm_unknown_claim_gates()
    hasproperty(mm, :nn_rate_rmse) || return false
    hasproperty(mm, :data_residual) || return false
    nn_ok = mm.nn_rate_rmse ≤ gates.nn_rate_rmse
    res_ok = mm.data_residual ≤ gates.data_residual
    return nn_ok && res_ok && !gates.applies_hill_recall &&
           !gates.applies_hill_f1_clean
end

function mm_unknown_hard_job_source_holds()
    path = joinpath(pkgdir(BioDynaX), "test", "test_recovery_hard.jl")
    src = read(path, String)
    start = findfirst("UDE unknown-edge MM recovery", src)
    start === nothing && return false
    rest = src[first(start):end]
    return occursin("nn_rate_rmse", rest) &&
           occursin("data_residual", rest) &&
           occursin("Hill recall 0.99 is not silently reused", rest) &&
           !occursin("support_recall ≥ RECOVERY_THRESHOLDS.support_recall",
               rest) &&
           !occursin("support_f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean", rest)
end

function mm_unknown_recovery_family_source_holds()
    path = joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
    src = read(path, String)
    start = findfirst("if :mm_unknown in wanted", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext("if :ablation in wanted", rest)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("family = :mm", body) &&
           occursin("mm_rate_truth", body) &&
           !occursin("family = :hill", body)
end

function mm_unknown_docs_hold()
    sentences = claim_metric_honesty_locked_sentences()
    unique_page = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "unique-claim.md"), String)
    benches = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "benchmarks.md"), String)
    metric_page = read(joinpath(pkgdir(BioDynaX), "docs", "src",
            "claim-metric-honesty.md"), String)
    return occursin(sentences.mm, unique_page) &&
           occursin(sentences.mm, metric_page) &&
           occursin("measured recall ~0.5", benches) &&
           occursin("F1 ~0.33", benches) &&
           !occursin("MM recall 0.99", benches)
end

function claim_metric_honesty_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "claim-metric-honesty.md")
end

function claim_metric_honesty_contract_holds()
    return ude_f1_attempt_live_row().holds &&
           extracted_hill_docs_hold() &&
           printed_protocol_not_structural_holds() &&
           identifiability_docs_not_structural_hold() &&
           unidentifiable_edge_formula_row().holds &&
           mm_unknown_hard_job_source_holds() &&
           mm_unknown_recovery_family_source_holds() &&
           mm_unknown_docs_hold() &&
           recovery_thresholds_hold() &&
           public_export_list_holds() &&
           validate_network_stays_open_source()
end
