@testset "UniqueClaimFingerprint types protocol versus smoke" begin
    fp = unique_claim_fingerprint()
    sm = unique_claim_fingerprint(; smoke = true)
    @test fp isa UniqueClaimFingerprint
    @test sm isa UniqueClaimFingerprint
    @test unique_claim_fingerprint(:protocol) == fp
    @test unique_claim_fingerprint(:smoke) == sm
    @test_throws ArgumentError unique_claim_fingerprint(:notebook)
    @test unique_claim_fingerprint_is_protocol(fp)
    @test unique_claim_fingerprint_is_smoke(sm)
    @test unique_claim_fingerprint_holds(fp)
    @test unique_claim_fingerprint_holds(sm)
    @test unique_claim_fingerprint_is_protocol(sm) == false
    @test unique_claim_fingerprint_is_smoke(fp) == false
    @test fp.kind === :protocol
    @test sm.kind === :smoke
    @test fp.n_ics == 9
    @test sm.n_ics == 1
    @test fp.n_points == 50
    @test sm.n_points == 8
    @test fp.bfgs_iterations == 50
    @test sm.bfgs_iterations == 0
    @test fp.bootstrap == 8
    @test sm.bootstrap === nothing
    @test fp.discovery_seed == 3
    @test sm.discovery_seed === nothing
    @test fp.observation_noise == 0.0
    @test fp.tspan == (0.0, 8.0)
    @test fp != sm
    named = unique_claim_fingerprint_namedtuple(fp)
    @test named.is_protocol
    @test named.n_ics == 9
    repro = unique_claim_reproduction(fp)
    @test repro.is_protocol
    @test repro.n_ics == fp.n_ics
    @test unique_claim_reproduction(sm).is_protocol == false
    text = format_unique_claim_fingerprint(fp)
    @test occursin("fingerprint_kind: protocol", text)
    @test occursin("is_protocol: true", text)
    @test occursin("n_ics: 9", format_unique_claim_fingerprint(fp))
    @test occursin("n_ics: 1", format_unique_claim_fingerprint(sm))
    @test !(:UniqueClaimFingerprint in names(BioDynaX))
    @test !(:unique_claim_fingerprint in names(BioDynaX))
end

@testset "format_protocol_result consumes the typed fingerprint" begin
    ident = (; unidentifiable_edge = true, production_param = :k_prod,
        collinearity = 0.997)
    fp = unique_claim_fingerprint()
    text = format_protocol_result(ident, fp;
        residual = 0.003, support_recall = 1.0, support_f1 = 0.57,
        extras = ("1", "r"), unknown_holes = 1)
    @test protocol_block_order_holds(text)
    @test format_protocol_result_field_order_holds(text)
    @test format_protocol_print_labels_hold(text)
    @test occursin("protocol_kind: protocol", text)
    @test occursin("n_ics: 9", text)
    @test occursin("n_points: 50", text)
    @test occursin("claim: recall_plus_data_residual", text)
    @test occursin("hybrid_data_residual: 0.003", text)
    smoke_txt = format_protocol_result(ident, unique_claim_fingerprint(; smoke = true);
        residual = Inf, extras = nothing)
    @test occursin("protocol_kind: smoke", smoke_txt)
    @test occursin("n_ics: 1", smoke_txt)
    @test occursin("n_points: 8", smoke_txt)
    @test occursin("bfgs_iters: 0", smoke_txt)
    @test occursin("extras: NA", smoke_txt)
    @test !occursin("1, r remain after the UDE F1 attempt", smoke_txt)
end

@testset "extras print distinguishes NA, none, and live leftovers" begin
    @test extras_print_label(nothing) == "NA"
    @test extras_print_label(String[]) == "(none)"
    @test extras_print_label(("1", "r")) == "1, r"
    @test extras_print_label(["1"]) == "1"
    @test extras_print_is_hardcoded_attempt("1, r remain after the UDE F1 attempt")
    @test extras_print_is_hardcoded_attempt(extras_print_label(nothing)) == false
    @test extras_print_is_hardcoded_attempt(extras_print_label(("1", "r"))) == false
    empty_txt = format_protocol_result((; unidentifiable_edge = true);
        extras = String[])
    @test occursin("extras: (none)", empty_txt)
    @test !occursin("1, r remain after the UDE F1 attempt", empty_txt)
    live_txt = format_protocol_result((; unidentifiable_edge = true);
        extras = ["1", "r"])
    @test occursin("extras: 1, r", live_txt)
    src = read(joinpath(pkgdir(BioDynaX), "src", "Recovery.jl"), String)
    @test occursin("return \"NA\"", src)
    @test occursin("return \"(none)\"", src)
    @test !occursin("1, r remain after the UDE F1 attempt", src)
end

@testset "protocol_result field order matches printed labels" begin
    @test protocol_result_field_to_print_label(:data_residual) ==
          "hybrid_data_residual"
    @test protocol_result_field_to_print_label(:claim) == "claim"
    @test_throws ArgumentError protocol_result_field_to_print_label(:support_precision)
    @test protocol_result_print_labels().extras == "extras"
    @test protocol_print_fields().FIT == (:hybrid_data_residual, :support_recall)
    @test :claim in protocol_print_fields().DISCOVERY
    order = protocol_result_print_order()
    @test first(order) === :unknown_holes
    @test :hybrid_data_residual in order
    @test :claim in order
    ude = (;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        identifiability = (; unidentifiable_edge = true))
    result = build_protocol_result(ude)
    text = format_protocol_result(ude.identifiability;
        residual = result.data_residual,
        support_recall = result.support_recall,
        support_f1 = result.support_f1,
        extras = result.extras,
        unknown_holes = result.unknown_holes,
        seed = 103, n_ics = 9, n_points = 50,
        adam_iters = 100, bfgs_iters = 50, bootstrap = 8,
        discovery_seed = 3, smoke = false)
    @test assert_format_matches_protocol_result(result, text) == text
    swapped = replace(text, "IDENTIFIABILITY" => "FIT\nIDENTIFIABILITY"; count = 1)
    @test format_protocol_result_field_order_holds(swapped) == false
    missing_ident = build_protocol_result((;
        data_residual = Inf, support_recall = 0.0))
    @test missing_ident.unidentifiable_edge == false
    @test missing_ident.coefficients_are_biological_constants
    miss_txt = format_protocol_result((; unidentifiable_edge = false);
        residual = Inf, support_recall = 0.0, support_f1 = 0.0,
        extras = nothing, unknown_holes = 0)
    @test occursin("unidentifiable_edge: false", miss_txt)
    @test occursin("coefficients_are_biological_constants: true", miss_txt)
    @test occursin("extras: NA", miss_txt)
    @test_throws ErrorException assert_format_matches_protocol_result(
        result, miss_txt)
end

@testset "recovery admission is independent of validate_network" begin
    one = build_hill_recovery_network(; known = false)
    @test unique_claim_recovery_admits(one)
    @test assert_unique_claim_recovery_network(one) === one
    @test unique_claim_compiler_stays_open(one)
    rng = MersenneTwister(0)
    model, _ = build_ude_model(rng, one)
    @test unique_claim_recovery_admits(model)
    @test recovery_suite_uses_single_hole_instrument()
    recovery_src = read(joinpath(pkgdir(BioDynaX), "src", "Recovery.jl"), String)
    @test occursin("admit_recovery_suite_network", recovery_src)
    @test occursin("only_unknown_destruction", recovery_src)
    suite_src = read(joinpath(pkgdir(BioDynaX), "benchmark", "recovery_suite.jl"), String)
    @test occursin("format_recovery_protocol", suite_src)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.seed", suite_src)
    @test occursin("extras_print_label", suite_src)
    @test !occursin("extras=(\"1\", \"r\")", suite_src)
    fake = (;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        identifiability = (; unidentifiable_edge = true),
        protocol_result = build_protocol_result((;
            data_residual = 0.003,
            support_recall = 1.0,
            support_f1 = 0.57,
            extras = ["1", "r"],
            identifiability = (; unidentifiable_edge = true))))
    recovery_txt = format_recovery_protocol(fake, unique_claim_fingerprint();
        equations = "D(z) = 1 + r")
    @test format_protocol_result_field_order_holds(recovery_txt)
    @test occursin("extras: 1, r", recovery_txt)
    @test occursin("n_ics: 9", recovery_txt)
end

@testset "F1 attempt contract is not the protocol fingerprint" begin
    contract = unique_claim_f1_attempt_contract()
    @test contract.is_protocol == false
    @test contract.trains_ude == false
    @test contract.n_ics == 0
    @test contract.uses_protocol_ics == false
    @test contract.new_atoms == false
    @test contract.library === :same_monomial
    @test contract.support_f1_ude == 0.50
    @test contract.support_f1_clean == 0.99
    @test contract.support_f1_ude == RECOVERY_THRESHOLDS.support_f1_ude
    @test contract.support_f1_clean == RECOVERY_THRESHOLDS.support_f1_clean
    @test unique_claim_f1_attempt_holds()
    violations = unique_claim_f1_attempt_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    row = unique_claim_f1_attempt_row(; extras = ["1", "r"], f1 = 0.57)
    @test row.is_protocol == false
    @test row.reaches_clean == false
    @test row.meets_skeleton
    @test row.verdict === :extras_remain_claim_stays_recall_plus_residual
    @test !(:UNIQUE_CLAIM_F1_ATTEMPT in names(BioDynaX))
    @test !(:unique_claim_f1_attempt_contract in names(BioDynaX))
end

@testset "example and docs lock the new protocol surfaces" begin
    src = read(unique_claim_example_path(), String)
    @test occursin("unique_claim_fingerprint", src)
    @test occursin("unique_claim_experiment_set", src)
    @test occursin("assert_unique_claim_recovery_network", src)
    @test occursin("format_protocol_result(ident, fingerprint", src)
    violations = unique_claim_example_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    page = read(joinpath(pkgdir(BioDynaX), "docs", "src", "unique-claim.md"), String)
    sentences = unique_claim_locked_sentences()
    for label in keys(sentences)
        @test occursin(sentences[label], page)
    end
    architecture = read(
        joinpath(pkgdir(BioDynaX), "docs", "src", "architecture.md"), String)
    @test occursin("reindexed to `1:n`", architecture) ||
          occursin("reindexes kept", architecture)
    @test occursin("IDENTIFIABILITY", architecture)
    @test occursin("UniqueClaimFingerprint", architecture)
    tutorial = read(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"), String)
    @test occursin("sample_unknown_destruction_grid", tutorial)
    @test occursin("unique_claim_fingerprint", tutorial)
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    @test occursin("unique_claim_fingerprint", howto)
    experimental = read(
        joinpath(pkgdir(BioDynaX), "docs", "src", "experimental.md"), String)
    @test occursin("UniqueClaimFingerprint", experimental)
    @test occursin("UNIQUE_CLAIM_F1_ATTEMPT", experimental)
    benchmarks = read(joinpath(pkgdir(BioDynaX), "docs", "src", "benchmarks.md"), String)
    @test occursin("UNIQUE_CLAIM_F1_ATTEMPT", benchmarks)
    @test occursin("extras_print_label", benchmarks)
    news = read(joinpath(pkgdir(BioDynaX), "NEWS.md"), String)
    @test occursin("UniqueClaimFingerprint", news)
    @test occursin("UNIQUE_CLAIM_F1_ATTEMPT", news)
    @test occursin("assert_unique_claim_recovery_network", news)
end
