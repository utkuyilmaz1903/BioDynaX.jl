@testset "suite section kinds lock unique-claim versus known kinetics" begin
    kinds = recovery_suite_section_kinds()
    @test kinds.ude_discovery === :unique_claim
    @test kinds.mm_unknown === :unique_claim
    @test kinds.ident_interventions === :unique_claim
    @test kinds.partial_obs === :unique_claim
    @test kinds.linear === :known_kinetics
    @test kinds.literature === :literature
    @test kinds.competitive_unknown === :analytical
    @test recovery_suite_section_requires_single_hole(:ude_discovery)
    @test recovery_suite_section_requires_single_hole(:mm_unknown)
    @test recovery_suite_section_requires_single_hole(:linear) == false
    @test recovery_suite_section_requires_single_hole(:ablation) == false
    @test recovery_suite_unique_claim_sections() ==
          (:ude_discovery, :mm_unknown, :ident_interventions, :partial_obs)
    @test_throws ArgumentError recovery_suite_section_kind(:not_a_section)
    @test !(:admit_recovery_suite_network in names(BioDynaX))
    @test !(:UniqueClaimProtocolRow in names(BioDynaX))
    @test !(:unique_claim_kpi_failure_symbols in names(BioDynaX))
end

@testset "admit_recovery_suite_network rejects 0/2 holes without training" begin
    closed = recovery_suite_rejects_zero_and_dual_holes(:ude_discovery)
    @test closed.holds
    @test closed.zero.unknown_holes == 0
    @test closed.two.unknown_holes == 2
    @test closed.one.unknown_holes == 1
    @test closed.zero.admitted == false
    @test closed.two.admitted == false
    @test closed.one.admitted
    @test closed.zero.validate_open
    @test closed.two.validate_open
    mm = recovery_suite_rejects_zero_and_dual_holes(:mm_unknown)
    @test mm.holds
    ident = recovery_suite_rejects_zero_and_dual_holes(:ident_interventions)
    @test ident.holds
    partial = recovery_suite_rejects_zero_and_dual_holes(:partial_obs)
    @test partial.holds
    zero = build_zero_unknown_linear_network()
    two = build_dual_unknown_network()
    @test validate_network(zero) === zero
    @test validate_network(two) === two
    @test_throws ErrorException admit_recovery_suite_network(:ude_discovery, zero)
    @test_throws ErrorException admit_recovery_suite_network(:ude_discovery, two)
    @test admit_recovery_suite_network(:linear, zero) === zero
    one = admit_recovery_suite_network(:ude_discovery)
    @test unique_claim_recovery_admits(one)
    @test recovery_suite_known_kinetics_admit_zero_holes().holds
end

@testset "run_recovery_suite source uses the admission helper" begin
    @test recovery_suite_uses_admission_helper()
    @test recovery_suite_uses_single_hole_instrument()
    violations = recovery_suite_admission_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    for section in recovery_suite_unique_claim_sections()
        net = recovery_suite_section_network(section)
        @test unique_claim_recovery_admits(net)
        @test admit_recovery_suite_network(section, net) === net
        @test count_unknown_destructions(net) == 1
    end
    hill_known = recovery_suite_section_network(:hill)
    @test count_unknown_destructions(hill_known) == 0
    @test admit_recovery_suite_network(:hill, hill_known) === hill_known
    @test validate_network_stays_open_source()
end

@testset "named KPI failures stay :unidentifiable_edge / :data_residual / :support_recall" begin
    contract = recovery_hard_named_kpi_contract()
    @test contract.symbols ==
          (:unidentifiable_edge, :data_residual, :support_recall)
    @test contract.f1_is_not_a_symbol
    @test contract.empty_label == "(none)"
    @test contract.miss_edge == "unidentifiable_edge"
    @test contract.miss_all == "unidentifiable_edge, data_residual, support_recall"
    hold = named_kpi_failure_row()
    @test hold.hold
    @test isempty(hold.failures)
    @test hold.label == "(none)"
    @test hold.message == "unique-claim KPIs hold"
    miss_edge = named_kpi_failure_row(; unidentifiable_edge = false)
    @test miss_edge.failures == [:unidentifiable_edge]
    @test occursin("unidentifiable_edge", miss_edge.message)
    miss_fit = named_kpi_failure_row(; data_residual = 0.31, support_recall = 0.4)
    @test :data_residual in miss_fit.failures
    @test :support_recall in miss_fit.failures
    @test !(:support_f1 in miss_fit.failures)
    @test unique_claim_kpi_failure_symbols_hold(miss_fit.failures)
    @test unique_claim_kpi_failure_symbols_hold([:support_f1]) == false
end

@testset "UniqueClaimProtocolRow joins fingerprint, extras, and named failures" begin
    row = unique_claim_protocol_row_from_fields()
    @test row isa UniqueClaimProtocolRow
    @test unique_claim_fingerprint_is_protocol(row.fingerprint)
    @test row.extras_label == "1, r"
    @test isempty(row.kpi_failures)
    @test assert_unique_claim_protocol_row(row) === row
    @test assert_unique_claim_protocol_row_holds(row) === row
    named = unique_claim_protocol_row_namedtuple(row)
    @test named.is_protocol
    @test named.n_ics == 9
    @test named.n_points == 50
    @test named.claim === :recall_plus_data_residual
    @test named.canonical_hill_from_nn == false
    smoke = unique_claim_protocol_row_from_fields(; smoke = true, extras = nothing)
    @test unique_claim_fingerprint_is_smoke(smoke.fingerprint)
    @test smoke.extras_label == "NA"
    @test assert_unique_claim_protocol_row(smoke) === smoke
    @test_throws ErrorException assert_unique_claim_protocol_row_holds(smoke)
    miss = unique_claim_protocol_row_from_fields(;
        unidentifiable_edge = false, data_residual = 0.31)
    @test :unidentifiable_edge in miss.kpi_failures
    @test :data_residual in miss.kpi_failures
    @test assert_unique_claim_protocol_row(miss) === miss
    @test_throws ErrorException assert_unique_claim_protocol_row_holds(miss)
    empty_extras = unique_claim_protocol_row_from_fields(; extras = String[])
    @test empty_extras.extras_label == "(none)"
    @test occursin("extras: (none)", empty_extras.text)
    @test format_protocol_result_field_order_holds(row.text)
    @test occursin("n_ics: 9", row.text)
    @test occursin("hybrid_data_residual: 0.003", row.text)
end

@testset "example and docs name the joint admission and datagen contracts" begin
    @test unique_claim_example_uses_experiment_set()
    src = read(unique_claim_example_path(), String)
    @test occursin("unique_claim_experiment_set", src)
    @test occursin("unique_claim_fingerprint", src)
    page = read(joinpath(pkgdir(BioDynaX), "docs", "src", "unique-claim.md"), String)
    sentences = unique_claim_locked_sentences()
    @test occursin(sentences.datagen, page)
    @test occursin(sentences.admission, page)
    @test occursin(sentences.protocol_row, page)
    architecture = read(
        joinpath(pkgdir(BioDynaX), "docs", "src", "architecture.md"), String)
    @test occursin("generate_from_compiled_model", architecture) ||
          occursin("compiled NN tree", architecture)
    @test occursin("admit_recovery_suite_network", architecture)
    news = read(joinpath(pkgdir(BioDynaX), "NEWS.md"), String)
    @test occursin("admit_recovery_suite_network", news)
    @test occursin("UniqueClaimProtocolRow", news)
    @test occursin("generate_from_compiled_model", news)
end
