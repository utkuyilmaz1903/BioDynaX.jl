@testset "identifiability product derives coefficients from the edge" begin
    edge_true = (; unidentifiable_edge = true, production_param = :k_prod,
        collinearity = 0.997)
    product = identifiability_product(edge_true)
    @test product.unidentifiable_edge
    @test product.coefficients_are_biological_constants == false
    @test product.production_param === :k_prod
    @test product.practical_not_structural
    @test product.unknown_holes == 1
    @test coefficients_are_biological_constants(edge_true) == false
    @test unique_claim_identifiability_holds(edge_true)
    @test assert_unique_claim_identifiability(edge_true) === edge_true

    edge_false = (; unidentifiable_edge = false, production_param = :k_prod)
    @test coefficients_are_biological_constants(edge_false)
    @test unique_claim_identifiability_holds(edge_false) == false
    @test identifiability_product(edge_false).coefficients_are_biological_constants
    @test_throws ErrorException assert_unique_claim_identifiability(edge_false)

    @test coefficients_are_biological_constants(nothing)
    @test unique_claim_identifiability_holds(nothing) == false
    @test_throws ErrorException assert_unique_claim_identifiability(nothing)
    missing_flag = (; production_param = :k_prod)
    @test coefficients_are_biological_constants(missing_flag)
    @test unique_claim_identifiability_holds(missing_flag) == false
    @test identifiability_product(missing_flag).unidentifiable_edge == false
    @test identifiability_product(nothing; unknown_holes = 0).unknown_holes == 0
    @test !(:identifiability_product in names(BioDynaX))
    @test !(:coefficients_are_biological_constants in names(BioDynaX))
    @test !(:assert_unique_claim_identifiability in names(BioDynaX))
end

@testset "protocol_result field order rejects swapped or extra keys" begin
    ude = (;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        identifiability = (; unidentifiable_edge = true))
    result = build_protocol_result(ude)
    @test Tuple(keys(result)) == protocol_result_field_order()
    @test Tuple(keys(result)) == PROTOCOL_RESULT_FIELDS
    @test assert_protocol_result_fields(result) === result
    @test result.coefficients_are_biological_constants == false
    @test result.canonical_hill_from_nn == false
    two_holes = build_protocol_result(ude; unknown_holes = 2)
    @test two_holes.unknown_holes == 2
    @test two_holes.unidentifiable_edge
    @test assert_protocol_result_fields(two_holes) === two_holes

    swapped = (;
        unidentifiable_edge = true,
        unknown_holes = 1,
        coefficients_are_biological_constants = false,
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        canonical_hill_from_nn = false,
        claim = :recall_plus_data_residual)
    @test Tuple(keys(swapped)) != PROTOCOL_RESULT_FIELDS
    @test_throws ErrorException assert_protocol_result_fields(swapped)

    hill_open = (;
        unknown_holes = 1,
        unidentifiable_edge = true,
        coefficients_are_biological_constants = false,
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.99,
        extras = String[],
        canonical_hill_from_nn = true,
        claim = :recall_plus_data_residual)
    @test_throws ErrorException assert_protocol_result_fields(hill_open)

    wrong_claim = (;
        unknown_holes = 1,
        unidentifiable_edge = true,
        coefficients_are_biological_constants = false,
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        canonical_hill_from_nn = false,
        claim = :canonical_hill)
    @test_throws ErrorException assert_protocol_result_fields(wrong_claim)
end

@testset "KPI helpers name the failed gate and ignore F1" begin
    hold = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.40,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpis_hold(hold)
    @test isempty(unique_claim_kpi_failures(hold))
    @test assert_unique_claim_kpis(hold) === hold
    @test unique_claim_residual_holds(hold.data_residual)
    @test unique_claim_recall_holds(hold.support_recall)
    @test unique_claim_f1_meets_skeleton_floor(0.57)
    @test unique_claim_f1_meets_skeleton_floor(0.40) == false
    @test unique_claim_f1_reaches_analytical_gate(0.57) == false
    @test unique_claim_f1_reaches_analytical_gate(0.99)
    @test UNIQUE_CLAIM_KPI_FIELDS == (
        :unidentifiable_edge, :data_residual, :support_recall)
    @test !(:support_f1 in UNIQUE_CLAIM_KPI_FIELDS)

    miss_edge = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = false)))
    @test unique_claim_kpi_failures(miss_edge) == [:unidentifiable_edge]
    @test_throws ErrorException assert_unique_claim_kpis(miss_edge)

    miss_residual = locked_ude_kpis((;
        data_residual = 0.31,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpi_failures(miss_residual) == [:data_residual]
    @test unique_claim_residual_holds(0.31) == false
    @test_throws ErrorException assert_unique_claim_residual(0.31)

    miss_recall = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 0.5,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpi_failures(miss_recall) == [:support_recall]
    @test unique_claim_recall_holds(0.5) == false
    @test_throws ErrorException assert_unique_claim_recall(0.5)
    @test assert_unique_claim_recall(1.0) == 1.0

    miss_all = locked_ude_kpis((;
        data_residual = 0.5,
        support_recall = 0.0,
        identifiability = nothing))
    @test unique_claim_kpi_failures(miss_all) == [
        :unidentifiable_edge, :data_residual, :support_recall]
    err = try
        assert_unique_claim_kpis(miss_all)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("unidentifiable_edge", err.msg)
    @test occursin("data_residual", err.msg)
    @test occursin("support_recall", err.msg)
    @test !occursin("support_f1", err.msg)
    @test !(:unique_claim_kpi_failures in names(BioDynaX))
    @test !(:assert_unique_claim_kpis in names(BioDynaX))
    @test !(:assert_unique_claim_recall in names(BioDynaX))
end

@testset "live extras on dirty Hill do not open Hill-from-NN" begin
    hill = unique_claim_truth_support(; family = :hill, order = 2)
    @test first(hill.numerator) == ((1,), (2,))
    mm = unique_claim_truth_support(; family = :mm)
    @test first(mm.numerator) == ((1,), (1,))
    @test_throws ArgumentError unique_claim_truth_support(; family = :competitive)

    r = collect(range(0.1, 2.0; length = 180))
    times = collect(range(0.0, 1.0; length = length(r)))
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    clean = discover_unknown_rate(
        reshape(r, 1, :), times, reshape(D, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = 1),
        verbose = false, strict = true)
    @test clean.success
    @test isempty(unique_claim_discovery_extras(clean))
    @test isempty(unique_claim_discovery_extras(clean.candidates[1]))
    clean_f1 = support_f1(
        clean.candidates[1], hill.numerator, hill.denominator).combined.f1
    @test unique_claim_f1_reaches_analytical_gate(clean_f1)

    dirty = discover_unknown_rate(
        reshape(r, 1, :), times, reshape(D .+ 0.04 .+ 0.04 .* r, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = 103),
        verbose = false, strict = false)
    @test dirty.success
    extras = unique_claim_discovery_extras(dirty)
    @test "1" in extras
    @test "r" in extras
    @test !("r^2" in extras)
    dirty_f1 = support_f1(
        dirty.candidates[1], hill.numerator, hill.denominator).combined.f1
    @test unique_claim_f1_reaches_analytical_gate(dirty_f1) == false
    @test unique_claim_f1_attempt_verdict(;
        extras, reaches_clean = false) ===
          :extras_remain_claim_stays_recall_plus_residual
    @test unique_claim_f1_attempt_verdict(;
        extras = String[], reaches_clean = true) ===
          :reopen_only_after_protocol_holds_clean
    @test unique_claim_f1_attempt_verdict(;
        extras = String[], reaches_clean = false) ===
          :no_extras_but_clean_gate_not_reached

    failed = discover_equations(
        reshape(collect(range(0.1, 0.2; length = 5)), 1, :),
        collect(range(0.0, 1.0; length = 5)),
        BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[]);
        derivatives = reshape(collect(range(0.1, 0.2; length = 5)), 1, :),
        verbose = false, strict = false)
    @test !failed.success
    @test unique_claim_discovery_extras(failed) == String[]
    @test !(:unique_claim_discovery_extras in names(BioDynaX))
    @test !(:unique_claim_f1_attempt_verdict in names(BioDynaX))
end

@testset "single-hole instrument does not close validate_network" begin
    @test validate_network_stays_open_source()
    rng = MersenneTwister(0)

    zero_net = build_linear_test_network()
    @test validate_network(zero_net) === zero_net
    @test count_unknown_destructions(zero_net) == 0
    zero_model, _ = build_ude_model(rng, zero_net)
    @test count_unknown_destructions(zero_model) == 0
    @test_throws ErrorException assert_single_unknown_destruction(zero_model)
    @test_throws ErrorException only_unknown_destruction(zero_model)

    one_net = build_hill_recovery_network(; known = false)
    @test validate_network(one_net) === one_net
    @test count_unknown_destructions(one_net) == 1
    one_model, _ = build_ude_model(rng, one_net)
    @test count_unknown_destructions(one_model) == 1
    @test assert_single_unknown_destruction(one_model) == 1
    @test only_unknown_destruction(one_model) isa NeuralDestructionTerm

    two_net = build_dual_unknown_network()
    @test validate_network(two_net) === two_net
    @test count_unknown_destructions(two_net) == 2
    two_model, two_p = build_ude_model(rng, two_net)
    @test count_unknown_destructions(two_model) == 2
    @test_throws ErrorException assert_single_unknown_destruction(two_model)
    @test_throws ErrorException only_unknown_destruction(two_model)
    dx = ude_system([0.2, 0.3, 0.4], two_p, 0.0, two_model)
    @test all(isfinite, dx)

    known_hill = build_hill_recovery_network(; known = true)
    @test validate_network(known_hill) === known_hill
    @test count_unknown_destructions(known_hill) == 0
    @test !(:count_unknown_destructions in names(BioDynaX))
end

@testset "protocol stdout splits into four product blocks" begin
    ident = (;
        unidentifiable_edge = true,
        production_param = :k_prod,
        collinearity = 0.99)
    sections = format_protocol_sections(ident;
        residual = 0.003,
        equations = "D(z) = vmax * r^2 / (K^2 + r^2)",
        extras = ("1", "r"),
        support_f1 = 0.57,
        support_recall = 1.0,
        unknown_holes = 1,
        seed = 103,
        n_ics = 9,
        n_points = 50,
        adam_iters = 100,
        bfgs_iters = 50,
        bootstrap = 8,
        discovery_seed = 3,
        smoke = false)
    @test sections.order_holds
    @test unique_claim_product_blocks() == UNIQUE_CLAIM_PRODUCT_BLOCKS
    @test protocol_block_order_holds(sections.text)
    @test startswith(sections.identifiability, "IDENTIFIABILITY")
    @test occursin("unidentifiable_edge: true", sections.identifiability)
    @test occursin("coefficients_are_biological_constants: false",
        sections.identifiability)
    @test !occursin("hybrid_data_residual", sections.identifiability)
    @test startswith(sections.fit, "FIT")
    @test occursin("hybrid_data_residual: 0.003", sections.fit)
    @test occursin("support_recall: 1.0", sections.fit)
    @test !occursin("canonical_hill_from_nn", sections.fit)
    @test startswith(sections.discovery, "DISCOVERY")
    @test occursin("extras: 1, r", sections.discovery)
    @test occursin("support_f1: 0.57", sections.discovery)
    @test occursin("canonical_hill_from_nn: false", sections.discovery)
    @test startswith(sections.reproduction, "REPRODUCTION")
    @test occursin("seed: 103", sections.reproduction)
    @test occursin("n_ics: 9", sections.reproduction)
    @test occursin("n_points: 50", sections.reproduction)
    @test occursin("protocol_kind: protocol", sections.reproduction)
    @test occursin("smoke: false", sections.reproduction)

    smoke_txt = format_protocol_result(ident; smoke = true, n_ics = 1, n_points = 8)
    @test occursin("protocol_kind: smoke", smoke_txt)
    @test protocol_block_order_holds(smoke_txt)
    @test !occursin("Note:", smoke_txt)

    fake_ude = (;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        identifiability = ident,
        protocol_result = build_protocol_result((;
            data_residual = 0.003,
            support_recall = 1.0,
            support_f1 = 0.57,
            extras = ["1", "r"],
            identifiability = ident)))
    recovery_txt = format_recovery_protocol(fake_ude; equations = "D(z) = 1 + r")
    @test protocol_block_order_holds(recovery_txt)
    @test occursin("extras: 1, r", recovery_txt)
    @test occursin("n_ics: 9", recovery_txt)
    @test occursin("canonical_hill_from_nn: false", recovery_txt)
    @test !(:format_protocol_sections in names(BioDynaX))
    @test !(:format_recovery_protocol in names(BioDynaX))
    @test !(:protocol_block_order_holds in names(BioDynaX))
end
