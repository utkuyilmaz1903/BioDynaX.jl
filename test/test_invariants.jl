@testset "extras printer does not invent F1-attempt leftovers" begin
    @test extras_print_label(nothing) == "NA"
    @test extras_print_label(String[]) == "(none)"
    @test extras_print_label(("1", "r")) == "1, r"
    @test extras_print_is_hardcoded_attempt("1, r remain after the UDE F1 attempt")
    @test extras_print_is_hardcoded_attempt(extras_print_label(nothing)) == false
end

@testset "typed fingerprint is not smoke and is not exported" begin
    fp = reference_protocol_fingerprint()
    @test reference_protocol_fingerprint_holds(fp)
    @test reference_protocol_fingerprint_is_protocol(fp)
    @test !(:ReferenceProtocolFingerprint in names(BioDynaX))
    @test !(:REFERENCE_PROTOCOL_F1_ATTEMPT in names(BioDynaX))
    @test !(:ReferenceProtocolRow in names(BioDynaX))
    @test !(:admit_recovery_suite_network in names(BioDynaX))
    @test reference_protocol_f1_attempt_spec().is_protocol == false
end

@testset "locked UDE KPI names" begin
    fake = (;
        data_residual = 0.01,
        support_recall = 1.0,
        support_f1 = 0.57,
        identifiability = (; unidentifiable_edge = true),
        extras = ("1", "r"))
    kpis = locked_ude_kpis(fake)
    @test kpis.claim === :recall_plus_data_residual
    @test kpis.data_residual == 0.01
    @test kpis.support_recall == 1.0
    @test kpis.unidentifiable_edge
end

@testset "protocol result field order is the reference protocol product" begin
    ude = (;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        identifiability = (; unidentifiable_edge = true))
    result = build_protocol_result(ude)
    @test Tuple(keys(result)) == (
        :unknown_holes,
        :unidentifiable_edge,
        :coefficients_are_biological_constants,
        :data_residual,
        :support_recall,
        :support_f1,
        :extras,
        :canonical_hill_from_nn,
        :claim)
    @test result.unknown_holes == 1
    @test result.unidentifiable_edge
    @test result.coefficients_are_biological_constants == false
    @test result.data_residual == 0.003
    @test result.support_recall == 1.0
    @test result.extras == ["1", "r"]
    @test result.canonical_hill_from_nn == false
    @test result.claim === :recall_plus_data_residual
    missing_ident = build_protocol_result((;
        data_residual = Inf, support_recall = 0.0))
    @test missing_ident.unidentifiable_edge == false
    @test missing_ident.coefficients_are_biological_constants
    @test !(:build_protocol_result in names(BioDynaX))
end

@testset "reference protocol KPI helpers encode the locked thresholds" begin
    passing = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = true)))
    @test reference_protocol_kpis_hold(passing)
    @test assert_reference_protocol_residual(0.003) == 0.003
    miss_edge = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = false)))
    @test reference_protocol_kpis_hold(miss_edge) == false
    miss_residual = locked_ude_kpis((;
        data_residual = 0.31,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = true)))
    @test reference_protocol_kpis_hold(miss_residual) == false
    miss_recall = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 0.5,
        identifiability = (; unidentifiable_edge = true)))
    @test reference_protocol_kpis_hold(miss_recall) == false
    @test_throws ErrorException assert_reference_protocol_residual(0.31)
    @test !(:reference_protocol_kpis_hold in names(BioDynaX))
    @test !(:assert_reference_protocol_residual in names(BioDynaX))
end

@testset "locked UDE KPIs survive early-fail rows" begin
    early = (;
        data_residual = Inf,
        support_recall = 0.0,
        support_f1 = 0.0)
    kpis = locked_ude_kpis(early)
    @test kpis.claim === :recall_plus_data_residual
    @test kpis.data_residual == Inf
    @test kpis.support_recall == 0.0
    @test kpis.unidentifiable_edge == false
end

@testset "production-destruction positivity invariant" begin
    rng = MersenneTwister(0)
    for network in (build_linear_test_network(),
        build_hill_recovery_network(; known = true, hill_order = 2),
        build_mm_test_network())
        model, params = build_ude_model(rng, network)
        n = model.compiled.nstates
        xs = ([0.4, 0.3, 0.2, 0.1][1:n],
            zeros(n),
            [0.0, 0.5, 0.0, 0.4][1:n])
        for x in xs
            for i in 1:n
                P = BioDynaX._state_production(
                    i, x, params, model.compiled.production_terms)
                D = BioDynaX._state_destruction(
                    i, x, params, model.compiled.destruction_terms,
                    model.nn, model.st)
                @test P ≥ -1e-14
                @test D ≥ -1e-14
            end
            dx = ude_system(x, params, 0.0, model)
            for i in 1:n
                if x[i] == 0
                    @test dx[i] ≥ -1e-12
                end
            end
        end
    end
end

@testset "graph-local library excludes distractor Z" begin
    net = build_six_state_unknown_network()
    local_spec = local_basis(net, 1; degree = 2, include_interactions = false,
        scope = :graph)
    global_spec = local_basis(net, 1; degree = 2, include_interactions = false,
        scope = :global)
    Z_in_local_library = 6 ∈ local_spec.variables
    Z_in_global_library = 6 ∈ global_spec.variables
    @test 2 ∈ local_spec.variables
    @test Z_in_local_library == false
    @test Z_in_global_library
    @test 2 ∈ candidate_parents(net, 1)
end

@testset "discovery retcode messages" begin
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    X = reshape(collect(range(0.1, 0.2; length = 5)), 1, :)
    times = collect(range(0.0, 1.0; length = 5))
    result = discover_equations(
        X, times, network; derivatives = X,
        verbose = false, strict = false)
    @test result.retcode === InsufficientSamples
    @test occursin("insufficient", lowercase(result.message))
    err = try
        discover_equations(
            X, times, network; derivatives = X, verbose = false, strict = true)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("insufficient", lowercase(err.msg))

    @test BioDynaX._discovery_retcode(ArgumentError(
        "empty support: no terms survived thresholding")) === EmptySupport
    @test BioDynaX._discovery_retcode(DomainError(
        0.0, "discovered denominator is singular")) === DenominatorUnsafe
    @test BioDynaX._discovery_retcode(LinearAlgebra.SingularException(1)) ===
          SingularLibrary
    @test BioDynaX._discovery_retcode(ErrorException("boom")) === DiscoveryFailed
end

@testset "ude_rhs! vs ude_system parity" begin
    rng = MersenneTwister(11)
    model, params = build_ude_model(rng, build_linear_test_network())
    cache = allocate_cache(model, Float64)
    u = [0.25, 0.15]
    ude_rhs!(cache.du, u, params, 0.0, model, cache)
    @test cache.du ≈ ude_system(u, params, 0.0, model)
end

@testset "analytical Hill discovery breaks at σ = 0.05" begin
    r = collect(range(0.1, 2.0; length = 180))
    rng = MersenneTwister(104)
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    amp = max(maximum(abs, D), eps(Float64))
    D_obs = D .+ 0.05 .* amp .* randn(rng, length(r))
    result = discover_unknown_rate(
        reshape(r, 1, :), collect(range(0.0, 1.0; length = length(r))),
        reshape(D_obs, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = 104),
        verbose = false, strict = false)
    truth = hill_rate_support(2)
    f1 = result.success ?
         support_f1(result.candidates[1], truth.numerator, truth.denominator).combined.f1 :
         0.0
    recall = result.success ?
             support_f1(
        result.candidates[1], truth.numerator, truth.denominator).combined.recall :
             0.0
    den = result.success ?
          denominator_violation_count(result.candidates[1], reshape(r, 1, :)) :
          typemax(Int)
    holds = result.success &&
            f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean &&
            recall ≥ RECOVERY_THRESHOLDS.support_recall &&
            den == 0
    @test holds == false
end
