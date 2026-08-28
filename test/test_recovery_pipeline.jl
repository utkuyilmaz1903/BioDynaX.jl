using BioDynaX: MechanismRecoveryResult,
    generate_recovery_experiments,
    consume_shared_suite_rng!,
    sample_destruction,
    evaluate_recovery,
    report_recovery,
    _evaluate_unknown_rate_recovery,
    run_recovery_suite,
    recovery_suite_section_body

function _mechanism_recovery_result(;
        extras = ["1", "r"],
        extras_denominator = nothing,
        discovery = nothing,
        term = nothing,
        identifiability = (; unidentifiable_edge = true),
        locked_kpis = nothing,
        protocol_result = nothing)
    return MechanismRecoveryResult(
        nn_correlation = 0.95,
        nn_rate_rmse = 0.05,
        success = true,
        retcode = DiscoverySuccess,
        message = "ok",
        support_f1 = 0.57,
        support_recall = 1.0,
        discovered_rate_rmse = 0.10,
        data_residual = 0.003,
        denominator_violations = 0,
        normalized_support_f1 = 0.60,
        normalized_support_recall = 1.0,
        extras = extras,
        extras_denominator = extras_denominator,
        discovery = discovery,
        term = term,
        identifiability = identifiability,
        locked_kpis = locked_kpis,
        protocol_result = protocol_result)
end

function _composer_early_evaled()
    return (;
        nn_correlation = 0.1,
        nn_rate_rmse = 1.0,
        success = false,
        retcode = DiscoveryFailed,
        message = "training did not identify the unknown edge",
        support_f1 = 0.0,
        support_recall = 0.0,
        discovered_rate_rmse = Inf,
        data_residual = Inf,
        denominator_violations = typemax(Int),
        normalized_support_f1 = 0.0,
        normalized_support_recall = 0.0,
        extras = String[],
        discovery = nothing,
        term = nothing)
end

function _composer_valid_evaled()
    return (;
        nn_correlation = 0.95,
        nn_rate_rmse = 0.05,
        success = true,
        retcode = DiscoverySuccess,
        message = "ok",
        support_f1 = 0.57,
        support_recall = 1.0,
        discovered_rate_rmse = 0.10,
        data_residual = 0.003,
        denominator_violations = 0,
        normalized_support_f1 = 0.60,
        normalized_support_recall = 1.0,
        extras = ["1", "r"],
        extras_denominator = (; train = 0, val = 0, domain = 0),
        discovery = :discovery_ran,
        term = :term)
end

function _real_composer_early_evaled()
    rng = MersenneTwister(1)
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    model, params = build_ude_model(rng, net)
    term = only_unknown_destruction(model)
    residual_calls = Ref(0)
    evaled = _evaluate_unknown_rate_recovery(
        model, params, term,
        r -> hill_rate_truth(r; vmax = 1.8, K = 0.55, n = 2);
        order = 2, family = :hill, noise_σ = 0.0,
        data_residual_fn = _ -> (residual_calls[] += 1; Inf))
    return evaled, residual_calls[], term
end

@testset "MechanismRecoveryResult stays internal" begin
    @test !(:MechanismRecoveryResult in names(BioDynaX))
    @test isdefined(BioDynaX, :MechanismRecoveryResult)
    @test public_export_list_holds()
    @test !isdefined(BioDynaX, :DestructionSamples)
    @test !isdefined(BioDynaX, :ExperimentSplit)
    @test isdefined(BioDynaX, :generate_recovery_experiments)
    @test isdefined(BioDynaX, :consume_shared_suite_rng!)
    @test isdefined(BioDynaX, :fit_unknown_destruction)
    @test !(:generate_recovery_experiments in names(BioDynaX))
    @test !(:consume_shared_suite_rng! in names(BioDynaX))
    @test !(:fit_unknown_destruction in names(BioDynaX))
    @test isdefined(BioDynaX, :sample_destruction)
    @test !(:sample_destruction in names(BioDynaX))
    @test isdefined(BioDynaX, :evaluate_recovery)
    @test !(:evaluate_recovery in names(BioDynaX))
    @test isdefined(BioDynaX, :report_recovery)
    @test !(:report_recovery in names(BioDynaX))
end

@testset "generate_recovery_experiments is the 9-IC unique-claim set" begin
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    proto = UNIQUE_CLAIM_PROTOCOL
    ics = BioDynaX._unknown_edge_ics()
    set = generate_recovery_experiments(
        MersenneTwister(7), truth_net, truth;
        tspan = proto.tspan, n_points = proto.n_points,
        noise_σ = proto.observation_noise)
    legacy = generate_experiment_set(
        MersenneTwister(7); network = truth_net, initial_conditions = ics,
        tspan = proto.tspan, n_points = proto.n_points,
        noise_σ = proto.observation_noise, truth_params = truth)
    @test length(set.experiments) == proto.n_ics
    @test length(set.experiments) == 9
    @test length(ics) == 9
    @test set.metadata[:n_ics] == 9
    @test set.metadata[:n_points] == proto.n_points
    @test set.metadata[:tspan] == proto.tspan
    @test !haskey(set.metadata, :unique_claim_fingerprint_kind)
    for (generated, expected, u0) in zip(set.experiments, legacy.experiments, ics)
        @test generated.u0 == u0
        @test length(generated.times) == proto.n_points
        @test first(generated.times) == first(proto.tspan)
        @test last(generated.times) == last(proto.tspan)
        @test generated.observations ≈ expected.observations
    end
end

@testset "consume_shared_suite_rng! matches discarded build_ude_model" begin
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    seed = 103
    discarded = build_ude_model(MersenneTwister(seed), truth_net)
    consumed = consume_shared_suite_rng!(MersenneTwister(seed), truth_net)
    @test discarded[2].nn ≈ consumed[2].nn
    rng_dummy = MersenneTwister(seed)
    rng_consume = MersenneTwister(seed)
    rng_skip = MersenneTwister(seed)
    build_ude_model(rng_dummy, truth_net)
    consume_shared_suite_rng!(rng_consume, truth_net)
    _, p_dummy = build_ude_model(rng_dummy, ude_net)
    _, p_consume = build_ude_model(rng_consume, ude_net)
    _, p_skip = build_ude_model(rng_skip, ude_net)
    @test p_dummy.nn ≈ p_consume.nn
    @test Vector(p_dummy.phys) ≈ Vector(p_consume.phys)
    @test !(p_skip.nn ≈ p_dummy.nn)
end

@testset "_train_unknown_edge remains the Recovery.jl compatibility wrapper" begin
    @test isdefined(BioDynaX, :_train_unknown_edge)
    @test !(:_train_unknown_edge in names(BioDynaX))
    @test BioDynaX.train_unknown_edge_reuses_warmup_source()
end

@testset "MechanismRecoveryResult keeps the current field surface" begin
    result = _mechanism_recovery_result()
    @test result.nn_correlation == 0.95
    @test result.nn_rate_rmse == 0.05
    @test result.success
    @test result.retcode === DiscoverySuccess
    @test result.message == "ok"
    @test result.support_f1 == 0.57
    @test result.support_recall == 1.0
    @test result.discovered_rate_rmse == 0.10
    @test result.data_residual == 0.003
    @test result.denominator_violations == 0
    @test result.normalized_support_f1 == 0.60
    @test result.normalized_support_recall == 1.0
    @test result.extras == ["1", "r"]
    @test result.extras_denominator === nothing
    @test result.discovery === nothing
    @test result.term === nothing
    @test result.identifiability.unidentifiable_edge
    @test result.locked_kpis === nothing
    @test result.protocol_result === nothing
    @test result.model === nothing
    @test result.params === nothing
    @test result.experiments === nothing
    fields = fieldnames(MechanismRecoveryResult)
    @test :nn_correlation in fields
    @test :locked_kpis in fields
    @test :protocol_result in fields
    @test :samples ∉ fields
    @test :r_range ∉ fields
    @test :holdout ∉ fields
    @test :train ∉ fields
    @test :functional_identifiability ∉ fields
    @test :independently_trained_D ∉ fields
    @test :uncertainty ∉ fields
    @test :hypothesis ∉ fields
end

@testset "MechanismRecoveryResult property getindex haskey keys" begin
    ident = (; unidentifiable_edge = true, collinearity = 0.97)
    result = _mechanism_recovery_result(; identifiability = ident)
    @test result.nn_correlation == result[:nn_correlation]
    @test result.locked_kpis === result[:locked_kpis]
    @test result[:identifiability] === ident
    @test haskey(result, :identifiability) || hasproperty(result, :identifiability)
    @test haskey(result, :nn_correlation)
    @test haskey(result, :protocol_result)
    @test haskey(result, :locked_kpis)
    @test hasproperty(result, :support_f1)
    @test :nn_correlation in keys(result)
    @test :protocol_result in keys(result)
    @test :samples ∉ keys(result)
    @test :holdout ∉ keys(result)
    @test :functional_identifiability ∉ keys(result)
    @test_throws KeyError result[:holdout]
    @test_throws KeyError result[:samples]
end

@testset "MechanismRecoveryResult is accepted by existing KPI helpers" begin
    result = _mechanism_recovery_result()
    kpis = locked_ude_kpis(result)
    proto = build_protocol_result(result)
    @test kpis.data_residual == result.data_residual
    @test kpis.support_recall == result.support_recall
    @test kpis.unidentifiable_edge
    @test kpis.claim === :recall_plus_data_residual
    @test proto.data_residual == result.data_residual
    @test proto.support_recall == result.support_recall
    @test proto.canonical_hill_from_nn === false
    @test Tuple(keys(proto)) == PROTOCOL_RESULT_FIELDS
    @test assert_protocol_result_fields(proto) === proto
    wrapped = _mechanism_recovery_result(;
        locked_kpis = kpis,
        protocol_result = proto)
    @test wrapped.locked_kpis === kpis
    @test wrapped.protocol_result === proto
    @test wrapped[:locked_kpis].data_residual == 0.003
    @test haskey(wrapped, :protocol_result)
end

@testset "sample_destruction returns the existing grid 3-tuple" begin
    rng = MersenneTwister(1)
    net = build_hill_recovery_network(; known = false)
    model, params = build_ude_model(rng, net)
    term = only_unknown_destruction(model)
    sampled = sample_destruction(model, params, term)
    expected = BioDynaX.sample_unknown_destruction_grid(model, params, term)
    @test sampled isa Tuple
    @test length(sampled) == 3
    R, D, chosen = sampled
    R_grid, D_grid, term_grid = expected
    @test R == R_grid
    @test D == D_grid
    @test chosen === term_grid
    @test chosen === term
    @test size(R) == (1, 80)
    @test size(D) == (1, 80)
    @test vec(R) == collect(range(0.05, 2.0; length = 80))
    r_range = range(0.1, 1.5; length = 11)
    R2, D2, term2 = sample_destruction(
        model, params, term; r_range = r_range, fill_value = 0.7)
    R2_grid, D2_grid, term2_grid = BioDynaX.sample_unknown_destruction_grid(
        model, params, term; r_range = r_range, fill_value = 0.7)
    @test R2 == R2_grid
    @test D2 == D2_grid
    @test term2 === term2_grid
    @test size(R2, 2) == 11
end

function _pipeline_function_body(name)
    src = read(joinpath(@__DIR__, "..", "src", "RecoveryPipeline.jl"), String)
    start = findfirst("function " * name, src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function _composer_function_body()
    src = read(joinpath(@__DIR__, "..", "src", "Recovery.jl"), String)
    start = findfirst("function _evaluate_unknown_rate_recovery", src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function _synthetic_discovery(success::Bool, candidates)
    return DiscoveryResult(
        success, success ? "ok" : "failed", nothing, nothing, nothing,
        candidates, (;))
end

@testset "evaluate_recovery is metric-only and unexported" begin
    @test isdefined(BioDynaX, :evaluate_recovery)
    @test !(:evaluate_recovery in names(BioDynaX))
    @test public_export_list_holds()
    body = _pipeline_function_body("evaluate_recovery")
    @test occursin("extras_denominator = ude_extras_denominator_row(", body)
    @test !occursin("discover_unknown_rate", body)
    @test !occursin("normalize_destruction_samples", body)
    @test !occursin("training_ok", body)
    @test !occursin("unique_claim_discovery_config", body)
    @test !occursin("times =", body)
    @test !occursin("RECOVERY_THRESHOLDS", body)
    composer = _composer_function_body()
    @test occursin("training_ok", composer)
    @test occursin("evaluate_recovery(", composer)
    @test occursin("success = discovery.success", composer)
    @test occursin("retcode = discovery.retcode", composer)
    @test occursin("message = discovery.message", composer)
end

@testset "evaluate_recovery does not call discovery or own training_ok" begin
    cand = BioDynaX.synthetic_safe_implicit_candidate()
    R = BioDynaX.regulator_grid(20)
    D = ones(size(R))
    truth = hill_rate_support(2)
    truth_rate = r -> hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    residual_calls = Ref(0)
    discovery = _synthetic_discovery(true, [cand])
    discovery_norm = _synthetic_discovery(false, ImplicitCandidate[])
    metrics = evaluate_recovery(
        R, D, discovery, discovery_norm, truth_rate, truth,
        d_hat -> (residual_calls[] += 1; 0.25))
    @test residual_calls[] == 1
    @test metrics.data_residual == 0.25
    @test :training_ok ∉ keys(metrics)
    @test :success ∉ keys(metrics)
    @test :retcode ∉ keys(metrics)
    @test :message ∉ keys(metrics)
    @test :nn_correlation ∉ keys(metrics)
    @test :nn_rate_rmse ∉ keys(metrics)
    @test :discovery ∉ keys(metrics)
    @test keys(metrics) == (
        :support_f1, :support_recall, :discovered_rate_rmse, :data_residual,
        :denominator_violations, :normalized_support_f1,
        :normalized_support_recall, :extras, :extras_denominator)
    failed = _synthetic_discovery(false, ImplicitCandidate[])
    fail_calls = Ref(0)
    failed_metrics = evaluate_recovery(
        R, D, failed, failed, truth_rate, truth,
        d_hat -> (fail_calls[] += 1; 0.0))
    @test fail_calls[] == 0
    @test failed_metrics.support_f1 == 0.0
    @test failed_metrics.support_recall == 0.0
    @test failed_metrics.discovered_rate_rmse == Inf
    @test failed_metrics.data_residual == Inf
    @test failed_metrics.denominator_violations == typemax(Int)
    @test failed_metrics.extras == String[]
    @test failed_metrics.extras_denominator === nothing
    @test failed_metrics.normalized_support_f1 == 0.0
    @test failed_metrics.normalized_support_recall == 0.0
    @test :training_ok ∉ keys(failed_metrics)
end

@testset "evaluate_recovery keeps current denominator and metric formulas" begin
    cand = BioDynaX.synthetic_unsafe_implicit_candidate()
    norm_cand = BioDynaX.synthetic_safe_implicit_candidate()
    R = BioDynaX.regulator_grid(40)
    D = ones(size(R))
    truth = hill_rate_support(2)
    truth_rate = r -> hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    discovery = _synthetic_discovery(true, [cand])
    discovery_norm = _synthetic_discovery(true, [norm_cand])
    extras = discovered_support_extras(cand, truth.numerator, truth.denominator)
    expected_f1 = support_f1(cand, truth.numerator, truth.denominator)
    expected_norm = support_f1(norm_cand, truth.numerator, truth.denominator)
    r = vec(R)
    D_true = truth_rate(r)
    d_hat = equation_to_function(cand)
    D_hat = [d_hat([rj]) for rj in r]
    expected_row = ude_extras_denominator_row(cand, R; extras = extras)
    metrics = evaluate_recovery(
        R, D, discovery, discovery_norm, truth_rate, truth, _ -> 0.11)
    @test metrics.support_f1 == expected_f1.combined.f1
    @test metrics.support_recall == expected_f1.combined.recall
    @test metrics.discovered_rate_rmse == rate_rel_rmse(D_hat, D_true)
    @test metrics.data_residual == 0.11
    @test metrics.denominator_violations ==
          denominator_violation_count(cand, R)
    @test metrics.extras == extras
    @test metrics.extras_denominator.train == expected_row.train
    @test metrics.extras_denominator.val == expected_row.val
    @test metrics.extras_denominator.domain == expected_row.domain
    @test metrics.extras_denominator.total == expected_row.total
    @test metrics.extras_denominator.any == expected_row.any
    @test metrics.extras_denominator.holds == expected_row.holds
    @test metrics.normalized_support_f1 == expected_norm.combined.f1
    @test metrics.normalized_support_recall == expected_norm.combined.recall
    @test metrics.denominator_violations > 0
    @test metrics.extras_denominator.any
end

@testset "report_recovery produces MechanismRecoveryResult with filled KPIs" begin
    evaled = _composer_valid_evaled()
    ident = (; unidentifiable_edge = true, production_param = :k_prod)
    ude_row = (; evaled..., identifiability = ident)
    expected_kpis = locked_ude_kpis(ude_row)
    expected_proto = build_protocol_result(ude_row)
    reported = report_recovery(
        evaled, ident; model = :model, params = :params, experiments = :set)
    @test reported isa MechanismRecoveryResult
    @test reported.locked_kpis !== nothing
    @test reported.protocol_result !== nothing
    @test reported.locked_kpis == expected_kpis
    @test reported.protocol_result == expected_proto
    @test Tuple(keys(reported.protocol_result)) == PROTOCOL_RESULT_FIELDS
    @test reported.protocol_result.canonical_hill_from_nn === false
    @test reported.protocol_result.claim === :recall_plus_data_residual
    @test reported.nn_correlation == evaled.nn_correlation
    @test reported.nn_rate_rmse == evaled.nn_rate_rmse
    @test reported.success == evaled.success
    @test reported.retcode === evaled.retcode
    @test reported.message == evaled.message
    @test reported.support_f1 == evaled.support_f1
    @test reported.support_recall == evaled.support_recall
    @test reported.discovered_rate_rmse == evaled.discovered_rate_rmse
    @test reported.data_residual == evaled.data_residual
    @test reported.denominator_violations == evaled.denominator_violations
    @test reported.normalized_support_f1 == evaled.normalized_support_f1
    @test reported.normalized_support_recall == evaled.normalized_support_recall
    @test reported.extras == evaled.extras
    @test reported.extras_denominator == evaled.extras_denominator
    @test reported.discovery === evaled.discovery
    @test reported.term === evaled.term
    @test reported.identifiability === ident
    @test reported.model === :model
    @test reported.params === :params
    @test reported.experiments === :set
    legacy = (;
        ude_row...,
        locked_kpis = expected_kpis,
        protocol_result = expected_proto)
    @test format_recovery_protocol(reported; equations = "D(z) = 1 + r") ==
          format_recovery_protocol(legacy; equations = "D(z) = 1 + r")
    row = unique_claim_protocol_row(reported)
    @test row.protocol_result === reported.protocol_result
    @test row.kpis === reported.locked_kpis
end

@testset "report_recovery accepts discovery nothing and missing extras_denominator" begin
    evaled = _composer_early_evaled()
    @test evaled.discovery === nothing
    @test !hasproperty(evaled, :extras_denominator)
    ident = (; unidentifiable_edge = true)
    reported = report_recovery(evaled, ident)
    @test reported isa MechanismRecoveryResult
    @test reported.discovery === nothing
    @test reported.extras_denominator === nothing
    @test reported.locked_kpis !== nothing
    @test reported.protocol_result !== nothing
    @test reported.success == false
    @test reported.data_residual == Inf
    @test reported.support_recall == 0.0
    @test reported.extras == String[]
    @test reported.protocol_result.canonical_hill_from_nn === false
    @test Tuple(keys(reported.protocol_result)) == PROTOCOL_RESULT_FIELDS
    ude_row = (; evaled..., identifiability = ident)
    @test reported.locked_kpis == locked_ude_kpis(ude_row)
    @test reported.protocol_result == build_protocol_result(ude_row)
end

@testset "haskey and hasproperty do not prove report field values" begin
    bare = _mechanism_recovery_result()
    @test haskey(bare, :protocol_result)
    @test hasproperty(bare, :protocol_result)
    @test bare.protocol_result === nothing
    @test haskey(bare, :locked_kpis)
    @test hasproperty(bare, :locked_kpis)
    @test bare.locked_kpis === nothing
    @test haskey(bare, :discovery)
    @test hasproperty(bare, :discovery)
    @test bare.discovery === nothing
    @test haskey(bare, :extras_denominator)
    @test hasproperty(bare, :extras_denominator)
    @test bare.extras_denominator === nothing
    txt = format_recovery_protocol(bare; equations = "D(z) = 1")
    @test occursin("canonical_hill_from_nn: false", txt)
    @test occursin("hybrid_data_residual: 0.003", txt)
    row = unique_claim_protocol_row(bare)
    @test row.protocol_result !== nothing
    @test row.kpis !== nothing
    @test Tuple(keys(row.protocol_result)) == PROTOCOL_RESULT_FIELDS
    @test row.protocol_result.canonical_hill_from_nn === false
    evaled = _composer_early_evaled()
    @test hasproperty(evaled, :discovery)
    @test evaled.discovery === nothing
    reported = report_recovery(evaled, (; unidentifiable_edge = false))
    @test haskey(reported, :discovery)
    @test hasproperty(reported, :discovery)
    @test reported.discovery === nothing
    @test haskey(reported, :extras_denominator)
    @test reported.extras_denominator === nothing
end

@testset "report_recovery does not introduce M2 or M3 fields" begin
    reported = report_recovery(
        _composer_valid_evaled(), (; unidentifiable_edge = true))
    fields = fieldnames(typeof(reported))
    @test :holdout ∉ fields
    @test :train ∉ fields
    @test :samples ∉ fields
    @test :r_range ∉ fields
    @test :functional_identifiability ∉ fields
    @test :independently_trained_D ∉ fields
    @test :uncertainty ∉ fields
    @test :hypothesis ∉ fields
    @test :occupancy ∉ fields
    @test :q4 ∉ fields
    @test :q7 ∉ fields
    @test :data_residual_holdout ∉ fields
    @test :d_rmse_holdout ∉ fields
    @test :holdout ∉ keys(reported)
    @test :samples ∉ keys(reported)
    @test :functional_identifiability ∉ keys(reported)
    @test !isdefined(BioDynaX, :DestructionSamples)
    @test !isdefined(BioDynaX, :ExperimentSplit)
    @test !isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)
    @test public_export_list_holds()
end

@testset "live unique-claim sections construct results through report_recovery" begin
    ude_body = recovery_suite_section_body(:ude_discovery)
    mm_body = recovery_suite_section_body(:mm_unknown)
    for body in (ude_body, mm_body)
        @test occursin("report_recovery(", body)
        @test occursin("_train_unknown_edge", body)
        @test occursin("_evaluate_unknown_rate_recovery", body)
        @test occursin("report_production_destruction_tradeoff", body)
        @test !occursin("(; evaled..., identifiability", body)
        @test !occursin("locked_ude_kpis(ude_row)", body)
        @test !occursin("locked_ude_kpis(mm_row)", body)
        @test !occursin("build_protocol_result(ude_row)", body)
        @test !occursin("build_protocol_result(mm_row)", body)
        @test !occursin("sample_destruction(", body)
        @test !occursin("evaluate_recovery(", body)
        @test !occursin("discover_unknown_rate(", body)
        @test !occursin("normalize_destruction_samples", body)
    end
    @test occursin("admit_recovery_suite_network(:ude_discovery)", ude_body)
    @test occursin("admit_recovery_suite_network(:mm_unknown)", mm_body)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.tspan", ude_body)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.n_points", ude_body)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.tspan", mm_body)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.n_points", mm_body)
    @test occursin("family = :mm", mm_body)
    @test findfirst("_train_unknown_edge", ude_body) <
          findfirst("_evaluate_unknown_rate_recovery", ude_body)
    @test findfirst("_evaluate_unknown_rate_recovery", ude_body) <
          findfirst("report_production_destruction_tradeoff", ude_body)
    @test findfirst("report_production_destruction_tradeoff", ude_body) <
          findfirst("report_recovery(", ude_body)
    @test findfirst("_train_unknown_edge", mm_body) <
          findfirst("_evaluate_unknown_rate_recovery", mm_body)
    @test findfirst("_evaluate_unknown_rate_recovery", mm_body) <
          findfirst("report_production_destruction_tradeoff", mm_body)
    @test findfirst("report_production_destruction_tradeoff", mm_body) <
          findfirst("report_recovery(", mm_body)
end

@testset "real composer early-return can be reported safely" begin
    evaled, residual_calls, term = _real_composer_early_evaled()
    @test residual_calls == 0
    @test evaled.discovery === nothing
    @test evaled.data_residual === Inf
    @test !hasproperty(evaled, :extras_denominator)
    @test evaled.success == false
    @test evaled.retcode === DiscoveryFailed
    @test evaled.support_recall == 0.0
    @test evaled.extras == String[]
    @test evaled.term === term
    ident = (; unidentifiable_edge = true, production_param = :k_prod)
    reported = report_recovery(evaled, ident)
    @test reported isa MechanismRecoveryResult
    @test reported.discovery === nothing
    @test reported.data_residual === Inf
    @test reported.extras_denominator === nothing
    @test hasproperty(reported, :locked_kpis) && reported.locked_kpis !== nothing
    @test hasproperty(reported, :protocol_result) &&
          reported.protocol_result !== nothing
    @test reported.success == false
    @test reported.support_recall == 0.0
    @test reported.extras == String[]
    @test reported.identifiability === ident
    @test reported.protocol_result.canonical_hill_from_nn === false
    @test Tuple(keys(reported.protocol_result)) == PROTOCOL_RESULT_FIELDS
    ude_row = (; evaled..., identifiability = ident)
    @test reported.locked_kpis == locked_ude_kpis(ude_row)
    @test reported.protocol_result == build_protocol_result(ude_row)
    txt = format_recovery_protocol(reported)
    @test occursin("hybrid_data_residual: Inf", txt)
    @test occursin("canonical_hill_from_nn: false", txt)
    row = unique_claim_protocol_row(reported)
    @test row.protocol_result !== nothing
    @test row.kpis !== nothing
end

@testset "live :ude_discovery and :mm_unknown paths use MechanismRecoveryResult" begin
    report = run_recovery_suite(MersenneTwister(1);
        ude_adam = 0, ude_bfgs = 0,
        sections = (:ude_discovery, :mm_unknown))
    @test report isa Dict{Symbol,Any}
    @test issetequal(keys(report), Set((:ude_discovery, :mm_unknown)))
    for section in (:ude_discovery, :mm_unknown)
        row = report[section]
        @test row isa MechanismRecoveryResult
        @test hasproperty(row, :locked_kpis) && row.locked_kpis !== nothing
        @test hasproperty(row, :protocol_result) && row.protocol_result !== nothing
        @test row[:data_residual] === row.data_residual
        @test row[:locked_kpis] === row.locked_kpis
        @test row[:protocol_result] === row.protocol_result
        @test haskey(row, :identifiability)
        @test haskey(row, :nn_correlation)
        @test haskey(row, :support_f1)
        @test row.data_residual === Inf
        @test row.discovery === nothing
        @test row.protocol_result.canonical_hill_from_nn === false
        @test row.protocol_result.claim === :recall_plus_data_residual
        @test Tuple(keys(row.protocol_result)) == PROTOCOL_RESULT_FIELDS
        fields = fieldnames(typeof(row))
        @test :holdout ∉ fields
        @test :train ∉ fields
        @test :samples ∉ fields
        @test :r_range ∉ fields
        @test :functional_identifiability ∉ fields
        @test :independently_trained_D ∉ fields
        @test :uncertainty ∉ fields
        @test :hypothesis ∉ fields
        @test :occupancy ∉ fields
        @test :q4 ∉ fields
        @test :q7 ∉ fields
        @test :data_residual_holdout ∉ fields
        @test :d_rmse_holdout ∉ fields
        @test :holdout ∉ keys(row)
        @test :samples ∉ keys(row)
        @test :functional_identifiability ∉ keys(row)
        txt = format_recovery_protocol(row)
        @test occursin("IDENTIFIABILITY", txt)
        @test occursin("FIT", txt)
        @test occursin("DISCOVERY", txt)
        @test occursin("REPRODUCTION", txt)
        @test occursin("canonical_hill_from_nn: false", txt)
        proto_row = unique_claim_protocol_row(row)
        @test proto_row.protocol_result !== nothing
        @test proto_row.kpis !== nothing
        @test proto_row.protocol_result === row.protocol_result
        @test proto_row.kpis === row.locked_kpis
    end
    @test !isdefined(BioDynaX, :DestructionSamples)
    @test !isdefined(BioDynaX, :ExperimentSplit)
    @test !isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)
    @test public_export_list_holds()
end
