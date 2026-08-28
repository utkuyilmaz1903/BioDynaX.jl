using BioDynaX: MechanismRecoveryResult,
    generate_recovery_experiments,
    consume_shared_suite_rng!

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
    @test !isdefined(BioDynaX, :sample_destruction)
    @test !isdefined(BioDynaX, :evaluate_recovery)
    @test !isdefined(BioDynaX, :report_recovery)
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
