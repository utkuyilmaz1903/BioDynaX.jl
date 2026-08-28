using BioDynaX: MechanismRecoveryResult

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
    @test !isdefined(BioDynaX, :generate_recovery_experiments)
    @test !isdefined(BioDynaX, :fit_unknown_destruction)
    @test !isdefined(BioDynaX, :sample_destruction)
    @test !isdefined(BioDynaX, :evaluate_recovery)
    @test !isdefined(BioDynaX, :report_recovery)
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
