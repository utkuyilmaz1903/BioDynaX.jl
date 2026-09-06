# Deprecated names of 0.12 forward to their replacements with a warning and
# are removed in 0.13.

@testset "deprecated names forward to their replacements" begin
    # A deprecated binding warns through the runtime, not the logging system,
    # so its state is checked directly.
    @test Base.isdeprecated(BioDynaX, :UNIQUE_CLAIM_PROTOCOL)
    @test Base.invokelatest(() -> BioDynaX.UNIQUE_CLAIM_PROTOCOL) ===
          BioDynaX.REFERENCE_PROTOCOL
    @test !Base.isdeprecated(BioDynaX, :REFERENCE_PROTOCOL)
    truth_net = BioDynaX.build_hill_recovery_network(; known = true, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    ics = [[0.25, 0.20], [0.80, 0.35]]
    old = @test_deprecated BioDynaX.unique_claim_experiment_set(
        MersenneTwister(103), truth_net; smoke = true, truth_params = truth,
        initial_conditions = ics)
    new = BioDynaX.reference_protocol_experiment_set(
        MersenneTwister(103), truth_net; smoke = true, truth_params = truth,
        initial_conditions = ics)
    @test old.experiments[1].observations == new.experiments[1].observations
    old_config = @test_deprecated BioDynaX.unique_claim_discovery_config()
    @test old_config.seed == BioDynaX.reference_protocol_discovery_config().seed
end
