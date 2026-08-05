@testset "positive production-destruction UDE" begin
    rng = MersenneTwister(12)
    network = build_network()
    nn, nn_parameters, nn_state = build_ude_nn(rng)
    parameters = pack_parameters(
        (α_p53 = 0.9, β_mdm2 = 1.1, γ_mdm2 = 1.5, signal = 1.0),
        nn_parameters)

    derivative = ude_system([0.0, 0.0], parameters, 0.0, nn, nn_state)
    @test all(isfinite, derivative)
    @test all(≥(0), derivative)

    negative_probe = ude_system(
        [-1e-6, -1e-6], parameters, 0.0, nn, nn_state)
    @test all(isfinite, negative_probe)
    @test all(≥(0), negative_probe)

    model = compile_network(network, nn, nn_state)
    @test ude_system([0.2, 0.1], parameters, 0.0, model) ≈
          ude_system([0.2, 0.1], parameters, 0.0, nn, nn_state)

    objective = p -> sum(abs2,
        ude_system([0.2, 0.1], p, 0.0, nn, nn_state))
    gradient = Zygote.gradient(objective, parameters)[1]
    @test all(isfinite, gradient)
    @test any(!iszero, gradient)

    times = collect(range(0.0, 0.5; length = 6))
    synthetic = predict_ude(
        parameters, [0.2, 0.1], (0.0, 0.5), times, nn, nn_state)
    trajectory_loss = p -> loss_mse(
        p, synthetic, times, [0.2, 0.1], (0.0, 0.5), nn, nn_state)
    adjoint_gradient = Zygote.gradient(trajectory_loss, parameters)[1]
    @test all(isfinite, adjoint_gradient)

    strategy = AugmentedLagrangianConfig(smoothness = 1e-2)
    constraints = BioDynaX._constraint_values(
        [-0.2 0.1; 0.5 1.0], strategy)
    @test constraints[1] > 0
    @test constraints[2] < 0
    @test constraints[1] > constraints[2]
end
