@testset "scientific release qualification" begin
    rng = MersenneTwister(2026)
    network = DEFAULT_EXAMPLE_NETWORK
    model, params = build_ude_model(rng, network)
    tspan = (0.0, 5.0)
    times, _, noisy, _ = generate_data(
        rng; tspan = tspan, n_points = 20, noise_σ = 0.02)
    trained = train_ude(
        params, noisy, times, [0.2, 0.1], tspan, model;
        adam_iters = 6, bfgs_iters = 0, log_every = 100, verbose = false)
    @test trained.converged || trained.final_loss ≤ trained.initial_loss

    checkpoint_path = joinpath(tempdir(), "biodynax_release_ckpt.bin")
    metadata = (
        run = RunMetadata(seed = 2026,
                          data_hash = data_fingerprint(noisy, times, [0.2, 0.1]),
                          config = Dict(:phase => :release)),
        dual = zeros(2), rho = 1.0, outer = 1, stage = 1,
        stage_iteration = 0, previous_residual = Inf)
    save_checkpoint(
        checkpoint_path,
        Checkpoint(BioDynaX.CHECKPOINT_SCHEMA_VERSION, trained.params, nothing, 1, metadata))
    resumed = resume_training(
        load_checkpoint(checkpoint_path),
        noisy, times, [0.2, 0.1], tspan, model;
        adam_iters = 2, bfgs_iters = 0, log_every = 100, verbose = false)
    @test resumed.final_loss ≤ trained.final_loss + 1e-6

    discovery = discover_equations(
        trained.params, model; tspan = tspan, n_samples = 40, verbose = false)
    @test discovery isa DiscoveryResult
    rm(checkpoint_path; force = true)
end

@testset "multi-topology E2E" begin
    rng = MersenneTwister(77)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    times, _, noisy, _ = generate_data(
        rng; network = network, u0 = [0.2, 0.1], tspan = (0.0, 2.0),
        n_points = 12, noise_σ = 0.01)
    result = train_ude(
        params, noisy, times, [0.2, 0.1], (0.0, 2.0), model;
        adam_iters = 4, bfgs_iters = 0, verbose = false)
    @test result.final_loss ≤ result.initial_loss
end
