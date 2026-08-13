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
    @test isfinite(trained.initial_loss)
    @test isfinite(trained.final_loss)
    @test trained.final_loss ≤ max(2 * trained.initial_loss,
                                   trained.initial_loss + 1e-3)

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
    @test isfinite(resumed.final_loss)
    @test resumed.final_loss ≤ max(2 * trained.final_loss,
                                   trained.final_loss + 1e-3)

    discovery = discover_equations(
        trained.params, model; tspan = tspan, n_samples = 40, verbose = false)
    @test discovery isa DiscoveryResult
    @test discovery.retcode isa BioDynaX.DiscoveryRetcode
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
    @test isfinite(result.initial_loss)
    @test isfinite(result.final_loss)
    # Four Adam steps are a smoke run, not a convergence proof. Reject only
    # exploding losses; tiny non-monotone steps are numerically expected.
    @test result.final_loss ≤ max(2 * result.initial_loss,
                                  result.initial_loss + 1e-3)
end
