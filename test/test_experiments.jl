@testset "experiment contracts" begin
    times = collect(range(0.0, 1.0; length = 5))
    observations = [times'; (2 .* times)']
    mask = trues(size(observations))
    mask[2, 3] = false
    experiment = Experiment(
        :replicate_1, times, observations, [0.0, 0.0]; mask)
    set = ExperimentSet(
        [experiment], [:x1, :x2]; units = [:μM, :μM])

    @test length(set) == 1
    @test set[1].mask == mask
    @test length(experiment_fingerprint(set)) == 64
    @test_throws ArgumentError Experiment(
        :bad, reverse(times), observations, [0.0, 0.0])
end

@testset "execution backends and generated replicates" begin
    rng = MersenneTwister(9)
    set = generate_experiment_set(
        rng; initial_conditions = [[0.2, 0.1], [0.3, 0.15]],
        n_points = 5, noise_σ = 0.0, tspan = (0.0, 1.0))
    serial = execute_experiments(
        experiment -> sum(experiment.observations), set;
        config = ExecutionConfig(backend = :serial))
    threaded = execute_experiments(
        experiment -> sum(experiment.observations), set;
        config = ExecutionConfig(backend = :threads))
    @test length(set) == 2
    @test length(experiment_batches(set, 1)) == 2
    @test length(experiment_batches(set, 2)) == 1
    @test serial == threaded
    @test gpu_available() isa Bool
end

@testset "versioned checkpoint" begin
    path = tempname()
    checkpoint = Checkpoint(
        BioDynaX.CHECKPOINT_SCHEMA_VERSION, [1.0, 2.0], nothing, 10,
        RunMetadata(seed = 42))
    save_checkpoint(path, checkpoint)
    restored = load_checkpoint(path)
    @test restored.iteration == 10
    @test restored.params == [1.0, 2.0]
    rm(path; force = true)

    result_path = tempname()
    result = TrainingResult(
        [1.0], [2.0], 2.0, 1.0, RunMetadata(seed = 1),
        (mse = 1.0,), true, :success)
    save_result(result_path, result)
    @test load_result(result_path).retcode == :success
    rm(result_path; force = true)
end
