@testset "phase 3 optimization problem" begin
    rng = MersenneTwister(3)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    times, _, noisy, _ = generate_data(
        rng; network = network, u0 = [0.2, 0.1], tspan = (0.0, 1.0),
        n_points = 8, noise_σ = 0.01)
    config = TrainingConfig(
        adam_iterations = 4, bfgs_iterations = 0,
        solver = SolverConfig(abstol = 1e-5, reltol = 1e-5),
        horizon_schedule = [1.0])
    prob, objective = build_optimization_problem(
        model, params, noisy, times, [0.2, 0.1], (0.0, 1.0); config = config)
    @test prob isa Optimization.OptimizationProblem
    @test isfinite(objective(params, nothing))
    opt = train_via_optimization(
        model, params, noisy, times, [0.2, 0.1], (0.0, 1.0);
        config = config, maxiters = 8, verbose = false)
    @test opt isa TrainingResult
    @test isfinite(opt.final_loss)
    @test opt.final_loss ≤ opt.initial_loss + 1e-8
    @test opt.diagnostics.bfgs.attempted
end

@testset "phase 3 sensealg recommendations" begin
    rng = MersenneTwister(5)
    linear_model, _ = build_ude_model(rng, build_linear_test_network())
    p53_model, _ = build_ude_model(rng, build_network())
    mech = recommend_sensealg(linear_model; n_observations = 20)
    neural = recommend_sensealg(p53_model; n_observations = 20)
    @test mech.name == :backsolve_mechanistic
    @test neural.name == :interpolating_neural
    @test occursin("BacksolveAdjoint", string(typeof(mech.sensealg)))
    @test occursin("InterpolatingAdjoint", string(typeof(neural.sensealg)))
end

@testset "phase 3 horizon curriculum" begin
    rng = MersenneTwister(9)
    fixture = build_linear_test_network()
    model, params = build_ude_model(rng, fixture)
    times, _, noisy, _ = generate_data(
        rng; network = fixture, u0 = [0.2, 0.1], tspan = (0.0, 0.8),
        n_points = 10, noise_σ = 0.01)
    curriculum = HorizonCurriculum(
        fractions = [0.3, 0.6, 1.0], min_points = 3, minimum_fraction = 0.2)
    config = TrainingConfig(
        adam_iterations = 6, bfgs_iterations = 0, log_every = 100,
        horizon_schedule = curriculum,
        solver = SolverConfig(abstol = 1e-5, reltol = 1e-5))
    result = train_ude(
        params, noisy, times, [0.2, 0.1], (0.0, 0.8), model;
        config = config, verbose = false)
    @test isfinite(result.final_loss)
end

@testset "phase 3 weighted experiments" begin
    rng = MersenneTwister(11)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    times1, _, noisy1, _ = generate_data(
        rng; network = network, u0 = [0.2, 0.1], tspan = (0.0, 1.0),
        n_points = 6, noise_σ = 0.01)
    times2, _, noisy2, _ = generate_data(
        rng; network = network, u0 = [0.3, 0.15], tspan = (0.0, 1.0),
        n_points = 6, noise_σ = 0.05)
    exp1 = Experiment(:a, times1, noisy1, [0.2, 0.1];
        metadata = Dict(:weight => 2.0, :noise_σ => 0.01))
    exp2 = Experiment(:b, times2, noisy2, [0.3, 0.15];
        metadata = Dict(:weight => 1.0, :noise_σ => 0.05))
    set = ExperimentSet([exp1, exp2], [:A, :B])
    @test experiment_weight(exp1) == 2.0
    @test experiment_noise_scale(exp2) == 0.05
    result = train_experiments(
        params, set, model;
        config = TrainingConfig(
            adam_iterations = 4, bfgs_iterations = 0, log_every = 100,
            solver = SolverConfig(abstol = 1e-5, reltol = 1e-5),
            horizon_schedule = [1.0]),
        execution = ExecutionConfig(batch_size = 2, deterministic = true),
        verbose = false)
    @test isfinite(result.final_loss)
end

@testset "phase 3 nn presets" begin
    rng = MersenneTwister(13)
    for preset in (:small, :medium, :large)
        nn, ps, st = build_ude_nn(rng; preset = preset)
        out, _ = nn([0.5], ps, st)
        @test out[1] > 0
        @test eltype(ps.layer_1.weight) === Float64
    end
end
