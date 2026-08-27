using SciMLSensitivity: BacksolveAdjoint, ZygoteVJP

@testset "training reuse helpers are not exported" begin
    @test !(:TrainingSolveSession in names(BioDynaX))
    @test !(:lock_training_solver in names(BioDynaX))
    @test !(:predict_ude_session in names(BioDynaX))
    @test !(:warmup_first_experiment in names(BioDynaX))
    @test !(:with_compile_network_counter in names(BioDynaX))
    @test !(:train_experiments_with_warmup in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "recommend_sensealg honesty matrix" begin
    matrix = BioDynaX.training_sensealg_honesty_matrix()
    @test matrix.holds
    @test matrix.linear.neural == false
    @test matrix.hill.neural
    @test matrix.hill.zygote_kind === :interpolating
    @test matrix.remap.neural
    @test matrix.two.neural
    @test matrix.hill.backsolve_forbidden_for_neural
    locked = BioDynaX.lock_training_solver(build_ude_model(
        MersenneTwister(0), build_hill_recovery_network(; known = false))[1])
    @test BioDynaX.training_sensealg_kind(locked) === :interpolating
end

@testset "session remake matches predict_ude without compiling" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    report = BioDynaX.training_session_remake_agreement(
        model, params, [0.22, 0.14]; tspan = (0.0, 1.0), n_points = 8)
    @test report.holds
    @test report.matches
    @test report.no_compile
    @test report.remake_count ≥ 2
end

@testset "multi-IC session remakes and does not compile" begin
    rng = MersenneTwister(11)
    net = build_linear_test_network()
    model, params = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(11); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18], [0.18, 0.12]],
        tspan = (0.0, 1.0), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    report = BioDynaX.training_session_multi_ic_agreement(model, params, set)
    @test report.holds
    @test report.matches
    @test report.counter == 0
    @test report.remake_count == 3
end

@testset "train_experiments with a compiled model does not compile per IC" begin
    rng = MersenneTwister(13)
    net = build_linear_test_network()
    model, p0 = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(13); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        tspan = (0.0, 0.8), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    report = BioDynaX.train_experiments_compile_report(init, set, model)
    @test report.holds
    @test report.with_model == 0
    @test report.with_nn_st == 1
    @test report.n_ics == 2
end

@testset "Augmented-Lagrangian path still does not compile per IC" begin
    rng = MersenneTwister(17)
    net = build_linear_test_network()
    model, p0 = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(17); network = net,
        initial_conditions = [[0.22, 0.14], [0.28, 0.16]],
        tspan = (0.0, 0.6), n_points = 5, noise_σ = 0.0,
        truth_params = truth)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    cfg = TrainingConfig(
        adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6,
        constraint = AugmentedLagrangianConfig(outer_iterations = 1))
    n = BioDynaX.with_compile_network_counter() do counter
        train_experiments(init, set, model; config = cfg, verbose = false)
        counter[]
    end
    @test n == 0
    @test BioDynaX.al_constraint_passes_model_source()
end

@testset "warmup exposes Optimisers state for train_experiments" begin
    rng = MersenneTwister(19)
    net = build_linear_test_network()
    model, p0 = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(19); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        tspan = (0.0, 0.6), n_points = 5, noise_σ = 0.0,
        truth_params = truth)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    report = BioDynaX.warmup_state_reuse_report(init, set, model; adam_iterations = 2)
    @test report.holds
    @test report.warmup_has_state
    @test report.sensealg_locked
    warm = BioDynaX.warmup_first_experiment(
        init, set, model;
        config = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0,
            log_every = 10^6), verbose = false)
    @test BioDynaX.optimizer_state_from_result(warm.result) !== nothing
end

@testset "neural unique-claim warmup does not compile" begin
    path = BioDynaX.unique_claim_warmup_compile_path(; smoke = true)
    @test path.holds
    @test path.compiles == 0
    @test path.sensealg.neural
    @test path.compiled_once
end

@testset "two-regulator and remapped sessions remake without compile" begin
    rng = MersenneTwister(23)
    two = build_two_regulator_unknown_network()
    model, params = build_ude_model(rng, two)
    packed = pack_parameters((k_es = 0.8, k_i = 0.5, k_e = 0.4), params.nn)
    report = BioDynaX.training_session_remake_agreement(
        model, packed, [0.25, 0.20, 0.15]; tspan = (0.0, 0.5), n_points = 6)
    @test report.holds
    remap = build_remapped_two_regulator_network()
    rmodel, rparams = build_ude_model(MersenneTwister(13), remap)
    rpacked = pack_parameters(remapped_two_regulator_phys_truth(), rparams.nn)
    rreport = BioDynaX.training_session_remake_agreement(
        rmodel, rpacked, remapped_two_regulator_state();
        tspan = (0.0, 0.5), n_points = 6)
    @test rreport.holds
    @test BioDynaX.neural_training_requires_interpolating(rmodel)
end

@testset "ProductionAD inplace forward stays unlocked as sensealg=nothing" begin
    rng = MersenneTwister(29)
    model, _ = build_ude_model(rng, build_linear_test_network())
    solver = SolverConfig(ad_policy = ProductionAD(), sensealg = nothing)
    locked = BioDynaX.lock_training_solver(model, solver)
    @test locked.sensealg === nothing
    @test BioDynaX.training_sensealg_is_locked(model, locked)
    @test BioDynaX._forward_inplace(locked)
end

@testset "BacksolveAdjoint is rejected on a neural hole" begin
    rng = MersenneTwister(31)
    model, _ = build_ude_model(
        rng, build_hill_recovery_network(; known = false, hill_order = 2))
    bad = SolverConfig(
        ad_policy = ZygoteAD(),
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()))
    @test BioDynaX.training_sensealg_is_locked(model, bad) == false
    @test_throws ErrorException BioDynaX.assert_training_sensealg(model, bad)
    good = BioDynaX.lock_training_solver(model, bad)
    @test BioDynaX.training_sensealg_kind(good) === :interpolating
    @test BioDynaX.assert_training_sensealg(model, good) === good
end

@testset "training reuse contract and docs hold" begin
    @test BioDynaX.training_reuse_source_holds()
    @test BioDynaX.al_constraint_passes_model_source()
    @test BioDynaX.train_experiments_accepts_optimizer_state_source()
    @test BioDynaX.train_unknown_edge_reuses_warmup_source()
    @test BioDynaX.training_reuse_docs_hold()
    @test BioDynaX.training_reuse_landing_docs_hold()
    @test BioDynaX.training_reuse_contract_holds()
    violations = BioDynaX.training_reuse_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
end

@testset "session remake matches generate_from_compiled_model" begin
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, build_linear_test_network())
    report = BioDynaX.training_session_matches_generate(
        model, params, [0.22, 0.14]; tspan = (0.0, 0.8), n_points = 8)
    @test report.holds
    @test report.matches_generate
    @test report.no_compile
end

@testset "train_ude with a compiled model does not compile" begin
    fixture = BioDynaX.linear_training_fixture()
    report = BioDynaX.train_ude_compile_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    @test report.holds
    @test report.with_model == 0
    @test report.with_nn_st == 1
end

@testset "frozen_phys survives warmup reuse" begin
    fixture = BioDynaX.linear_training_fixture()
    report = BioDynaX.frozen_phys_warmup_report(
        fixture.init, fixture.set, fixture.model)
    @test report.holds
    @test report.frozen_held
end

@testset "masked experiments still do not compile per IC" begin
    fixture = BioDynaX.linear_training_fixture()
    report = BioDynaX.masked_experiment_compile_report(
        fixture.init, fixture.set, fixture.model)
    @test report.holds
    @test report.compiles == 0
end

@testset "sensealg n_obs honesty: lock follows 100 observations" begin
    matrix = BioDynaX.sensealg_nobs_honesty_matrix()
    @test matrix.holds
    @test matrix.linear.small_name === :backsolve_mechanistic
    @test matrix.linear.large_name === :interpolating_default
    @test matrix.hill.small_name === :interpolating_neural
end

@testset "six-state, MM-unknown, and competitive sessions remake" begin
    six = BioDynaX.six_state_session_path()
    @test six.holds
    @test six.nstates == 6
    mm = BioDynaX.mm_unknown_session_path()
    @test mm.holds
    competitive = BioDynaX.competitive_session_path()
    @test competitive.holds
    @test competitive.n_heads == 0
end

@testset "horizon curriculum and optimizer-state roundtrip do not compile" begin
    fixture = BioDynaX.linear_training_fixture()
    horizon = BioDynaX.horizon_curriculum_session_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    @test horizon.holds
    roundtrip = BioDynaX.optimizer_state_roundtrip_report(
        fixture.init, fixture.set, fixture.model)
    @test roundtrip.holds
    resume = BioDynaX.resume_from_diagnostics_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    @test resume.holds
    @test resume.compiles == 0
end

@testset "extended training-reuse matrix holds" begin
    matrix = BioDynaX.training_reuse_extended_matrix()
    @test matrix.holds
end

@testset "reuse path does not loosen locked claim numbers" begin
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test recovery_thresholds_lock() == RECOVERY_THRESHOLDS
    @test issetequal(names(BioDynaX), collect(locked_public_names()))
    zero = build_zero_unknown_linear_network()
    @test validate_network(zero) === zero
end
