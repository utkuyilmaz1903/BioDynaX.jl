@testset "generate_data source uses compiled NN tree" begin
    @test datagen_compiled_source_holds()
    violations = datagen_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    @test generate_data_uses_stored_ground_truth_model()
    @test generate_from_compiled_model_uses_sciml_odeproblem()
    @test generate_experiment_set_uses_compiled_once()
    @test datagen_experiment_set_source_holds()
    src = read(datagen_source_path(), String)
    @test occursin("function generate_from_compiled_model", src)
    @test occursin("function compile_ground_truth_model", src)
    @test occursin("SciMLBase.ODEProblem(model", src)
    @test occursin("build_ude_model", src)
    @test !occursin("Lux.Dense(1 => 1", src)
    @test !(:generate_from_compiled_model in names(BioDynaX))
    @test !(:unique_claim_experiment_set in names(BioDynaX))
    @test !(:build_remapped_two_regulator_network in names(BioDynaX))
    @test !(:compile_ground_truth_model in names(BioDynaX))
    @test !(:generate_experiment_set_from_compiled_model in names(BioDynaX))
end

@testset "GroundTruthModel integrates the stored compiled model" begin
    rng = MersenneTwister(41)
    net = build_linear_test_network()
    model, params = build_ude_model(rng, net)
    truth = GroundTruthModel(net, model, params, :compiled_mechanism)
    u0 = [0.22, 0.14]
    tspan = (0.0, 1.0)
    times, clean, _, used = generate_data(
        truth, MersenneTwister(2); u0, tspan, n_points = 8, noise_σ = 0.0)
    @test used === params
    @test length(times) == 8
    @test size(clean) == (2, 8)
    @test all(isfinite, clean)
    times2, clean2, _, used2 = generate_from_compiled_model(
        model, params, MersenneTwister(3); u0, tspan, n_points = 8, noise_σ = 0.0)
    @test used2 === params
    @test clean ≈ clean2
    prob = ODEProblem(model, u0, tspan, params)
    sol = Array(solve(prob, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9))
    @test clean ≈ sol
    @test_throws ArgumentError generate_from_compiled_model(
        model, params, MersenneTwister(4);
        u0 = [0.2], tspan, n_points = 8, noise_σ = 0.0)
end

@testset "remapped skipped duplicate generates with custom truth_params" begin
    net = build_skipped_duplicate_unknown_network()
    @test validate_network(net) === net
    snap = generate_compiled_snapshot(
        MersenneTwister(13), net;
        u0 = [0.2, 0.3, 0.4], truth_params = (k_ca = 0.8, k_b = 0.5, k_c = 0.4))
    @test snap.finite
    @test snap.matches_solve
    @test snap.matches_stored_truth
    @test snap.arch.n_heads == 2
    @test snap.arch.dense
    @test snap.arch.matches
    @test snap.model.nn isa MultiHeadNetwork
    @test packed_nn_head_count(snap.params) == 2
    @test unique_claim_recovery_admits(net) == false
    named = generate_data_namedtuple_snapshot(
        MersenneTwister(17), net;
        u0 = [0.2, 0.3, 0.4],
        truth_params = (k_ca = 0.8, k_b = 0.5, k_c = 0.4))
    @test named.finite
    @test named.nstates == 3
    @test hasproperty(named.packed.nn, :head_1)
    @test hasproperty(named.packed.nn, :head_2)
    set_snap = generate_experiment_set_snapshot(
        MersenneTwister(21), net;
        initial_conditions = [[0.2, 0.3, 0.4], [0.15, 0.25, 0.35]],
        truth_params = (k_ca = 0.8, k_b = 0.5, k_c = 0.4))
    @test set_snap.n_experiments == 2
    @test set_snap.finite
    @test set_snap.n_points == [8, 8]
end

@testset "two-regulator generate_data sizes the NN input to 2" begin
    net = build_two_regulator_unknown_network()
    snap = generate_compiled_snapshot(
        MersenneTwister(19), net;
        u0 = [0.25, 0.20, 0.15],
        truth_params = (k_es = 0.8, k_i = 0.5, k_e = 0.4))
    @test snap.arch.n_heads == 1
    @test snap.arch.arities == [2]
    @test snap.arch.packed_dims == [2]
    @test snap.finite
    @test snap.matches_solve
    @test unique_claim_recovery_admits(net)
    named = generate_data_namedtuple_snapshot(
        MersenneTwister(20), net;
        u0 = [0.25, 0.20, 0.15],
        truth_params = (k_es = 0.8, k_i = 0.5, k_e = 0.4))
    @test packed_nn_input_dim(named.packed.nn) == 2
    @test named.arch.matches
end

@testset "remapped multi-head and two-regulator D(S,I) generate together" begin
    net = build_remapped_two_regulator_network()
    @test validate_network(net) === net
    compiled = compile_mechanism(net)
    @test neural_head_count(compiled) == 2
    @test neural_index_is_dense(compiled)
    @test neural_regulator_arities(compiled) == [1, 2]
    @test unique_claim_recovery_admits(net) == false
    @test unique_claim_compiler_stays_open(net)
    @test_throws ErrorException assert_unique_claim_recovery_network(net)
    @test remapped_two_regulator_contract_holds()
    row = joint_datagen_compiler_row(
        net; rng = MersenneTwister(13),
        u0 = remapped_two_regulator_state(),
        truth_params = remapped_two_regulator_phys_truth())
    @test row.joint_holds
    @test row.arities == [1, 2]
    @test row.packed_dims == [1, 2]
    @test row.snap.model.nn isa MultiHeadNetwork
    @test length(row.snap.model.nn.heads) == 2
    @test packed_nn_head_input_dims(row.snap.params) == [1, 2]
    @test packed_nn_head_input_dims(row.named.packed) == [1, 2]
    @test row.default_matches
    @test row.default_finite
    @test row.recovery_admits == false
    @test row.validate_open
    @test row.set.finite
    @test size(row.snap.cache.nn_inputs, 1) >= 2
    @test size(row.snap.cache.nn_inputs, 2) == 2
end

@testset "unique_claim_experiment_set reads the fingerprint" begin
    rng = MersenneTwister(103)
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    smoke = unique_claim_experiment_set(
        rng, net; smoke = true, truth_params = truth)
    @test unique_claim_experiment_set_matches_fingerprint(smoke; smoke = true)
    @test length(smoke.experiments) == 1
    @test size(first(smoke.experiments).observations, 2) == 8
    @test all(isfinite, first(smoke.experiments).observations)
    @test unique_claim_example_uses_experiment_set()
    two = build_two_regulator_unknown_network()
    @test_throws ArgumentError unique_claim_experiment_set(
        MersenneTwister(1), two; smoke = true)
    custom = unique_claim_experiment_set(
        MersenneTwister(2), two; smoke = true,
        initial_conditions = [[0.25, 0.20, 0.15]],
        truth_params = (k_es = 0.8, k_i = 0.5, k_e = 0.4))
    @test length(custom.experiments) == 1
    @test size(first(custom.experiments).observations, 1) == 3
    @test size(first(custom.experiments).observations, 2) == 8
end

@testset "default_parameters match remapped and two-regulator heads" begin
    rng = MersenneTwister(31)
    skipped = build_skipped_duplicate_unknown_network()
    two = build_two_regulator_unknown_network()
    joint = build_remapped_two_regulator_network()
    for net in (skipped, two, joint)
        model, _ = build_ude_model(rng, net)
        params = default_parameters(model; rng = MersenneTwister(32))
        @test default_parameters_match_compiled(model, params)
        @test all(
            isfinite, ude_system(
                fill(0.2, model.compiled.nstates), params, 0.0, model))
    end
    dual = build_dual_unknown_network()
    dual_model, _ = build_ude_model(rng, dual)
    dual_p = default_parameters(dual_model; rng = MersenneTwister(33))
    @test packed_nn_head_count(dual_p) == 2
    @test default_parameters_match_compiled(dual_model, dual_p)
end
