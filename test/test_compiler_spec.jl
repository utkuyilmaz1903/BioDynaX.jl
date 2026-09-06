@testset "compile_mechanism remaps kept nn_index in source" begin
    @test compile_mechanism_reindexes_source()
    violations = compile_mechanism_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    @test occursin("_reindex_neural_destruction!",
        read(compile_mechanism_source_path(), String))
    @test !occursin("assert_single_unknown_destruction",
        read(compile_mechanism_source_path(), String))
    @test !(:assert_dense_neural_index in names(BioDynaX))
    @test !(:build_skipped_duplicate_unknown_network in names(BioDynaX))
    @test !(:build_two_regulator_unknown_network in names(BioDynaX))
end

@testset "skipped duplicate unknown edge keeps dense heads" begin
    @test default_example_has_duplicate_unknown_declaration()
    default_compiled = compile_mechanism(DEFAULT_EXAMPLE_NETWORK)
    @test neural_head_count(default_compiled) == 1
    @test neural_index_is_dense(default_compiled)
    @test assert_dense_neural_index(default_compiled) === default_compiled

    snap = compile_unknown_topology(
        build_skipped_duplicate_unknown_network(); rng = MersenneTwister(13),
        x = [0.2, 0.3, 0.4])
    @test snap.validate_open
    @test snap.n_heads == 2
    @test snap.indices == [1, 2]
    @test neural_index_is_dense(snap.compiled)
    @test snap.rhs.finite
    @test snap.rhs.parity
    @test snap.rhs.dense
    @test snap.rhs.cache_matches
    @test snap.rhs.multihead_matches
    @test snap.schema.nn_heads == 2
    @test snap.model.nn isa MultiHeadNetwork
    @test length(snap.model.nn.heads) == 2
    @test reference_protocol_recovery_admits(snap.network) == false
    @test reference_protocol_compiler_stays_open(snap.network)
    @test_throws ErrorException assert_reference_protocol_recovery_network(snap.network)
end

@testset "skipped middle unknown keeps slots 1:2 not 1 and 3" begin
    snap = compile_unknown_topology(
        build_skipped_middle_unknown_network(); rng = MersenneTwister(17),
        x = [0.2, 0.3, 0.4, 0.5])
    @test snap.n_heads == 3
    @test snap.indices == [1, 2, 3]
    @test neural_nn_indices(snap.compiled) != [1, 3, 4]
    @test maximum(neural_nn_indices(snap.compiled)) == snap.n_heads
    @test snap.rhs.finite
    @test snap.rhs.parity
    @test snap.rhs.cache_matches
    cache = allocate_cache(snap.model, Float64)
    @test size(cache.nn_inputs, 2) == 3
    @test neural_cache_matches_heads(snap.model, cache)
    @test reference_protocol_recovery_admits(snap.network) == false
    @test count_unknown_destructions(snap.network) == 3
end

@testset "two-regulator unknown compiles without custom truth_params" begin
    net = build_two_regulator_unknown_network()
    @test validate_network(net) === net
    compiled = compile_mechanism(net)
    terms = neural_destruction_terms(compiled)
    @test length(terms) == 1
    @test length(only(terms).regulators) == 2
    @test neural_regulator_arities(compiled) == [2]
    @test neural_index_is_dense(compiled)

    rng = MersenneTwister(19)
    model, params = build_ude_model(rng, net)
    @test neural_head_count(model) == 1
    @test !(model.nn isa MultiHeadNetwork)
    cache = allocate_cache(model, Float64)
    @test size(cache.nn_inputs, 1) >= 2
    @test neural_cache_matches_heads(model, cache)
    rhs = evaluate_compiled_rhs(model, params, [0.3, 0.4, 0.2])
    @test rhs.finite
    @test rhs.parity

    times, clean, noisy, _ = generate_data(
        rng; network = net, u0 = [0.25, 0.20, 0.15],
        tspan = (0.0, 1.0), n_points = 8, noise_σ = 0.0)
    @test length(times) == 8
    @test size(clean, 1) == 3
    @test all(isfinite, clean)
    @test all(isfinite, noisy)
    set = generate_experiment_set(
        rng; network = net, initial_conditions = [[0.25, 0.20, 0.15]],
        tspan = (0.0, 1.0), n_points = 8, noise_σ = 0.0)
    @test length(set.experiments) == 1
    @test all(isfinite, first(set.experiments).observations)
    @test reference_protocol_recovery_admits(net)
    @test assert_reference_protocol_recovery_network(net) === net
end

@testset "dual-unknown and zero-hole stay legal at compile" begin
    rng = MersenneTwister(0)
    zero_net = build_zero_unknown_linear_network()
    two_net = build_dual_unknown_network()
    zero_adm = reference_protocol_recovery_admission(zero_net)
    two_adm = reference_protocol_recovery_admission(two_net)
    @test zero_adm.unknown_holes == 0
    @test two_adm.unknown_holes == 2
    @test zero_adm.validate_open
    @test two_adm.validate_open
    @test zero_adm.recovery_admits == false
    @test two_adm.recovery_admits == false
    @test zero_adm.single_hole_in_validate_network == false
    @test two_adm.single_hole_in_validate_network == false
    @test reference_protocol_compiler_stays_open(zero_net)
    @test reference_protocol_compiler_stays_open(two_net)
    zero_model, zero_p = build_ude_model(rng, zero_net)
    two_model, two_p = build_ude_model(rng, two_net)
    @test neural_index_is_dense(zero_model)
    @test neural_index_is_dense(two_model)
    @test evaluate_compiled_rhs(zero_model, zero_p, [0.2, 0.1]).finite
    @test evaluate_compiled_rhs(two_model, two_p, [0.2, 0.3, 0.4]).finite
    @test_throws ErrorException assert_reference_protocol_recovery_network(zero_net)
    @test_throws ErrorException assert_reference_protocol_recovery_network(two_net)
    gapped = BioDynaX.CompiledMechanism(
        1, [1], Dict(1 => 1),
        (BioDynaX.InputProductionTerm(1, :k, :s, 1.0),),
        (NeuralDestructionTerm(1, 1, 3, 1.0, [1]),))
    @test neural_index_is_dense(gapped) == false
    @test_throws ErrorException assert_dense_neural_index(gapped)
end

@testset "gapped index would miss cache columns" begin
    rng = MersenneTwister(13)
    model, params = build_ude_model(rng, build_skipped_duplicate_unknown_network())
    cache = allocate_cache(model, Float64)
    @test size(cache.nn_inputs, 2) == neural_head_count(model)
    @test maximum(neural_nn_indices(model)) == size(cache.nn_inputs, 2)
    gapped = NeuralDestructionTerm(1, 1, 3, 1.0, [1])
    @test gapped.nn_index > size(cache.nn_inputs, 2)
    @test neural_cache_matches_heads(model, cache)
    dx = ude_system([0.2, 0.3, 0.4], params, 0.0, model)
    ude_rhs!(cache.du, [0.2, 0.3, 0.4], params, 0.0, model, cache)
    @test Vector(cache.du) ≈ dx
end
