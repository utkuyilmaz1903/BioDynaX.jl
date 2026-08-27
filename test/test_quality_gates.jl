using Test: @inferred

@testset "type stability and allocation gates" begin
    rng = MersenneTwister(77)
    linear_network = build_linear_test_network()
    nn, nn_ps, st = build_ude_nn(rng)
    linear_model = compile_network(linear_network, nn, st)
    linear_parameters = pack_parameters(
        (k_ba = 0.8, k_a = 1.2, k_b = 0.5), nn_ps)
    u = [0.2, 0.1]
    linear_cache = allocate_cache(linear_model, eltype(u))

    @test @inferred(ude_rhs!(
        linear_cache.du, u, linear_parameters, 0.0, linear_model, linear_cache)) ===
        linear_cache.du
    @test all(isfinite, linear_cache.du)

    for _ in 1:50
        ude_rhs!(linear_cache.du, u, linear_parameters, 0.0, linear_model, linear_cache)
    end
    warm = @allocated ude_rhs!(
        linear_cache.du, u, linear_parameters, 0.0, linear_model, linear_cache)
    hot = @allocated ude_rhs!(
        linear_cache.du, u, linear_parameters, 0.0, linear_model, linear_cache)
    @test hot ≤ 512
    @test warm ≥ hot

    p53_network = build_network()
    p53_model, _ = build_ude_model(rng, p53_network)
    p53_parameters = default_parameters(p53_network, p53_model)
    p53_cache = allocate_cache(p53_model, eltype(u))
    for _ in 1:20
        ude_rhs!(p53_cache.du, u, p53_parameters, 0.0, p53_model, p53_cache)
    end
    p53_hot = @allocated ude_rhs!(
        p53_cache.du, u, p53_parameters, 0.0, p53_model, p53_cache)
    @test p53_hot ≤ 4096

    out = ude_system(u, p53_parameters, 0.0, p53_model)
    @test out isa Vector{Float64}
    @test out ≈ Vector(p53_cache.du)
end

@testset "unexported workspace allocation gates" begin
    @test BioDynaX.allocation_gates_contract_holds()
    pack = BioDynaX.pack_parameters_allocation_row()
    @test pack.holds
    @test pack.typed.hot ≤ BioDynaX.ALLOCATION_GATE_LIMITS.pack_parameters
    pos = BioDynaX.positive_parameter_allocation_row()
    @test pos.holds
    @test pos.typed.hot == 0
    stlsq = BioDynaX.stlsq_workspace_reuse_row()
    @test stlsq.holds
    @test stlsq.same_shape
    @test BioDynaX.schema_type_row().holds
end
