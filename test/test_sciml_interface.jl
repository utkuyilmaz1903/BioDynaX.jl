@testset "README SciML ODE snippet" begin
    using BioDynaX, SciMLBase, OrdinaryDiffEq, Random

    network = BiologicalNetwork(
        [NodeSpec(name = :A), NodeSpec(name = :B)],
        EdgeSpec[];
        reactions = [
            ReactionSpec(name = :b_drives_a,
                         stoichiometry = Dict(1 => 1.0), regulators = [2],
                         metadata = MassActionMetadata(rate_param = :k_ba)),
            ReactionSpec(name = :a_decay,
                         stoichiometry = Dict(1 => -1.0), regulators = Int[],
                         metadata = LinearDecayMetadata(rate_param = :k_a)),
            ReactionSpec(name = :b_decay,
                         stoichiometry = Dict(2 => -1.0), regulators = Int[],
                         metadata = LinearDecayMetadata(rate_param = :k_b)),
        ])
    model, p = build_ude_model(MersenneTwister(0), network)
    prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), p)
    sol = solve(prob, Tsit5(); saveat = 0:0.5:10.0)
    @test SciMLBase.successful_retcode(sol)
    @test size(Array(sol), 1) == 2
end

@testset "SciMLBase ODEProblem contract" begin
    rng = MersenneTwister(17)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    u0 = [0.2, 0.1]
    tspan = (0.0, 1.0)
    times = collect(range(tspan...; length = 10))

    prob_oop = SciMLBase.ODEProblem(model, u0, tspan, params)
    @test prob_oop isa SciMLBase.ODEProblem
    @test !SciMLBase.isinplace(prob_oop.f)

    cache = allocate_cache(model, Float64)
    prob_ip = SciMLBase.ODEProblem(
        model, u0, tspan, params; inplace = true, cache = cache)
    @test SciMLBase.isinplace(prob_ip.f)

    sol_oop = solve(prob_oop, Tsit5(); saveat = times, sensealg = nothing)
    sol_ip = solve(prob_ip, Tsit5(); saveat = times, sensealg = nothing)
    @test SciMLBase.successful_retcode(sol_oop)
    @test SciMLBase.successful_retcode(sol_ip)
    @test Array(sol_oop) ≈ Array(sol_ip)

    remade = SciMLBase.remake(prob_oop; p = params)
    sol_remade = solve(remade, Tsit5(); saveat = times, sensealg = nothing)
    @test Array(sol_remade) ≈ Array(sol_oop)
end

@testset "SciML forward and adjoint contracts" begin
    rng = MersenneTwister(23)
    model, params = build_ude_model(rng, build_linear_test_network())
    u0 = [0.2, 0.1]
    tspan = (0.0, 0.5)
    times = collect(range(tspan...; length = 6))
    synthetic = predict_ude(
        params, u0, tspan, times, model;
        solver_config = SolverConfig(ad_policy = ZygoteAD()))

    loss = p -> loss_mse(
        p, synthetic, times, u0, tspan, model;
        solver_config = SolverConfig(ad_policy = ZygoteAD()))
    grad_z = Zygote.gradient(loss, params)[1]
    @test all(isfinite, grad_z)

    forward_cfg = SolverConfig(ad_policy = ProductionAD(), sensealg = nothing)
    cache = allocate_cache(model, Float64)
    synthetic_prod = predict_ude(
        params, u0, tspan, times, model;
        solver_config = forward_cfg, cache = cache)
    @test synthetic_prod ≈ synthetic
    @test BioDynaX._forward_inplace(forward_cfg)

    adjoint_cfg = default_solver_config(model; ad_policy = ProductionAD())
    @test adjoint_cfg.sensealg isa InterpolatingAdjoint
    @test !BioDynaX._forward_inplace(adjoint_cfg)
end

@testset "build_ude_function and SciML solve" begin
    rng = MersenneTwister(41)
    model, params = build_ude_model(rng)
    u0 = [0.2, 0.1]
    tspan = (0.0, 0.25)
    times = collect(range(tspan...; length = 5))

    f = build_ude_function(model)
    @test f([0.2, 0.1], params, 0.0) ≈ ude_system([0.2, 0.1], params, 0.0, model)

    sol = SciMLBase.solve(model, u0, tspan, params; saveat = times)
    @test all(isfinite, Array(sol))
end

@testset "SciML ODEProblem on remapped multi-head and two-regulator D(S,I)" begin
    remap = remapped_two_regulator_compiled_path()
    @test remap.holds
    @test remap.row.sciml.matches_odeproblem
    @test remap.row.sciml.matches_inplace
    @test remap.row.sciml.matches_remake
    @test remap.row.arch.n_heads == 2
    @test remap.row.arch.arities == [1, 2]
    two = two_regulator_sciml_path()
    @test two.holds
    skipped = skipped_duplicate_sciml_path()
    @test skipped.holds
    linear = build_linear_test_network()
    model, params = build_ude_model(MersenneTwister(0), linear)
    times, clean, _, used = generate_from_compiled_model(
        model, params, MersenneTwister(0);
        u0 = [0.2, 0.1], tspan = (0.0, 0.5), n_points = 5, noise_σ = 0.0)
    prob = SciMLBase.ODEProblem(model, [0.2, 0.1], (0.0, 0.5), used)
    sol = solve(prob, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9)
    @test clean ≈ Array(sol)
end
