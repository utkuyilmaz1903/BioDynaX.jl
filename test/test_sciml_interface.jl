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
