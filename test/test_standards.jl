using Test: @inferred

# Isolated SciML / industry-bar tests. Not included from runtests.jl.
# Do not use `@test_broken` here: a failing assertion is the recorded threshold.

const STANDARDS_FORBIDDEN_SOLVERS = (
    "Rodas5", "Rodas4", "KenCarp4", "Rosenbrock23", "TRBDF2", "QNDF")

function standards_missing_docstrings(mod::Module)
    missing = Symbol[]
    for name in names(mod)
        name === nameof(mod) && continue
        doc = Base.Docs.doc(Base.Docs.Binding(mod, name))
        text = strip(repr(doc))
        if isempty(text) || occursin("No documentation found", text)
            push!(missing, name)
        end
    end
    return missing
end

function standards_chain_network(nstates::Int)
    nstates ≥ 1 || throw(ArgumentError("nstates must be ≥ 1"))
    nodes = [NodeSpec(name = Symbol(:S, i)) for i in 1:nstates]
    reactions = ReactionSpec[]
    for i in 1:nstates
        push!(reactions,
            ReactionSpec(
                name = Symbol(:decay, i),
                stoichiometry = Dict(i => -1.0),
                regulators = Int[],
                metadata = LinearDecayMetadata(rate_param = Symbol(:k, i))))
    end
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

function standards_sciml_agreement(model, params, u0;
        tspan = (0.0, 0.4), n_points::Int = 5)
    u = Float64.(u0)
    times = collect(range(first(tspan), last(tspan); length = n_points))
    cfg = SolverConfig(
        algorithm = Tsit5(),
        sensealg = nothing,
        abstol = 1e-10,
        reltol = 1e-10)
    prob = SciMLBase.ODEProblem(model, u, tspan, params)
    @test prob isa SciMLBase.ODEProblem
    @test !SciMLBase.isinplace(prob.f)
    @test prob.f.mass_matrix == LinearAlgebra.I
    sol = solve(prob, Tsit5(); saveat = times, abstol = 1e-10, reltol = 1e-10,
        sensealg = nothing)
    @test SciMLBase.successful_retcode(sol)
    @test sol.retcode == SciMLBase.ReturnCode.Success
    remade = SciMLBase.remake(prob; u0 = u, p = params, tspan = tspan)
    @test remade.f === prob.f
    sol_r = solve(remade, Tsit5(); saveat = times, abstol = 1e-10,
        reltol = 1e-10, sensealg = nothing)
    @test Array(sol_r)≈Array(sol) rtol=1e-10 atol=1e-12
    cache = allocate_cache(model, Float64)
    prob_ip = SciMLBase.ODEProblem(
        model, u, tspan, params; inplace = true, cache = cache)
    @test SciMLBase.isinplace(prob_ip.f)
    sol_ip = solve(prob_ip, Tsit5(); saveat = times, abstol = 1e-10,
        reltol = 1e-10, sensealg = nothing)
    @test Array(sol_ip)≈Array(sol) rtol=1e-10 atol=1e-12
    sol_m = SciMLBase.solve(
        model, u, tspan, params; saveat = times, solver_config = cfg)
    @test Array(sol_m)≈Array(sol) rtol=1e-10 atol=1e-12
    @test length(sol.t) == n_points
    @test length(sol.u[1]) == model.compiled.nstates
    @test cfg.algorithm isa Tsit5
    return sol
end

@testset "public API and recovery locks stay untouched" begin
    @test public_export_list_holds()
    @test issetequal(names(BioDynaX), collect(locked_public_names()))
    @test recovery_thresholds_hold()
    @test recovery_thresholds_lock() == RECOVERY_THRESHOLDS
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test validate_network_stays_open_source()
    src = read(joinpath(pkgdir(BioDynaX), "src", "SciMLInterface.jl"), String)
    for solver in STANDARDS_FORBIDDEN_SOLVERS
        @test !occursin(solver, src)
    end
    @test occursin("function recommend_sensealg", src)
    runtests = read(joinpath(pkgdir(BioDynaX), "test", "runtests.jl"), String)
    @test !occursin("test_standards.jl", runtests)
    @test !occursin("quality.jl", runtests)
end

@testset "every exported name has a docstring" begin
    missing = standards_missing_docstrings(BioDynaX)
    @test isempty(missing)
end

@testset "recommend_sensealg observation and state boundaries" begin
    rng = MersenneTwister(7)
    linear, _ = build_ude_model(rng, build_linear_test_network())
    hill, _ = build_ude_model(rng, build_hill_recovery_network(; known = false))
    eight, _ = build_ude_model(rng, standards_chain_network(8))
    nine, _ = build_ude_model(rng, standards_chain_network(9))

    @test recommend_sensealg(linear; n_observations = 64).name ===
          :backsolve_mechanistic
    @test recommend_sensealg(linear; n_observations = 65).name ===
          :interpolating_default
    @test recommend_sensealg(eight; n_observations = 20).name ===
          :backsolve_mechanistic
    @test recommend_sensealg(nine; n_observations = 20).name ===
          :interpolating_default
    @test recommend_sensealg(hill; n_observations = 1).name ===
          :interpolating_neural
    @test recommend_sensealg(linear; policy = ProductionAD()).name ===
          :interpolating_production
    @test auto_sensealg(linear; n_observations = 64) ==
          recommend_sensealg(linear; n_observations = 64).sensealg

    @test_throws ArgumentError recommend_sensealg(linear; n_observations = 0)
    @test_throws ArgumentError recommend_sensealg(linear; n_observations = -1)
end

@testset "ODEProblem remake solve agreement on extra fixtures" begin
    rng = MersenneTwister(19)
    fixtures = (
        (build_linear_test_network(), [0.22, 0.14]),
        (build_hill_recovery_network(; known = true, hill_order = 2),
            [0.30, 0.20]),
        (build_mm_test_network(), [0.40, 0.25]),
        (build_competitive_test_network(), [0.35, 0.20, 0.15]),
        (build_repressilator_network(), [0.40, 0.30, 0.20]),
        (build_zero_unknown_linear_network(), [0.22, 0.14]),
        (standards_chain_network(8), fill(0.18, 8)),
        (standards_chain_network(9), fill(0.16, 9)))
    for (network, u0) in fixtures
        model, params = build_ude_model(rng, network)
        standards_sciml_agreement(model, params, u0)
    end
end

@testset "SciML eltype and Dual contracts" begin
    rng = MersenneTwister(21)
    model, params = build_ude_model(rng, build_linear_test_network())
    times = collect(0.0:0.1:0.4)

    u32 = Float32[0.2, 0.1]
    tspan32 = (0.0f0, 0.4f0)
    prob32 = SciMLBase.ODEProblem(model, u32, tspan32, params)
    sol32 = solve(prob32, Tsit5(); saveat = Float32.(times), sensealg = nothing)
    @test SciMLBase.successful_retcode(sol32)
    @test eltype(sol32.u[1]) === Float32
    @test eltype(sol32.t) === Float32

    uSA = SA[0.2, 0.1]
    probSA = SciMLBase.ODEProblem(model, uSA, (0.0, 0.4), params)
    solSA = solve(probSA, Tsit5(); saveat = times, sensealg = nothing)
    @test SciMLBase.successful_retcode(solSA)
    @test solSA.u[1] isa SVector{2, Float64}

    tspan_int = (0, 1)
    prob_int = SciMLBase.ODEProblem(model, [0.2, 0.1], tspan_int, params)
    sol_int = solve(prob_int, Tsit5(); saveat = 0:0.25:1, sensealg = nothing)
    @test SciMLBase.successful_retcode(sol_int)
    @test eltype(sol_int.t) <: AbstractFloat

    dual_u = [ForwardDiff.Dual(0.2, 1.0), ForwardDiff.Dual(0.1, 0.0)]
    du = ude_system(dual_u, params, 0.0, model)
    @test length(du) == 2
    @test eltype(du) <: ForwardDiff.Dual
    @test all(isfinite, ForwardDiff.value.(du))
end

@testset "ODEProblem rejects a rank-mismatched u0" begin
    rng = MersenneTwister(23)
    model, params = build_ude_model(rng, build_linear_test_network())
    @test_throws DimensionMismatch begin
        prob = SciMLBase.ODEProblem(model, [0.2], (0.0, 0.2), params)
        solve(prob, Tsit5(); saveat = [0.0, 0.2], sensealg = nothing)
    end
end

@testset "stricter allocation and inference on the public RHS" begin
    rng = MersenneTwister(29)
    network = build_linear_test_network()
    nn, nn_ps, st = build_ude_nn(rng)
    model = compile_network(network, nn, st)
    params = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), nn_ps)
    u = [0.2, 0.1]
    cache = allocate_cache(model, Float64)
    for _ in 1:40
        ude_rhs!(cache.du, u, params, 0.0, model, cache)
    end
    hot_ip = @allocated ude_rhs!(cache.du, u, params, 0.0, model, cache)
    # Existing quality gate allows 512 bytes. This file requires zero.
    @test hot_ip == 0

    f = build_ude_function(model)
    for _ in 1:20
        f(u, params, 0.0)
    end
    hot_oop = @allocated f(u, params, 0.0)
    # One length-2 Vector{Float64} plus no extras.
    @test hot_oop ≤ 96

    uSA = SVector{2, Float64}(0.2, 0.1)
    for _ in 1:20
        ude_system(uSA, params, 0.0, model)
    end
    hot_sa = @allocated ude_system(uSA, params, 0.0, model)
    @test hot_sa == 0
    @test @inferred(ude_system(uSA, params, 0.0, model)) isa SVector{2, Float64}
    @test @inferred(ude_rhs!(
        cache.du, u, params, 0.0, model, cache)) === cache.du
    @test @inferred(recommend_sensealg(model; n_observations = 20)) isa
          BioDynaX.SensealgRecommendation
    @test @inferred(build_ude_function(model)) isa SciMLBase.ODEFunction
end

@testset "compose_hybrid_rhs stays on the SciML problem surface" begin
    rng = MersenneTwister(31)
    model, params = build_ude_model(
        rng, build_hill_recovery_network(; known = false, hill_order = 2))
    term = only(filter(t -> t isa NeuralDestructionTerm,
        model.compiled.destruction_terms))
    rate_fn = r -> 0.25 * r[1] / (0.4 + r[1])
    rhs = compose_hybrid_rhs(model, params, term, rate_fn)
    u = [0.3, 0.2]
    du = rhs(u, nothing, 0.0)
    @test length(du) == 2
    @test all(isfinite, du)
    @test @inferred(rhs(u, nothing, 0.0)) isa Vector{Float64}
    tspan = (0.0, 0.4)
    times = collect(range(tspan...; length = 5))
    prob = SciMLBase.ODEProblem(rhs, u, tspan)
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    @test SciMLBase.successful_retcode(sol)
    @test size(Array(sol), 1) == 2
end
