@testset "known-term IR matches export contract" begin
    linear = compile_mechanism(build_linear_test_network())
    @test any(t -> t isa BioDynaX.MassActionProductionTerm && t.param === :k_ba,
              linear.production_terms)
    @test any(t -> t isa BioDynaX.LinearDestructionTerm && t.param === :k_a,
              linear.destruction_terms)
    rng = MersenneTwister(0)
    model, p0 = build_ude_model(rng, build_linear_test_network())
    p = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    x = [0.3, 0.4]
    dx = ude_system(x, p, 0.0, model)
    k_ba = positive_parameter(p.phys.k_ba)
    k_a = positive_parameter(p.phys.k_a)
    k_b = positive_parameter(p.phys.k_b)
    @test dx[1] ≈ k_ba * x[2] - k_a * x[1]
    @test dx[2] ≈ -k_b * x[2]
    hill = compile_mechanism(build_hill_recovery_network(; known = true, hill_order = 2))
    @test any(t -> t isa BioDynaX.HillDestructionTerm && t.hill_order == 2,
              hill.destruction_terms)
    unknown = compile_mechanism(build_hill_recovery_network(; known = false))
    @test any(t -> t isa BioDynaX.NeuralDestructionTerm, unknown.destruction_terms)
    competitive = compile_mechanism(build_competitive_test_network())
    @test any(t -> t isa BioDynaX.CompetitiveDestructionTerm,
              competitive.destruction_terms)
end

@testset "known linear parameter recovery" begin
    rng = MersenneTwister(101)
    report = run_recovery_suite(rng;
        linear_adam = 30, linear_bfgs = 15,
        sections = (:linear,))
    @test report[:linear].rmse < 0.25
    @test all(<(0.4), values(report[:linear].rel))
    @test isfinite(report[:linear].final_loss)
end

@testset "known Michaelis–Menten parameter recovery" begin
    rng = MersenneTwister(102)
    report = run_recovery_suite(rng;
        mm_adam = 40, mm_bfgs = 20,
        sections = (:mm,))
    @test report[:mm].rmse < 0.45
    @test isfinite(report[:mm].final_loss)
    mm_model, _ = build_ude_model(MersenneTwister(1), build_mm_test_network())
    names = parameter_schema(mm_model).phys_names
    @test :vmax in names
    @test :km in names
end

@testset "known Hill parameter recovery" begin
    rng = MersenneTwister(105)
    report = run_recovery_suite(rng;
        hill_adam = 40, hill_bfgs = 20,
        sections = (:hill,))
    @test report[:hill].rmse < 0.45
    @test isfinite(report[:hill].final_loss)
end

@testset "known competitive parameter recovery" begin
    rng = MersenneTwister(106)
    report = run_recovery_suite(rng;
        competitive_adam = 40, competitive_bfgs = 20,
        sections = (:competitive,))
    @test report[:competitive].rmse < 0.55
    @test isfinite(report[:competitive].final_loss)
end

@testset "UDE to discovery executable RHS" begin
    rng = MersenneTwister(103)
    report = run_recovery_suite(rng;
        ude_adam = 40, ude_bfgs = 15,
        sections = (:ude_discovery,))
    ude = report[:ude_discovery]
    @test ude.retcode isa BioDynaX.DiscoveryRetcode
    if ude.success
        @test ude.correlation > 0.75
    else
        @test ude.retcode in (DenominatorUnsafe, EmptySupport, DiscoveryFailed)
    end
end

@testset "graph-local vs global library ablation" begin
    rng = MersenneTwister(104)
    report = run_recovery_suite(rng; sections = (:ablation,))
    ablation = report[:ablation]
    @test ablation.local_terms < ablation.global_terms
    @test 3 ∉ ablation.local_variables
    @test 3 ∈ ablation.global_variables
end

@testset "DataDrivenSparse backend requires extension" begin
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    x = collect(range(0.1, 2.0; length = 40))
    X = reshape(x, 1, :)
    dX = reshape(1.2 .* x, 1, :)
    times = collect(range(0.0, 1.0; length = 40))
    ext = Base.get_extension(BioDynaX, :BioDynaXDataDrivenSparseExt)
    if ext === nothing
        result = discover_equations(
            X, times, network; derivatives = dX,
            config = DiscoveryConfig(backend = DataDrivenSparseSTLSQ()),
            verbose = false, strict = false)
        @test !result.success
        @test occursin("DataDrivenSparse", result.message)
    else
        result = discover_equations(
            X, times, network; derivatives = dX,
            config = DiscoveryConfig(backend = DataDrivenSparseSTLSQ(threshold = 1e-4)),
            verbose = false, strict = false)
        @test result.success
        @test result.candidates isa Vector{ExplicitCandidate}
    end
end

@testset "discovery retcode is not silent" begin
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    X = reshape(collect(range(0.1, 0.2; length = 5)), 1, :)
    times = collect(range(0.0, 1.0; length = 5))
    result = discover_equations(
        X, times, network; derivatives = X,
        verbose = false, strict = false)
    @test !result.success
    @test result.retcode === InsufficientSamples
    @test_throws ArgumentError discover_equations(
        X, times, network; derivatives = X, verbose = false, strict = true)
end
