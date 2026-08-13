@testset "CSV experiment roundtrip" begin
    times = collect(0.0:0.5:4.0)
    obs = [exp.(-0.3 .* times)'; (0.2 .+ 0.1 .* times)']
    experiment = Experiment(:csv_roundtrip, times, obs, obs[:, 1])
    path = joinpath(tempdir(), "biodynax_csv_roundtrip.csv")
    write_experiment_csv(path, experiment; state_names = [:S, :R])
    loaded, names = experiment_from_csv(path)
    @test names == [:S, :R]
    @test loaded.times ≈ times
    @test loaded.observations ≈ obs
    @test loaded.u0 ≈ obs[:, 1]
    rm(path; force = true)
end

@testset "golden path export_rhs resimulation" begin
    network = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
    x = collect(range(0.2, 2.0; length = 80))
    X = reshape(x, 1, :)
    times = collect(range(0.0, 2.0; length = length(x)))
    vmax, km = 1.9, 0.7
    dX = reshape(vmax .* x ./ (km .+ x), 1, :)
    result = discover_equations(
        X, times, network; derivatives = dX,
        config = DiscoveryConfig(
            backend = ImplicitSINDyPI(
                threshold = 1e-6, max_degree = 1, max_hill_degree = 1,
                bootstrap_samples = 0, domain_samples = 16),
            include_interactions = false),
        verbose = false, strict = true)
    @test result.success
    @test result.retcode === DiscoverySuccess
    rhs = export_rhs(result)
    u0 = [x[1]]
    prob = SciMLBase.ODEProblem((u, p, t) -> rhs(u), u0, (0.0, 0.5))
    sol = solve(prob, Tsit5(); saveat = 0.0:0.1:0.5, sensealg = nothing)
    @test SciMLBase.successful_retcode(sol)
    @test all(isfinite, Array(sol))
end
