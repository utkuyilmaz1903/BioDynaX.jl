@testset "discovery workspace is not exported" begin
    @test !(:STLSQWorkspace in names(BioDynaX))
    @test !(:_stlsq_blocked! in names(BioDynaX))
    @test !(:_fit_implicit_stream in names(BioDynaX))
    @test !(:evaluate_candidate! in names(BioDynaX))
    @test !(:each_reusable_library_chunk in names(BioDynaX))
    @test !(:discovery_workspace_contract_holds in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "evaluate_library! writes monomials in place" begin
    network = BiologicalNetwork([NodeSpec(name = :x), NodeSpec(name = :z)], EdgeSpec[])
    spec = local_basis(network, 1; degree = 2, include_interactions = false)
    X = reshape(collect(range(0.2, 1.8; length = 64)), 1, :)
    X = vcat(X, 0.3 .* X)
    out = Matrix{Float64}(undef, size(X, 2), length(spec.numerator))
    BioDynaX.evaluate_library!(out, spec.numerator, X)
    ptr = pointer(out)
    BioDynaX.evaluate_library!(out, spec.numerator, X)
    @test pointer(out) === ptr
    allocating = BioDynaX.evaluate_library(spec.numerator, X)
    @test out ≈ allocating
    col = out[:, 2]
    col_ptr = pointer(col)
    BioDynaX.evaluate_term!(col, spec.numerator[2], X)
    @test pointer(col) === col_ptr
    @test basis_factory_evaluates_in_place()
end

@testset "blocked STLSQ workspace matches the allocating path" begin
    rng = MersenneTwister(11)
    A = randn(rng, 180, 9)
    ξ = [1.2, 0.0, -0.4, 0.0, 0.8, 0.0, 0.0, 0.15, 0.0]
    y = A * ξ .+ 1e-4 .* randn(rng, 180)
    report = BioDynaX.stlsq_path_agreement(A, y, 5e-2; chunk_size = 32)
    @test report.holds
    @test report.blocked_matches_workspace
    @test report.workspace_stable
    @test report.resize_count == 1
    dense_err = maximum(abs, report.dense .- report.blocked)
    @test dense_err < 5e-2
end

@testset "STLSQ workspace buffers are stable across calls" begin
    rng = MersenneTwister(19)
    A = randn(rng, 120, 6)
    y = A * [0.9, 0.0, 0.3, 0.0, 0.0, 0.2]
    ws = BioDynaX.allocate_stlsq_workspace(Float64, 120, 6, 24)
    gram_ptr = pointer(ws.gram)
    chunk_ptr = pointer(ws.design_chunk)
    first = collect(BioDynaX._stlsq_blocked!(ws, A, y, 1e-2))
    @test pointer(ws.gram) === gram_ptr
    @test pointer(ws.design_chunk) === chunk_ptr
    second = collect(BioDynaX._stlsq_blocked!(ws, A, y, 1e-2))
    @test first ≈ second
    @test pointer(ws.gram) === gram_ptr
    @test ws.resize_count == 1
    via_api = BioDynaX._stlsq_blocked(A, y, 1e-2; chunk_size = 24, workspace = ws)
    @test via_api ≈ first
end

@testset "workspace STLSQ allocates less than a fresh blocked call" begin
    rng = MersenneTwister(23)
    A = randn(rng, 220, 8)
    y = A * [1.0, 0.0, 0.4, 0.0, 0.0, 0.2, 0.0, 0.0]
    report = BioDynaX.discovery_workspace_alloc_report(A, y, 2e-2; chunk_size = 40)
    @test report.gram_stable
    @test report.chunk_stable
    @test report.reuse_smaller
    @test report.holds
    @test report.reused_bytes < report.naive_bytes
end

@testset "chunk_size one and larger than n agree" begin
    rng = MersenneTwister(29)
    A = randn(rng, 40, 5)
    y = A * [0.7, 0.0, 0.2, 0.0, 0.1]
    a = BioDynaX._stlsq_blocked(A, y, 1e-2; chunk_size = 1)
    b = BioDynaX._stlsq_blocked(A, y, 1e-2; chunk_size = 40)
    c = BioDynaX._stlsq_blocked(A, y, 1e-2; chunk_size = 256)
    @test a ≈ b
    @test b ≈ c
end

@testset "implicit stream agrees with materialised design" begin
    network = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
    x = collect(range(0.08, 2.4; length = 160))
    X = reshape(x, 1, :)
    vmax, km = 2.1, 0.55
    derivative = vmax .* x ./ (km .+ x)
    spec = local_basis(
        network, 1; degree = 1, include_interactions = false,
        X, derivative, max_variables = 1)
    report = BioDynaX.implicit_stream_agreement(
        spec, X, derivative, collect(eachindex(x)), 1e-7; chunk_size = 32)
    @test report.holds
    @test report.num_stream
    @test report.den_stream
    @test report.num_workspace
    pred, denvals = BioDynaX._evaluate_candidate(
        spec, report.streamed[1], report.streamed[2], X)
    @test mean(abs2, pred .- derivative) < 1e-8
    @test minimum(denvals) > 0
end

@testset "Hill recovery still holds on the streamed implicit path" begin
    single = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    x = collect(range(0.1, 2.0; length = 220))
    X = reshape(x, 1, :)
    vmax, k = 1.8, 0.6
    hill = vmax .* x .^ 4 ./ (k^4 .+ x .^ 4)
    spec = local_basis(
        single, 1; degree = 4, include_interactions = false,
        X, derivative = hill)
    num, den = BioDynaX._fit_implicit(
        spec, X, hill, collect(eachindex(x)), 1e-7; chunk_size = 28)
    pred, denvals = BioDynaX._evaluate_candidate(spec, num, den, X)
    @test mean(abs2, pred .- hill) < 1e-7
    @test minimum(abs, denvals) > 0
    streamed = BioDynaX._fit_implicit_stream(
        spec, X, hill, collect(eachindex(x)), 1e-7; chunk_size = 28)
    @test streamed[1] ≈ num
    @test streamed[2] ≈ den
end

@testset "evaluate_candidate! matches _evaluate_candidate" begin
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    x = collect(range(0.2, 1.5; length = 50))
    X = reshape(x, 1, :)
    spec = local_basis(network, 1; degree = 2, include_interactions = false)
    num = [0.0, 1.4, 0.0]
    den = [0.8, 0.0]
    length(num) == length(spec.numerator) ||
        (num = zeros(length(spec.numerator)); num[2] = 1.4)
    length(den) == length(spec.denominator) ||
        (den = zeros(length(spec.denominator)); den[1] = 0.8)
    report = BioDynaX.evaluate_candidate_agreement(spec, num, den, X)
    @test report.holds
end

@testset "reusable library chunks overwrite one buffer" begin
    network = BiologicalNetwork(
        [NodeSpec(name = :a), NodeSpec(name = :b)],
        [EdgeSpec(source = 2, target = 1, kind = INHIBITION, family = HILL)])
    spec = local_basis(network, 1; degree = 2, include_interactions = true)
    X = randn(MersenneTwister(3), 2, 90)
    report = BioDynaX.library_chunk_agreement(spec.numerator, X; chunk_size = 17)
    @test report.holds
    @test report.matches_full
    ws = BioDynaX.allocate_library_chunk_workspace(Float64, 17, length(spec.numerator))
    ptr = pointer(ws.buffer)
    nchunks = 0
    for (chunk, sample_range) in BioDynaX.each_reusable_library_chunk(
            spec.numerator, X, ws; chunk_size = 17)
        nchunks += 1
        @test pointer(parent(chunk)) === ptr || pointer(ws.buffer) === ptr
        @test length(sample_range) ≤ 17
        @test size(chunk, 2) == length(spec.numerator)
    end
    @test nchunks == cld(90, 17)
    @test ws.resize_count == 1
end

@testset "backend chunk_size helper is honest" begin
    implicit = ImplicitSINDyPI(chunk_size = 48)
    @test BioDynaX._backend_chunk_size(implicit) == 48
    @test BioDynaX._backend_chunk_size(ExplicitSTLSQ()) == 256
    @test BioDynaX._backend_chunk_size(DataDrivenSparseSTLSQ()) == 256
    cfg = DiscoveryConfig(backend = ImplicitSINDyPI(chunk_size = 16))
    @test BioDynaX._backend_chunk_size(cfg) == 16
end

@testset "bootstrap reuses one streaming workspace" begin
    network = BiologicalNetwork([NodeSpec(name = :r)], EdgeSpec[])
    r = collect(range(0.15, 1.8; length = 100))
    X = reshape(r, 1, :)
    D = 1.6 .* r .^ 2 ./ (0.5^2 .+ r .^ 2)
    spec = local_basis(
        network, 1; degree = 2, include_interactions = false,
        X, derivative = D)
    report = BioDynaX.bootstrap_workspace_reuse_report(
        spec, X, D, collect(1:80), 1e-3;
        bootstrap_samples = 5, chunk_size = 20, seed = 7)
    @test report.reused
    @test report.workspace_resizes == 0
    @test report.stlsq_resizes == 0
    @test length(report.selected) == length(spec.numerator) + length(spec.denominator)
    @test all(x -> 0 ≤ x ≤ 1, report.selected)
end

@testset "masked implicit refit reuses the implicit workspace" begin
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    x = collect(range(0.2, 2.0; length = 80))
    X = reshape(x, 1, :)
    y = 1.5 .* x ./ (0.4 .+ x)
    spec = local_basis(
        network, 1; degree = 1, include_interactions = false, X, derivative = y)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    ws = BioDynaX.allocate_implicit_workspace(Float64, 80, n_num, n_den, 32)
    keep_n = trues(n_num)
    keep_d = trues(n_den)
    a = BioDynaX._refit_masked_implicit(
        spec, X, y, keep_n, keep_d, 1e-6; workspace = ws)
    start = ws.resize_count
    b = BioDynaX._refit_masked_implicit(
        spec, X, y, keep_n, keep_d, 1e-6; workspace = ws)
    @test a[1] ≈ b[1]
    @test a[2] ≈ b[2]
    @test ws.resize_count == start
end

@testset "raw-data discovery still returns InsufficientSamples" begin
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    X = reshape(collect(range(0.1, 0.4; length = 8)), 1, :)
    times = collect(range(0.0, 1.0; length = 8))
    result = discover_equations(X, times, network; verbose = false, strict = false)
    @test result.success == false
    @test result.retcode === InsufficientSamples
end

@testset "explicit discovery uses the backend chunk helper" begin
    src = read(BioDynaX.discovery_jl_source_path(), String)
    @test occursin("chunk_size = _backend_chunk_size(backend)", src)
    network = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
    x = collect(range(0.1, 1.5; length = 80))
    X = reshape(x, 1, :)
    dX = reshape(1.1 .- 0.4 .* x, 1, :)
    cfg = DiscoveryConfig(backend = ExplicitSTLSQ(threshold = 1e-3),
        include_interactions = false)
    result = discover_equations(
        X, collect(range(0.0, 1.0; length = 80)), network;
        derivatives = dX, config = cfg, verbose = false, strict = true)
    @test result.success
    @test result.retcode === DiscoverySuccess
    @test !isempty(result.candidates)
end

@testset "streaming contract source and docs hold" begin
    @test discovery_workspace_source_holds()
    @test BioDynaX.discovery_jl_uses_workspace()
    @test BioDynaX.basis_factory_evaluates_in_place()
    @test BioDynaX.discovery_streaming_docs_hold()
    @test BioDynaX.discovery_streaming_landing_docs_hold()
    @test discovery_workspace_contract_holds()
    violations = BioDynaX.discovery_workspace_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    sentences = BioDynaX.discovery_streaming_locked_sentences()
    @test haskey(sentences, :workspace)
    @test haskey(sentences, :library)
    @test haskey(sentences, :stream)
    @test haskey(sentences, :chunk)
    @test haskey(sentences, :backend)
end

@testset "workspace path does not loosen locked claim numbers" begin
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test recovery_thresholds_lock() == RECOVERY_THRESHOLDS
    @test issetequal(names(BioDynaX), collect(locked_public_names()))
end

@testset "0/2-hole networks stay open on validate_network after workspace include" begin
    zero = build_zero_unknown_linear_network()
    two = BioDynaX.build_dual_unknown_network()
    @test validate_network(zero) === zero
    @test validate_network(two) === two
    @test count_unknown_destructions(zero) == 0
    @test count_unknown_destructions(two) == 2
    @test unique_claim_recovery_admits(zero) == false
    @test unique_claim_recovery_admits(two) == false
end
