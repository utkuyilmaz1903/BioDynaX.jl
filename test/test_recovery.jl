@testset "recovery metric helpers" begin
    @test rate_rel_rmse([1.0, 2.0], [1.0, 2.0]) == 0
    @test rate_rel_rmse([2.0, 2.0], [1.0, 1.0]) ≈ 1.0
    key = ((1,), (2,))
    @test support_f1(Set([key]), Set([key])).f1 == 1.0
    @test support_f1(Set([((1,), (1,))]), Set([key])).f1 == 0.0
    hill = hill_rate_support(2)
    @test first(hill.numerator) == key
    @test first(hill.denominator) == key
    mm = mm_rate_support()
    @test first(mm.numerator) == ((1,), (1,))
    r = collect(range(0.1, 2.0; length = 50))
    D = hill_rate_truth(r; vmax = 1.5, K = 0.5, n = 2)
    @test rate_rel_rmse(D, D) == 0
    spec = local_basis(build_rate_discovery_network(), 1;
                       degree = 2, include_interactions = false)
    @test term_key(spec.numerator[1]) == ((), ())
    @test term_key(spec.denominator[2]) == ((1,), (2,))
end

@testset "DiscoveryConfig basis_scope plumbing" begin
    @test DiscoveryConfig().basis_scope === :graph
    @test DiscoveryConfig(basis_scope = :global).basis_scope === :global
    @test_throws ArgumentError DiscoveryConfig(basis_scope = :parents)
    net = build_rate_ablation_network()
    local_spec = local_basis(net, 1; degree = 2, include_interactions = false,
                             scope = :graph)
    global_spec = local_basis(net, 1; degree = 2, include_interactions = false,
                              scope = :global)
    @test 1 ∈ local_spec.variables
    @test 2 ∉ local_spec.variables
    @test 2 ∈ global_spec.variables
    @test candidate_count(local_spec) < candidate_count(global_spec)
    cfg = rate_discovery_config(scope = :global)
    @test cfg.basis_scope === :global
end

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

@testset "sample_unknown_destruction matches compiled neural D" begin
    rng = MersenneTwister(1)
    net = build_hill_recovery_network(; known = false)
    model, p = build_ude_model(rng, net)
    X = [0.3 0.4 0.5; 0.2 0.6 1.0]
    R, D, term = sample_unknown_destruction(model, p, X)
    @test size(R) == (1, 3)
    @test size(D) == (1, 3)
    @test term isa NeuralDestructionTerm
    @test vec(R) == vec(X[term.regulator, :])
    @test all(≥(0), D)
end

@testset "known linear parameter recovery" begin
    rng = MersenneTwister(101)
    report = run_recovery_suite(rng;
        linear_adam = 30, linear_bfgs = 15,
        sections = (:linear,))
    # Measured floors on seed 101. Tightening requires a new table; do not guess.
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

@testset "discovered support extras labels leftover monomials" begin
    @test monomial_key_label(((), ())) == "1"
    @test monomial_key_label(((1,), (1,))) == "r"
    @test monomial_key_label(((1,), (2,))) == "r^2"
    r = collect(range(0.1, 2.0; length = 180))
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    times = collect(range(0.0, 1.0; length = length(r)))
    truth = hill_rate_support(2)
    clean = discover_unknown_rate(
        reshape(r, 1, :), times, reshape(D, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = 1),
        verbose = false, strict = true)
    @test clean.success
    @test isempty(discovered_support_extras(
        clean.candidates[1], truth.numerator, truth.denominator))
    nn_like = D .+ 0.04 .+ 0.04 .* r
    dirty = discover_unknown_rate(
        reshape(r, 1, :), times, reshape(nn_like, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = 103),
        verbose = false, strict = false)
    @test dirty.success
    extras = discovered_support_extras(
        dirty.candidates[1], truth.numerator, truth.denominator)
    @test "1" in extras
    @test "r" in extras
    @test !("r^2" in extras)
    @test !(:discovered_support_extras in names(BioDynaX))
    @test !(:monomial_key_label in names(BioDynaX))
end

@testset "recovery metrics on analytical Hill rate" begin
    r = collect(range(0.1, 2.0; length = 120))
    D = hill_rate_truth(r; vmax = 1.8, K = 0.6, n = 2)
    R = reshape(r, 1, :)
    dX = reshape(D, 1, :)
    times = collect(range(0.0, 1.0; length = length(r)))
    result = discover_unknown_rate(
        R, times, dX; config = rate_discovery_config(bootstrap = 0, seed = 1),
        verbose = false, strict = true)
    @test result.success
    @test result.retcode === DiscoverySuccess
    truth = hill_rate_support(2)
    metrics = support_f1(result.candidates[1], truth.numerator, truth.denominator)
    @test metrics.combined.f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean
    d_hat = equation_to_function(result.candidates[1])
    pred = [d_hat([rj]) for rj in r]
    @test rate_rel_rmse(pred, D) < 0.05
    @test denominator_violation_count(result.candidates[1], R) == 0
end

@testset "Occam prune recovers sparse Hill on noisy rate samples" begin
    r = collect(range(0.1, 2.0; length = 180))
    rng = MersenneTwister(104)
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    amp = maximum(abs, D)
    D_noisy = D .+ 0.005 .* amp .* randn(rng, length(r))
    R = reshape(r, 1, :)
    times = collect(range(0.0, 1.0; length = length(r)))
    result = discover_unknown_rate(
        R, times, reshape(D_noisy, 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = 4),
        verbose = false, strict = true)
    truth = hill_rate_support(2)
    metrics = support_f1(result.candidates[1], truth.numerator, truth.denominator)
    @test metrics.combined.recall ≥ RECOVERY_THRESHOLDS.support_recall
    @test metrics.combined.f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean
    @test denominator_violation_count(result.candidates[1], R) == 0
end

@testset "graph vs global rate discovery ablation" begin
    report = run_recovery_suite(MersenneTwister(104); sections = (:ablation,))
    ablation = report[:ablation]
    @test ablation.local_terms < ablation.global_terms
    @test 1 ∈ ablation.local_variables
    @test 2 ∉ ablation.local_variables
    @test 2 ∈ ablation.global_variables
    @test ablation.local_success
    @test ablation.local_false_parent == false
    @test ablation.local_f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean
    # After Occam, graph and global F1 can both be 1.00. The locked prior is
    # library membership of the distractor, not an F1 gap.
    @test 2 ∉ ablation.local_variables
    @test 2 ∈ ablation.global_variables
    @test ablation.local_denominator_violations ≤ ablation.global_denominator_violations
end

@testset "3-state graph prior vs global distractors" begin
    report = run_recovery_suite(MersenneTwister(204); sections = (:three_state,))
    three = report[:three_state]
    @test 2 ∈ three.graph_parents
    @test 3 ∉ three.graph_parents
    @test 4 ∉ three.graph_parents
    @test three.local_success
    @test three.local_has_true_parent
    @test three.local_false_parent == false
end

@testset "wrong-graph negative control misses the true parent" begin
    report = run_recovery_suite(MersenneTwister(214); sections = (:wrong_graph,))
    wrong = report[:wrong_graph]
    @test 3 ∈ wrong.graph_parents
    @test 2 ∉ wrong.graph_parents
    @test wrong.local_has_true_parent == false
end

@testset "6-state graph prior vs global distractors" begin
    report = run_recovery_suite(MersenneTwister(224); sections = (:six_state,))
    six = report[:six_state]
    @test six.nstates == 6
    @test 2 ∈ six.graph_parents
    @test 3 ∉ six.graph_parents
    @test 6 ∉ six.graph_parents
    @test six.local_success
    @test six.local_has_true_parent
    @test six.local_false_parent == false
    @test six.Z_in_local_library == false
    @test six.Z_in_global_library
    @test six.distractor_in_local == false
    @test six.distractor_in_global
end

@testset "6-state wrong-graph negative control misses the true parent" begin
    report = run_recovery_suite(MersenneTwister(234);
                               sections = (:six_state_wrong_graph,))
    wrong = report[:six_state_wrong_graph]
    @test wrong.nstates == 6
    @test 3 ∈ wrong.graph_parents
    @test 2 ∉ wrong.graph_parents
    @test wrong.local_has_true_parent == false
end

@testset "k_prod vs D practical identifiability is reported" begin
    report = run_recovery_suite(MersenneTwister(205); sections = (:identifiability,))
    ident = report[:identifiability]
    @test isfinite(ident.condition_number) || isinf(ident.condition_number)
    @test ident.unidentifiable_edge isa Bool
    @test ident.production_param === :k_prod
end

@testset "identifiability interventions do not break the scale tradeoff" begin
    report = run_recovery_suite(MersenneTwister(215); sections = (:ident_interventions,))
    ident = report[:ident_interventions]
    @test ident.normalized_analytical_f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean
    @test ident.frozen_k_prod_unchanged
    @test ident.tradeoff_broken == false
    @test ident.nominal_unidentifiable || ident.freeze_unidentifiable ||
          ident.perturbation_unidentifiable
    @test !isfinite(ident.nominal_collinearity) || ident.nominal_collinearity ≥ 0.95
    @test !isfinite(ident.freeze_collinearity) || ident.freeze_collinearity ≥ 0.95
end

@testset "partial observation mask and subsampled Hill parent" begin
    report = run_recovery_suite(MersenneTwister(206); sections = (:partial_obs,))
    part = report[:partial_obs]
    @test part.subsample_success
    @test part.subsample_recall ≥ RECOVERY_THRESHOLDS.support_recall
    @test part.mask_used
    @test part.masked_train_finite
    @test part.closed_loop_vs_data
    @test part.ude_mask_train_claimed == false
    rng = MersenneTwister(9)
    times = collect(0.0:0.5:2.0)
    data = [0.2 0.3 0.4 0.5 0.6; 0.1 0.12 0.11 0.13 0.14]
    exp = Experiment(:raw, times, data, [0.2, 0.1])
    masked = BioDynaX.subsample_state_mask(exp, 1, 0.5, rng)
    @test masked.mask[1, 1]
    @test count(masked.mask[1, :]) < size(data, 2)
end

@testset "competitive unknown edge is 2D D(S,I) and keeps true parents" begin
    compiled = compile_mechanism(build_competitive_test_network(; known = false))
    nn_terms = [t for t in compiled.destruction_terms if t isa NeuralDestructionTerm]
    @test length(nn_terms) == 1
    @test nn_terms[1].regulators == [2, 3]
    rng = MersenneTwister(8)
    model, p = build_ude_model(rng, build_competitive_test_network(; known = false))
    dx = ude_system([0.3, 0.4, 0.2], p, 0.0, model)
    @test all(isfinite, dx)
    report = run_recovery_suite(MersenneTwister(304); sections = (:competitive_unknown,))
    comp = report[:competitive_unknown]
    @test comp.compiled_regulators == [2, 3]
    @test comp.two_parent_success
    @test comp.has_substrate
    @test comp.has_inhibitor
    @test comp.local_false_parent == false
    @test comp.canonical_f1_claimed == false
end

@testset "Elowitz repressilator is a synthetic literature fixture" begin
    report = run_recovery_suite(MersenneTwister(207); sections = (:literature,))
    lit = report[:literature]
    @test lit.experimental_csv == false
    @test lit.unique_claim_protocol == false
    @test lit.licensed_experimental_series == false
    @test lit.finite_trajectory
    @test lit.nonnegative
    @test lit.nstates == 3
    @test occursin("Elowitz", lit.source)
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

@testset "golden-path example matches recovery CI protocol" begin
    src = read(joinpath(@__DIR__, "..", "examples", "unknown_inhibition.jl"), String)
    proto = UNIQUE_CLAIM_PROTOCOL
    @test proto.seed == 103
    @test proto.adam_iterations == 100
    @test proto.bfgs_iterations == 50
    @test proto.n_points == 50
    @test proto.smoke_n_points == 8
    @test proto.tspan == (0.0, 8.0)
    @test proto.bootstrap == 8
    @test proto.discovery_seed == 3
    @test occursin("UNIQUE_CLAIM_PROTOCOL.seed", src)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.adam_iterations", src)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.bfgs_iterations", src)
    @test occursin("unique_claim_discovery_config", src)
    @test occursin("production_destruction_tradeoff", src)
    @test occursin("assert_single_unknown_destruction", src)
    @test occursin("format_protocol_result", src)
    @test occursin("_unknown_edge_ics()", src)
    @test occursin("smoke ? ics_all[1:1]", src)
    @test occursin("smoke_n_points", src)
    @test occursin("sample_unknown_destruction_grid", src)
    @test occursin("_regulator_grid", src)
    @test occursin("ReactionSpec", src)
    @test occursin("HillMetadata", src)
    @test !occursin("build_hill_recovery_network", src)
    @test !occursin("Note:", src)
    @test !(:UNIQUE_CLAIM_PROTOCOL in names(BioDynaX))
    @test !(:unique_claim_discovery_config in names(BioDynaX))
    cfg = unique_claim_discovery_config()
    @test cfg.seed == proto.discovery_seed
    @test cfg.backend.bootstrap_samples == proto.bootstrap
    @test BioDynaX._unknown_edge_ics() == [
        [0.25, 0.20], [0.80, 0.35], [0.40, 1.10], [1.20, 0.70], [0.15, 0.90],
        [0.50, 0.15], [0.90, 1.50], [0.20, 0.50], [1.50, 1.20]]
end

@testset "unique-claim protocol requires exactly one unknown D" begin
    rng = MersenneTwister(0)
    zero_model, _ = build_ude_model(rng, build_linear_test_network())
    @test_throws ErrorException assert_single_unknown_destruction(zero_model)
    one_model, _ = build_ude_model(rng, build_hill_recovery_network(; known = false))
    @test assert_single_unknown_destruction(one_model) == 1
    two_model, _ = build_ude_model(rng, build_dual_unknown_network())
    @test_throws ErrorException assert_single_unknown_destruction(two_model)
end

@testset "protocol formatter prints identifiability before equations" begin
    ident = (;
        unidentifiable_edge = true,
        production_param = :k_prod,
        collinearity = 0.99)
    text = format_protocol_result(ident;
        residual = 0.003,
        equations = "D(z) = vmax * r^2 / (K^2 + r^2)",
        extras = ("1", "r"),
        support_f1 = 0.57,
        support_recall = 1.0,
        unknown_holes = 1,
        seed = 103,
        n_ics = 9,
        adam_iters = 100,
        bfgs_iters = 50,
        bootstrap = 8,
        discovery_seed = 3,
        smoke = false)
    ident_at = findfirst("IDENTIFIABILITY", text)
    coeff_at = findfirst("coefficients_are_biological_constants: false", text)
    edge_at = findfirst("unidentifiable_edge: true", text)
    eq_at = findfirst("D(z) = vmax", text)
    @test ident_at !== nothing
    @test coeff_at !== nothing
    @test eq_at !== nothing
    @test first(ident_at) < first(coeff_at) < first(eq_at)
    @test edge_at !== nothing && first(ident_at) < first(edge_at)
    @test occursin("canonical_hill_from_nn: false", text)
    @test occursin("extras: 1, r", text)
    @test !occursin("Note:", text)
    @test startswith(text, "IDENTIFIABILITY")
    identifiable = format_protocol_result((; unidentifiable_edge = false);
        equations = "D(z) = 1")
    @test occursin("coefficients_are_biological_constants: true", identifiable)
    @test first(findfirst("IDENTIFIABILITY", identifiable)) <
          first(findfirst("D(z) = 1", identifiable))
end

@testset "recovery thresholds and protocol helpers stay unexported" begin
    @test RECOVERY_THRESHOLDS == (
        nn_correlation = 0.90,
        nn_rate_rmse = 0.12,
        support_f1_clean = 0.99,
        support_f1_ude = 0.50,
        support_f1_noisy = 0.50,
        support_recall = 0.99,
        discovered_rate_rmse = 0.20,
        data_residual = 0.30)
    @test !(:format_protocol_result in names(BioDynaX))
    @test !(:assert_single_unknown_destruction in names(BioDynaX))
end
