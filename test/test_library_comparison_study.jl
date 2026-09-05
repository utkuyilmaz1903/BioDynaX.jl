# Library comparison study: the multi-seed, multi-noise generalisation of the
# trained-model library check. The smoke configuration (one seed, one noise
# level, the smoke training budget) runs here; the full study is
# benchmark/library_comparison_study.jl.

const _LCS = BioDynaX
const _LCS_TRUTH = _LCS.hill_rate_support(_LCS.LIBRARY_STUDY_TRUTH.n;
    variable = _LCS.LIBRARY_STUDY_TRUTH.variable)

function _lcs_supports(evidence)
    return map((evidence.graph_discovery, evidence.global_discovery,
        evidence.wrong_graph_discovery)) do discovery
        discovery.success || return nothing
        support = _LCS.active_support(discovery.candidates[1])
        return (sort(collect(support.numerator); by = string),
            sort(collect(support.denominator); by = string))
    end
end

@testset "library comparison study" begin
    @testset "single-run defaults reproduce the original run" begin
        default = evaluate_trained_graph_local(kind = :smoke)
        explicit = evaluate_trained_graph_local(kind = :smoke,
            seed = _LCS.M4B_SMOKE.seed, noise_σ = _LCS.M4B_SMOKE.noise_σ)
        @test default.params_nn_fingerprint == explicit.params_nn_fingerprint
        @test default.D == explicit.D
        @test default.X == explicit.X
        @test _lcs_supports(default) == _lcs_supports(explicit)
        set = _LCS.library_study_training_set(:smoke)
        @test [experiment.u0 for experiment in set.experiments] ==
              _LCS.m4b_initial_conditions(:smoke)
        @test length(set.experiments[1].times) == _LCS.M4B_SMOKE.n_points
        # The generated data is a pure function of the seed and the noise level.
        again = _LCS.library_study_training_set(:smoke)
        @test set.experiments[1].observations == again.experiments[1].observations
        noisy = _LCS.library_study_training_set(:smoke; noise_σ = 0.02)
        @test noisy.experiments[1].observations != set.experiments[1].observations
    end

    @testset "smoke study reuses the single run and reports every column" begin
        rows = _LCS.library_comparison_smoke()
        @test length(rows) == 3
        @test [row.library for row in rows] == collect(_LCS.LIBRARY_STUDY_LIBRARIES)
        @test all(propertynames(row) == _LCS.LIBRARY_STUDY_COLUMNS for row in rows)
        @test all(row.seed == _LCS.M4B_SMOKE.seed for row in rows)
        @test all(row.noise == _LCS.M4B_SMOKE.noise_σ for row in rows)
        # The three rows share one training.
        @test all(row.train_time_s == rows[1].train_time_s for row in rows)
        @test all(row.run_time_s == rows[1].run_time_s for row in rows)
        @test all(row.nn_rate_rmse == rows[1].nn_rate_rmse for row in rows)
        @test 0 < rows[1].train_time_s <= rows[1].run_time_s

        evidence = evaluate_trained_graph_local(kind = :smoke)
        names = [node.name for node in evidence.model.network.nodes]
        @test names == [:S, :R, :Q, :Z]
        for (row, discovery) in zip(rows,
            (evidence.graph_discovery,
                evidence.global_discovery, evidence.wrong_graph_discovery))
            scores = _LCS.library_study_scores(discovery, _LCS_TRUTH; names)
            @test row.success == scores.success
            @test row.support_recall == scores.support_recall
            @test row.support_precision == scores.support_precision
            @test row.support_f1 == scores.support_f1
            @test row.extra_terms == scores.extra_terms
            @test row.extras == join(scores.extras, ";")
            @test 0 <= row.support_recall <= 1
            @test 0 <= row.support_precision <= 1
            @test 0 <= row.support_f1 <= 1
            @test row.extra_terms == length(scores.extras)
            if scores.success
                p, r = row.support_precision, row.support_recall
                expected = p + r == 0 ? 0.0 : 2p * r / (p + r)
                @test row.support_f1 ≈ expected
                @test isfinite(row.data_residual) || row.data_residual == Inf
            else
                @test isnan(row.data_residual)
                @test isnan(row.holdout_residual)
            end
        end
        r = vec(evidence.X[_LCS.LIBRARY_STUDY_TRUTH.variable, :])
        truth_rate = _LCS.hill_rate_truth(r; vmax = _LCS.LIBRARY_STUDY_TRUTH.vmax,
            K = _LCS.LIBRARY_STUDY_TRUTH.K, n = _LCS.LIBRARY_STUDY_TRUTH.n)
        @test rows[1].nn_rate_rmse == _LCS.rate_rel_rmse(vec(evidence.D), truth_rate)
    end

    @testset "term labels and scores" begin
        names = [:S, :R, :Q, :Z]
        @test _LCS._library_study_term_label(((), ()), names) == "1"
        @test _LCS._library_study_term_label(((2,), (2,)), names) == "R^2"
        @test _LCS._library_study_term_label(((3, 4), (1, 2)), names) == "Q*Z^2"
        @test _LCS_TRUTH.numerator == Set([((2,), (2,))])
        @test _LCS_TRUTH.denominator == Set([((2,), (2,))])
        @test_throws ArgumentError _LCS.library_comparison_run(
            seed = 1, noise_σ = 0.0, kind = :smoke, libraries = (:everything,))
    end

    @testset "CSV round trip, resume, and summary" begin
        rows = _LCS.library_comparison_smoke()
        path = joinpath(mktempdir(), "study.csv")
        @test isempty(_LCS.read_library_study_csv(path))
        for row in rows
            _LCS.append_library_study_row(path, row)
        end
        lines = readlines(path)
        @test lines[1] == _LCS.library_study_csv_header()
        @test length(lines) == 4
        back = _LCS.read_library_study_csv(path)
        @test length(back) == 3
        for (row, read_row) in zip(rows, back)
            for column in _LCS.LIBRARY_STUDY_COLUMNS
                @test isequal(getproperty(row, column), getproperty(read_row, column))
            end
        end
        keys = _LCS.library_study_keys(back)
        @test length(keys) == 3
        # Resuming from a complete CSV trains nothing.
        resumed = _LCS.library_comparison_smoke(
            skip = (seed, noise, library) -> (seed, noise, library) in keys,
            training_call = (args...; kwargs...) -> error("must not train"))
        @test isempty(resumed)
        # A partially written pair is rerun for the missing libraries only.
        partial = _LCS.library_comparison_smoke(
            skip = (seed, noise, library) -> library !== :wrong_graph)
        @test length(partial) == 1
        @test partial[1].library === :wrong_graph
        @test partial[1].support_f1 == rows[3].support_f1

        summary = _LCS.library_study_summary(back)
        @test length(summary) == 3
        @test [entry.library for entry in summary] ==
              collect(_LCS.LIBRARY_STUDY_LIBRARIES)
        @test all(entry.n == 1 for entry in summary)
        @test summary[1].support_f1_median == rows[1].support_f1
        @test summary[1].support_f1_q25 == rows[1].support_f1
        table = _LCS.format_library_study_summary(summary)
        @test occursin("| graph_local | 0.0 | 1 |", table)
        @test occursin("| wrong_graph |", table)
        @test count(==('\n'), table) == 5
    end
end
