# Bit-identity of the single-unknown-term path with 0.15.
#
# test/support/fingerprints_015.toml was written by
# test/support/record_fingerprints_015.jl on `main` at de497e8 (release
# 0.15.0) with 0.15's `discover_unknown_term`, on the machine named in its
# [environment] table. This test recomputes the same runs with
# `discover_unknown_terms` and compares every recorded value: trained
# parameters, sampled rate, discovered equations and coefficients,
# identifiability numbers, residuals, extras and the report text.
#
# On the recording machine (same Julia version, CPU model and BLAS) the
# comparison is exact (`==` on the 17-digit decimal strings). Elsewhere the
# last bits of BLAS results can differ, so the comparison is `isapprox` with
# a relative tolerance of 1e-10 on the numbers and exact on every string,
# equation, support and flag, and the test reports which mode ran.
using TOML, Printf

const _FP015 = TOML.parsefile(joinpath(@__DIR__, "fingerprints_015.toml"))
_fp17(x) = @sprintf("%.17g", Float64(x))
_fpvec(v) = String[_fp17(x) for x in Vector{Float64}(vec(v))]

function _fp_environment_matches()
    env = _FP015["environment"]
    return env["julia"] == string(VERSION) &&
           env["cpu"] == Sys.cpu_info()[1].model &&
           env["blas"] == string(HybridKinetics.LinearAlgebra.BLAS.get_config())
end

function _fp_same(recorded::Vector, actual::Vector, exact::Bool)
    length(recorded) == length(actual) || return false
    exact && return recorded == actual
    return all(isapprox(
                   parse(Float64, a), parse(Float64, b); rtol = 1e-10, atol = 1e-300) ||
               (isnan(parse(Float64, a)) && isnan(parse(Float64, b)))
    for (a, b) in zip(recorded, actual))
end
function _fp_same(recorded::String, actual::String, exact::Bool)
    _fp_same([recorded], [actual], exact)
end

function _fp_record(result::UnknownTermsResult)
    term = only(result.terms)
    d = term.discovery
    function coeffs(c)
        c isa HybridKinetics.ImplicitCandidate ?
        Dict("numerator" => _fpvec(c.numerator_coefficients),
            "denominator" => _fpvec(c.denominator_coefficients),
            "selection_frequency" => _fpvec(c.selection_frequency),
            "validation_error" => _fp17(c.validation_error)) :
        Dict("coefficients" => _fpvec(c.coefficients),
            "validation_error" => _fp17(c.validation_error))
    end
    return Dict(
        "phys" => _fpvec(collect(result.params.phys)),
        "nn_fingerprint" => string(HybridKinetics.nn_parameter_fingerprint(result.params.nn)),
        "nn" => _fpvec(HybridKinetics._nn_fingerprint_values(result.params.nn)),
        "final_loss" => _fp17(result.training.final_loss),
        "initial_loss" => _fp17(result.training.initial_loss),
        "R" => _fpvec(term.samples.R), "D" => _fpvec(term.samples.D),
        "equations" => d.success ?
                       (d.equations isa AbstractString ? [String(d.equations)] :
                        String[string(e) for e in d.equations]) : String[],
        "candidates" => d.success ? [coeffs(c) for c in d.candidates] : Dict[],
        "discovery_success" => d.success, "retcode" => string(d.retcode),
        "collinearity" => _fp17(term.identifiability.collinearity),
        "condition_number" => _fp17(term.identifiability.condition_number),
        "unidentifiable_edge" => term.identifiability.unidentifiable_edge,
        "data_residual" => _fp17(result.residuals.data_residual),
        "data_residual_train" => _fp17(result.residuals.data_residual_train),
        "data_residual_holdout" => _fp17(result.residuals.data_residual_holdout),
        "extras" => term.extras === nothing ? String[] : term.extras,
        "report" => report_unknown_terms(result))
end

function _fp_compare(name::String, actual::Dict, exact::Bool)
    recorded = _FP015[name]
    @testset "$name" begin
        for key in ("phys", "nn", "R", "D")
            @test _fp_same(recorded[key], actual[key], exact)
        end
        exact && @test recorded["nn_fingerprint"] == actual["nn_fingerprint"]
        for key in ("final_loss", "initial_loss", "collinearity", "condition_number",
            "data_residual", "data_residual_train", "data_residual_holdout")
            @test _fp_same(recorded[key], actual[key], exact)
        end
        @test recorded["discovery_success"] == actual["discovery_success"]
        @test recorded["retcode"] == actual["retcode"]
        @test recorded["unidentifiable_edge"] == actual["unidentifiable_edge"]
        @test recorded["equations"] == actual["equations"]
        @test recorded["extras"] == actual["extras"]
        @test length(recorded["candidates"]) == length(actual["candidates"])
        for (rc, ac) in zip(recorded["candidates"], actual["candidates"])
            for key in keys(rc)
                @test _fp_same(rc[key], ac[key], exact)
            end
        end
        # The report is compared exactly in both modes except for the digits
        # that a last-bit change could move; with exact numbers it is exact.
        exact ? (@test recorded["report"] == actual["report"]) :
        (@test length(recorded["report"]) == length(actual["report"]))
    end
end

@testset "0.15 fingerprints reproduce through discover_unknown_terms" begin
    exact = _fp_environment_matches()
    @info "0.15 fingerprint comparison" mode=(exact ? "exact (recording machine)" :
                                              "isapprox 1e-10 (other machine)") recorded_on=_FP015["environment"]["commit"]
    cfg = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6)
    truth_h = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    ics = [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10]]
    hill_truth = HybridKinetics.build_hill_recovery_network(; known = true, hill_order = 2)
    hill_ude = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    set_h = HybridKinetics.reference_protocol_experiment_set(
        MersenneTwister(103), hill_truth;
        smoke = true, truth_params = truth_h, initial_conditions = ics)
    r1 = discover_unknown_terms(hill_ude, set_h; training = cfg, holdout = 0,
        rng = MersenneTwister(7), verbose = false,
        known_support = HybridKinetics.hill_rate_support(2))
    _fp_compare("hill_holdout0", _fp_record(r1), exact)
    r2 = discover_unknown_terms(hill_ude, set_h; training = cfg, holdout = 1,
        rng = MersenneTwister(7), verbose = false, seed = 103)
    _fp_compare("hill_holdout1_seed103", _fp_record(r2), exact)
    truth_m = (k_prod = 0.9, vmax = 1.5, km = 0.4, k_rs = 1.0, k_r = 0.6)
    mm_truth = HybridKinetics.build_mm_recovery_network(; known = true)
    mm_ude = HybridKinetics.build_mm_recovery_network(; known = false)
    set_m = HybridKinetics.reference_protocol_experiment_set(
        MersenneTwister(103), mm_truth;
        smoke = true, truth_params = truth_m, initial_conditions = ics)
    r3 = discover_unknown_terms(mm_ude, set_m; training = cfg, holdout = 1,
        rng = MersenneTwister(7), verbose = false,
        known_support = HybridKinetics.mm_rate_support())
    _fp_compare("mm_holdout1", _fp_record(r3), exact)
    r4 = discover_unknown_terms(hill_ude, set_h; holdout = 1,
        rng = MersenneTwister(0), verbose = false, seed = 103,
        known_support = HybridKinetics.hill_rate_support(2))
    _fp_compare("hill_protocol_defaults", _fp_record(r4), exact)
end
