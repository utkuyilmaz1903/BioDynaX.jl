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
# The comparison has three modes, chosen from the [environment] table:
#
# * exact: same Julia version, CPU model and BLAS as the recording. Every
#   value is compared as its 17-digit decimal string. This is the
#   bit-identity evidence.
# * tolerance: same Julia version, another CPU, BLAS or set of package
#   versions (CI resolves the newest compatible packages). The seeded random
#   streams are the same, so the runs start from the same data and initial
#   parameters and drift only in the last digits; numbers are compared at
#   the scale-relative tolerance `FP_CROSS_ENV_RTOL` (largest deviation over
#   the largest magnitude of the vector) and the deviations are logged.
#   Every support, flag, extra and the number of candidates stay exact; the
#   equation strings and the report are compared with their numeric literals
#   masked, because a coefficient printed at five significant digits can sit
#   on a rounding boundary (a drift of 1e-7 turned `-2.208` into `-2.2079`
#   on one CI runner) while the coefficients themselves are compared at the
#   tolerance. A change of the call sequence (seed, holdout, warm-up,
#   library) moves the numbers by orders of magnitude more than the
#   tolerance.
# * version: another Julia version. Julia does not keep seeded random
#   streams stable across versions, so the initial parameters and the noise
#   differ and no trained value is comparable (measured on 1.12.7 against
#   the 1.10.12 recording: the initial loss differs by an order of
#   magnitude). Only the shapes and the sampling grid are checked, and the
#   mode is logged.
using TOML, Printf

const FP_CROSS_ENV_RTOL = 1e-3

const _FP015 = TOML.parsefile(joinpath(@__DIR__, "fingerprints_015.toml"))
_fp17(x) = @sprintf("%.17g", Float64(x))
_fpvec(v) = String[_fp17(x) for x in Vector{Float64}(vec(v))]

function _fp_mode()
    env = _FP015["environment"]
    env["julia"] == string(VERSION) || return :version
    env["cpu"] == Sys.cpu_info()[1].model &&
        env["blas"] == string(HybridKinetics.LinearAlgebra.BLAS.get_config()) &&
        return :exact
    return :tolerance
end

# Largest deviation over the largest magnitude of the recorded vector (NaN
# pairs and matching Inf pairs ignored; a mismatched non-finite value is Inf).
function _fp_max_dev(recorded::Vector, actual::Vector)
    length(recorded) == length(actual) || return Inf
    xs = parse.(Float64, recorded)
    ys = parse.(Float64, actual)
    scale = max(maximum(abs, filter(isfinite, xs); init = 0.0), 1e-300)
    worst = 0.0
    for (x, y) in zip(xs, ys)
        if !isfinite(x) || !isfinite(y)
            (isnan(x) && isnan(y)) && continue
            x == y && continue
            return Inf
        end
        worst = max(worst, abs(x - y) / scale)
    end
    return worst
end
_fp_max_dev(recorded::String, actual::String) = _fp_max_dev([recorded], [actual])

# Text with every numeric literal replaced by `#`, so that two printouts of
# the same structure with last-digit differences compare equal. Digits that
# are structure, not values (a monomial's exponent after `^`, a state index
# after `[`), are kept.
function _fp_mask_numbers(s::AbstractString)
    replace(s, r"(?<![\^\[])-?\d+(\.\d+)?([eE][+-]?\d+)?" => "#")
end
_fp_mask_numbers(v::Vector) = String[_fp_mask_numbers(s) for s in v]

function _fp_same(recorded::Vector, actual::Vector, mode::Symbol)
    length(recorded) == length(actual) || return false
    mode === :exact && return recorded == actual
    mode === :version && return true
    return _fp_max_dev(recorded, actual) <= FP_CROSS_ENV_RTOL
end
function _fp_same(recorded::String, actual::String, mode::Symbol)
    _fp_same([recorded], [actual], mode)
end

function _fp_record(result::DiscoveryRun)
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

function _fp_compare(name::String, actual::Dict, mode::Symbol)
    recorded = _FP015[name]
    numeric = ("phys", "nn", "R", "D", "final_loss", "initial_loss", "collinearity",
        "condition_number", "data_residual", "data_residual_train",
        "data_residual_holdout")
    if mode !== :exact
        worst = Dict(key => _fp_max_dev(recorded[key], actual[key]) for key in numeric)
        for (i, (rc, ac)) in enumerate(zip(recorded["candidates"], actual["candidates"]))
            for key in keys(rc)
                worst["candidate_$(i)_$(key)"] = _fp_max_dev(rc[key], ac[key])
            end
        end
        @info "0.15 fingerprint deviation (largest, scale-relative)" run=name mode=mode tolerance=FP_CROSS_ENV_RTOL worst=maximum(values(worst)) per_field=sort(
            collect(worst); by = last, rev = true)
    end
    @testset "$name" begin
        for key in numeric
            @test _fp_same(recorded[key], actual[key], mode)
        end
        mode === :exact && @test recorded["nn_fingerprint"] == actual["nn_fingerprint"]
        if mode === :version
            # The sampling grid comes from the data settings, not from a
            # trained value, and the recorded shapes must still hold.
            @test _fp_max_dev(recorded["R"], actual["R"]) <= FP_CROSS_ENV_RTOL
            @test length(recorded["report"]) > 0 && length(actual["report"]) > 0
            return
        end
        @test recorded["discovery_success"] == actual["discovery_success"]
        @test recorded["retcode"] == actual["retcode"]
        @test recorded["unidentifiable_edge"] == actual["unidentifiable_edge"]
        mode === :exact ? (@test recorded["equations"] == actual["equations"]) :
        (@test _fp_mask_numbers(recorded["equations"]) ==
               _fp_mask_numbers(actual["equations"]))
        @test recorded["extras"] == actual["extras"]
        @test length(recorded["candidates"]) == length(actual["candidates"])
        for (rc, ac) in zip(recorded["candidates"], actual["candidates"])
            for key in keys(rc)
                @test _fp_same(rc[key], ac[key], mode)
            end
        end
        # The report is compared exactly with exact numbers; with drifting
        # last digits its text must still agree once numbers are masked.
        mode === :exact ? (@test recorded["report"] == actual["report"]) :
        (@test _fp_mask_numbers(recorded["report"]) == _fp_mask_numbers(actual["report"]))
    end
end

@testset "0.15 fingerprints reproduce through discover_unknown_terms" begin
    mode = _fp_mode()
    @info "0.15 fingerprint comparison" mode=mode recorded_on=_FP015["environment"]["commit"] recorded_julia=_FP015["environment"]["julia"] tolerance=FP_CROSS_ENV_RTOL
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
    _fp_compare("hill_holdout0", _fp_record(r1), mode)
    r2 = discover_unknown_terms(hill_ude, set_h; training = cfg, holdout = 1,
        rng = MersenneTwister(7), verbose = false, seed = 103)
    _fp_compare("hill_holdout1_seed103", _fp_record(r2), mode)
    truth_m = (k_prod = 0.9, vmax = 1.5, km = 0.4, k_rs = 1.0, k_r = 0.6)
    mm_truth = HybridKinetics.build_mm_recovery_network(; known = true)
    mm_ude = HybridKinetics.build_mm_recovery_network(; known = false)
    set_m = HybridKinetics.reference_protocol_experiment_set(
        MersenneTwister(103), mm_truth;
        smoke = true, truth_params = truth_m, initial_conditions = ics)
    r3 = discover_unknown_terms(mm_ude, set_m; training = cfg, holdout = 1,
        rng = MersenneTwister(7), verbose = false,
        known_support = HybridKinetics.mm_rate_support())
    _fp_compare("mm_holdout1", _fp_record(r3), mode)
    r4 = discover_unknown_terms(hill_ude, set_h; holdout = 1,
        rng = MersenneTwister(0), verbose = false, seed = 103,
        known_support = HybridKinetics.hill_rate_support(2))
    _fp_compare("hill_protocol_defaults", _fp_record(r4), mode)
end
