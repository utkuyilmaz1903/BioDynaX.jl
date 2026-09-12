# Records the 0.15 numbers of the single-unknown-term path so that 0.16 can
# assert bit-identity. Run against the package at `main` (the base worktree).
using Pkg;
Pkg.activate(ENV["PKG_DIR"]);
using HybridKinetics, Random, TOML, Printf;
const HK = HybridKinetics
f17(x) = @sprintf("%.17g", Float64(x))
fvec(v) = String[f17(x) for x in Vector{Float64}(vec(v))]

function record(result)
    d = result.discovery
    function coeffs(c)
        c isa HK.ImplicitCandidate ?
        Dict("numerator" => fvec(c.numerator_coefficients),
            "denominator" => fvec(c.denominator_coefficients),
            "selection_frequency" => fvec(c.selection_frequency),
            "validation_error" => f17(c.validation_error)) :
        Dict("coefficients" => fvec(c.coefficients),
            "validation_error" => f17(c.validation_error))
    end
    cands = d.success ? [coeffs(c) for c in d.candidates] : Dict[]
    return Dict(
        "phys" => fvec(collect(result.params.phys)),
        "nn_fingerprint" => string(HK.nn_parameter_fingerprint(result.params.nn)),
        "nn" => fvec(HK._nn_fingerprint_values(result.params.nn)),
        "final_loss" => f17(result.training.final_loss),
        "initial_loss" => f17(result.training.initial_loss),
        "R" => fvec(result.samples.R), "D" => fvec(result.samples.D),
        "equations" => d.success ?
                       (d.equations isa AbstractString ? [String(d.equations)] :
                        String[string(e) for e in d.equations]) : String[],
        "candidates" => cands,
        "discovery_success" => d.success, "retcode" => string(d.retcode),
        "collinearity" => f17(result.identifiability.collinearity),
        "condition_number" => f17(result.identifiability.condition_number),
        "unidentifiable_edge" => result.identifiability.unidentifiable_edge,
        "data_residual" => f17(result.residuals.data_residual),
        "data_residual_train" => f17(result.residuals.data_residual_train),
        "data_residual_holdout" => f17(result.residuals.data_residual_holdout),
        "extras" => result.extras === nothing ? String[] : result.extras,
        "report" => report_unknown_term(result))
end

out = Dict{String, Any}()
out["environment"] = Dict(
    "julia" => string(VERSION), "package" => string(HK.PACKAGE_VERSION),
    "cpu" => Sys.cpu_info()[1].model, "blas" => string(HK.LinearAlgebra.BLAS.get_config()),
    "recorded" => "2026-09-11", "commit" => ENV["PKG_SHA"])

cfg = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6)
truth_h = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
ics = [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10]]
hill_truth = HK.build_hill_recovery_network(; known = true, hill_order = 2)
hill_ude = HK.build_hill_recovery_network(; known = false, hill_order = 2)
set_h = HK.reference_protocol_experiment_set(
    MersenneTwister(103), hill_truth; smoke = true,
    truth_params = truth_h, initial_conditions = ics)
t = @elapsed r1 = discover_unknown_term(hill_ude, set_h; training = cfg, holdout = 0,
    rng = MersenneTwister(7), verbose = false, known_support = HK.hill_rate_support(2))
println("hill holdout0 ", round(t; digits = 1), " s");
out["hill_holdout0"] = record(r1);
t = @elapsed r2 = discover_unknown_term(hill_ude, set_h; training = cfg, holdout = 1,
    rng = MersenneTwister(7), verbose = false, seed = 103)
println("hill holdout1 ", round(t; digits = 1), " s");
out["hill_holdout1_seed103"] = record(r2);

truth_m = (k_prod = 0.9, vmax = 1.5, km = 0.4, k_rs = 1.0, k_r = 0.6)
mm_truth = HK.build_mm_recovery_network(; known = true)
mm_ude = HK.build_mm_recovery_network(; known = false)
set_m = HK.reference_protocol_experiment_set(MersenneTwister(103), mm_truth; smoke = true,
    truth_params = truth_m, initial_conditions = ics)
t = @elapsed r3 = discover_unknown_term(mm_ude, set_m; training = cfg, holdout = 1,
    rng = MersenneTwister(7), verbose = false, known_support = HK.mm_rate_support())
println("mm holdout1 ", round(t; digits = 1), " s");
out["mm_holdout1"] = record(r3);

# the reference-protocol smoke path with the protocol defaults (Adam 100, BFGS 50)
t = @elapsed r4 = discover_unknown_term(hill_ude, set_h; holdout = 1,
    rng = MersenneTwister(0), verbose = false, seed = 103,
    known_support = HK.hill_rate_support(2))
println("hill protocol-defaults smoke ", round(t; digits = 1), " s");
out["hill_protocol_defaults"] = record(r4);

open(ENV["OUT"], "w") do io
    TOML.print(io, out)
end
println("written ", ENV["OUT"])
