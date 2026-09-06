#!/usr/bin/env julia
# Replays the nuisance terms left by trained-model discovery (a constant and
# a linear term) on the same monomial library, with subset selection and
# scale normalization, to document whether they can be removed without
# changing the library. No new atoms are added to the library. Does not train
# a model and does not change any threshold. Prints F1, recall and extras per
# variant. Not run in CI.
# Runtime: about a minute.
# Run:  julia --project=. benchmark/ude_f1_attempt.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using BioDynaX:
                hill_rate_truth, hill_rate_support, support_f1, rate_discovery_config,
                discover_unknown_rate, normalize_destruction_samples, RECOVERY_THRESHOLDS,
                REFERENCE_PROTOCOL, REFERENCE_PROTOCOL_F1_ATTEMPT
using Printf

function discover_f1(D, r; seed = 103)
    result = discover_unknown_rate(
        reshape(r, 1, :), collect(range(0.0, 1.0; length = length(r))),
        reshape(vec(D), 1, :);
        config = rate_discovery_config(bootstrap = 0, seed = seed),
        verbose = false, strict = false)
    truth = hill_rate_support(2)
    metrics = result.success ?
              support_f1(result.candidates[1], truth.numerator, truth.denominator) :
              nothing
    extras = result.success ?
             BioDynaX.reference_protocol_discovery_extras(result.candidates[1]) :
             String[]
    f1 = metrics === nothing ? 0.0 : metrics.combined.f1
    return (;
        success = result.success,
        f1,
        recall = metrics === nothing ? 0.0 : metrics.combined.recall,
        extras,
        reaches_clean = BioDynaX.reference_protocol_f1_reaches_analytical_threshold(f1),
        meets_skeleton = BioDynaX.reference_protocol_f1_meets_skeleton_floor(f1),
        verdict = BioDynaX.reference_protocol_f1_attempt_verdict(;
            extras,
            reaches_clean = BioDynaX.reference_protocol_f1_reaches_analytical_threshold(f1)))
end

function main()
    proto = REFERENCE_PROTOCOL
    attempt = REFERENCE_PROTOCOL_F1_ATTEMPT
    r = collect(range(0.1, 2.0; length = 180))
    hill = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    # Extras that remain on the trained-NN rate (seed 103): constant and linear r.
    nn_like = hill .+ 0.04 .+ 0.04 .* r
    nn_norm, _ = normalize_destruction_samples(nn_like)
    rows = (
        exact_hill = discover_f1(hill, r),
        nn_like_extras = discover_f1(nn_like, r),
        nn_like_normalized = discover_f1(nn_norm, r)
    )
    println("UDE combined-F1 attempt (same library; no new atoms)")
    println("  REFERENCE_PROTOCOL_F1_ATTEMPT is_protocol=$(attempt.is_protocol)",
        " trains_ude=$(attempt.trains_ude) n_ics=$(attempt.n_ics)",
        " new_atoms=$(attempt.new_atoms)")
    println("  not the recovery protocol: seed=$(proto.seed) n_ics=$(proto.n_ics)",
        " n_points=$(proto.n_points) (smoke is $(proto.smoke_n_ics) IC / ",
        "$(proto.smoke_n_points) points)")
    println("  skeleton floor support_f1_ude=$(RECOVERY_THRESHOLDS.support_f1_ude)",
        " analytical threshold support_f1_clean=$(RECOVERY_THRESHOLDS.support_f1_clean)")
    for (name, row) in pairs(rows)
        @printf "  %-22s F1=%.3f recall=%.3f extras=%s clean_threshold=%s skeleton=%s verdict=%s\n" string(name) row.f1 row.recall string(row.extras) string(row.reaches_clean) string(row.meets_skeleton) string(row.verdict)
    end
    reached = rows.nn_like_extras.reaches_clean || rows.nn_like_normalized.reaches_clean
    if reached
        println("  RESULT: extras dropped on this surrogate. Re-open the",
            " canonical-Hill-from-NN sentence only after the seed-103",
            " UDE protocol also holds support_f1_clean.")
    else
        println("  RESULT: extras remain after Occam and scale-normalization.",
            " Public claim stays recall + hybrid residual versus data.",
            " support_f1_ude is not tightened. support_f1_clean is not loosened.")
    end
    return rows
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
