#!/usr/bin/env julia
# One F1 attempt on the same monomial library. No new atoms.
# Does not change RECOVERY_THRESHOLDS or the default Occam path.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using BioDynaX:
    hill_rate_truth, hill_rate_support, support_f1, rate_discovery_config,
    discover_unknown_rate, normalize_destruction_samples, RECOVERY_THRESHOLDS
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
    extras = String[]
    if result.success
        rec = BioDynaX.active_support(result.candidates[1])
        truth_keys = union(truth.numerator, truth.denominator)
        for key in union(rec.numerator, rec.denominator)
            key in truth_keys && continue
            push!(extras, string(key))
        end
    end
    return (;
        success = result.success,
        f1 = metrics === nothing ? 0.0 : metrics.combined.f1,
        recall = metrics === nothing ? 0.0 : metrics.combined.recall,
        extras,
        reaches_clean = metrics !== nothing &&
            metrics.combined.f1 ≥ RECOVERY_THRESHOLDS.support_f1_clean)
end

function main()
    r = collect(range(0.1, 2.0; length = 180))
    hill = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    # Extras that remain on the trained-NN rate (seed 103): constant and linear r.
    nn_like = hill .+ 0.04 .+ 0.04 .* r
    nn_norm, _ = normalize_destruction_samples(nn_like)
    rows = (
        exact_hill = discover_f1(hill, r),
        nn_like_extras = discover_f1(nn_like, r),
        nn_like_normalized = discover_f1(nn_norm, r),
    )
    println("UDE combined-F1 attempt (same library; no new atoms)")
    for (name, row) in pairs(rows)
        @printf "  %-22s F1=%.3f recall=%.3f extras=%s clean_gate=%s\n" string(name) row.f1 row.recall string(row.extras) string(row.reaches_clean)
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
