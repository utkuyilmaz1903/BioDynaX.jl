#!/usr/bin/env julia
# Same (R, D) + distractor z. Columns differ only by prior / backend.
# DataDrivenDiffEq is optional and is never a CI dependency.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using BioDynaX:
                run_recovery_suite, discover_equations, DiscoveryConfig,
                DataDrivenSparseSTLSQ, build_rate_ablation_network, hill_rate_truth,
                hill_rate_support, support_f1, denominator_violation_count,
                rate_rel_rmse, equation_to_function, support_uses_variable
using Printf
using Random

function _try_datadriven(X, dX, times)
    ext = Base.get_extension(BioDynaX, :BioDynaXDataDrivenSparseExt)
    ext === nothing && return nothing
    t0 = time()
    result = discover_equations(
        X, times, build_rate_ablation_network();
        derivatives = dX, targets = 1,
        config = DiscoveryConfig(
            backend = DataDrivenSparseSTLSQ(threshold = 1e-3),
            basis_scope = :global, seed = 4),
        verbose = false, strict = false)
    elapsed = time() - t0
    return (; success = result.success, elapsed, result)
end

report = run_recovery_suite(MersenneTwister(104); sections = (:ablation,))
a = report[:ablation]
println("Internal ablation: BioDynaX graph vs global (same y, only basis_scope differs)")
println("F1 after Occam is not the prior; library membership of z is.")
@printf "  %-22s %10s %10s %10s %10s %8s\n" "prior" "F1" "false_par" "den_viol" "rate_rmse" "sec"
@printf "  %-22s %10.3f %10s %10s %10.3f %8.3f\n" "BioDynaX graph" a.local_f1 string(a.local_false_parent) string(a.local_denominator_violations) a.local_rate_rmse a.local_time
@printf "  %-22s %10.3f %10s %10s %10.3f %8.3f\n" "BioDynaX global" a.global_f1 string(a.global_false_parent) string(a.global_denominator_violations) a.global_rate_rmse a.global_time

r = collect(range(0.1, 2.0; length = 180))
rng_ab = MersenneTwister(104)
D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
amp = max(maximum(abs, D), eps(Float64))
D_noisy = D .+ 0.005 .* amp .* randn(rng_ab, length(r))
z = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(rng_ab, length(r))
X_ab = permutedims(hcat(r, z))
dX_ab = vcat(reshape(D_noisy, 1, :), reshape(-0.5 .* z, 1, :))
X_ab, dX_ab = BioDynaX._permute_rate_samples(X_ab, dX_ab, 104)
times_ab = collect(range(0.0, 1.0; length = length(r)))
dd = _try_datadriven(X_ab, dX_ab, times_ab)
if dd === nothing
    println("  DataDrivenSparse global  skipped (package not loaded; not a CI dep)")
    println("  Frozen row in docs: unavailable (DataDrivenSparse resolve conflicts with this preview; not a win)")
else
    truth = hill_rate_support(2; variable = 1)
    cand = dd.success && !isempty(dd.result.candidates) ?
           dd.result.candidates[1] : nothing
    f1 = cand === nothing ? 0.0 :
         support_f1(cand, truth.numerator, truth.denominator).combined.f1
    fp = cand !== nothing && support_uses_variable(cand; variable = 2)
    den = cand === nothing ? typemax(Int) : denominator_violation_count(cand, X_ab)
    rmse = cand === nothing ? Inf :
           rate_rel_rmse(
        [equation_to_function(cand)(X_ab[:, j])
         for j in axes(X_ab, 2)], vec(dX_ab[1, :]))
    @printf "  %-22s %10.3f %10s %10s %10.3f %8.3f\n" "DataDrivenSparse global" f1 string(fp) string(den) rmse dd.elapsed
end
