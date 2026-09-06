#!/usr/bin/env julia
# Research benchmark: the functional-identifiability diagnostic on the
# reference protocol (seed 103, nine experiments split 7/2) with five fixed
# training restarts (seeds 201 to 205). Prints the full diagnostic report:
# every restart including failures, pairwise rate and trajectory agreement,
# and the derived status. The status is a diagnostic and not an acceptance criterion.
# The script is not run in CI.
# Runtime: not measured for this release; five trainings of the reference
# protocol, so expect several times the runtime of the example.
# Run:  julia --project=. benchmark/functional_identifiability.jl

if !isdefined(Main, :BioDynaX)
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using BioDynaX
using BioDynaX:
                FUNCTIONAL_ID_RESTART_SEEDS,
                REFERENCE_PROTOCOL,
                assess_functional_identifiability,
                build_hill_recovery_network,
                format_functional_identifiability_diagnostic,
                generate_recovery_experiments,
                reference_protocol_experiment_split
using Random

truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
set = generate_recovery_experiments(
    MersenneTwister(REFERENCE_PROTOCOL.seed), truth_net, truth_params;
    tspan = REFERENCE_PROTOCOL.tspan,
    n_points = REFERENCE_PROTOCOL.n_points,
    noise_σ = REFERENCE_PROTOCOL.observation_noise)
split = reference_protocol_experiment_split(set)
diagnostic = assess_functional_identifiability(
    split, ude_net; restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS)
report = format_functional_identifiability_diagnostic(diagnostic)
println(report)
(; diagnostic, report)
