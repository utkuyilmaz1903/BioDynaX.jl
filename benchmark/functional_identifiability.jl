#!/usr/bin/env julia
# Research / nightly functional-identifiability benchmark.
#
# Calls the live assess_functional_identifiability entry on the locked
# M2 7/2 Hill split. Five fixed restart seeds (201, 202, 203, 204, 205).
# Q4 is a practical diagnostic. It is not a PR gate, not a unique-claim
# success gate, and not structural identifiability. Nightly scheduling
# is deferred to M7.
#
# Not a public API. The five-restart UDE workload is not part of ordinary
# PR tests. Failures stay in the five-attempt accounting. The printed
# status is not a success decision.

if !isdefined(Main, :BioDynaX)
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
end

using BioDynaX
using BioDynaX:
                FUNCTIONAL_ID_RESTART_SEEDS,
                UNIQUE_CLAIM_PROTOCOL,
                assess_functional_identifiability,
                build_hill_recovery_network,
                format_functional_identifiability_diagnostic,
                generate_recovery_experiments,
                unique_claim_experiment_split
using Random

truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
set = generate_recovery_experiments(
    MersenneTwister(UNIQUE_CLAIM_PROTOCOL.seed), truth_net, truth_params;
    tspan = UNIQUE_CLAIM_PROTOCOL.tspan,
    n_points = UNIQUE_CLAIM_PROTOCOL.n_points,
    noise_σ = UNIQUE_CLAIM_PROTOCOL.observation_noise)
split = unique_claim_experiment_split(set)
diagnostic = assess_functional_identifiability(
    split, ude_net; restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS)
report = format_functional_identifiability_diagnostic(diagnostic)
println(report)
(; diagnostic, report)
