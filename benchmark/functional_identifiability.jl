#!/usr/bin/env julia
# Official M3 functional-identifiability benchmark entry point.
# Calls assess_functional_identifiability on the locked M2 7/2 split.
# Not a fabricated diagnostic table and not a unique-claim gate.

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
proto = UNIQUE_CLAIM_PROTOCOL
set = generate_recovery_experiments(
    MersenneTwister(proto.seed), truth_net, truth_params;
    tspan = proto.tspan,
    n_points = proto.n_points,
    noise_σ = proto.observation_noise)
split = unique_claim_experiment_split(set)
diagnostic = assess_functional_identifiability(
    split, ude_net; restart_seeds = FUNCTIONAL_ID_RESTART_SEEDS)
report = format_functional_identifiability_diagnostic(diagnostic)
println(report)
(; diagnostic, report)
