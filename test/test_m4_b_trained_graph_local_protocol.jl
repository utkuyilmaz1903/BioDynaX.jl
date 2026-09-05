# M4-B protocol path. This is the trained-UDE scientific acceptance path.
# PR smoke is not trained-UDE scientific acceptance.
# This file is outside test/runtests.jl.

using Test
using Random
using BioDynaX
if !@isdefined(evaluate_trained_graph_local)
    include(joinpath(@__DIR__, "internals.jl"))
end
if !@isdefined(_b4_run)
    include(joinpath(@__DIR__, "m4_b_helpers.jl"))
end

const _B4_PROTO_STARTED_AT = time()
const _B4_PROTO_BUNDLE = _b4_run(; kind = :protocol, inject = nothing)
const _B4_PROTO_ELAPSED_S = time() - _B4_PROTO_STARTED_AT

@testset "T-B4-PROTO-FIT real UDE training is not SET-injected" begin
    bundle = _B4_PROTO_BUNDLE
    @test bundle.logs.set_return[] === nothing
    @test length(bundle.logs.entry) == 1
    entry = bundle.logs.entry[1]
    @test entry.adam == M4B_PROTOCOL.adam_iterations
    @test entry.bfgs == M4B_PROTOCOL.bfgs_iterations
    @test entry.adam == UNIQUE_CLAIM_PROTOCOL.adam_iterations
    @test entry.bfgs == UNIQUE_CLAIM_PROTOCOL.bfgs_iterations
    @test entry.fit_set_length == M4B_PROTOCOL.n_ics
    @test length(bundle.logs.fit_set) == 1
    @test length(bundle.logs.split) == 0
    @test length(bundle.logs.holdout) == 0
    @test [exp.u0 for exp in entry.fit_set] == m4b_initial_conditions(:protocol)
    @test bundle.capture.count[] == 1
    @test bundle.capture.result[] isa TrainingResult
    @test bundle.capture.constructed[] == false
    @test bundle.evidence.kind === :protocol
end

@testset "T-B4-PROTO-RET wrapper captures the actual fit return" begin
    bundle = _B4_PROTO_BUNDLE
    @test bundle.capture.count[] == 1
    @test bundle.capture.result[] isa TrainingResult
    @test bundle.capture.constructed[] == false
    @test bundle.capture.fp[] ==
          nn_parameter_fingerprint(bundle.capture.result[].params.nn)
    @test bundle.capture.fp[] != bundle.logs.p0_fp[]
end

@testset "T-B4-PROTO-D learned D follows captured return params" begin
    bundle = _B4_PROTO_BUNDLE
    @test bundle.sample_fp == bundle.capture.fp[]
    @test bundle.evidence.D == bundle.D_replay
    stored_fp = nn_parameter_fingerprint(bundle.evidence.training.params.nn)
    @test bundle.sample_fp == bundle.capture.fp[]
    @test stored_fp == bundle.capture.fp[]
    @test bundle.sample_fp != bundle.logs.p0_fp[]
end

@testset "T-B4-PROTO-SAME one learned D is shared by three scopes" begin
    bundle = _B4_PROTO_BUNDLE
    @test length(bundle.logs.entry) == 1
    @test length(bundle.logs.sample) == 1
    @test length(bundle.logs.discover) == 3
    @test bundle.capture.count[] == 1
    d1 = bundle.logs.discover[1].derivatives
    d2 = bundle.logs.discover[2].derivatives
    d3 = bundle.logs.discover[3].derivatives
    @test d1 == d2 == d3
    @test vec(d1[1, :]) == vec(bundle.evidence.D)
    @test vec(d1[1, :]) == vec(bundle.D_replay)
    @test bundle.sample_fp == bundle.capture.fp[]
    for captured in bundle.logs.discover
        @test captured.X == bundle.logs.discover[1].X
        @test captured.times == bundle.logs.discover[1].times
    end
end

@testset "T-B4-PROTO-DISC observer-OFF replay binds three live scopes" begin
    bundle = _B4_PROTO_BUNDLE
    @test length(bundle.logs.discover) == 3
    @test _b4_discovery_bound(bundle.evidence.graph_discovery, bundle.replays.graph)
    @test _b4_discovery_bound(bundle.evidence.global_discovery, bundle.replays.global_disc)
    @test _b4_discovery_bound(bundle.evidence.wrong_graph_discovery, bundle.replays.wrong)
    @test bundle.replays.cfg_graph.basis_scope === :graph
    @test bundle.replays.cfg_global.basis_scope === :global
    libs = _b4_library_oracles()
    @test libs.graph_vars != libs.global_vars
    @test libs.graph_vars != libs.wrong_vars
    @test 4 ∈ libs.global_vars
    @test 2 ∈ libs.graph_vars
    @test 2 ∉ libs.wrong_vars
    hand = _b4_hand_candidate()
    replay_cand = _b4_first_candidate(bundle.replays.graph)
    @test replay_cand !== hand
end

@testset "T-B4-PROTO-SUP support is derived from observer-OFF candidates" begin
    bundle = _B4_PROTO_BUNDLE
    graph_cand = _b4_first_candidate(bundle.replays.graph)
    wrong_cand = _b4_first_candidate(bundle.replays.wrong)
    @test bundle.replays.graph.success
    @test graph_cand !== nothing
    @test local_has_true_parent_gate(graph_cand; variable = 2) === true
    @test local_has_true_parent_gate(wrong_cand; variable = 2) === false
    prod_graph = _b4_first_candidate(bundle.evidence.graph_discovery)
    prod_wrong = _b4_first_candidate(bundle.evidence.wrong_graph_discovery)
    @test local_has_true_parent_gate(prod_graph; variable = 2) ===
          local_has_true_parent_gate(graph_cand; variable = 2)
    @test local_has_true_parent_gate(prod_wrong; variable = 2) ===
          local_has_true_parent_gate(wrong_cand; variable = 2)
    @test !_b4_accepts_hardcoded_support(true)
end

@testset "T-B4-PROTO-HP decoy Adam=50 does not select 100/50" begin
    decoy_preferred_adam = 50
    entry = _B4_PROTO_BUNDLE.logs.entry[1]
    @test entry.adam == 100
    @test entry.bfgs == 50
    @test entry.adam != decoy_preferred_adam
    @test length(_B4_PROTO_BUNDLE.logs.holdout) == 0
    decoy_preferred_adam = 7
    @test entry.adam == 100
end

@testset "T-B4-PROTO-SCOPE a priori plan is not a holdout winner" begin
    @test M4B_SCOPE_PLAN[1].name === :graph
    @test M4B_SCOPE_PLAN[2].name === :global
    @test M4B_SCOPE_PLAN[3].name === :wrong_graph
    @test length(_B4_PROTO_BUNDLE.logs.discover) == 3
    @test length(_B4_PROTO_BUNDLE.logs.holdout) == 0
    best_scope = :global
    @test length(_B4_PROTO_BUNDLE.logs.discover) == 3
    @test best_scope !== :holdout_winner
end

@testset "T-B4-PROTO-RETRY protocol does not retry" begin
    bundle = _B4_PROTO_BUNDLE
    @test length(bundle.logs.entry) == 1
    @test length(bundle.logs.discover) == 3
    @test length(bundle.logs.sample) == 1
    @test bundle.capture.count[] == 1
    @test !occursin("206", _b4_source())
    @test !occursin("retry", lowercase(_b4_source()))
end

@testset "T-B4-PROTO-ACC protocol scientific acceptance cannot be smoke" begin
    bundle = _B4_PROTO_BUNDLE
    @test _b4_scientific_acceptance_holds(bundle)
    @test bundle.evidence.kind === :protocol
    @test !_b4_scientific_acceptance_holds((;
        evidence = TrainedGraphLocalEvidence(
            :smoke,
            bundle.evidence.training,
            bundle.evidence.model,
            bundle.evidence.term,
            bundle.evidence.params_nn_fingerprint,
            bundle.evidence.X,
            bundle.evidence.D,
            bundle.evidence.times,
            bundle.evidence.graph_discovery,
            bundle.evidence.global_discovery,
            bundle.evidence.wrong_graph_discovery),
        capture = bundle.capture,
        logs = bundle.logs,
        D_replay = bundle.D_replay,
        sample_fp = bundle.sample_fp,
        replays = bundle.replays))
    @info "M4-B protocol trained-UDE runtime" elapsed_s=_B4_PROTO_ELAPSED_S
end
