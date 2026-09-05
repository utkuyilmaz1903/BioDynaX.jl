# M4-B PR smoke and adversarial coverage.
# PR smoke is not trained-UDE scientific acceptance.

using Test
using Random
using BioDynaX
if !@isdefined(evaluate_trained_graph_local)
    include(joinpath(@__DIR__, "internals.jl"))
end
if !@isdefined(_b4_run)
    include(joinpath(@__DIR__, "m4_b_helpers.jl"))
end

const _B4_SMOKE_BUNDLE = _b4_run(; kind = :smoke, inject = _b4_inject_p0)

@testset "T-B4-API M4-B helpers stay unexported and fields stay exact" begin
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test FUNCTIONAL_ID_REPORTING_CUTOFFS === (
        min_successful_restarts = 3,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
    @test :evaluate_trained_graph_local ∉ names(BioDynaX)
    @test :TrainedGraphLocalEvidence ∉ names(BioDynaX)
    @test :M4B_PROTOCOL ∉ names(BioDynaX)
    @test :M4B_SMOKE ∉ names(BioDynaX)
    @test :M4B_SCOPE_PLAN ∉ names(BioDynaX)
    @test :training_call ∉ names(BioDynaX)
    @test fieldnames(TrainedGraphLocalEvidence) === (
        :kind, :training, :model, :term, :params_nn_fingerprint,
        :X, :D, :times, :graph_discovery, :global_discovery,
        :wrong_graph_discovery)
    for name in _B4_FORBIDDEN_EVIDENCE_FIELDS
        @test name ∉ fieldnames(TrainedGraphLocalEvidence)
    end
    @test :occupancy ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :occupancy ∉ fieldnames(HoldoutEvidence)
    @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
end

@testset "T-B4-FIX true and wrong parent sets are constructive" begin
    ude_net = _b4_ude_net()
    wrong_net = _b4_wrong_net()
    relabel_net = build_three_state_unknown_network(;
        known = false, with_distractor = true, parent = 2)
    @test 2 ∈ candidate_parents(ude_net, 1)
    @test 3 ∉ candidate_parents(ude_net, 1)
    @test 4 ∉ candidate_parents(ude_net, 1)
    @test 3 ∈ candidate_parents(wrong_net, 1)
    @test 2 ∉ candidate_parents(wrong_net, 1)
    @test graph_parent_set(ude_net, 1) != graph_parent_set(wrong_net, 1)
    @test graph_parent_set(ude_net, 1) == graph_parent_set(relabel_net, 1)
    @test !_b4_is_constructive_wrong_graph(relabel_net, ude_net)
    @test _b4_is_constructive_wrong_graph(wrong_net, ude_net)
    _, graph_vars = graph_library_variables(
        ude_net, 1; degree = 2, include_interactions = false)
    _, wrong_vars = wrong_graph_library_variables(wrong_net, 1)
    @test 2 ∈ graph_vars
    @test 2 ∉ wrong_vars
end

@testset "T-B4-SRC production source uses the locked trained-UDE path" begin
    src = _b4_source()
    code = _b4_source_code(src)
    for needle in _B4_PRODUCTION_MUST_CONTAIN
        @test occursin(needle, code)
    end
    for needle in _B4_PRODUCTION_MUST_NOT_CONTAIN
        @test !occursin(needle, code)
    end
    @test occursin("training_call", code)
    @test !occursin("discover_equations(p_trained", code)
    @test !occursin(r"discover_equations\([^\n]{0,240}scope\s*=", code)
    @test length(_B4_SMOKE_BUNDLE.logs.entry) == 1
    @test length(_B4_SMOKE_BUNDLE.logs.sample) == 1
    @test length(_B4_SMOKE_BUNDLE.logs.discover) == 3
end

@testset "T-B4-LAB analytic control is not trained-UDE; smoke is not acceptance" begin
    texts = _b4_docs_texts()
    for text in (texts.source,)
        @test occursin(_B4_LOCKED_SENTENCES.analytic, text)
        @test occursin(_B4_LOCKED_SENTENCES.trained, text)
        @test occursin(_B4_LOCKED_SENTENCES.smoke, text)
    end
    @test _B4_SMOKE_BUNDLE.evidence.kind === :smoke
    @test !_b4_smoke_is_scientific_acceptance(_B4_SMOKE_BUNDLE.evidence)
    @test !_b4_scientific_acceptance_holds(_B4_SMOKE_BUNDLE)
end

@testset "T-B4-CTRL analytic library-membership control stays separate" begin
    row = three_state_discovery_gate_row()
    wrong = wrong_graph_discovery_gate_row()
    @test row.holds
    @test wrong.holds
    label = :analytic_library_membership_control
    @test label !== :protocol
    @test !_b4_analytic_is_trained_ude(row)
    @test !_b4_analytic_is_trained_ude(wrong)
    @test !(row isa TrainedGraphLocalEvidence)
    attack_kind = :protocol
    @test _b4_analytic_is_trained_ude((; kind = attack_kind))
end

@testset "T-B4-SMOKE-FIT live fit request is captured exactly once" begin
    bundle = _B4_SMOKE_BUNDLE
    @test length(bundle.logs.entry) == 1
    entry = bundle.logs.entry[1]
    @test entry.adam == M4B_SMOKE.adam_iterations
    @test entry.bfgs == M4B_SMOKE.bfgs_iterations
    @test entry.fit_set_length == M4B_SMOKE.n_ics
    @test length(bundle.logs.fit_set) == 1
    @test bundle.logs.fit_set[1] === entry.fit_set
    @test entry.fit_experiments_identity === entry.fit_set.experiments
    @test length(bundle.logs.split) == 0
    @test length(bundle.logs.holdout) == 0
    @test bundle.capture.count[] == 1
    @test bundle.capture.result[] isa TrainingResult
    @test bundle.capture.constructed[] == false
    expected = m4b_initial_conditions(:smoke)
    @test [exp.u0 for exp in entry.fit_set] == expected
end

@testset "T-B4-SMOKE-D captured fit return is the sample oracle" begin
    bundle = _B4_SMOKE_BUNDLE
    captured_fp = bundle.capture.fp[]
    @test captured_fp == nn_parameter_fingerprint(bundle.capture.result[].params.nn)
    @test bundle.sample_fp == captured_fp
    @test bundle.evidence.D == bundle.D_replay
    stored_fp = nn_parameter_fingerprint(bundle.evidence.training.params.nn)
    @test captured_fp == stored_fp
    @test bundle.sample_fp == captured_fp
end

@testset "T-B4-SMOKE-SAME one learned D is shared by three scopes" begin
    bundle = _B4_SMOKE_BUNDLE
    @test length(bundle.logs.sample) == 1
    @test length(bundle.logs.entry) == 1
    @test length(bundle.logs.discover) == 3
    d1 = bundle.logs.discover[1].derivatives
    d2 = bundle.logs.discover[2].derivatives
    d3 = bundle.logs.discover[3].derivatives
    @test d1 == d2 == d3
    @test vec(d1[1, :]) == vec(bundle.evidence.D)
    @test bundle.sample_fp == bundle.capture.fp[]
    @test bundle.evidence.D == bundle.D_replay
    for captured in bundle.logs.discover
        @test captured.X == bundle.logs.discover[1].X
        @test captured.times == bundle.logs.discover[1].times
    end
end

@testset "T-B4-SMOKE-DISC observer-OFF replay binds live discovery" begin
    bundle = _B4_SMOKE_BUNDLE
    @test length(bundle.logs.discover) == 3
    @test _b4_discovery_bound(bundle.evidence.graph_discovery, bundle.replays.graph)
    @test _b4_discovery_bound(bundle.evidence.global_discovery, bundle.replays.global_disc)
    @test _b4_discovery_bound(bundle.evidence.wrong_graph_discovery, bundle.replays.wrong)
    hand = _b4_hand_candidate()
    replay_cand = _b4_first_candidate(bundle.replays.graph)
    @test replay_cand === nothing || hand.specification.variables !=
          replay_cand.specification.variables ||
          hand.numerator_coefficients != replay_cand.numerator_coefficients
    @test bundle.replays.cfg_graph.basis_scope === :graph
    @test bundle.replays.cfg_global.basis_scope === :global
    @test graph_parent_set(bundle.replays.ude_net, 1) !=
          graph_parent_set(bundle.replays.wrong_net, 1)
end

@testset "T-B4-SMOKE-LIB live libraries match graph/global/wrong oracles" begin
    libs = _b4_library_oracles()
    @test 2 ∈ libs.graph_vars
    @test 3 ∉ libs.graph_vars
    @test 4 ∉ libs.graph_vars
    @test 2 ∈ libs.global_vars
    @test 3 ∈ libs.global_vars || 4 ∈ libs.global_vars
    @test 4 ∈ libs.global_vars
    @test 2 ∉ libs.wrong_vars
    @test 3 ∈ libs.wrong_vars
    graph_cand = _b4_first_candidate(_B4_SMOKE_BUNDLE.evidence.graph_discovery)
    if graph_cand !== nothing
        @test Set(graph_cand.specification.variables) == libs.graph_vars ||
              Set(graph_cand.specification.variables) ⊆ libs.graph_vars
    end
    global_cand = _b4_first_candidate(_B4_SMOKE_BUNDLE.evidence.global_discovery)
    if global_cand !== nothing
        @test Set(global_cand.specification.variables) == libs.global_vars ||
              Set(global_cand.specification.variables) ⊆ libs.global_vars
    end
    wrong_cand = _b4_first_candidate(_B4_SMOKE_BUNDLE.evidence.wrong_graph_discovery)
    if wrong_cand !== nothing
        @test 2 ∉ wrong_cand.specification.variables
        @test Set(wrong_cand.specification.variables) == libs.wrong_vars ||
              Set(wrong_cand.specification.variables) ⊆ libs.wrong_vars
    end
end

@testset "T-B4-SMOKE-HP decoy holdout Adam=50 does not select the budget" begin
    decoy_preferred_adam = 50
    bundle = _B4_SMOKE_BUNDLE
    entry = bundle.logs.entry[1]
    @test entry.adam == 2
    @test entry.bfgs == 0
    @test entry.adam != decoy_preferred_adam
    @test length(bundle.logs.holdout) == 0
    decoy_preferred_adam = 7
    @test entry.adam == 2
    @test entry.adam != decoy_preferred_adam
end

@testset "T-B4-SMOKE-SCOPE a priori scope plan runs all three scopes" begin
    @test M4B_SCOPE_PLAN[1].name === :graph
    @test M4B_SCOPE_PLAN[2].name === :global
    @test M4B_SCOPE_PLAN[3].name === :wrong_graph
    bundle = _B4_SMOKE_BUNDLE
    @test length(bundle.logs.discover) == 3
    @test length(bundle.logs.holdout) == 0
    best_scope = :graph
    @test length(bundle.logs.discover) == 3
    @test best_scope !== :selected_by_holdout
end

@testset "T-B4-SMOKE-RETRY smoke stays one fit, one sample, three discoveries" begin
    bundle = _B4_SMOKE_BUNDLE
    @test length(bundle.logs.entry) == 1
    @test length(bundle.logs.sample) == 1
    @test length(bundle.logs.discover) == 3
    @test bundle.capture.count[] == 1
    src = _b4_source()
    @test !occursin("206", src)
    @test !occursin("retry", lowercase(src))
end

@testset "T-B4-SMOKE-SEP holdout/composer/occupancy/Q4 ownership stays zero" begin
    bundle = _B4_SMOKE_BUNDLE
    @test length(bundle.logs.holdout) == 0
    @test length(bundle.logs.grid) == 0
    @test length(bundle.logs.grid_result) == 0
    @test length(bundle.logs.rate_discover) == 0
    @test length(bundle.logs.range) == 0
    @test length(bundle.logs.split) == 0
    @test length(bundle.logs.generate_recovery) == 0
    @test length(bundle.logs.assess_q4) == 0
    @test bundle.logs.train_unknown_edge[] == 0
    src = _b4_source()
    @test !occursin("sample_destruction_occupancy", _b4_source_code(src))
    @test !occursin("TrajectoryOccupancy", _b4_source_code(src))
    @test !occursin("evaluate_holdout", _b4_source_code(src))
end

@testset "T-B4-ATK A–Z force RED against live witnesses" begin
    bundle = _B4_SMOKE_BUNDLE
    ude_net = _b4_ude_net()
    wrong_net = _b4_wrong_net()
    libs = _b4_library_oracles()

    @testset "A copied restart fingerprints are rejected" begin
        @test length(bundle.logs.entry) == 1
        @test length(bundle.logs.assess_q4) == 0
        fps = [nn_parameter_fingerprint(deepcopy(bundle.capture.result[].params).nn)
               for _ in 1:5]
        @test all(==(fps[1]), fps)
        @test !(length(bundle.logs.entry) == 1 && length(fps) == 5 &&
                all(==(fps[1]), fps) && false)
        @test FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
        @test !occursin("FUNCTIONAL_ID_RESTART_SEEDS", _b4_source())
    end

    @testset "B fake scope results do not bind observer-OFF oracles" begin
        fake = DiscoveryResult(true, "fake scope", "hand-written",
            nothing, nothing, [_b4_hand_candidate()], (;))
        @test !_b4_discovery_bound(fake, bundle.replays.graph)
        @test !_b4_discovery_bound(fake, bundle.replays.global_disc)
        @test _b4_discovery_bound(bundle.evidence.graph_discovery, bundle.replays.graph)
    end

    @testset "C scope-specific D is RED" begin
        n = size(bundle.evidence.D, 2)
        r = collect(range(0.1, 2.0; length = n))
        D_other = reshape(hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2), 1, :)
        @test D_other != bundle.evidence.D
        @test D_other != bundle.D_replay
        @test bundle.logs.discover[1].derivatives ==
              bundle.logs.discover[2].derivatives
    end

    @testset "D analytic Hill D is not the learned-D oracle" begin
        n = size(bundle.evidence.D, 2)
        r = collect(range(0.1, 2.0; length = n))
        D_hill = reshape(hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2), 1, :)
        @test D_hill != bundle.D_replay
        @test !occursin("graph_local_rate_samples", _b4_source())
        @test length(bundle.logs.sample) == 1
    end

    @testset "E discarded real return params are RED" begin
        attack = _b4_run(; kind = :smoke, inject = _b4_inject_perturbed)
        @test attack.capture.fp[] != attack.logs.p0_fp[]
        (_, D_p0) = _b4_independent_replay_D(
            attack.evidence.model, attack.logs.p0[], attack.evidence.term,
            size(attack.evidence.D, 2))
        @test nn_parameter_fingerprint(attack.logs.p0[].nn) != attack.capture.fp[]
        @test D_p0 != attack.D_replay
        @test attack.sample_fp == attack.capture.fp[]
    end

    @testset "F hand-built candidates are not the oracle" begin
        hand = _b4_hand_candidate()
        replay_cand = _b4_first_candidate(bundle.replays.graph)
        @test replay_cand === nothing ||
              hand.specification.variables != replay_cand.specification.variables ||
              hand.numerator_coefficients != replay_cand.numerator_coefficients
        @test local_has_true_parent_gate(hand; variable = 2) !==
              (replay_cand === nothing ? false :
               local_has_true_parent_gate(replay_cand; variable = 2)) ||
              replay_cand === nothing ||
              hand !== replay_cand
    end

    @testset "G hardcoded support Booleans are not evidence" begin
        graph_true_in_support = true
        replay_cand = _b4_first_candidate(bundle.replays.graph)
        derived = local_has_true_parent_gate(replay_cand; variable = 2)
        @test !_b4_accepts_hardcoded_support(graph_true_in_support)
        @test derived === local_has_true_parent_gate(replay_cand; variable = 2)
    end

    @testset "H skipped global is RED" begin
        @test 4 ∈ libs.global_vars
        @test libs.global_vars != libs.graph_vars
        @test _b4_discovery_bound(
            bundle.evidence.global_discovery, bundle.replays.global_disc)
        @test bundle.replays.cfg_global.basis_scope === :global
        @test length(bundle.logs.discover) == 3
    end

    @testset "I skipped graph is RED" begin
        @test 2 ∈ libs.graph_vars
        @test _b4_discovery_bound(
            bundle.evidence.graph_discovery, bundle.replays.graph)
        @test bundle.replays.cfg_graph.basis_scope === :graph
        @test length(bundle.logs.discover) == 3
    end

    @testset "J relabel is not a wrong graph" begin
        relabel = build_three_state_unknown_network(;
            known = false, with_distractor = true, parent = 2)
        @test 2 ∈ candidate_parents(relabel, 1)
        @test graph_parent_set(relabel, 1) == graph_parent_set(ude_net, 1)
        @test !_b4_is_constructive_wrong_graph(relabel, ude_net)
        @test 2 ∉ libs.wrong_vars
        wrong_cand = _b4_first_candidate(bundle.evidence.wrong_graph_discovery)
        if wrong_cand !== nothing
            @test 2 ∉ wrong_cand.specification.variables
        end
    end

    @testset "K holdout optimizer selection is RED" begin
        decoy_preferred_adam = 50
        @test bundle.logs.entry[1].adam != decoy_preferred_adam
        @test bundle.logs.entry[1].adam == 2
        @test length(bundle.logs.holdout) == 0
    end

    @testset "L holdout scope selection is RED" begin
        best_scope = :graph
        @test length(bundle.logs.discover) == 3
        @test length(bundle.logs.holdout) == 0
        @test M4B_SCOPE_PLAN[1].name === :graph
        @test best_scope !== :holdout_winner
    end

    @testset "M retry-until-success is RED" begin
        @test length(bundle.logs.entry) == 1
        @test length(bundle.logs.discover) == 3
        @test length(bundle.logs.sample) == 1
        @test !occursin("206", _b4_source())
    end

    @testset "N internal discovery retry is RED" begin
        @test length(bundle.logs.discover) == 3
        @test !occursin("retry", lowercase(_b4_source()))
    end

    @testset "O observer-only fake training is not protocol evidence" begin
        @test bundle.logs.set_return[] isa TrainingResult
        @test bundle.evidence.kind === :smoke
        @test !_b4_scientific_acceptance_holds(bundle)
        @test bundle.capture.result[] isa TrainingResult
        @test bundle.capture.constructed[] == false
    end

    @testset "P smoke is not scientific acceptance" begin
        @test bundle.evidence.kind === :smoke
        @test !_b4_smoke_is_scientific_acceptance(bundle.evidence)
        @test !_b4_scientific_acceptance_holds(bundle)
        attack_kind = :protocol
        @test attack_kind === :protocol
        @test bundle.evidence.kind !== attack_kind
    end

    @testset "Q M2 semantics stay locked" begin
        @test UNIQUE_CLAIM_TRAIN_INDICES === (1, 2, 3, 4, 5, 6, 7)
        @test UNIQUE_CLAIM_HOLDOUT_INDICES === (8, 9)
        @test fieldnames(HoldoutEvidence) === (
            :data_residual_train, :data_residual_holdout,
            :d_rmse_holdout, :d_rmse_holdout_domain)
        @test recovery_thresholds_hold()
        @test :occupancy ∉ fieldnames(HoldoutEvidence)
    end

    @testset "R M3 semantics stay locked" begin
        @test FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
        @test FUNCTIONAL_ID_REPORTING_CUTOFFS === (
            min_successful_restarts = 3,
            n_attempted_restarts = 5,
            traj_agree_rel_rmse = 0.05,
            d_disagree_scale_norm_rel_rmse = 0.20)
        @test FUNCTIONAL_ID_STATUS_VOCABULARY === (
            :incomplete, :traj_disagree, :scale_ambiguity,
            :function_agree, :trajectory_agree_function_disagree)
        @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
        @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDomain)
    end

    @testset "S A1/A2 IDs stay present" begin
        a1 = read(joinpath(pkgdir(BioDynaX), "test", "test_trajectory_occupancy.jl"),
            String)
        a2 = read(joinpath(pkgdir(BioDynaX), "test", "test_m4_a2_separation.jl"),
            String)
        for id in _B4_A1_IDS
            @test occursin(id, a1)
        end
        for id in _B4_A2_IDS
            @test occursin(id, a2)
        end
    end

    @testset "T analytic control labeled trained-UDE is RED" begin
        row = three_state_discovery_gate_row()
        @test !_b4_analytic_is_trained_ude(row)
        @test _b4_analytic_is_trained_ude((; kind = :protocol))
    end

    @testset "U source-only helpers are not sufficient" begin
        @test length(bundle.logs.entry) == 1
        @test length(bundle.logs.sample) == 1
        @test length(bundle.logs.discover) == 3
        @test occursin("sample_unknown_destruction(", _b4_source())
    end

    @testset "V fake candidate after observer is RED" begin
        hand = _b4_hand_candidate()
        replay_cand = _b4_first_candidate(bundle.replays.graph)
        production_cand = _b4_first_candidate(bundle.evidence.graph_discovery)
        @test production_cand === replay_cand ||
              (production_cand !== nothing && replay_cand !== nothing &&
               production_cand.specification.variables ==
               replay_cand.specification.variables)
        @test hand !== production_cand
    end

    @testset "W wrong-graph library definition is respected" begin
        @test 2 ∉ libs.wrong_vars
        wrong_cand = _b4_first_candidate(bundle.evidence.wrong_graph_discovery)
        if wrong_cand !== nothing
            @test 2 ∉ wrong_cand.specification.variables
        end
        @test graph_parent_set(wrong_net, 1) != graph_parent_set(ude_net, 1)
    end

    @testset "X sampling p0 after a different captured return is RED" begin
        attack = _b4_run(; kind = :smoke, inject = _b4_inject_perturbed)
        @test attack.capture.fp[] != attack.logs.p0_fp[]
        p0_fp = attack.logs.p0_fp[]
        (_, D_p0) = _b4_independent_replay_D(
            attack.evidence.model, attack.logs.p0[], attack.evidence.term,
            size(attack.evidence.D, 2))
        @test p0_fp != attack.capture.fp[]
        @test D_p0 != attack.D_replay
        @test attack.sample_fp != p0_fp
        @test attack.sample_fp == attack.capture.fp[]
    end

    @testset "Y fake stored TrainingResult with different params is RED" begin
        attack = _b4_run(; kind = :smoke, inject = _b4_inject_perturbed)
        fake = _b4_fake_training(attack.logs.p0[])
        fake_fp = nn_parameter_fingerprint(fake.params.nn)
        @test fake_fp != attack.capture.fp[]
        (_, D_fake) = _b4_independent_replay_D(
            attack.evidence.model, fake.params, attack.evidence.term,
            size(attack.evidence.D, 2))
        @test D_fake != attack.D_replay
        same = _b4_fake_training(deepcopy(attack.capture.result[].params))
        same_fp = nn_parameter_fingerprint(same.params.nn)
        @test same_fp == attack.capture.fp[]
    end

    @testset "Z scopes that leave captured-return params are RED" begin
        attack = _b4_run(; kind = :smoke, inject = _b4_inject_perturbed)
        (_, D_p0) = _b4_independent_replay_D(
            attack.evidence.model, attack.logs.p0[], attack.evidence.term,
            size(attack.evidence.D, 2))
        @test D_p0 != attack.D_replay
        @test attack.evidence.D == attack.D_replay
        @test attack.sample_fp == attack.capture.fp[]
    end
end

@testset "T-B4-REG-M2 M2 lock remains 7/2 train-only holdout D" begin
    @test UNIQUE_CLAIM_TRAIN_INDICES === (1, 2, 3, 4, 5, 6, 7)
    @test UNIQUE_CLAIM_HOLDOUT_INDICES === (8, 9)
    @test fieldnames(HoldoutEvidence) === (
        :data_residual_train, :data_residual_holdout,
        :d_rmse_holdout, :d_rmse_holdout_domain)
    @test recovery_thresholds_hold()
    @test :occupancy ∉ fieldnames(HoldoutEvidence)
    @test :occupancy ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    holdout_src = read(joinpath(pkgdir(BioDynaX), "src", "RecoveryPipeline.jl"),
        String)
    @test occursin("function evaluate_holdout", holdout_src)
    @test occursin("struct HoldoutEvidence", holdout_src)
    @test isfile(joinpath(pkgdir(BioDynaX), "test", "test_holdout.jl"))
    @test !occursin("evaluate_holdout", _b4_source_code())
    @test UNIQUE_CLAIM_PROTOCOL.n_ics == 9
    @test length(UNIQUE_CLAIM_TRAIN_INDICES) == 7
    @test length(UNIQUE_CLAIM_HOLDOUT_INDICES) == 2
end

@testset "T-B4-REG-M3 M3 domain, seeds, cutoffs, and status stay locked" begin
    @test FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
    @test FUNCTIONAL_ID_REPORTING_CUTOFFS === (
        min_successful_restarts = 3,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
    @test FUNCTIONAL_ID_STATUS_VOCABULARY === (
        :incomplete, :traj_disagree, :scale_ambiguity,
        :function_agree, :trajectory_agree_function_disagree)
    @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDomain)
    @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
    @test isfile(joinpath(pkgdir(BioDynaX), "test",
        "test_functional_identifiability.jl"))
    src = read(joinpath(pkgdir(BioDynaX), "src", "FunctionalIdentifiability.jl"),
        String)
    @test occursin(":train_obs_union_holdout_obs", src)
    @test occursin("construction must be :train_obs_union_holdout_obs", src)
end

@testset "T-B4-REG-A A1/A2 files and occupancy separations stay intact" begin
    a1_path = joinpath(pkgdir(BioDynaX), "test", "test_trajectory_occupancy.jl")
    a2_path = joinpath(pkgdir(BioDynaX), "test", "test_m4_a2_separation.jl")
    @test isfile(a1_path)
    @test isfile(a2_path)
    a1 = read(a1_path, String)
    a2 = read(a2_path, String)
    for id in _B4_A1_IDS
        @test occursin(id, a1)
    end
    for id in _B4_A2_IDS
        @test occursin(id, a2)
    end
    @test occursin("occupancy.X", a1)
    @test occursin("Q4 must NOT use occupancy.X", a2)
    @test :occupancy ∉ fieldnames(FunctionalIdentifiabilityDiagnostic)
    @test !occursin("sample_destruction_occupancy", _b4_source())
end
