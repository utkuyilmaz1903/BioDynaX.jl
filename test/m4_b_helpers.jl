# Shared M4-B test helpers. Not a scientific result and not a public API.

const _B4_LOCKED_SENTENCES = (
    analytic = "analytic library-membership control uses hill_rate_truth and is not trained-UDE evidence.",
    trained = "trained-UDE graph-local evidence samples D from the captured fit_unknown_destruction return params via sample_unknown_destruction.",
    smoke = "PR smoke is not trained-UDE scientific acceptance.")

const _B4_A1_IDS = (
    "T-A-API", "T-A-SRC", "T-A-XNEQ", "T-A-PROV", "T-A-SPLIT",
    "T-A-LEN", "T-A-R", "T-A-Q4SEP", "T-A-SAMP", "T-A-DTRUTH",
    "T-A-M1", "T-A-TIME", "T-A-M2", "T-A-RES", "T-A-INTACT",
    "T-A-VECTOR")

const _B4_A2_IDS = (
    "T-A2-M1", "T-A2-M1-TIME", "T-A2-Q4", "T-A2-Q4SEP",
    "T-A2-M2", "T-A2-M2-D")

const _B4_FORBIDDEN_EVIDENCE_FIELDS = (
    :success, :holdout, :occupancy, :z, :domain, :payload, :extra,
    :misc, :status, :best_scope)

const _B4_PRODUCTION_MUST_CONTAIN = (
    "generate_experiment_set(",
    "training_call",
    "fit_unknown_destruction",
    "sample_unknown_destruction(",
    "discover_equations",
    "rate_discovery_config",
    "targets = 1",
    "basis_scope")

const _B4_PRODUCTION_MUST_NOT_CONTAIN = (
    "hill_rate_truth",
    "graph_local_rate_samples",
    "sample_unknown_destruction_grid",
    "sample_destruction(",
    "sample_learned_function",
    "discover_unknown_rate",
    "equation_to_function",
    "unique_claim_experiment_split",
    "evaluate_holdout",
    "_evaluate_unknown_rate_recovery",
    "_unique_claim_rate_recovery",
    "sample_destruction_occupancy",
    "TrajectoryOccupancy",
    "FunctionalIdentifiabilityDomain",
    "select_discovery_config",
    "run_recovery_suite",
    "_train_unknown_edge",
    "generate_recovery_experiments",
    "ROBUSTNESS_SEEDS")

function _b4_truth_net()
    return BioDynaX.build_three_state_unknown_network(;
        known = true, with_distractor = true, parent = 2)
end

function _b4_ude_net()
    return BioDynaX.build_three_state_unknown_network(;
        known = false, with_distractor = true, parent = 2)
end

function _b4_wrong_net()
    return BioDynaX.build_wrong_graph_unknown_network(;
        known = false, with_distractor = true)
end

function wrong_graph_library_variables(wrong_net, target::Int = 1;
        degree::Int = 2, include_interactions::Bool = false)
    return graph_library_variables(wrong_net, target;
        degree = degree, include_interactions = include_interactions)
end

function _b4_fake_training(params)
    return TrainingResult(
        params, Float64[], 1.0, 0.4,
        RunMetadata(seed = 0), (;), true, BioDynaX.Success)
end

function _b4_perturb_params(params; δ = 0.37)
    q = deepcopy(params)
    q.nn .+= δ
    return q
end

function _b4_hand_candidate()
    spec = BioDynaX.LocalBasisSpec(1, [2], BioDynaX.MonomialTerm[],
        BioDynaX.MonomialTerm[])
    return ImplicitCandidate(
        1, spec,
        Float64[1.0],
        Float64[1.0],
        Float64[1.0],
        0.0, 1.0)
end

function _b4_source()
    return read(joinpath(pkgdir(BioDynaX), "src", "TrainedGraphLocal.jl"), String)
end

function _b4_source_code(text = _b4_source())
    stripped = replace(text, r"\"\"\"[\s\S]*?\"\"\"" => "")
    lines = String[]
    for line in split(stripped, r"\r\n|\n")
        code = replace(line, r"#.*" => "")
        push!(lines, code)
    end
    return join(lines, '\n')
end

function _b4_new_logs()
    return (;
        entry = Any[],
        fit_set = Any[],
        sample = Any[],
        discover = Any[],
        holdout = Any[],
        grid = Any[],
        grid_result = Any[],
        rate_discover = Any[],
        range = Any[],
        split = Any[],
        generate_recovery = Any[],
        assess_q4 = Any[],
        set_return = Ref{Any}(nothing),
        p0 = Ref{Any}(nothing),
        p0_fp = Ref{UInt64}(0),
        train_unknown_edge = Ref(0))
end

function _b4_new_capture()
    result = Ref{Any}(nothing)
    count = Ref(0)
    fp = Ref{UInt64}(0)
    constructed = Ref(false)
    function training_call(args...; kwargs...)
        returned = BioDynaX.fit_unknown_destruction(args...; kwargs...)
        result[] = returned
        fp[] = nn_parameter_fingerprint(returned.params.nn)
        count[] += 1
        return returned
    end
    return (; result, count, fp, constructed, training_call)
end

function _b4_with_observers(f, logs; inject = nothing)
    return with_fit_unknown_destruction_entry_observer(obs -> begin
            push!(logs.entry, obs)
            logs.p0[] = obs.p0
            logs.p0_fp[] = nn_parameter_fingerprint(obs.p0.nn)
            nothing
        end) do
        with_fit_unknown_destruction_observer(set -> begin
                push!(logs.fit_set, set)
                injected = inject === nothing ? nothing :
                           inject(set, logs.p0[])
                logs.set_return[] = injected
                return injected
            end) do
            with_sample_unknown_destruction_observer(obs -> begin
                    push!(logs.sample, obs)
                    nothing
                end) do
                with_discover_equations_observer((X, times, derivatives) -> begin
                        push!(logs.discover, (;
                            X = copy(X),
                            times = copy(times),
                            derivatives = derivatives === nothing ?
                                nothing : copy(derivatives)))
                        return nothing
                    end) do
                    with_evaluate_holdout_observer((args...) -> begin
                            push!(logs.holdout, args)
                            nothing
                        end) do
                        with_sample_unknown_destruction_grid_observer(r_range -> begin
                                push!(logs.grid, r_range)
                                nothing
                            end) do
                            with_sample_unknown_destruction_result_observer(obs -> begin
                                    push!(logs.grid_result, obs)
                                    nothing
                                end) do
                                with_discover_unknown_rate_observer(
                                    (R, times, D, config) -> begin
                                        push!(logs.rate_discover,
                                            (; R, times, D, config))
                                        return nothing
                                    end) do
                                    with_evaluate_unknown_rate_recovery_range_observer(
                                        r_range -> begin
                                            push!(logs.range, r_range)
                                            nothing
                                        end) do
                                        with_unique_claim_experiment_split_observer(
                                            split -> begin
                                                push!(logs.split, split)
                                                nothing
                                            end) do
                                            with_generate_recovery_experiments_observer(
                                                set -> begin
                                                    push!(logs.generate_recovery, set)
                                                    nothing
                                                end) do
                                                with_assess_functional_identifiability_observer(
                                                    args -> begin
                                                        push!(logs.assess_q4, args)
                                                        nothing
                                                    end) do
                                                    with_train_unknown_edge_counter() do counter
                                                        result = f()
                                                        logs.train_unknown_edge[] = counter[]
                                                        result
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function _b4_inject_p0(set, p0)
    return _b4_fake_training(p0)
end

function _b4_inject_perturbed(set, p0)
    return _b4_fake_training(_b4_perturb_params(p0))
end

function _b4_first_candidate(result)
    return result.success && !isempty(result.candidates) ?
           result.candidates[1] : nothing
end

function _b4_discovery_bound(production, replay)
    production.success == replay.success || return false
    production.retcode == replay.retcode || return false
    if production.success
        pc = production.candidates[1]
        rc = replay.candidates[1]
        pc.specification.variables == rc.specification.variables || return false
        local_has_true_parent_gate(pc; variable = 2) ==
            local_has_true_parent_gate(rc; variable = 2) || return false
        support_uses_variable(pc; variable = 2) ==
            support_uses_variable(rc; variable = 2) || return false
        pc.numerator_coefficients ≈ rc.numerator_coefficients || return false
        pc.denominator_coefficients ≈ rc.denominator_coefficients || return false
    end
    return true
end

function _b4_observer_off_replays(X, times, dX; seed = 11)
    ude_net = _b4_ude_net()
    wrong_net = _b4_wrong_net()
    cfg_graph = rate_discovery_config(scope = :graph, bootstrap = 0, seed = seed)
    cfg_global = rate_discovery_config(scope = :global, bootstrap = 0, seed = seed)
    graph = discover_equations(
        X, times, ude_net;
        derivatives = dX, targets = 1, config = cfg_graph, verbose = false)
    global_disc = discover_equations(
        X, times, ude_net;
        derivatives = dX, targets = 1, config = cfg_global, verbose = false)
    wrong = discover_equations(
        X, times, wrong_net;
        derivatives = dX, targets = 1, config = cfg_graph, verbose = false)
    return (;
        graph, global_disc, wrong, cfg_graph, cfg_global,
        ude_net, wrong_net)
end

function _b4_independent_replay_D(model, params, term, n)
    X = designed_trained_graph_local_coordinates(n; x_seed = 619)
    (_, D, _) = sample_unknown_destruction(model, params, X; term = term)
    return X, Matrix{Float64}(D)
end

function _b4_run(; kind::Symbol, inject = nothing)
    capture = _b4_new_capture()
    logs = _b4_new_logs()
    evidence = _b4_with_observers(logs; inject = inject) do
        evaluate_trained_graph_local(;
            kind = kind,
            training_call = capture.training_call)
    end
    n = size(evidence.D, 2)
    X_ind, D_replay = _b4_independent_replay_D(
        evidence.model, capture.result[].params, evidence.term, n)
    captured = logs.discover[1]
    replays = _b4_observer_off_replays(
        captured.X, captured.times, captured.derivatives;
        seed = 11)
    sample_fp = nn_parameter_fingerprint(logs.sample[1].params.nn)
    return (;
        evidence, capture, logs, X_ind, D_replay, replays, sample_fp)
end

function _b4_library_oracles()
    ude_net = _b4_ude_net()
    wrong_net = _b4_wrong_net()
    graph_spec, graph_vars = graph_library_variables(
        ude_net, 1; degree = 2, include_interactions = false)
    global_spec, global_vars = global_library_variables(
        ude_net, 1; degree = 2, include_interactions = false)
    wrong_spec, wrong_vars = wrong_graph_library_variables(wrong_net, 1)
    return (;
        ude_net, wrong_net, graph_spec, graph_vars, global_spec, global_vars,
        wrong_spec, wrong_vars)
end

function _b4_docs_texts()
    return (source = _b4_source(),)
end

function _b4_analytic_is_trained_ude(result)
    return result isa TrainedGraphLocalEvidence ||
           (result isa NamedTuple && haskey(result, :kind) &&
            (result.kind === :protocol || result.kind === :trained_ude))
end

function _b4_smoke_is_scientific_acceptance(evidence)
    return evidence isa TrainedGraphLocalEvidence &&
           evidence.kind === :protocol
end

function _b4_accepts_hardcoded_support(::Bool)
    return false
end

function _b4_is_constructive_wrong_graph(candidate_net, true_net)
    return graph_parent_set(candidate_net, 1) != graph_parent_set(true_net, 1) &&
           2 ∉ candidate_parents(candidate_net, 1) &&
           3 ∈ candidate_parents(candidate_net, 1)
end

function _b4_scientific_acceptance_holds(bundle)
    bundle.evidence.kind === :protocol || return false
    bundle.capture.count[] == 1 || return false
    bundle.capture.result[] isa TrainingResult || return false
    bundle.capture.constructed[] == false || return false
    bundle.logs.set_return[] === nothing || return false
    length(bundle.logs.entry) == 1 || return false
    length(bundle.logs.sample) == 1 || return false
    length(bundle.logs.discover) == 3 || return false
    bundle.sample_fp == bundle.capture.fp[] || return false
    bundle.evidence.D == bundle.D_replay || return false
    d1 = bundle.logs.discover[1].derivatives
    d2 = bundle.logs.discover[2].derivatives
    d3 = bundle.logs.discover[3].derivatives
    d1 == d2 == d3 || return false
    _b4_discovery_bound(bundle.evidence.graph_discovery, bundle.replays.graph) ||
        return false
    _b4_discovery_bound(bundle.evidence.global_discovery, bundle.replays.global_disc) ||
        return false
    _b4_discovery_bound(bundle.evidence.wrong_graph_discovery, bundle.replays.wrong) ||
        return false
    return true
end
