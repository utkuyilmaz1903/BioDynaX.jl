###############################################################################
# Training reuse: compiled model, remake session, locked sensealg (not exported).
#
# train_experiments used to drop the Adam optimiser state after a first-IC
# warmup and, on the Augmented-Lagrangian path, called predict_ude without
# the compiled model (recompile per IC). This file owns the session that
# holds one UDEModel, remakes SciMLBase.ODEProblem across ICs, and locks
# the adjoint already chosen by recommend_sensealg.
#
# Does not drop protocol ICs, points, or seeds. Does not grow exports.
###############################################################################

"""Source strings that prove the training reuse path stays wired."""
const TRAINING_REUSE_MUST_CONTAIN = (
    "mutable struct TrainingSolveSession",
    "function lock_training_solver",
    "function predict_ude_session",
    "function warmup_first_experiment",
    "function with_compile_network_counter",
    "function training_session_matches_generate",
    "function train_ude_compile_report",
    "function frozen_phys_warmup_report",
    "function sensealg_nobs_honesty_row",
    "optimizer_state")

const TRAINING_REUSE_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function training_reuse_locked_sentences()
    return (;
        session = "A TrainingSolveSession remakes one SciMLBase.ODEProblem across ICs; it does not compile_network per IC.",
        warmup = "First-IC warmup hands its Optimisers state to train_experiments; Adam momentum is not discarded.",
        sensealg = "Neural UDE training locks InterpolatingAdjoint with ZygoteVJP; BacksolveAdjoint is not used on a neural hole.",
        al = "The Augmented-Lagrangian constraint path calls predict_ude with the compiled model.",
        counter = "with_compile_network_counter fails the suite if the training path compiles per IC.",
        generate = "predict_ude_session must match generate_from_compiled_model at noise 0 without compile_network.",
        nobs = "The training lock asks recommend_sensealg for 100 observations; a short mechanistic horizon may still recommend BacksolveAdjoint.")
end

function training_reuse_source_path()
    joinpath(pkgdir(BioDynaX), "src", "TrainingReuse.jl")
end

function training_jl_source_path()
    joinpath(pkgdir(BioDynaX), "src", "Training.jl")
end

# -- compile_network counter (state lives in MechanismCompiler.jl) ------------

"""
    with_compile_network_counter(f)

Run `f(counter)` while `compile_network` increments `counter`. Nested
calls restore the previous counter. Used to lock the training path:
a compiled `UDEModel` must not compile again per IC.
"""
function with_compile_network_counter(f)
    counter = Ref(0)
    previous = COMPILE_NETWORK_COUNTER[]
    COMPILE_NETWORK_COUNTER[] = counter
    try
        return f(counter)
    finally
        COMPILE_NETWORK_COUNTER[] = previous
    end
end

function compile_network_call_count()
    counter = COMPILE_NETWORK_COUNTER[]
    return counter === nothing ? 0 : counter[]
end

# -- Sensealg lock ------------------------------------------------------------

function training_sensealg_kind(sensealg)
    sensealg === nothing && return :none
    sensealg isa InterpolatingAdjoint && return :interpolating
    sensealg isa BacksolveAdjoint && return :backsolve
    return :other
end

function training_sensealg_kind(solver::SolverConfig)
    return training_sensealg_kind(solver.sensealg)
end

function neural_training_requires_interpolating(model::UDEModel)
    return model.n_neural > 0
end

"""
    lock_training_solver(model, solver) -> SolverConfig

Return a solver config whose adjoint matches `recommend_sensealg` for
`solver.ad_policy`. Neural holes keep `InterpolatingAdjoint`. A
`ProductionAD` forward pass with `sensealg === nothing` is left in place
(in-place RHS; no adjoint).
"""
function lock_training_solver(model::UDEModel, solver::SolverConfig)
    if solver.ad_policy isa ProductionAD && solver.sensealg === nothing
        return solver
    end
    sa = locked_training_sensealg(model, solver.ad_policy, 100)
    return SolverConfig(
        solver.algorithm, sa, solver.abstol, solver.reltol,
        solver.maxiters, solver.ad_policy)
end

function lock_training_solver(model::UDEModel; ad_policy::AbstractADPolicy = ZygoteAD(),
        n_observations::Int = 100)
    return default_solver_config(model; ad_policy = ad_policy,
        n_observations = n_observations)
end

function lock_training_config(model::UDEModel, config::TrainingConfig)
    solver = lock_training_solver(model, config.solver)
    return TrainingConfig(
        config.adam_iterations, config.adam_learning_rate, config.bfgs_iterations,
        config.gradient_clip, config.log_every, config.constraint, solver,
        config.horizon_schedule, config.frozen_phys)
end

"""
    training_sensealg_is_locked(model, solver) -> Bool

Honesty check. Neural models must not use `BacksolveAdjoint`. The
recommended kind from `recommend_sensealg` must match the stored
`sensealg`, except the ProductionAD in-place forward (`sensealg === nothing`).
"""
function training_sensealg_is_locked(model::UDEModel, solver::SolverConfig)
    if solver.ad_policy isa ProductionAD && solver.sensealg === nothing
        return true
    end
    rec = recommend_sensealg(model; policy = solver.ad_policy)
    kind = training_sensealg_kind(solver)
    rec_kind = training_sensealg_kind(rec.sensealg)
    neural_training_requires_interpolating(model) && kind === :backsolve &&
        return false
    return kind === rec_kind
end

function training_sensealg_is_locked(model::UDEModel, config::TrainingConfig)
    return training_sensealg_is_locked(model, config.solver)
end

function assert_training_sensealg(model::UDEModel, solver::SolverConfig)
    training_sensealg_is_locked(model, solver) || throw(ErrorException(
        "training sensealg $(training_sensealg_kind(solver)) is not locked to recommend_sensealg for this model"))
    return solver
end

function recommend_sensealg_honesty_row(model::UDEModel;
        n_observations::Int = 100)
    zy = recommend_sensealg(model; policy = ZygoteAD(), n_observations = n_observations)
    prod = recommend_sensealg(
        model; policy = ProductionAD(), n_observations = n_observations)
    neural = neural_training_requires_interpolating(model)
    locked = lock_training_solver(model, SolverConfig())
    return (;
        neural,
        zygote_kind = training_sensealg_kind(zy.sensealg),
        zygote_name = zy.name,
        production_kind = training_sensealg_kind(prod.sensealg),
        production_name = prod.name,
        locked_kind = training_sensealg_kind(locked),
        backsolve_forbidden_for_neural = neural,
        holds = (neural ? zy.name === :interpolating_neural :
                 zy.name === :backsolve_mechanistic ||
                 zy.name === :interpolating_default) &&
                prod.name === :interpolating_production &&
                training_sensealg_is_locked(model, locked) &&
                !(neural && training_sensealg_kind(locked) === :backsolve))
end

# -- Solve session ------------------------------------------------------------

"""
    TrainingSolveSession

One compiled `UDEModel`, one template `ODEProblem`, and the locked
solver. `predict_ude_session` remakes `u0` / `tspan` / `p` and does not
call `compile_network`.
"""
mutable struct TrainingSolveSession{M, P, C, S}
    model::M
    template::P
    cache::C
    solver::S
    inplace::Bool
    remake_count::Int
    predict_count::Int
    sensealg_kind::Symbol
end

function training_solve_session(model::UDEModel, u0, tspan, p, solver::SolverConfig)
    assert_training_sensealg(model, solver)
    inplace = _forward_inplace(solver)
    template = SciMLBase.ODEProblem(model, u0, tspan, p; inplace = inplace)
    return TrainingSolveSession(
        model, template, nothing, solver, inplace, 0, 0,
        training_sensealg_kind(solver))
end

function training_solve_session(model::UDEModel, u0, tspan, p;
        solver::SolverConfig = lock_training_solver(model, SolverConfig()),
        cache::Union{Nothing, UDEModelCache} = nothing)
    cache === nothing &&
        return training_solve_session(model, u0, tspan, p, solver)
    assert_training_sensealg(model, solver)
    inplace = _forward_inplace(solver)
    local_cache = cache
    if inplace && local_cache === nothing
        local_cache = allocate_cache(model, Float64)
    end
    template = SciMLBase.ODEProblem(
        model, u0, tspan, p; inplace = inplace, cache = local_cache)
    return TrainingSolveSession(
        model,
        template,
        local_cache,
        solver,
        inplace,
        0,
        0,
        training_sensealg_kind(solver))
end

function training_solve_session(model::UDEModel, experiment::Experiment, p;
        kwargs...)
    tspan = (first(experiment.times), last(experiment.times))
    return training_solve_session(
        model, experiment.u0, tspan, p; kwargs...)
end

function training_solve_session(model::UDEModel, set::ExperimentSet, p; kwargs...)
    return training_solve_session(model, first(set.experiments), p; kwargs...)
end

"""
    predict_ude_session(session, p, u0, tspan, saveat)

Integrate by `SciMLBase.remake` of the session template. Does not compile.
"""
function predict_ude_session(session::TrainingSolveSession, p, u0, tspan, saveat)
    session.remake_count += 1
    session.predict_count += 1
    remade = SciMLBase.remake(session.template; u0 = u0, tspan = tspan, p = p)
    sol = solve(
        remade, session.solver.algorithm;
        saveat = saveat,
        abstol = session.solver.abstol,
        reltol = session.solver.reltol,
        maxiters = session.solver.maxiters,
        sensealg = session.solver.sensealg,
        dense = false,
        save_everystep = false)
    prediction = Array(sol)
    ignore_derivatives() do
        _validate_solution(sol, prediction, saveat, tspan)
    end
    return prediction
end

function session_predicts_without_compile(session::TrainingSolveSession, p, u0,
        tspan, saveat)
    n = with_compile_network_counter() do counter
        predict_ude_session(session, p, u0, tspan, saveat)
        counter[]
    end
    return n == 0
end

function resolve_training_model(model::Union{Nothing, UDEModel}, nn, st,
        network::BiologicalNetwork)
    model === nothing || return model
    return compile_network(network, nn, st)
end

# -- Warmup -------------------------------------------------------------------

"""
    warmup_first_experiment(p_init, set, model; config, ...)

Train on the first IC only (`bfgs_iterations = 0`). Returns
`(result, optimizer_state, session)`. The Optimisers state is the object
`train_experiments` must reuse.
"""
function warmup_first_experiment(p_init, set::ExperimentSet, model::UDEModel;
        config::TrainingConfig = TrainingConfig(),
        verbose::Bool = false,
        seed::Integer = 0)
    isempty(set.experiments) &&
        throw(ArgumentError("warmup_first_experiment needs a non-empty ExperimentSet"))
    locked = lock_training_config(model, config)
    first_exp = first(set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    session = training_solve_session(
        model, first_exp, p_init; solver = locked.solver)
    warm_config = TrainingConfig(locked;
        bfgs_iterations = 0,
        adam_iterations = max(1, locked.adam_iterations))
    result = train_ude(
        p_init, first_exp.observations, first_exp.times, first_exp.u0,
        tspan, model;
        config = warm_config,
        verbose = verbose,
        seed = seed,
        session = session)
    state = hasproperty(result.diagnostics, :optimizer_state) ?
            result.diagnostics.optimizer_state : nothing
    return (;
        result,
        params = result.params,
        optimizer_state = state,
        session,
        n_ics = 1,
        locked_config = warm_config)
end

function optimizer_state_from_result(result::TrainingResult)
    hasproperty(result.diagnostics, :optimizer_state) || return nothing
    return result.diagnostics.optimizer_state
end

# -- Joint warmup + multi-IC train --------------------------------------------

"""
    train_experiments_with_warmup(p_init, set, model; config, ...)

Warmup on IC 1, then `train_experiments` with the same compiled model,
locked solver, and Optimisers state. This is the unique-claim training
entry used by `_train_unknown_edge`.
"""
function train_experiments_with_warmup(p_init, set::ExperimentSet, model::UDEModel;
        config::TrainingConfig = TrainingConfig(),
        execution::ExecutionConfig = ExecutionConfig(),
        verbose::Bool = false,
        seed::Integer = 0,
        warmup::Bool = true)
    locked = lock_training_config(model, config)
    session = training_solve_session(model, set, p_init; solver = locked.solver)
    if !warmup || length(set) == 1
        return train_experiments(
            p_init, set, model;
            config = locked, execution = execution, verbose = verbose,
            seed = seed, session = session)
    end
    warm = warmup_first_experiment(
        p_init, set, model; config = locked, verbose = verbose, seed = seed)
    return train_experiments(
        warm.params, set, model;
        config = locked, execution = execution, verbose = verbose,
        seed = seed,
        optimizer_state = warm.optimizer_state,
        session = session)
end

function unique_claim_training_config(;
        adam_iterations = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
        bfgs_iterations = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
        frozen_phys::Vector{Symbol} = Symbol[],
        model::Union{Nothing, UDEModel} = nothing)
    base = TrainingConfig(
        adam_iterations = adam_iterations,
        bfgs_iterations = bfgs_iterations,
        horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0]),
        log_every = 10^6,
        frozen_phys = frozen_phys)
    model === nothing && return base
    return lock_training_config(model, base)
end

# -- Agreement / reports ------------------------------------------------------

function training_session_remake_agreement(model::UDEModel, params, u0;
        tspan = (0.0, 1.0), n_points::Int = 8,
        solver::SolverConfig = lock_training_solver(model, SolverConfig()))
    times = collect(range(first(tspan), last(tspan); length = n_points))
    direct = predict_ude(
        params, u0, tspan, times, model; solver_config = solver)
    session = training_solve_session(
        model, u0, tspan, params; solver = solver)
    remade = predict_ude_session(session, params, u0, tspan, times)
    remade2 = predict_ude_session(session, params, u0, tspan, times)
    compiled = with_compile_network_counter() do counter
        predict_ude_session(session, params, u0, tspan, times)
        counter[]
    end
    return (;
        direct,
        remade,
        remade2,
        remake_count = session.remake_count,
        predict_count = session.predict_count,
        sensealg_kind = session.sensealg_kind,
        matches = direct ≈ remade && remade ≈ remade2,
        no_compile = compiled == 0,
        holds = direct ≈ remade && remade ≈ remade2 && compiled == 0 &&
                session.remake_count ≥ 2)
end

function training_session_multi_ic_agreement(model::UDEModel, params, set::ExperimentSet;
        solver::SolverConfig = lock_training_solver(model, SolverConfig()))
    session = training_solve_session(model, set, params; solver = solver)
    n = with_compile_network_counter() do counter
        for experiment in set.experiments
            tspan = (first(experiment.times), last(experiment.times))
            pred = predict_ude_session(
                session, params, experiment.u0, tspan, experiment.times)
            direct = predict_ude(
                params, experiment.u0, tspan, experiment.times, model;
                solver_config = solver)
            pred ≈ direct || return (;
                counter = counter[], matches = false, holds = false)
        end
        return (;
            counter = counter[],
            remake_count = session.remake_count,
            matches = true,
            holds = counter[] == 0 &&
                    session.remake_count == length(set.experiments))
    end
    return n
end

function train_experiments_compile_report(p_init, set::ExperimentSet, model::UDEModel;
        config::TrainingConfig = TrainingConfig(
            adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
        constraint = nothing)
    cfg = constraint === nothing ? config :
          TrainingConfig(config; constraint = constraint)
    locked = lock_training_config(model, cfg)
    n = with_compile_network_counter() do counter
        train_experiments(
            p_init, set, model;
            config = locked, verbose = false)
        counter[]
    end
    n_nn = with_compile_network_counter() do counter
        train_experiments(
            p_init, set, model.nn, model.st;
            network = model.network,
            config = locked, verbose = false)
        counter[]
    end
    return (;
        with_model = n,
        with_nn_st = n_nn,
        n_ics = length(set),
        holds = n == 0 && n_nn == 1)
end

function warmup_state_reuse_report(p_init, set::ExperimentSet, model::UDEModel;
        adam_iterations::Int = 2)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations,
            bfgs_iterations = 0,
            log_every = 10^6))
    warm = warmup_first_experiment(
        p_init, set, model; config = config, verbose = false)
    reused = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = warm.optimizer_state)
    fresh = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = nothing)
    return (;
        warmup_has_state = warm.optimizer_state !== nothing,
        reused_history = length(reused.history),
        fresh_history = length(fresh.history),
        sensealg_locked = training_sensealg_is_locked(model, config),
        holds = warm.optimizer_state !== nothing &&
                training_sensealg_is_locked(model, config))
end

function al_constraint_passes_model_source()
    src = read(training_jl_source_path(), String)
    start = findfirst("function train_experiments(p_init, set::ExperimentSet, nn, st", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(
        "function train_experiments(p_init, set::ExperimentSet, model::UDEModel",
        rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("predict_ude(", body) &&
           occursin("model = resolved_model", body) &&
           occursin("session = local_session", body)
end

function train_experiments_accepts_optimizer_state_source()
    src = read(training_jl_source_path(), String)
    return occursin("optimizer_state = nothing", src) &&
           occursin("session = nothing", src) &&
           occursin("optimizer_state = current_optimizer_state", src)
end

function train_unknown_edge_reuses_warmup_source()
    path = joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
    src = read(path, String)
    start = findfirst("function _train_unknown_edge", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("train_experiments_with_warmup", body) &&
           occursin("lock_training_config", body) &&
           !occursin("bfgs_iterations = 0,\n            horizon_schedule", body)
end

# -- Linear / neural fixtures for sensealg honesty ----------------------------

function linear_sensealg_honesty()
    rng = MersenneTwister(7)
    model, _ = build_ude_model(rng, build_linear_test_network())
    return recommend_sensealg_honesty_row(model)
end

function hill_ude_sensealg_honesty()
    rng = MersenneTwister(11)
    model, _ = build_ude_model(
        rng, build_hill_recovery_network(; known = false, hill_order = 2))
    return recommend_sensealg_honesty_row(model)
end

function remapped_sensealg_honesty()
    rng = MersenneTwister(13)
    model, _ = build_ude_model(rng, build_remapped_two_regulator_network())
    return recommend_sensealg_honesty_row(model)
end

function two_regulator_sensealg_honesty()
    rng = MersenneTwister(19)
    model, _ = build_ude_model(rng, build_two_regulator_unknown_network())
    return recommend_sensealg_honesty_row(model)
end

function training_sensealg_honesty_matrix()
    linear = linear_sensealg_honesty()
    hill = hill_ude_sensealg_honesty()
    remap = remapped_sensealg_honesty()
    two = two_regulator_sensealg_honesty()
    return (;
        linear,
        hill,
        remap,
        two,
        holds = linear.holds && hill.holds && remap.holds && two.holds &&
                linear.neural == false &&
                hill.neural && hill.zygote_kind === :interpolating &&
                remap.neural && two.neural)
end

function unique_claim_warmup_compile_path(; smoke::Bool = true)
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    truth = build_hill_recovery_network(; known = true, hill_order = 2)
    rng = MersenneTwister(103)
    model, p0 = build_ude_model(rng, net)
    set = unique_claim_experiment_set(
        MersenneTwister(103), truth;
        smoke = smoke,
        truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6))
    names = Tuple(parameter_schema(model).phys_names)
    guess = NamedTuple{names}(ntuple(_ -> 0.8, length(names)))
    p_init = pack_parameters(guess, p0.nn)
    config = unique_claim_training_config(
        model = model,
        adam_iterations = 1,
        bfgs_iterations = 0)
    n = with_compile_network_counter() do counter
        warmup_first_experiment(
            p_init, set, model; config = config, verbose = false)
        counter[]
    end
    rec = recommend_sensealg_honesty_row(model)
    return (;
        n_ics = length(set.experiments),
        compiles = n,
        sensealg = rec,
        compiled_once = experiment_set_is_compiled_once(set),
        holds = n == 0 && rec.holds && rec.neural &&
                experiment_set_is_compiled_once(set))
end

function dual_unknown_session_path()
    net = build_dual_unknown_network()
    rng = MersenneTwister(21)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_cb = 0.9, k_c = 0.5), params.nn)
    set = generate_experiment_set(
        MersenneTwister(21); network = net,
        initial_conditions = [[0.22, 0.18, 0.16], [0.30, 0.24, 0.20]],
        tspan = (0.0, 0.8), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
    remake = training_session_multi_ic_agreement(model, packed, set)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        remake,
        sense,
        n_heads = neural_head_count(model),
        recovery_admits = unique_claim_recovery_admits(net),
        holds = remake.holds && sense.holds && sense.neural &&
                neural_head_count(model) == 2 &&
                unique_claim_recovery_admits(net) == false)
end

function zero_hole_session_path()
    net = build_zero_unknown_linear_network()
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    report = training_session_remake_agreement(
        model, packed, [0.22, 0.14]; tspan = (0.0, 0.8), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        report,
        sense,
        n_heads = neural_head_count(model),
        validate_open = validate_network(net) === net,
        holds = report.holds && sense.holds && sense.neural == false &&
                neural_head_count(model) == 0)
end

function skipped_duplicate_session_path()
    net = build_skipped_duplicate_unknown_network()
    rng = MersenneTwister(13)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_b = 0.5, k_c = 0.4), params.nn)
    report = training_session_remake_agreement(
        model, packed, [0.2, 0.3, 0.4]; tspan = (0.0, 0.6), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        report,
        sense,
        n_heads = neural_head_count(model),
        dense = neural_index_is_dense(model),
        holds = report.holds && sense.holds && sense.neural &&
                neural_index_is_dense(model))
end

function training_reuse_fixture_matrix()
    linear = training_session_remake_agreement(
        build_ude_model(MersenneTwister(7), build_linear_test_network())...,
        [0.22, 0.14])
    dual = dual_unknown_session_path()
    zero = zero_hole_session_path()
    skipped = skipped_duplicate_session_path()
    claim = unique_claim_warmup_compile_path()
    sense = training_sensealg_honesty_matrix()
    return (;
        linear,
        dual,
        zero,
        skipped,
        claim,
        sense,
        holds = linear.holds && dual.holds && zero.holds && skipped.holds &&
                claim.holds && sense.holds)
end

function train_ude_optimizer_state_is_recorded(p_init, data, times, u0, tspan,
        model; adam_iterations::Int = 2)
    result = train_ude(
        p_init, data, times, u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6),
        verbose = false)
    return (;
        has_state = optimizer_state_from_result(result) !== nothing,
        history = length(result.history),
        holds = optimizer_state_from_result(result) !== nothing &&
                length(result.history) == adam_iterations)
end

function resume_training_compile_report(p_init, data, times, u0, tspan, model;
        checkpoint_dir)
    path = joinpath(checkpoint_dir, "reuse.ckpt")
    result = train_ude(
        p_init, data, times, u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6),
        verbose = false,
        checkpoint_path = path,
        checkpoint_every = 1,
        seed = 3)
    isfile(path) || return (; holds = false, compiles = typemax(Int))
    ckpt = load_checkpoint(path)
    n = with_compile_network_counter() do counter
        resume_training(
            ckpt, data, times, u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        schema = ckpt.schema_version,
        holds = n == 0)
end

# -- Generate / train_ude / mask / n_obs honesty ------------------------------

"""Tight solver used only to compare a session remake against generate."""
function _generate_match_solver(model::UDEModel)
    base = lock_training_solver(model, SolverConfig())
    return SolverConfig(
        algorithm = Tsit5(),
        ad_policy = base.ad_policy,
        sensealg = base.sensealg,
        abstol = 1e-9,
        reltol = 1e-9)
end

"""
    training_session_matches_generate(model, params, u0; tspan, n_points)

`predict_ude_session` versus `generate_from_compiled_model` at noise 0.
Forward trajectories must agree; `compile_network` must stay at zero.
"""
function training_session_matches_generate(model::UDEModel, params, u0;
        tspan = (0.0, 0.8), n_points::Int = 8,
        rng::AbstractRNG = MersenneTwister(0))
    solver = _generate_match_solver(model)
    times, clean, _, used = generate_from_compiled_model(
        model, params, rng;
        u0 = Float64.(u0), tspan = tspan, n_points = n_points, noise_σ = 0.0)
    session = training_solve_session(
        model, Float64.(u0), tspan, used; solver = solver)
    remade = predict_ude_session(session, used, Float64.(u0), tspan, times)
    direct = predict_ude(
        used, Float64.(u0), tspan, times, model; solver_config = solver)
    compiled = with_compile_network_counter() do counter
        predict_ude_session(session, used, Float64.(u0), tspan, times)
        counter[]
    end
    return (;
        times,
        clean,
        remade,
        direct,
        remake_count = session.remake_count,
        matches_generate = remade ≈ clean,
        matches_direct = remade ≈ direct,
        no_compile = compiled == 0,
        holds = remade ≈ clean && remade ≈ direct && compiled == 0)
end

function train_ude_compile_report(p_init, data, times, u0, tspan, model;
        adam_iterations::Int = 1)
    locked = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6))
    n = with_compile_network_counter() do counter
        train_ude(
            p_init, data, times, u0, tspan, model;
            config = locked, verbose = false)
        counter[]
    end
    n_nn = with_compile_network_counter() do counter
        train_ude(
            p_init, data, times, u0, tspan, model.nn, model.st;
            network = model.network, config = locked, verbose = false)
        counter[]
    end
    return (;
        with_model = n,
        with_nn_st = n_nn,
        holds = n == 0 && n_nn == 1)
end

function frozen_phys_warmup_report(p_init, set::ExperimentSet, model::UDEModel;
        frozen::Vector{Symbol} = [:k_ba],
        adam_iterations::Int = 2)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations,
            bfgs_iterations = 0,
            log_every = 10^6,
            frozen_phys = frozen))
    names = Tuple(parameter_schema(model).phys_names)
    before = NamedTuple{names}(ntuple(
        i -> Float64(p_init.phys[i]), length(names)))
    warm = warmup_first_experiment(
        p_init, set, model; config = config, verbose = false)
    reused = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = warm.optimizer_state)
    after = NamedTuple{names}(ntuple(
        i -> Float64(reused.params.phys[i]), length(names)))
    frozen_held = all(name -> getfield(before, name) ≈ getfield(after, name),
        frozen)
    return (;
        frozen,
        before,
        after,
        frozen_held,
        warmup_has_state = warm.optimizer_state !== nothing,
        holds = frozen_held && warm.optimizer_state !== nothing &&
                training_sensealg_is_locked(model, config))
end

function masked_experiment_compile_report(p_init, set::ExperimentSet,
        model::UDEModel; hide_state::Int = 1)
    experiments = map(set.experiments) do experiment
        mask = copy(experiment.mask)
        mask[hide_state, :] .= false
        Experiment(
            experiment.name, experiment.times, experiment.observations,
            experiment.u0; mask = mask, metadata = experiment.metadata)
    end
    masked = ExperimentSet(experiments, set.state_names; units = set.units,
        metadata = set.metadata)
    locked = lock_training_config(
        model, TrainingConfig(
            adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6))
    n = with_compile_network_counter() do counter
        train_experiments(
            p_init, masked, model; config = locked, verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        hidden = hide_state,
        n_ics = length(masked),
        holds = n == 0)
end

"""
    sensealg_nobs_honesty_row(model)

`recommend_sensealg` at 20 observations versus the training lock, which
asks for 100. Mechanistic models may prefer `BacksolveAdjoint` on a short
horizon; the training lock still writes the 100-observation adjoint.
Neural holes stay interpolating at both widths.
"""
function sensealg_nobs_honesty_row(model::UDEModel)
    small = recommend_sensealg(model; n_observations = 20)
    large = recommend_sensealg(model; n_observations = 100)
    locked = lock_training_solver(model, SolverConfig())
    neural = neural_training_requires_interpolating(model)
    return (;
        neural,
        small_name = small.name,
        large_name = large.name,
        locked_kind = training_sensealg_kind(locked),
        lock_follows_nobs_100 = training_sensealg_kind(large.sensealg) ===
                                training_sensealg_kind(locked),
        holds = training_sensealg_kind(large.sensealg) ===
                training_sensealg_kind(locked) &&
                (neural ?
                 small.name === :interpolating_neural &&
                 large.name === :interpolating_neural :
                 small.name === :backsolve_mechanistic &&
                 large.name === :interpolating_default))
end

function sensealg_nobs_honesty_matrix()
    linear = sensealg_nobs_honesty_row(
        build_ude_model(MersenneTwister(7), build_linear_test_network())[1])
    hill = sensealg_nobs_honesty_row(
        build_ude_model(MersenneTwister(11),
        build_hill_recovery_network(; known = false, hill_order = 2))[1])
    remap = sensealg_nobs_honesty_row(
        build_ude_model(MersenneTwister(13),
        build_remapped_two_regulator_network())[1])
    return (;
        linear,
        hill,
        remap,
        holds = linear.holds && hill.holds && remap.holds &&
                linear.neural == false && hill.neural && remap.neural)
end

function six_state_session_path()
    net = build_six_state_unknown_network(; known = false)
    rng = MersenneTwister(41)
    model, params = build_ude_model(rng, net)
    u0 = [0.22, 0.18, 0.16, 0.14, 0.12, 0.10]
    remake = training_session_remake_agreement(
        model, params, u0; tspan = (0.0, 0.5), n_points = 6)
    generate = training_session_matches_generate(
        model, params, u0; tspan = (0.0, 0.5), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    nobs = sensealg_nobs_honesty_row(model)
    return (;
        remake,
        generate,
        sense,
        nobs,
        n_heads = neural_head_count(model),
        nstates = model.compiled.nstates,
        holds = remake.holds && generate.holds && sense.holds && nobs.holds &&
                neural_head_count(model) == 1 &&
                model.compiled.nstates == 6)
end

function mm_unknown_session_path()
    net = build_mm_recovery_network(; known = false)
    rng = MersenneTwister(43)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), params.nn)
    remake = training_session_remake_agreement(
        model, packed, [0.30, 0.25]; tspan = (0.0, 0.6), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        remake,
        sense,
        n_heads = neural_head_count(model),
        recovery_admits = unique_claim_recovery_admits(net),
        holds = remake.holds && sense.holds && sense.neural &&
                neural_head_count(model) == 1 &&
                unique_claim_recovery_admits(net))
end

function competitive_session_path()
    net = build_competitive_test_network(; known = true)
    rng = MersenneTwister(47)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters(
        (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6, k_s = 0.8, k_i = 0.5),
        params.nn)
    remake = training_session_remake_agreement(
        model, packed, [0.25, 0.45, 0.20]; tspan = (0.0, 0.6), n_points = 6)
    generate = training_session_matches_generate(
        model, packed, [0.25, 0.45, 0.20]; tspan = (0.0, 0.6), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        remake,
        generate,
        sense,
        n_heads = neural_head_count(model),
        validate_open = validate_network(net) === net,
        holds = remake.holds && generate.holds && sense.holds &&
                sense.neural == false && neural_head_count(model) == 0)
end

function horizon_curriculum_session_report(p_init, data, times, u0, tspan,
        model; adam_iterations::Int = 1)
    curriculum = HorizonCurriculum(
        fractions = [0.5, 1.0], min_points = 3, minimum_fraction = 0.4)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6, horizon_schedule = curriculum))
    n = with_compile_network_counter() do counter
        train_ude(
            p_init, data, times, u0, tspan, model;
            config = config, verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        fractions = curriculum.fractions,
        holds = n == 0 && training_sensealg_is_locked(model, config))
end

function optimizer_state_roundtrip_report(p_init, set::ExperimentSet,
        model::UDEModel; adam_iterations::Int = 2)
    config = lock_training_config(model,
        TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6))
    warm = warmup_first_experiment(
        p_init, set, model; config = config, verbose = false)
    reused = train_experiments(
        warm.params, set, model;
        config = config, verbose = false,
        optimizer_state = warm.optimizer_state)
    from_warm = optimizer_state_from_result(warm.result)
    from_reused = optimizer_state_from_result(reused)
    return (;
        warmup_has_state = from_warm !== nothing,
        reused_has_state = from_reused !== nothing,
        same_type = typeof(from_warm) === typeof(from_reused),
        holds = from_warm !== nothing && from_reused !== nothing &&
                typeof(from_warm) === typeof(from_reused))
end

function resume_from_diagnostics_report(p_init, data, times, u0, tspan, model;
        adam_iterations::Int = 2)
    first = train_ude(
        p_init, data, times, u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = adam_iterations, bfgs_iterations = 0,
            log_every = 10^6),
        verbose = false)
    state = optimizer_state_from_result(first)
    n = with_compile_network_counter() do counter
        train_ude(
            first.params, data, times, u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            optimizer_state = state, verbose = false)
        counter[]
    end
    return (;
        has_state = state !== nothing,
        compiles = n,
        holds = state !== nothing && n == 0)
end

function linear_training_fixture()
    rng = MersenneTwister(53)
    net = build_linear_test_network()
    model, p0 = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(53); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        tspan = (0.0, 0.6), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    exp = first(set.experiments)
    tspan = (first(exp.times), last(exp.times))
    return (;
        net, model, p0, set, init, exp, tspan,
        data = exp.observations, times = exp.times, u0 = exp.u0)
end

function training_reuse_extended_matrix()
    fixture = linear_training_fixture()
    generate = training_session_matches_generate(
        fixture.model, fixture.init, fixture.u0;
        tspan = fixture.tspan, n_points = length(fixture.times))
    ude = train_ude_compile_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    frozen = frozen_phys_warmup_report(
        fixture.init, fixture.set, fixture.model)
    masked = masked_experiment_compile_report(
        fixture.init, fixture.set, fixture.model)
    nobs = sensealg_nobs_honesty_matrix()
    six = six_state_session_path()
    mm = mm_unknown_session_path()
    competitive = competitive_session_path()
    horizon = horizon_curriculum_session_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    roundtrip = optimizer_state_roundtrip_report(
        fixture.init, fixture.set, fixture.model)
    resume = resume_from_diagnostics_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    return (;
        generate,
        ude,
        frozen,
        masked,
        nobs,
        six,
        mm,
        competitive,
        horizon,
        roundtrip,
        resume,
        holds = generate.holds && ude.holds && frozen.holds && masked.holds &&
                nobs.holds && six.holds && mm.holds && competitive.holds &&
                horizon.holds && roundtrip.holds && resume.holds)
end

# -- Docs / source locks ------------------------------------------------------

function training_reuse_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "training-reuse.md")
end

function training_reuse_source_holds()
    src = read(training_reuse_source_path(), String)
    impl = read(training_jl_source_path(), String)
    docs = isfile(training_reuse_docs_path()) ?
           read(training_reuse_docs_path(), String) : ""
    return all(occursin(needle, src) for needle in TRAINING_REUSE_MUST_CONTAIN) &&
           !any(occursin(needle, impl) || occursin(needle, docs)
    for needle in TRAINING_REUSE_MUST_NOT_CONTAIN)
end

function training_reuse_docs_hold()
    path = training_reuse_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(training_reuse_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("training-reuse.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function training_reuse_landing_docs_hold()
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sentences = training_reuse_locked_sentences()
    return occursin("training-reuse", sciml) &&
           occursin("TrainingSolveSession", howto) &&
           occursin(sentences.session, sciml)
end

function training_reuse_source_violations()
    src = read(training_reuse_source_path(), String)
    impl = read(training_jl_source_path(), String)
    docs = isfile(training_reuse_docs_path()) ?
           read(training_reuse_docs_path(), String) : ""
    missing = [s for s in TRAINING_REUSE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s
                 for s in TRAINING_REUSE_MUST_NOT_CONTAIN
                 if occursin(s, impl) || occursin(s, docs)]
    return (; missing, forbidden)
end

function training_reuse_contract_holds()
    return training_reuse_source_holds() &&
           al_constraint_passes_model_source() &&
           train_experiments_accepts_optimizer_state_source() &&
           train_unknown_edge_reuses_warmup_source() &&
           training_reuse_docs_hold() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source()
end
