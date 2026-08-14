using OrdinaryDiffEq: ODEProblem, solve, Tsit5
using SciMLSensitivity: InterpolatingAdjoint, ZygoteVJP

mutable struct LossDiagnostics
    mse::Float64
    constraint::Float64
    total::Float64
    primal_residual::Float64
    bfgs_attempted::Bool
    bfgs_success::Bool
    bfgs_retcode::Symbol
    bfgs_message::String
    gradient_norm_history::Vector{Float64}
    gradient_failure::Bool
    LossDiagnostics() = new(NaN, NaN, NaN, NaN, false, false, :none, "", Float64[], false)
end

@inline function _record!(d::LossDiagnostics, mse, constraint, total, residual)
    d.mse = Float64(mse)
    d.constraint = Float64(constraint)
    d.total = Float64(total)
    d.primal_residual = Float64(residual)
    return nothing
end

@inline function _record_bfgs!(d::LossDiagnostics, attempted, success, retcode, message)
    d.bfgs_attempted = attempted
    d.bfgs_success = success
    d.bfgs_retcode = retcode
    d.bfgs_message = message
    return nothing
end

"""Return true when the solver config requests an in-place forward pass."""
function _forward_inplace(solver_config::SolverConfig)
    solver_config.ad_policy isa ProductionAD || return false
    # Zygote-based adjoints require the out-of-place RHS during differentiation.
    return solver_config.sensealg === nothing
end

function predict_ude(p, u0, tspan, saveat, nn, st;
                     model::Union{Nothing,UDEModel} = nothing,
                     network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
                     solver_config::SolverConfig = SolverConfig(),
                     cache::Union{Nothing,UDEModelCache} = nothing)
    resolved = model === nothing ?
        Zygote.@ignore(compile_network(network, nn, st)) : model
    inplace = _forward_inplace(solver_config)
    prob = SciMLBase.ODEProblem(
        resolved, u0, tspan, p; inplace = inplace, cache = cache)
    sol  = solve(
        prob, solver_config.algorithm;
        saveat   = saveat,
        abstol   = solver_config.abstol,
        reltol   = solver_config.reltol,
        maxiters = solver_config.maxiters,
        sensealg = solver_config.sensealg,
        dense = false,
        save_everystep = false,
    )
    prediction = Array(sol)
    Zygote.@ignore _validate_solution(sol, prediction, saveat, tspan)
    return prediction
end

@inline _smooth_violation(value, temperature) =
    temperature * softplus(-value / temperature)

function _validate_solution(solution, prediction, saveat, tspan)
    SciMLBase.successful_retcode(solution) ||
        throw(ErrorException("ODE solve failed with retcode $(solution.retcode)"))
    size(prediction, 2) == length(saveat) ||
        throw(ErrorException("ODE solve did not produce all requested samples"))
    isapprox(solution.t[end], tspan[2]; atol = 10eps(float(tspan[2])),
             rtol = 10eps(float(tspan[2]))) ||
        throw(ErrorException("ODE solve terminated before the final time"))
    all(isfinite, prediction) ||
        throw(ErrorException("ODE solve returned non-finite states"))
    return nothing
end

function _constraint_values(prediction, strategy::StructuralPositivity)
    return zeros(eltype(prediction), size(prediction, 1))
end

function _constraint_values(prediction, strategy::AugmentedLagrangianConfig)
    temperature = strategy.smoothness
    return map(axes(prediction, 1)) do state
        values = .-(@view prediction[state, :])
        maximum_value = maximum(values)
        maximum_value + temperature * log(mean(
            exp.((values .- maximum_value) ./ temperature)))
    end
end

function _masked_mse(prediction, data, mask)
    count(mask) > 0 || throw(ArgumentError("observation mask is empty"))
    residual = ifelse.(mask, prediction .- data, zero(eltype(prediction)))
    return sum(abs2, residual) / count(mask)
end

@inline _smooth_projection(value, temperature) =
    temperature * softplus(value / temperature)

function _augmented_term(constraints, dual, ρ, temperature)
    isempty(constraints) && return zero(eltype(constraints))
    shifted = dual .+ ρ .* constraints
    projected = _smooth_projection.(shifted, temperature)
    return sum((projected .^ 2 .- dual .^ 2) ./ (2ρ))
end

function predict_ude(p, u0, tspan, saveat, model::UDEModel;
                     solver_config::SolverConfig = SolverConfig(),
                     cache::Union{Nothing,UDEModelCache} = nothing)
    return predict_ude(
        p, u0, tspan, saveat, model.nn, model.st;
        model = model, network = model.network,
        solver_config = solver_config, cache = cache)
end

function loss_mse(p, data, t_data, u0, tspan, nn, st;
                  model::Union{Nothing,UDEModel} = nothing,
                  network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
                  constraint::AbstractConstraintStrategy =
                      StructuralPositivity(),
                  dual = zeros(eltype(p), size(data, 1)),
                  ρ = one(eltype(p)),
                  solver_config::SolverConfig = SolverConfig(),
                  mask = trues(size(data)),
                  diagnostics::Union{Nothing,LossDiagnostics} = nothing)
    prediction = predict_ude(p, u0, tspan, t_data, nn, st;
                             model = model, network = network,
                             solver_config = solver_config)
    size(prediction) == size(data) ||
        throw(ErrorException("ODE solve terminated before all observations"))
    all(isfinite, prediction) ||
        throw(ErrorException("ODE solve returned non-finite states"))
    mse = _masked_mse(prediction, data, mask)
    constraints = _constraint_values(prediction, constraint)
    constraint_loss = constraint isa AugmentedLagrangianConfig ?
        _augmented_term(
            constraints, dual, ρ, constraint.smoothness) : zero(mse)
    total = mse + constraint_loss
    residual = isempty(constraints) ? zero(mse) :
        max(zero(mse), maximum(constraints))

    if diagnostics !== nothing
        Zygote.@ignore _record!(
            diagnostics, mse, constraint_loss, total, residual)
    end
    return total
end

function _training_retcode(converged::Bool, diagnostics::LossDiagnostics)
    converged && return Success
    diagnostics.gradient_failure && return GradientFailure
    diagnostics.bfgs_attempted && !diagnostics.bfgs_success && return BFGSFailure
    return NotConverged
end

function _optimize_stage(p_init, loss_closure, config, history, diagnostics;
                         verbose, optimizer_state = nothing,
                         checkpoint_hook = nothing)
    callback = function (_state, loss)
        push!(history, Float64(loss))
        if verbose && length(history) % config.log_every == 0
            println("       iter ", lpad(length(history), 4),
                    " | total=", round(loss; sigdigits = 7),
                    " | mse=", round(diagnostics.mse; sigdigits = 7),
                    " | constraint=",
                    round(diagnostics.constraint; sigdigits = 7),
                    " | primal=",
                    round(diagnostics.primal_residual; sigdigits = 5))
        end
        return false
    end
    optimizer = Optimisers.OptimiserChain(
        Optimisers.ClipGrad(config.gradient_clip),
        Optimisers.Adam(config.adam_learning_rate))
    params = p_init
    frozen_ref = p_init
    state = optimizer_state === nothing ?
        Optimisers.setup(optimizer, params) : optimizer_state
    for _ in 1:config.adam_iterations
        loss, gradients = Zygote.withgradient(
            value -> loss_closure(value, nothing), params)
        gradient = only(gradients)
        all(isfinite, gradient) ||
            begin
                diagnostics.gradient_failure = true
                throw(ErrorException("optimizer produced a non-finite gradient"))
            end
        gradient = _zero_frozen_phys_gradient(gradient, config.frozen_phys)
        push!(diagnostics.gradient_norm_history,
              Float64(sqrt(sum(abs2, gradient))))
        state, params = Optimisers.update(state, params, gradient)
        callback(nothing, loss)
        checkpoint_hook === nothing ||
            checkpoint_hook(params, state, length(history))
    end
    if config.bfgs_iterations > 0
        objective = Optimization.OptimizationFunction(
            loss_closure, Optimization.AutoZygote())
        refinement = Optimization.OptimizationProblem(objective, params)
        bfgs_result = try
            Optimization.solve(
                refinement, OptimizationOptimJL.BFGS();
                callback, maxiters = config.bfgs_iterations)
        catch error
            _record_bfgs!(diagnostics, true, false, :failure,
                          sprint(showerror, error))
            @warn "BFGS refinement failed; retaining Adam result." exception = error
            nothing
        end
        if bfgs_result === nothing
            diagnostics.bfgs_attempted || _record_bfgs!(
                diagnostics, true, false, :failure, "BFGS solve failed")
        else
            params = _restore_frozen_phys(
                bfgs_result.u, frozen_ref, config.frozen_phys)
            _record_bfgs!(diagnostics, true, true, :success, "BFGS refinement completed")
        end
    end
    return params, state
end

function _zero_frozen_phys_gradient(gradient, frozen::AbstractVector{Symbol})
    isempty(frozen) && return gradient
    hasproperty(gradient, :phys) || return gradient
    phys = gradient.phys
    updates = Pair{Symbol,Any}[]
    changed = false
    for name in propertynames(phys)
        value = getproperty(phys, name)
        if name in frozen
            push!(updates, name => zero(value))
            changed = true
        else
            push!(updates, name => value)
        end
    end
    changed || return gradient
    return ComponentVector(phys = NamedTuple(updates), nn = gradient.nn)
end

function _restore_frozen_phys(params, reference, frozen::AbstractVector{Symbol})
    isempty(frozen) && return params
    hasproperty(params, :phys) || return params
    updates = Pair{Symbol,Any}[]
    for name in propertynames(params.phys)
        value = name in frozen ?
            getproperty(reference.phys, name) : getproperty(params.phys, name)
        push!(updates, name => value)
    end
    return ComponentVector(phys = NamedTuple(updates), nn = params.nn)
end

function _stage_config(config::TrainingConfig, stages::Int, final_stage::Bool)
    adam_iterations = max(1, cld(config.adam_iterations, stages))
    horizon = config.horizon_schedule isa HorizonCurriculum ?
        config.horizon_schedule :
        HorizonCurriculum(fractions = collect(_horizon_fractions(config.horizon_schedule)))
    return TrainingConfig(config;
        adam_iterations = adam_iterations,
        bfgs_iterations = final_stage ? config.bfgs_iterations : 0,
        horizon_schedule = horizon)
end

function _training_converged(final_loss, initial_loss, config, diagnostics)
    isfinite(final_loss) && isfinite(initial_loss) || return false
    if config.constraint isa AugmentedLagrangianConfig
        return diagnostics.primal_residual ≤ config.constraint.tolerance ||
               final_loss ≤ initial_loss
    end
    return final_loss ≤ initial_loss
end

function train_ude(p_init, data, t_data, u0, tspan, nn, st;
                   model::Union{Nothing,UDEModel} = nothing,
                   network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
                   config::Union{Nothing,TrainingConfig} = nothing,
                   adam_iters::Int = 300,
                   adam_lr::Float64 = 0.01,
                   bfgs_iters::Int = 100,
                   log_every::Int = 20,
                   verbose::Bool = true,
                   seed::Integer = 0,
                   optimizer_state = nothing,
                   checkpoint_path::Union{Nothing,AbstractString} = nothing,
                   checkpoint_every::Int = 0,
                   initial_iteration::Int = 0,
                   dual_init = nothing,
                   rho_init = nothing,
                   initial_outer::Int = 1,
                   initial_stage::Int = 1,
                   initial_stage_iteration::Int = 0,
                   previous_residual_init = Inf)
    training_config = isnothing(config) ? TrainingConfig(
        adam_iterations = adam_iters,
        adam_learning_rate = adam_lr,
        bfgs_iterations = bfgs_iters,
        log_every = log_every) : config
    diag = LossDiagnostics()
    state_count = size(data, 1)
    dual = dual_init === nothing ?
        zeros(eltype(p_init), state_count) : copy(dual_init)
    ρ = rho_init === nothing ?
        (training_config.constraint isa AugmentedLagrangianConfig ?
         training_config.constraint.initial_ρ : zero(eltype(p_init))) :
        rho_init
    function make_loss(local_data, local_times, local_span)
        return (p, _) -> loss_mse(
            p, local_data, local_times, u0, local_span, nn, st;
            model = model, network = network,
            constraint = training_config.constraint, dual, ρ,
            solver_config = training_config.solver, diagnostics = diag)
    end
    full_loss = make_loss(data, t_data, tspan)
    initial_loss = try
        full_loss(p_init, nothing)
    catch error
        first_fraction = minimum(_horizon_fractions(training_config.horizon_schedule))
        min_pts = _horizon_min_points(training_config.horizon_schedule)
        count = clamp(round(Int, first_fraction * length(t_data)), min_pts,
                      length(t_data))
        @warn "Full-horizon initial solve failed; starting curriculum." exception = error
        make_loss(data[:, 1:count], t_data[1:count],
                  (tspan[1], t_data[count]))(p_init, nothing)
    end
    history = Float64[]
    params = p_init
    current_optimizer_state = optimizer_state
    current_outer = Ref(initial_outer)
    current_stage = Ref(initial_stage)
    current_stage_iteration = Ref(initial_stage_iteration)
    current_previous_residual = Ref(previous_residual_init)
    checkpoint_hook = if checkpoint_path === nothing || checkpoint_every ≤ 0
        nothing
    else
        function (current_params, current_state, _)
            current_stage_iteration[] += 1
            absolute_iteration = initial_iteration + length(history)
            absolute_iteration % checkpoint_every == 0 || return nothing
            metadata = (
                run = RunMetadata(
                    seed = seed,
                    data_hash = data_fingerprint(data, t_data, u0),
                    config = Dict(:training => training_config)),
                dual = copy(dual),
                rho = ρ,
                outer = current_outer[],
                stage = current_stage[],
                stage_iteration = current_stage_iteration[],
                previous_residual = current_previous_residual[],
            )
            save_checkpoint(
                checkpoint_path,
                Checkpoint(CHECKPOINT_SCHEMA_VERSION, current_params,
                           current_state, absolute_iteration, metadata))
            return nothing
        end
    end
    outer_iterations = training_config.constraint isa
        AugmentedLagrangianConfig ?
        training_config.constraint.outer_iterations : 1
    previous_residual = previous_residual_init
    for outer in initial_outer:outer_iterations
        min_frac = _horizon_minimum_fraction(training_config.horizon_schedule)
        min_pts = _horizon_min_points(training_config.horizon_schedule)
        schedule = sort(unique(clamp.(
            _horizon_fractions(training_config.horizon_schedule),
            min_frac, 1.0)))
        for (stage, fraction) in pairs(schedule)
            outer == initial_outer && stage < initial_stage && continue
            current_outer[] = outer
            current_stage[] = stage
            count = clamp(round(Int, fraction * length(t_data)), min_pts,
                          length(t_data))
            local_times = t_data[1:count]
            local_data = data[:, 1:count]
            local_span = (tspan[1], local_times[end])
            final_stage = outer == outer_iterations &&
                          stage == length(schedule)
            stage_config = _stage_config(
                training_config, length(schedule), final_stage)
            if outer == initial_outer && stage == initial_stage &&
               initial_stage_iteration > 0
                stage_config = TrainingConfig(stage_config;
                    adam_iterations = max(
                        0, stage_config.adam_iterations -
                           initial_stage_iteration))
            else
                current_stage_iteration[] = 0
            end
            verbose && println(
                "  → AL $outer/$outer_iterations, horizon ",
                round(fraction; digits = 2))
            params, current_optimizer_state = _optimize_stage(
                params, make_loss(local_data, local_times, local_span),
                stage_config, history, diag; verbose,
                optimizer_state = current_optimizer_state,
                checkpoint_hook)
        end
        if training_config.constraint isa AugmentedLagrangianConfig
            prediction = predict_ude(
                params, u0, tspan, t_data, nn, st;
                model = model, network = network,
                solver_config = training_config.solver)
            constraints = _constraint_values(
                prediction, training_config.constraint)
            residual = maximum(constraints)
            dual .= max.(zero(eltype(dual)), dual .+ ρ .* constraints)
            strategy = training_config.constraint
            if residual > strategy.progress_ratio * previous_residual
                ρ = min(strategy.max_ρ, strategy.growth * ρ)
            end
            previous_residual = residual
            current_previous_residual[] = residual
            residual ≤ strategy.tolerance && break
        end
    end
    final_loss = full_loss(params, nothing)
    metadata = RunMetadata(
        seed = seed,
        data_hash = data_fingerprint(data, t_data, u0),
        config = Dict(:training => training_config))
    diagnostics = (
        mse = diag.mse, constraint = diag.constraint,
        primal_residual = diag.primal_residual, dual = copy(dual), ρ = ρ,
        final_gradient_norm = isempty(diag.gradient_norm_history) ? NaN :
            last(diag.gradient_norm_history),
        gradient_norm_history = copy(diag.gradient_norm_history),
        bfgs = (attempted = diag.bfgs_attempted, success = diag.bfgs_success,
                retcode = diag.bfgs_retcode, message = diag.bfgs_message))
    converged = _training_converged(
        final_loss, initial_loss, training_config, diag)
    retcode = _training_retcode(converged, diag)
    return TrainingResult(params, history, initial_loss, final_loss,
                          metadata, diagnostics, converged, retcode)
end

"""
    train_ude(p_init, data, t_data, u0, tspan, model::UDEModel; kwargs...)

Fit physical (and optional neural) parameters of a compiled UDE to one
trajectory. Use `train_experiments` for masked multi-replicate data.
"""
function train_ude(p_init, data, t_data, u0, tspan, model::UDEModel; kwargs...)
    return train_ude(
        p_init, data, t_data, u0, tspan, model.nn, model.st;
        model = model, network = model.network, kwargs...)
end

function loss_mse(p, data, t_data, u0, tspan, model::UDEModel; kwargs...)
    return loss_mse(p, data, t_data, u0, tspan, model.nn, model.st;
                    model = model, network = model.network, kwargs...)
end

"""
    train_experiments(p_init, set, model; config, execution, ...)

Fit a UDE to an `ExperimentSet`. Adam follows `execution.batch_size`.
BFGS always refines the joint loss over every experiment, not the last
minibatch.
"""
function train_experiments(p_init, set::ExperimentSet, nn, st;
                           model::Union{Nothing,UDEModel} = nothing,
                           network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
                           config::TrainingConfig = TrainingConfig(),
                           execution::ExecutionConfig = ExecutionConfig(),
                           verbose::Bool = true, seed::Integer = 0)
    diag = LossDiagnostics()
    dual = zeros(eltype(p_init), length(set.state_names))
    ρ = config.constraint isa AugmentedLagrangianConfig ?
        config.constraint.initial_ρ : zero(eltype(p_init))
    function batch_loss(p, experiments)
        total = zero(eltype(p))
        weight_sum = zero(eltype(p))
        for experiment in experiments
            weight = Zygote.@ignore experiment_weight(experiment)
            scale = Zygote.@ignore experiment_noise_scale(experiment)
            total += weight * loss_mse(
                p, experiment.observations, experiment.times, experiment.u0,
                (first(experiment.times), last(experiment.times)), nn, st;
                model = model, network = network,
                constraint = config.constraint, dual, ρ,
                solver_config = config.solver, mask = experiment.mask,
                diagnostics = nothing) / (scale^2)
            weight_sum += weight
        end
        return total / max(weight_sum, one(eltype(p_init)))
    end
    objective = (p, _) -> batch_loss(p, set.experiments)
    initial = objective(p_init, nothing)
    history = Float64[]
    params = p_init
    optimizer_state = nothing
    outer_iterations = config.constraint isa AugmentedLagrangianConfig ?
        config.constraint.outer_iterations : 1
    previous_residual = Inf
    for outer in 1:outer_iterations
        batches = experiment_batches(
            set, min(execution.batch_size, length(set));
            shuffle = !execution.deterministic,
            rng = MersenneTwister(seed + outer - 1))
        for batch in batches
            batch_objective = (p, _) -> batch_loss(p, batch)
            # Adam only on this minibatch. Joint BFGS runs once after the
            # outer loop so the last IC cannot monopolize the second-order step.
            batch_config = _stage_config(
                config, outer_iterations * length(batches), false)
            params, optimizer_state = _optimize_stage(
                params, batch_objective, batch_config, history, diag; verbose,
                optimizer_state)
        end
        if config.constraint isa AugmentedLagrangianConfig
            experiment_constraints = map(set.experiments) do experiment
                prediction = predict_ude(
                    params, experiment.u0,
                    (first(experiment.times), last(experiment.times)),
                    experiment.times, nn, st; solver_config = config.solver)
                _constraint_values(prediction, config.constraint)
            end
            constraints = reduce((left, right) -> max.(left, right),
                                 experiment_constraints)
            residual = max(zero(eltype(constraints)), maximum(constraints))
            dual .= max.(zero(eltype(dual)), dual .+ ρ .* constraints)
            if residual >
               config.constraint.progress_ratio * previous_residual
                ρ = min(
                    config.constraint.max_ρ,
                    config.constraint.growth * ρ)
            end
            previous_residual = residual
            residual ≤ config.constraint.tolerance && break
        end
    end
    if config.bfgs_iterations > 0
        polish = TrainingConfig(config;
            adam_iterations = 0,
            bfgs_iterations = config.bfgs_iterations)
        params, optimizer_state = _optimize_stage(
            params, objective, polish, history, diag; verbose,
            optimizer_state)
    end
    final = objective(params, nothing)
    metadata = RunMetadata(
        seed = seed, data_hash = experiment_fingerprint(set),
        config = Dict(:training => config, :experiments => length(set)))
    converged = _training_converged(final, initial, config, diag)
    retcode = _training_retcode(converged, diag)
    return TrainingResult(
        params, history, initial, final, metadata,
        (experiment_count = length(set), dual = copy(dual), ρ = ρ,
         primal_residual =
             config.constraint isa AugmentedLagrangianConfig ?
             max(zero(ρ), previous_residual) : zero(ρ),
         final_gradient_norm = isempty(diag.gradient_norm_history) ? NaN :
             last(diag.gradient_norm_history),
         gradient_norm_history = copy(diag.gradient_norm_history),
         bfgs = (attempted = diag.bfgs_attempted, success = diag.bfgs_success,
                 retcode = diag.bfgs_retcode, message = diag.bfgs_message)),
        converged, retcode)
end

function train_experiments(p_init, set::ExperimentSet, model::UDEModel; kwargs...)
    return train_experiments(
        p_init, set, model.nn, model.st;
        model = model, network = model.network, kwargs...)
end

function save_checkpoint(path::AbstractString, checkpoint::Checkpoint)
    directory = dirname(abspath(path))
    isdir(directory) ||
        throw(ArgumentError("checkpoint directory does not exist: $directory"))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        serialize(io, checkpoint)
    end
    mv(temporary, path; force = true)
    return path
end

function load_checkpoint(path::AbstractString)
    checkpoint = open(deserialize, path)
    checkpoint isa Checkpoint ||
        throw(ArgumentError("file does not contain a BioDynaX checkpoint"))
    checkpoint.schema_version.major == CHECKPOINT_SCHEMA_VERSION.major ||
        throw(ArgumentError("incompatible checkpoint schema"))
    return checkpoint
end

"""
    resume_training(checkpoint, args...; kwargs...)

Resume Adam training from versioned parameters, Optimisers state, iteration,
Augmented-Lagrangian dual variables and penalty state. BFGS is intentionally a
terminal refinement stage and is restarted if requested after resumption.
"""
function resume_training(checkpoint::Checkpoint, args...; kwargs...)
    metadata = checkpoint.metadata
    dual = hasproperty(metadata, :dual) ? metadata.dual : nothing
    rho = hasproperty(metadata, :rho) ? metadata.rho : nothing
    outer = hasproperty(metadata, :outer) ? metadata.outer : 1
    stage = hasproperty(metadata, :stage) ? metadata.stage : 1
    stage_iteration = hasproperty(metadata, :stage_iteration) ?
        metadata.stage_iteration : 0
    previous_residual = hasproperty(metadata, :previous_residual) ?
        metadata.previous_residual : Inf
    return train_ude(
        checkpoint.params, args...;
        optimizer_state = checkpoint.optimizer_state,
        initial_iteration = checkpoint.iteration,
        dual_init = dual,
        rho_init = rho,
        initial_outer = outer,
        initial_stage = stage,
        initial_stage_iteration = stage_iteration,
        previous_residual_init = previous_residual,
        kwargs...)
end
