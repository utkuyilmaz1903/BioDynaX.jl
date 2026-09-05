"""
SciML Optimization.jl integration for UDE parameter estimation.
"""

"""
    build_optimization_problem(model, p_init, data, t_data, u0, tspan;
                               config=TrainingConfig(), mask=trues(data))

Construct an `Optimization.OptimizationProblem` for a compiled UDE.
"""
function build_optimization_problem(
        model::UDEModel, p_init, data, t_data, u0, tspan;
        config::TrainingConfig = TrainingConfig(),
        mask = trues(size(data)),
        dual = zeros(eltype(p_init), size(data, 1)),
        ρ = config.constraint isa AugmentedLagrangianConfig ?
            config.constraint.initial_ρ : zero(eltype(p_init)))
    function objective(p, _)
        loss_mse(
            p, data, t_data, u0, tspan, model;
            constraint = config.constraint, dual, ρ,
            solver_config = config.solver, mask = mask)
    end
    return Optimization.OptimizationProblem(
        Optimization.OptimizationFunction(objective, Optimization.AutoZygote()),
        p_init),
    objective
end

"""
    solve_optimization(prob; maxiters, algorithm=OptimizationOptimJL.BFGS())

Integrate a BioDynaX optimization problem with Optimization.jl defaults.
"""
function solve_optimization(prob::Optimization.OptimizationProblem;
        maxiters::Int = 100,
        algorithm = OptimizationOptimJL.BFGS())
    return Optimization.solve(prob, algorithm; maxiters = maxiters)
end

"""
    train_via_optimization(model, p_init, data, t_data, u0, tspan; kwargs...)

One-shot training through Optimization.jl (BFGS by default).
"""
function train_via_optimization(model::UDEModel, p_init, data, t_data, u0, tspan;
        config::TrainingConfig = TrainingConfig(),
        maxiters::Int = 100,
        algorithm = OptimizationOptimJL.BFGS(),
        seed::Integer = 0,
        verbose::Bool = false,
        kwargs...)
    prob, objective = build_optimization_problem(
        model, p_init, data, t_data, u0, tspan; config = config, kwargs...)
    initial_loss = objective(p_init, nothing)
    verbose && println("Optimization initial loss: ", initial_loss)
    opt_result = solve_optimization(prob; maxiters = maxiters, algorithm = algorithm)
    final_loss = objective(opt_result.u, nothing)
    metadata = RunMetadata(
        seed = seed,
        data_hash = data_fingerprint(data, t_data, u0),
        config = Dict(:training => config))
    converged = isfinite(final_loss) && final_loss ≤ initial_loss
    retcode = converged ? Success : NotConverged
    return TrainingResult(
        opt_result.u, Float64[initial_loss, final_loss],
        initial_loss, final_loss, metadata,
        (mse = final_loss, constraint = 0.0, primal_residual = 0.0,
            final_gradient_norm = NaN, gradient_norm_history = Float64[],
            bfgs = (attempted = true, success = converged, retcode = :success,
                message = "Optimization.jl solve")),
        converged, retcode)
end
