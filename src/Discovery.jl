struct ImplicitCandidate{T}
    target::Int
    specification::LocalBasisSpec
    numerator_coefficients::Vector{T}
    denominator_coefficients::Vector{T}
    selection_frequency::Vector{T}
    validation_error::T
    denominator_minimum::T
end

function discover_equations(p_trained, model::UDEModel; kwargs...)
    return discover_equations(
        p_trained, model.nn, model.st; network = model.network, kwargs...)
end

function sample_learned_function(p_trained, nn, st;
                                 regulator_index::Int = 2,
                                 input_range = range(1e-3, 2.5; length = 400))
    xs = collect(input_range)
    ys = map(xs) do value
        output, _ = nn([value], p_trained.nn, st)
        output[1]
    end
    keep = isfinite.(ys)
    return reshape(xs[keep], 1, :), reshape(ys[keep], 1, :)
end

function sample_learned_function(p_trained, model::UDEModel; kwargs...)
    return sample_learned_function(p_trained, model.nn, model.st; kwargs...)
end

function _collect_trajectory_data(p_trained, nn, st, u0, tspan, n_samples;
                                  model::Union{Nothing,UDEModel} = nothing,
                                  solver = SolverConfig())
    times = collect(range(tspan[1], tspan[2]; length = n_samples))
    if model === nothing
        X = predict_ude(p_trained, u0, tspan, times, nn, st;
                        solver_config = solver)
        derivatives = reduce(hcat,
            (ude_system(@view(X[:, i]), p_trained, times[i], nn, st)
             for i in 1:size(X, 2)))
    else
        X = predict_ude(p_trained, u0, tspan, times, model;
                        solver_config = solver)
        derivatives = reduce(hcat,
            (ude_system(@view(X[:, i]), p_trained, times[i], model)
             for i in 1:size(X, 2)))
    end
    keep = vec(all(isfinite, X; dims = 1) .&
               all(isfinite, derivatives; dims = 1))
    return X[:, keep], derivatives[:, keep], times[1:size(X, 2)][keep]
end

function _collect_trajectory_data(p_trained, model::UDEModel, u0, tspan, n_samples;
                                  solver = SolverConfig())
    return _collect_trajectory_data(
        p_trained, model.nn, model.st, u0, tspan, n_samples;
        model = model, solver = solver)
end

function _collect_multi_trajectory_data(p_trained, model::UDEModel,
                                          experiments::Vector{Experiment},
                                          n_samples; solver = SolverConfig())
    blocks = map(experiments) do experiment
        span = (first(experiment.times), last(experiment.times))
        X, derivatives, _ = _collect_trajectory_data(
            p_trained, model, experiment.u0, span, n_samples;
            solver = solver)
        (X, derivatives)
    end
    X = hcat(first.(blocks)...)
    derivatives = hcat(last.(blocks)...)
    return X, derivatives
end

function _stlsq(A, y, threshold; max_iterations = 20, ridge = 1e-10)
    scales = vec(sqrt.(sum(abs2, A; dims = 1)))
    scales .= max.(scales, eps(eltype(scales)))
    normalized = A ./ reshape(scales, 1, :)
    active = trues(size(A, 2))
    coefficients = zeros(eltype(A), size(A, 2))
    for _ in 1:max_iterations
        indices = findall(active)
        isempty(indices) && break
        local_A = @view normalized[:, indices]
        # QR on an augmented system avoids squaring the condition number as
        # normal equations do.
        augmented_A = vcat(local_A,
                           sqrt(ridge) * Matrix{eltype(A)}(
                               I, length(indices), length(indices)))
        augmented_y = vcat(y, zeros(eltype(y), length(indices)))
        local_coefficients = augmented_A \ augmented_y
        coefficients .= zero(eltype(coefficients))
        coefficients[indices] .= local_coefficients
        next_active = abs.(coefficients) .≥ threshold
        next_active == active && break
        active = next_active
    end
    return coefficients ./ scales
end

function _block_bootstrap_indices(rng, indices, block_length)
    output = Int[]
    while length(output) < length(indices)
        start = rand(rng, 1:max(1, length(indices) - block_length + 1))
        stop = min(length(indices), start + block_length - 1)
        append!(output, @view indices[start:stop])
    end
    resize!(output, length(indices))
    return output
end

function _fit_implicit(spec::LocalBasisSpec, X, derivative, indices,
                       threshold)
    local_X = @view X[:, indices]
    y = @view derivative[indices]
    numerator = evaluate_library(spec.numerator, local_X)
    denominator = evaluate_library(spec.denominator, local_X)
    design = hcat(numerator, -(reshape(y, :, 1) .* denominator))
    coefficients = _stlsq(design, y, threshold)
    split = length(spec.numerator)
    numerator_coefficients = coefficients[1:split]
    denominator_coefficients = coefficients[(split + 1):end]
    enforce_hierarchy!(
        numerator_coefficients, spec.numerator, threshold)
    enforce_hierarchy!(
        denominator_coefficients, spec.denominator, threshold)
    return numerator_coefficients, denominator_coefficients
end

function _evaluate_candidate(spec, numerator_coefficients,
                             denominator_coefficients, X)
    numerator = evaluate_library(spec.numerator, X) *
                numerator_coefficients
    denominator = one(eltype(X)) .+
                  evaluate_library(spec.denominator, X) *
                  denominator_coefficients
    return numerator ./ denominator, denominator
end

function _bootstrap_frequency(rng, spec, X, derivative, train_indices,
                              threshold, bootstrap_samples)
    term_count = length(spec.numerator) + length(spec.denominator)
    selected = zeros(Float64, term_count)
    bootstrap_samples == 0 && return selected
    block_length = max(2, round(Int, sqrt(length(train_indices))))
    for _ in 1:bootstrap_samples
        indices = _block_bootstrap_indices(
            rng, train_indices, block_length)
        numerator, denominator =
            _fit_implicit(spec, X, derivative, indices, threshold)
        selected .+= .!iszero.(vcat(numerator, denominator))
    end
    return selected ./ bootstrap_samples
end

function _consensus_refit(spec, X, derivative, indices, threshold,
                          frequencies; minimum_frequency = 0.8)
    local_X = @view X[:, indices]
    y = @view derivative[indices]
    numerator = evaluate_library(spec.numerator, local_X)
    denominator = evaluate_library(spec.denominator, local_X)
    design = hcat(numerator, -(reshape(y, :, 1) .* denominator))
    support = frequencies .≥ minimum_frequency
    any(support) || return _fit_implicit(
        spec, X, derivative, indices, threshold)
    coefficients = zeros(eltype(X), size(design, 2))
    coefficients[support] .= (@view design[:, support]) \ y
    split = length(spec.numerator)
    numerator_coefficients = coefficients[1:split]
    denominator_coefficients = coefficients[(split + 1):end]
    enforce_hierarchy!(
        numerator_coefficients, spec.numerator, threshold)
    enforce_hierarchy!(
        denominator_coefficients, spec.denominator, threshold)
    return numerator_coefficients, denominator_coefficients
end

function _format_side(coefficients, terms)
    pieces = String[]
    for (coefficient, term) in zip(coefficients, terms)
        abs(coefficient) ≤ 1e-10 && continue
        push!(pieces, string(round(coefficient; sigdigits = 5), "*",
                             term.label))
    end
    return isempty(pieces) ? "0" : join(pieces, " + ")
end

function format_equation(candidate::ImplicitCandidate)
    numerator = _format_side(candidate.numerator_coefficients,
                             candidate.specification.numerator)
    denominator = _format_side(candidate.denominator_coefficients,
                               candidate.specification.denominator)
    denominator = denominator == "0" ? "1" : "1 + " * denominator
    return "dx[$(candidate.target)]/dt = ($numerator) / ($denominator)"
end

function _discover_implicit(X, derivatives, network, backend::ImplicitSINDyPI,
                            config::DiscoveryConfig)
    rng = MersenneTwister(config.seed)
    sample_count = size(X, 2)
    validation_count = clamp(
        round(Int, backend.validation_fraction * sample_count), 1,
        sample_count - 2)
    # A contiguous hold-out block prevents temporal leakage from adjacent
    # points of the same trajectory.
    training_indices = collect(1:(sample_count - validation_count))
    validation_indices =
        collect((sample_count - validation_count + 1):sample_count)
    candidates = ImplicitCandidate[]

    for target in axes(X, 1)
        derivative = vec(@view derivatives[target, :])
        spec = local_basis(
            network, target;
            degree = max(backend.max_degree, backend.max_hill_degree),
            max_variables = backend.max_parents,
            include_interactions = config.include_interactions,
            X = X,
            derivative = derivative,
            extra_candidates = backend.extra_candidates)
        frequencies = _bootstrap_frequency(
            rng, spec, X, derivative, training_indices, backend.threshold,
            backend.bootstrap_samples)
        numerator, denominator = _consensus_refit(
            spec, X, derivative, training_indices, backend.threshold,
            frequencies)
        prediction, denominator_values = _evaluate_candidate(
            spec, numerator, denominator, @view X[:, validation_indices])
        denominator_minimum = minimum(denominator_values)
        denominator_minimum ≥ backend.denominator_floor ||
            throw(DomainError(
                denominator_minimum,
                "discovered denominator is singular on validation data"))
        error = mean(abs2,
                     prediction .- derivative[validation_indices])
        push!(candidates, ImplicitCandidate(
            target, spec, numerator, denominator, frequencies, error,
            denominator_minimum))
    end
    return candidates
end

"""
    discover_equations(p_trained, nn, st; ...)

Discover graph-local rational dynamics with SINDy-PI's implicit identity
`D(z)ẋ - N(z)=0`. Unknown numerator and denominator coefficients are learned
jointly; no fixed-denominator rational atoms or Taylor approximations are used.
"""
function discover_equations(p_trained, nn, st;
                            network::BiologicalNetwork = DEFAULT_EXAMPLE_NETWORK,
                            u0::Vector{Float64} = [0.2, 0.1],
                            tspan::Tuple{Float64,Float64} = (0.0, 20.0),
                            n_samples::Int = 300,
                            config::DiscoveryConfig = DiscoveryConfig(),
                            polynomial_degree::Int =
                                config.backend isa ImplicitSINDyPI ?
                                config.backend.max_degree : 3,
                            sparsity_threshold =
                                config.backend isa ImplicitSINDyPI ?
                                config.backend.threshold : 1e-2,
                            verbose::Bool = true,
                            kwargs...)
    n_samples ≥ 20 ||
        throw(ArgumentError("n_samples must be at least 20"))
    backend = config.backend isa ImplicitSINDyPI ?
        ImplicitSINDyPI(
            threshold = float(sparsity_threshold),
            max_degree = polynomial_degree,
            max_hill_degree = config.backend.max_hill_degree,
            max_parents = config.backend.max_parents,
            extra_candidates = config.backend.extra_candidates,
            bootstrap_samples = config.backend.bootstrap_samples,
            validation_fraction = config.backend.validation_fraction,
            denominator_floor = config.backend.denominator_floor) :
        config.backend
    try
        X, derivatives, _ = _collect_trajectory_data(
            p_trained, nn, st, u0, tspan, n_samples)
        size(X, 2) ≥ 20 ||
            throw(ArgumentError("insufficient finite trajectory samples"))
        backend isa ImplicitSINDyPI ||
            throw(ArgumentError("unsupported discovery backend $(typeof(backend))"))
        candidates = _discover_implicit(
            X, derivatives, network, backend, config)
        equation_text = join(format_equation.(candidates), "\n")
        verbose && println("\n[Discovery] Implicit rational system:\n",
                           equation_text)
        metadata = RunMetadata(
            seed = config.seed,
            data_hash = data_fingerprint(X, derivatives),
            config = Dict(:backend => string(typeof(backend)),
                          :samples => size(X, 2)))
        return DiscoveryResult(true, "ok", equation_text,
                               getfield.(candidates, :specification), nothing,
                               candidates, metadata)
    catch error
        message = "Implicit discovery failed: " * sprint(showerror, error)
        @warn message
        return DiscoveryResult(false, message, nothing, nothing, nothing,
                               ImplicitCandidate[], RunMetadata(seed = config.seed))
    end
end
