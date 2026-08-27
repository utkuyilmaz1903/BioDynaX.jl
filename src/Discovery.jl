"""
    ExplicitCandidate

Explicit STLSQ candidate (`ẋ = Φ(x)ξ`) for one target state.
"""
struct ExplicitCandidate{T}
    target::Int
    specification::LocalBasisSpec
    coefficients::Vector{T}
    validation_error::T
end

"""
    ImplicitCandidate

Implicit SINDy-PI candidate for a rational rate
`D(z)ẋ − N(z) = 0` on one target.
"""
struct ImplicitCandidate{T}
    target::Int
    specification::LocalBasisSpec
    numerator_coefficients::Vector{T}
    denominator_coefficients::Vector{T}
    selection_frequency::Vector{T}
    validation_error::T
    denominator_minimum::T
end

"""
    discover_equations(p_trained, model::UDEModel; kwargs...)
    discover_equations(p_trained, model, set::ExperimentSet; kwargs...)
    discover_equations(X, times, network; kwargs...)

Recover graph-local equations from a trained UDE, several experiments, or raw
trajectories. Failures set `DiscoveryResult.retcode`; pass `strict=true` to throw.
The raw-data method accepts `targets` to recover a subset of state rows.
"""
function discover_equations(p_trained, model::UDEModel; kwargs...)
    return discover_equations(
        p_trained, model.nn, model.st;
        network = model.network, model = model, kwargs...)
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
                                          experiments::AbstractVector{<:Experiment},
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

"""
    _stlsq_blocked(A, y, threshold; chunk_size=256, ...)

Streaming STLSQ that accumulates ridge Gram blocks (`A'A`, `A'y`) so large
design matrices need not be factored in one shot. Matches dense `_stlsq`
thresholding semantics.
"""
function _stlsq_blocked(A, y, threshold; max_iterations = 20, ridge = 1e-10,
                        chunk_size::Int = 256,
                        workspace::Union{Nothing,STLSQWorkspace} = nothing)
    n, p = size(A)
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    ws = workspace === nothing ?
        allocate_stlsq_workspace(eltype(A), n, p, chunk_size) :
        ensure_stlsq_workspace!(workspace, n, p, chunk_size)
    return collect(_stlsq_blocked!(
        ws, A, y, threshold; max_iterations = max_iterations, ridge = ridge))
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

function _implicit_design!(design, num_buf, den_buf, spec, local_X, y)
    evaluate_library!(num_buf, spec.numerator, local_X)
    evaluate_library!(den_buf, spec.denominator, local_X)
    n_num = size(num_buf, 2)
    design[:, 1:n_num] .= num_buf
    @inbounds for j in axes(den_buf, 2)
        design[:, n_num + j] .= .-(@view(den_buf[:, j]) .* y)
    end
    return design
end

function _fit_implicit(spec::LocalBasisSpec, X, derivative, indices,
                       threshold; chunk_size::Int = 256,
                       workspace::Union{Nothing,StreamingImplicitWorkspace} = nothing)
    return _fit_implicit_stream(
        spec, X, derivative, indices, threshold;
        chunk_size = chunk_size, workspace = workspace)
end

function _implicit_rss(spec, numerator, denominator, X, y)
    pred, denvals = _evaluate_candidate(spec, numerator, denominator, X)
    any(!isfinite, pred) && return Inf, pred, denvals
    return mean(abs2, pred .- y), pred, denvals
end

_unsafe_denominator(dvals, floor) =
    isempty(dvals) || any(!isfinite, dvals) || minimum(dvals) < floor

function _refit_masked_implicit(spec, X, y, num_keep, den_keep, threshold;
                                workspace::Union{Nothing,ImplicitLibraryWorkspace} = nothing)
    n = size(X, 2)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    ws = workspace === nothing ?
        allocate_implicit_workspace(eltype(X), n, n_num, n_den, 256) :
        ensure_implicit_workspace!(workspace, n, n_num, n_den, 256)
    return _refit_masked_implicit!(
        ws, spec, X, y, num_keep, den_keep, threshold)
end

function _refit_masked_implicit!(ws::ImplicitLibraryWorkspace, spec, X, y,
        num_keep, den_keep, threshold)
    n = size(X, 2)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    length(y) == n || throw(DimensionMismatch("y must match X columns"))
    design = implicit_design!(ws, spec, X, y)
    keep = vcat(num_keep, den_keep)
    idx = findall(keep)
    coeffs = zeros(eltype(X), n_num + n_den)
    if !isempty(idx)
        coeffs[idx] .= (@view design[:, idx]) \ y
    end
    numerator = coeffs[1:n_num]
    denominator = coeffs[(n_num + 1):end]
    numerator[.!num_keep] .= 0
    denominator[.!den_keep] .= 0
    pred, _ = evaluate_candidate!(ws, spec, numerator, denominator, X)
    if !isempty(idx) && all(isfinite, pred)
        pred_vec = @view ws.pred[1:n]
        design = implicit_design!(ws, spec, X, pred_vec)
        coeffs[idx] .= (@view design[:, idx]) \ y
        numerator = coeffs[1:n_num]
        denominator = coeffs[(n_num + 1):end]
        numerator[.!num_keep] .= 0
        denominator[.!den_keep] .= 0
    end
    enforce_hierarchy!(numerator, spec.numerator, threshold)
    enforce_hierarchy!(denominator, spec.denominator, threshold)
    return numerator, denominator
end

function _active_term_indices(numerator, denominator)
    return (findall(c -> abs(c) > 1e-8, numerator),
            findall(c -> abs(c) > 1e-8, denominator))
end

"""Drop nested monomials on the same library when held-out RSS / BIC do not suffer."""
function prune_nested_implicit(spec, numerator, denominator, X, y, threshold;
                               floor::Real = 1e-8, rtol::Real = 0.02,
                               workspace::Union{Nothing,ImplicitLibraryWorkspace} = nothing)
    numerator = copy(numerator)
    denominator = copy(denominator)
    # Bootstrap index views are not dense. Workspace mul! / design fills
    # assume contiguous sample columns.
    X = X isa Matrix ? X : Matrix(X)
    y = y isa Vector ? y : collect(y)
    rss0, _, denvals0 = _implicit_rss(spec, numerator, denominator, X, y)
    (!isfinite(rss0) || _unsafe_denominator(denvals0, floor)) &&
        return numerator, denominator
    num_idx, den_idx = _active_term_indices(numerator, denominator)
    n_act = length(num_idx) + length(den_idx)
    n_act == 0 && return numerator, denominator
    n = length(y)
    n_val = n ≥ 40 ? max(8, round(Int, 0.2 * n)) : 0
    ws = workspace === nothing ?
        allocate_implicit_workspace(
            eltype(X), n, length(spec.numerator), length(spec.denominator), 256) :
        ensure_implicit_workspace!(
            workspace, n, length(spec.numerator), length(spec.denominator), 256)
    if n_act ≤ 12
        return _subset_implicit_prune(
            spec, X, y, threshold, floor, rtol, rss0,
            num_idx, den_idx, n_val; workspace = ws)
    end
    return _greedy_implicit_prune(
        spec, numerator, denominator, X, y, threshold, floor, rtol, rss0;
        workspace = ws)
end

function _subset_implicit_prune(spec, X, y, threshold, floor, rtol, _rss0,
                                num_idx, den_idx, n_val;
                                workspace::Union{Nothing,ImplicitLibraryWorkspace} = nothing)
    n = length(y)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    n_num_act = length(num_idx)
    n_den_act = length(den_idx)
    n_act = n_num_act + n_den_act
    train = n_val == 0 ? (1:n) : (1:(n - n_val))
    val = n_val == 0 ? (1:n) : ((n - n_val + 1):n)
    Xtr = @view X[:, train]
    ytr = @view y[train]
    Xval = @view X[:, val]
    yval = @view y[val]
    candidates = NamedTuple[]
    for bits in 1:((1 << n_act) - 1)
        keep_n = falses(n_num)
        keep_d = falses(n_den)
        @inbounds for i in 1:n_num_act
            (bits & (1 << (i - 1))) != 0 && (keep_n[num_idx[i]] = true)
        end
        any(keep_n) || continue
        @inbounds for j in 1:n_den_act
            (bits & (1 << (n_num_act + j - 1))) != 0 && (keep_d[den_idx[j]] = true)
        end
        n2, d2 = _refit_masked_implicit(
            spec, Xtr, ytr, keep_n, keep_d, threshold; workspace = workspace)
        rss_fit, _, dvals_fit = _implicit_rss(spec, n2, d2, Xtr, ytr)
        rss_val, _, dvals_val = _implicit_rss(spec, n2, d2, Xval, yval)
        _unsafe_denominator(dvals_fit, floor) && continue
        _unsafe_denominator(dvals_val, floor) && continue
        (!isfinite(rss_fit) || !isfinite(rss_val)) && continue
        k = count(c -> abs(c) > 1e-8, vcat(n2, d2))
        # BIC on the fit set is the Occam score. Raw held-out RSS has no
        # complexity penalty and keeps extra monomials on 0.5% noise.
        # Negative BIC * (1 + rtol) previously emptied the admissible set.
        score = information_criterion(length(ytr), rss_fit * length(ytr), k;
                                      criterion = :bic)
        push!(candidates, (num = n2, den = d2, score = score, k = k, rss = rss_val))
    end
    isempty(candidates) && return _refit_masked_implicit(
        spec, X, y,
        [i in num_idx for i in 1:n_num],
        [i in den_idx for i in 1:n_den], threshold; workspace = workspace)
    finite_candidates = [c for c in candidates if isfinite(c.score) && isfinite(c.rss)]
    isempty(finite_candidates) && return _refit_masked_implicit(
        spec, X, y,
        [i in num_idx for i in 1:n_num],
        [i in den_idx for i in 1:n_den], threshold; workspace = workspace)
    best_score = minimum(c.score for c in finite_candidates)
    slack = max(abs(best_score) * rtol, 1e-12)
    admissible = [c for c in finite_candidates if c.score ≤ best_score + slack]
    isempty(admissible) && return _refit_masked_implicit(
        spec, X, y,
        [i in num_idx for i in 1:n_num],
        [i in den_idx for i in 1:n_den], threshold; workspace = workspace)
    chosen = argmin(c -> (c.k, c.rss), admissible)
    keep_n = abs.(chosen.num) .> 1e-8
    keep_d = abs.(chosen.den) .> 1e-8
    any(keep_n) || return chosen.num, chosen.den
    return _refit_masked_implicit(
        spec, X, y, keep_n, keep_d, threshold; workspace = workspace)
end

function _greedy_implicit_prune(spec, numerator, denominator, X, y, threshold,
                                floor, rtol, rss0;
                                workspace::Union{Nothing,ImplicitLibraryWorkspace} = nothing)
    n = length(y)
    k0 = count(c -> abs(c) > 1e-8, vcat(numerator, denominator))
    bic0 = information_criterion(n, rss0 * n, k0; criterion = :bic)
    for _ in 1:(length(numerator) + length(denominator))
        num_active = abs.(numerator) .> 1e-8
        den_active = abs.(denominator) .> 1e-8
        best_bic = bic0
        best_num = numerator
        best_den = denominator
        improved = false
        for i in findall(num_active)
            count(num_active) == 1 && continue
            keep_n = copy(num_active)
            keep_n[i] = false
            n2, d2 = _refit_masked_implicit(
                spec, X, y, keep_n, den_active, threshold; workspace = workspace)
            rss, _, dvals = _implicit_rss(spec, n2, d2, X, y)
            (_unsafe_denominator(dvals, floor) || rss > rss0 * (1 + rtol)) && continue
            k = count(c -> abs(c) > 1e-8, vcat(n2, d2))
            a = information_criterion(n, rss * n, k; criterion = :bic)
            if a < best_bic - 1e-9
                best_bic = a
                best_num, best_den = n2, d2
                improved = true
            end
        end
        for i in findall(den_active)
            keep_d = copy(den_active)
            keep_d[i] = false
            n2, d2 = _refit_masked_implicit(
                spec, X, y, num_active, keep_d, threshold; workspace = workspace)
            rss, _, dvals = _implicit_rss(spec, n2, d2, X, y)
            (_unsafe_denominator(dvals, floor) || rss > rss0 * (1 + rtol)) && continue
            k = count(c -> abs(c) > 1e-8, vcat(n2, d2))
            a = information_criterion(n, rss * n, k; criterion = :bic)
            if a < best_bic - 1e-9
                best_bic = a
                best_num, best_den = n2, d2
                improved = true
            end
        end
        improved || break
        numerator, denominator = best_num, best_den
        rss0, _, _ = _implicit_rss(spec, numerator, denominator, X, y)
        bic0 = best_bic
    end
    return numerator, denominator
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
                              threshold, bootstrap_samples;
                              chunk_size::Int = 256,
                              workspace::Union{Nothing,StreamingImplicitWorkspace} = nothing)
    term_count = length(spec.numerator) + length(spec.denominator)
    selected = zeros(Float64, term_count)
    bootstrap_samples == 0 && return selected
    block_length = max(2, round(Int, sqrt(length(train_indices))))
    n = length(train_indices)
    ws = workspace === nothing ?
        allocate_streaming_implicit_workspace(
            eltype(X), n, length(spec.numerator), length(spec.denominator),
            chunk_size) : workspace
    for _ in 1:bootstrap_samples
        indices = _block_bootstrap_indices(
            rng, train_indices, block_length)
        numerator, denominator =
            _fit_implicit(spec, X, derivative, indices, threshold;
                          chunk_size = chunk_size, workspace = ws)
        selected .+= .!iszero.(vcat(numerator, denominator))
    end
    return selected ./ bootstrap_samples
end

function _consensus_refit(spec, X, derivative, indices, threshold,
                          frequencies; minimum_frequency = 0.8,
                          chunk_size::Int = 256,
                          workspace::Union{Nothing,ImplicitLibraryWorkspace} = nothing)
    local_X = @view X[:, indices]
    n = length(indices)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    ws = workspace === nothing ?
        allocate_implicit_workspace(eltype(X), n, n_num, n_den, chunk_size) :
        ensure_implicit_workspace!(workspace, n, n_num, n_den, chunk_size)
    y = @view ws.y[1:n]
    @inbounds for i in 1:n
        y[i] = derivative[indices[i]]
    end
    design = implicit_design!(ws, spec, local_X, y)
    support = frequencies .≥ minimum_frequency
    any(support) || return _fit_implicit(
        spec, X, derivative, indices, threshold; chunk_size = chunk_size)
    coefficients = zeros(eltype(X), size(design, 2))
    coefficients[support] .= (@view design[:, support]) \ y
    split = n_num
    numerator_coefficients = coefficients[1:split]
    denominator_coefficients = coefficients[(split + 1):end]
    enforce_hierarchy!(
        numerator_coefficients, spec.numerator, threshold)
    enforce_hierarchy!(
        denominator_coefficients, spec.denominator, threshold)
    return prune_nested_implicit(
        spec, numerator_coefficients, denominator_coefficients,
        local_X, Vector(y), threshold; workspace = ws)
end

"""Deterministic orthant stress grid spanning observed data bounds."""
function _denominator_domain_grid(X; n::Int = 256, seed::Integer = 42)
    n ≤ 0 && return Matrix{eltype(X)}(undef, size(X, 1), 0)
    nstates = size(X, 1)
    lo = vec(minimum(X; dims = 2))
    hi = vec(maximum(X; dims = 2))
    span = hi .- lo
    pad = 0.05 .* max.(span, eps.(span))
    lo = max.(zero.(lo), lo .- pad)
    hi = hi .+ pad
    rng = MersenneTwister(seed)
    grid = Matrix{eltype(X)}(undef, nstates, n)
    @inbounds for j in 1:n
        for i in 1:nstates
            u = ((j - 1) + rand(rng)) / n
            grid[i, j] = lo[i] + u * (hi[i] - lo[i])
        end
    end
    return grid
end

function _denominator_minimum_over(spec, numerator_coefficients,
                                   denominator_coefficients, matrices...)
    minimum_value = typemax(eltype(first(matrices)))
    for matrix in matrices
        size(matrix, 2) == 0 && continue
        _, denominator_values = _evaluate_candidate(
            spec, numerator_coefficients, denominator_coefficients, matrix)
        minimum_value = min(minimum_value, minimum(denominator_values))
    end
    return minimum_value
end

function _check_denominator_safety(spec, numerator_coefficients,
                                   denominator_coefficients, train_X, val_X,
                                   domain_X, floor)
    minimum_value = _denominator_minimum_over(
        spec, numerator_coefficients, denominator_coefficients,
        train_X, val_X, domain_X)
    minimum_value ≥ floor ||
        throw(DomainError(
            minimum_value,
            "discovered denominator is singular on train/validation/domain data"))
    return minimum_value
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

function _latex_term_label(label::AbstractString)
    replaced = replace(label, r"x\[(\d+)\]" => s"x_{\1}")
    return replace(replaced, "*" => " ")
end

function _format_side_latex(coefficients, terms)
    pieces = String[]
    for (coefficient, term) in zip(coefficients, terms)
        abs(coefficient) ≤ 1e-10 && continue
        coeff = round(coefficient; sigdigits = 5)
        label = _latex_term_label(term.label)
        label == "1" ?
            push!(pieces, string(coeff)) :
            push!(pieces, string(coeff, " ", label))
    end
    return isempty(pieces) ? "0" : join(pieces, " + ")
end

function format_equation(candidate::ExplicitCandidate)
    side = _format_side(candidate.coefficients, candidate.specification.numerator)
    return "dx[$(candidate.target)]/dt = $side"
end

function format_equation(candidate::ImplicitCandidate)
    numerator = _format_side(candidate.numerator_coefficients,
                             candidate.specification.numerator)
    denominator = _format_side(candidate.denominator_coefficients,
                               candidate.specification.denominator)
    denominator = denominator == "0" ? "1" : "1 + " * denominator
    return "dx[$(candidate.target)]/dt = ($numerator) / ($denominator)"
end

"""
    equation_to_latex(candidate)

LaTeX form of a discovered candidate equation.
"""
function equation_to_latex(candidate::ExplicitCandidate)
    side = _format_side_latex(candidate.coefficients,
                              candidate.specification.numerator)
    return "\\dot{x}_{$(candidate.target)} = $side"
end

function equation_to_latex(candidate::ImplicitCandidate)
    numerator = _format_side_latex(candidate.numerator_coefficients,
                                   candidate.specification.numerator)
    denominator = _format_side_latex(candidate.denominator_coefficients,
                                     candidate.specification.denominator)
    denominator = denominator == "0" ? "1" : "1 + " * denominator
    return "\\dot{x}_{$(candidate.target)} = \\frac{$numerator}{$denominator}"
end

@inline function _eval_monomial(term::MonomialTerm, x::AbstractVector)
    isempty(term.variables) && return one(eltype(x))
    value = one(eltype(x))
    @inbounds for (variable, power) in zip(term.variables, term.powers)
        value *= x[variable]^power
    end
    return value
end

"""Executable closure `(x) -> dx_target` for a discovered candidate."""
function equation_to_function(candidate::ExplicitCandidate)
    coefficients = copy(candidate.coefficients)
    terms = candidate.specification.numerator
    return function (x::AbstractVector)
        value = zero(eltype(x))
        @inbounds for (coefficient, term) in zip(coefficients, terms)
            value += coefficient * _eval_monomial(term, x)
        end
        return value
    end
end

function equation_to_function(candidate::ImplicitCandidate)
    numerator_coefficients = copy(candidate.numerator_coefficients)
    denominator_coefficients = copy(candidate.denominator_coefficients)
    numerator_terms = candidate.specification.numerator
    denominator_terms = candidate.specification.denominator
    return function (x::AbstractVector)
        numerator = zero(eltype(x))
        @inbounds for (coefficient, term) in zip(numerator_coefficients,
                                                 numerator_terms)
            numerator += coefficient * _eval_monomial(term, x)
        end
        denominator = one(eltype(x))
        @inbounds for (coefficient, term) in zip(denominator_coefficients,
                                                 denominator_terms)
            denominator += coefficient * _eval_monomial(term, x)
        end
        return numerator / denominator
    end
end

"""
    export_rhs(result::DiscoveryResult)

Full-state RHS `(x) -> du` assembled from discovery candidates. Unknown-edge
recovery should use `compose_hybrid_rhs` instead: this exports a discovered
`ẋ`, not a compiled hybrid with known `P`.
"""
function export_rhs(result::DiscoveryResult)
    result.success ||
        throw(ArgumentError(
            "cannot export RHS from a failed discovery ($(result.retcode)): $(result.message)"))
    candidates = result.candidates
    isempty(candidates) &&
        throw(ArgumentError("DiscoveryResult has no candidates to export"))
    nstates = maximum(candidate.target for candidate in candidates)
    functions = Vector{Any}(undef, nstates)
    fill!(functions, nothing)
    for candidate in candidates
        functions[candidate.target] = equation_to_function(candidate)
    end
    any(isnothing, functions) &&
        throw(ArgumentError("missing candidate for one or more state indices"))
    return function (x::AbstractVector)
        length(x) == nstates ||
            throw(DimensionMismatch("state dimension mismatch"))
        du = similar(x)
        @inbounds for i in eachindex(du)
            du[i] = functions[i](x)
        end
        return du
    end
end

"""
    estimate_derivatives(X, times; method=:central)

Central finite differences on trajectory samples (one-sided at endpoints).
No external derivative package required.
"""
function estimate_derivatives(X::AbstractMatrix, times::AbstractVector;
                              method::Symbol = :central)
    method == :central ||
        throw(ArgumentError("unsupported derivative method $method"))
    nstates, n = size(X)
    length(times) == n ||
        throw(DimensionMismatch("times must match X columns"))
    n ≥ 2 || throw(ArgumentError("need at least two samples"))
    dX = similar(X)
    @inbounds for j in 1:n
        if j == 1
            dt = times[2] - times[1]
            dX[:, j] .= (X[:, 2] .- X[:, 1]) ./ dt
        elseif j == n
            dt = times[n] - times[n - 1]
            dX[:, j] .= (X[:, n] .- X[:, n - 1]) ./ dt
        else
            dt = times[j + 1] - times[j - 1]
            dX[:, j] .= (X[:, j + 1] .- X[:, j - 1]) ./ dt
        end
    end
    return dX
end

function information_criterion(n::Integer, rss, k::Integer; criterion::Symbol = :aic)
    n > 0 || throw(ArgumentError("n must be positive"))
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    rss_value = max(float(rss), eps(float(rss)))
    criterion == :aic && return n * log(rss_value / n) + 2k
    criterion == :bic && return n * log(rss_value / n) + k * log(n)
    throw(ArgumentError("criterion must be :aic or :bic"))
end

function score_candidate(candidate::ExplicitCandidate, X, derivative;
                         criterion::Symbol = :aic)
    prediction = evaluate_library(candidate.specification.numerator, X) *
                 candidate.coefficients
    rss = sum(abs2, prediction .- derivative)
    k = count(!iszero, candidate.coefficients)
    return information_criterion(length(derivative), rss, k; criterion)
end

function score_candidate(candidate::ImplicitCandidate, X, derivative;
                         criterion::Symbol = :aic)
    prediction, _ = _evaluate_candidate(
        candidate.specification, candidate.numerator_coefficients,
        candidate.denominator_coefficients, X)
    rss = sum(abs2, prediction .- derivative)
    k = count(!iszero, vcat(candidate.numerator_coefficients,
                            candidate.denominator_coefficients))
    return information_criterion(length(derivative), rss, k; criterion)
end

function _with_threshold(backend::ExplicitSTLSQ, threshold)
    return ExplicitSTLSQ(threshold = float(threshold))
end

function _with_threshold(backend::DataDrivenSparseSTLSQ, threshold)
    return DataDrivenSparseSTLSQ(
        threshold = float(threshold), ridge = backend.ridge)
end

function _with_threshold(backend::ImplicitSINDyPI, threshold)
    return ImplicitSINDyPI(
        threshold = float(threshold),
        max_degree = backend.max_degree,
        max_hill_degree = backend.max_hill_degree,
        max_parents = backend.max_parents,
        extra_candidates = backend.extra_candidates,
        bootstrap_samples = backend.bootstrap_samples,
        validation_fraction = backend.validation_fraction,
        denominator_floor = backend.denominator_floor,
        domain_samples = backend.domain_samples,
        chunk_size = backend.chunk_size)
end

function _backend_threshold(backend::ExplicitSTLSQ)
    return backend.threshold
end

function _backend_threshold(backend::DataDrivenSparseSTLSQ)
    return backend.threshold
end

function _fit_backend(A, y, backend::ExplicitSTLSQ; chunk_size::Int = 256)
    return _stlsq_blocked(A, y, backend.threshold; chunk_size = chunk_size)
end

function _fit_backend(A, y, backend::DataDrivenSparseSTLSQ; chunk_size::Int = 256)
    return _datadriven_sparse_fit(A, y, backend)
end

function _datadriven_sparse_fit(A, y, backend::DataDrivenSparseSTLSQ)
    extension = Base.get_extension(@__MODULE__, :BioDynaXDataDrivenSparseExt)
    extension === nothing &&
        throw(ArgumentError(
            "DataDrivenSparseSTLSQ requires `using DataDrivenSparse` " *
            "(BioDynaXDataDrivenSparseExt)"))
    return extension.sparse_coefficients(A, y, backend)
end

function _empty_candidates(backend)
    return backend isa ImplicitSINDyPI ? ImplicitCandidate[] : ExplicitCandidate[]
end

function _discovery_retcode(error)
    error isa DomainError && return DenominatorUnsafe
    error isa LinearAlgebra.SingularException && return SingularLibrary
    if error isa ArgumentError
        msg = error.msg
        occursin("insufficient", lowercase(msg)) && return InsufficientSamples
        occursin("empty support", lowercase(msg)) && return EmptySupport
    end
    return DiscoveryFailed
end

function _failed_discovery(error, config; prefix = "Discovery failed")
    message = prefix * ": " * sprint(showerror, error)
    @warn message
    return DiscoveryResult(
        false, message, nothing, nothing, nothing,
        _empty_candidates(config.backend),
        RunMetadata(seed = config.seed, package_version = PACKAGE_VERSION),
        _discovery_retcode(error))
end

@inline function _coefficient_vanishes(coefficient)
    coefficient isa Number || return true
    return abs(float(coefficient)) ≤ 1e-10
end

function _support_empty(candidate::ExplicitCandidate)
    coeffs = candidate.coefficients
    isempty(coeffs) && return true
    empty = true
    @inbounds for coefficient in coeffs
        empty &= _coefficient_vanishes(coefficient)
    end
    return empty
end

function _support_empty(candidate::ImplicitCandidate)
    empty = true
    @inbounds for coefficient in candidate.numerator_coefficients
        empty &= _coefficient_vanishes(coefficient)
    end
    @inbounds for coefficient in candidate.denominator_coefficients
        empty &= _coefficient_vanishes(coefficient)
    end
    return empty
end

function _target_indices(X, targets)
    targets === nothing && return collect(axes(X, 1))
    idxs = targets isa Integer ? Int[Int(targets)] : collect(Int, targets)
    isempty(idxs) && throw(ArgumentError("targets cannot be empty"))
    nstates = size(X, 1)
    for target in idxs
        1 ≤ target ≤ nstates ||
            throw(ArgumentError("target $target is out of bounds for $nstates states"))
    end
    return idxs
end

function _discover_explicit(X, derivatives, network, backend,
                            config::DiscoveryConfig; targets = nothing)
    sample_count = size(X, 2)
    validation_count = clamp(
        round(Int, 0.2 * sample_count), 1, sample_count - 2)
    training_indices = collect(1:(sample_count - validation_count))
    validation_indices =
        collect((sample_count - validation_count + 1):sample_count)
    candidates = ExplicitCandidate{eltype(X)}[]
    chunk_size = _backend_chunk_size(backend)

    for target in _target_indices(X, targets)
        derivative = vec(@view derivatives[target, :])
        spec = local_basis(
            network, target;
            degree = 3,
            max_variables = 8,
            include_interactions = config.include_interactions,
            X = X,
            derivative = derivative,
            extra_candidates = 0,
            scope = config.basis_scope)
        train_X = @view X[:, training_indices]
        n_train = length(training_indices)
        library = Matrix{eltype(X)}(undef, n_train, length(spec.numerator))
        evaluate_library!(library, spec.numerator, train_X)
        coefficients = _fit_backend(
            library, derivative[training_indices], backend;
            chunk_size = chunk_size)
        val_X = @view X[:, validation_indices]
        prediction = evaluate_library(spec.numerator, val_X) * coefficients
        error = mean(abs2, prediction .- derivative[validation_indices])
        push!(candidates, ExplicitCandidate(
            target, spec, coefficients, error))
    end
    return candidates
end

function _discover_implicit(X, derivatives, network, backend::ImplicitSINDyPI,
                            config::DiscoveryConfig; targets = nothing)
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
    train_X = @view X[:, training_indices]
    val_X = @view X[:, validation_indices]
    domain_X = _denominator_domain_grid(
        X; n = backend.domain_samples, seed = config.seed)
    candidates = ImplicitCandidate{eltype(X)}[]
    denominator_errors = Exception[]

    for target in _target_indices(X, targets)
        derivative = vec(@view derivatives[target, :])
        spec = local_basis(
            network, target;
            degree = max(backend.max_degree, backend.max_hill_degree),
            max_variables = backend.max_parents,
            include_interactions = config.include_interactions,
            X = X,
            derivative = derivative,
            extra_candidates = backend.extra_candidates,
            scope = config.basis_scope)
        try
            frequencies = _bootstrap_frequency(
                rng, spec, X, derivative, training_indices, backend.threshold,
                backend.bootstrap_samples; chunk_size = backend.chunk_size)
            numerator, denominator = _consensus_refit(
                spec, X, derivative, training_indices, backend.threshold,
                frequencies; chunk_size = backend.chunk_size)
            prediction, _ = _evaluate_candidate(
                spec, numerator, denominator, val_X)
            denominator_minimum = _check_denominator_safety(
                spec, numerator, denominator, train_X, val_X, domain_X,
                backend.denominator_floor)
            error = mean(abs2,
                         prediction .- derivative[validation_indices])
            push!(candidates, ImplicitCandidate(
                target, spec, numerator, denominator, frequencies, error,
                denominator_minimum))
        catch error
            error isa DomainError || rethrow()
            push!(denominator_errors, error)
        end
    end
    isempty(candidates) && !isempty(denominator_errors) &&
        throw(first(denominator_errors))
    return candidates
end

function _run_discovery(X, derivatives, network, backend, config::DiscoveryConfig;
                        targets = nothing)
    size(X, 2) ≥ 20 ||
        throw(ArgumentError("insufficient finite trajectory samples"))
    if backend isa ImplicitSINDyPI
        candidates = _discover_implicit(
            X, derivatives, network, backend, config; targets = targets)
    elseif backend isa ExplicitSTLSQ || backend isa DataDrivenSparseSTLSQ
        candidates = _discover_explicit(
            X, derivatives, network, backend, config; targets = targets)
    else
        throw(ArgumentError("unsupported discovery backend $(typeof(backend))"))
    end
    (isempty(candidates) || all(_support_empty, candidates)) &&
        throw(ArgumentError("empty support: no terms survived thresholding"))
    equation_text = join(format_equation.(candidates), "\n")
    basis = getfield.(candidates, :specification)
    metadata = RunMetadata(
        seed = config.seed,
        package_version = PACKAGE_VERSION,
        data_hash = data_fingerprint(X, derivatives),
        config = (; backend = string(typeof(backend)), samples = size(X, 2)))
    return DiscoveryResult(true, "ok", equation_text, basis, nothing,
                           candidates, metadata, DiscoverySuccess)
end

function discover_equations(p_trained, model::UDEModel, set::ExperimentSet;
                            n_samples::Int = 300,
                            config::DiscoveryConfig = DiscoveryConfig(),
                            solver::SolverConfig = SolverConfig(),
                            verbose::Bool = true,
                            strict::Bool = false,
                            kwargs...)
    isempty(set) &&
        throw(ArgumentError("ExperimentSet cannot be empty"))
    n_samples ≥ 20 ||
        throw(ArgumentError("n_samples must be at least 20"))
    try
        X, derivatives = _collect_multi_trajectory_data(
            p_trained, model, collect(set.experiments), n_samples;
            solver = solver)
        result = _run_discovery(
            X, derivatives, model.network, config.backend, config)
        verbose && println("\n[Discovery] Recovered system:\n", result.equations)
        return result
    catch error
        strict && rethrow()
        return _failed_discovery(
            error, config; prefix = "Multi-trajectory discovery failed")
    end
end

"""
    discover_equations(X, times, network; config, ...)

UDE-free discovery from observed trajectories. Derivatives are estimated by
central finite differences unless `derivatives` is supplied. Pass `targets`
to recover a subset of state rows (used for rate-only unknown-edge discovery).
"""
function discover_equations(X::AbstractMatrix, times::AbstractVector,
                            network::BiologicalNetwork;
                            derivatives = nothing,
                            targets = nothing,
                            config::DiscoveryConfig = DiscoveryConfig(),
                            verbose::Bool = true,
                            strict::Bool = false)
    size(X, 2) == length(times) ||
        throw(DimensionMismatch("X columns must match times"))
    dX = derivatives === nothing ?
        estimate_derivatives(X, times) : derivatives
    size(dX) == size(X) ||
        throw(DimensionMismatch("derivatives must match X"))
    keep = vec(all(isfinite, X; dims = 1) .& all(isfinite, dX; dims = 1))
    X_clean = X[:, keep]
    dX_clean = dX[:, keep]
    try
        result = _run_discovery(
            X_clean, dX_clean, network, config.backend, config;
            targets = targets)
        verbose && println("\n[Discovery] Recovered system:\n", result.equations)
        return result
    catch error
        strict && rethrow()
        return _failed_discovery(
            error, config; prefix = "Raw-data discovery failed")
    end
end

"""
    select_discovery_config(p_trained, model; thresholds, criterion=:aic, ...)

Sweep sparsity thresholds, score recovered candidates with AIC/BIC, and return
the best `DiscoveryResult` (selection summary in `metadata.config`).
"""
function select_discovery_config(p_trained, model::UDEModel;
                                 thresholds = (1e-1, 1e-2, 1e-3),
                                 criterion::Symbol = :aic,
                                 n_samples::Int = 300,
                                 u0::Vector{Float64} = [0.2, 0.1],
                                 tspan::Tuple{Float64,Float64} = (0.0, 20.0),
                                 config::DiscoveryConfig = DiscoveryConfig(),
                                 solver::SolverConfig = SolverConfig(),
                                 verbose::Bool = false)
    X, derivatives, _ = _collect_trajectory_data(
        p_trained, model, u0, tspan, n_samples; solver = solver)
    best_result = nothing
    best_score = Inf
    best_threshold = first(thresholds)
    score_table = NamedTuple[]
    for threshold in thresholds
        backend = _with_threshold(config.backend, threshold)
        local_config = DiscoveryConfig(
            backend, config.include_constant, config.include_interactions,
            config.max_interaction_order, config.seed, config.basis_scope)
        result = try
            _run_discovery(X, derivatives, model.network, backend, local_config)
        catch error
            push!(score_table, (
                threshold = float(threshold), score = Inf,
                success = false, message = sprint(showerror, error)))
            continue
        end
        total = sum(
            score_candidate(
                candidate, X, vec(@view derivatives[candidate.target, :]);
                criterion = criterion)
            for candidate in result.candidates; init = 0.0)
        push!(score_table, (
            threshold = float(threshold), score = total,
            success = true, message = "ok"))
        if total < best_score
            best_score = total
            best_result = result
            best_threshold = threshold
        end
    end
    if best_result === nothing
        return DiscoveryResult(
            false, "no discovery configuration succeeded", nothing, nothing,
            nothing, _empty_candidates(config.backend),
            RunMetadata(seed = config.seed, package_version = PACKAGE_VERSION,
                        config = Dict(:selection => score_table,
                                      :criterion => criterion)),
            DiscoveryFailed)
    end
    metadata = RunMetadata(
        seed = config.seed,
        package_version = PACKAGE_VERSION,
        data_hash = best_result.metadata.data_hash,
        config = Dict(
            :backend => string(typeof(config.backend)),
            :samples => size(X, 2),
            :criterion => criterion,
            :selected_threshold => float(best_threshold),
            :selected_score => best_score,
            :selection => score_table))
    verbose && println(
        "\n[Discovery] Selected threshold ", best_threshold,
        " with ", criterion, "=", best_score)
    return DiscoveryResult(
        best_result.success, best_result.message, best_result.equations,
        best_result.basis, best_result.solution, best_result.candidates,
        metadata, best_result.retcode)
end

"""
    discover_equations(p_trained, nn, st; ...)

Discover graph-local rational dynamics with SINDy-PI's implicit identity
`D(z)ẋ - N(z)=0`. Unknown numerator and denominator coefficients are learned
jointly; no fixed-denominator rational atoms or Taylor approximations are used.
"""
function discover_equations(p_trained, nn, st;
                            model::Union{Nothing,UDEModel} = nothing,
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
                            strict::Bool = false,
                            kwargs...)
    n_samples ≥ 20 ||
        throw(ArgumentError("n_samples must be at least 20"))
    backend = config.backend
    if backend isa ImplicitSINDyPI
        backend = ImplicitSINDyPI(
            threshold = float(sparsity_threshold),
            max_degree = polynomial_degree,
            max_hill_degree = config.backend.max_hill_degree,
            max_parents = config.backend.max_parents,
            extra_candidates = config.backend.extra_candidates,
            bootstrap_samples = config.backend.bootstrap_samples,
            validation_fraction = config.backend.validation_fraction,
            denominator_floor = config.backend.denominator_floor,
            domain_samples = config.backend.domain_samples,
            chunk_size = config.backend.chunk_size)
    end
    try
        if model !== nothing
            X, derivatives, _ = _collect_trajectory_data(
                p_trained, model, u0, tspan, n_samples; solver = SolverConfig())
        else
            X, derivatives, _ = _collect_trajectory_data(
                p_trained, nn, st, u0, tspan, n_samples;
                model = nothing, solver = SolverConfig())
        end
        result = _run_discovery(X, derivatives, network, backend, config)
        verbose && println("\n[Discovery] Recovered system:\n", result.equations)
        return result
    catch error
        strict && rethrow()
        return _failed_discovery(error, config)
    end
end
