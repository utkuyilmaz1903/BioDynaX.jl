struct MonomialTerm
    variables::Vector{Int}
    powers::Vector{Int}
    label::String
end

struct LocalBasisSpec
    target::Int
    variables::Vector{Int}
    numerator::Vector{MonomialTerm}
    denominator::Vector{MonomialTerm}
end

function _monomial(variable::Int, power::Int)
    suffix = power == 1 ? "" : "^$power"
    return MonomialTerm([variable], [power], "x[$variable]$suffix")
end

function _interaction(a::Int, b::Int)
    return MonomialTerm([a, b], [1, 1], "x[$a]*x[$b]")
end

"""
    screen_variables(X, derivative, candidates, max_variables)

Deterministic derivative-correlation screening. Graph parents remain the prior;
screening only bounds the local library when indegree is unexpectedly large.
"""
function screen_variables(X, derivative, candidates, max_variables)
    length(candidates) ≤ max_variables && return collect(candidates)
    centered_y = derivative .- mean(derivative)
    scores = map(candidates) do index
        centered_x = @view(X[index, :]) .- mean(@view X[index, :])
        denominator = sqrt(sum(abs2, centered_x) * sum(abs2, centered_y))
        denominator == 0 ? 0.0 : abs(dot(centered_x, centered_y) / denominator)
    end
    order = sortperm(scores; rev = true)
    return sort(collect(candidates)[order[1:max_variables]])
end

function local_basis(network::BiologicalNetwork, target::Int;
                     degree::Int = 3,
                     max_variables::Int = 8,
                     include_interactions::Bool = true,
                     X = nothing,
                     derivative = nothing,
                     extra_candidates::Int = 0)
    dynamic_nodes = state_nodes(network)
    1 ≤ target ≤ length(dynamic_nodes) ||
        throw(ArgumentError("target must index a dynamic state"))
    target_node = dynamic_nodes[target]
    parent_nodes = filter(in(dynamic_nodes),
                          candidate_parents(network, target_node))
    row_by_node = Dict(node => row for (row, node) in pairs(dynamic_nodes))
    parent_rows = Int[row_by_node[node] for node in parent_nodes]
    graph_candidates = unique([target; parent_rows])
    if X !== nothing && derivative !== nothing
        graph_candidates = screen_variables(
            X, derivative, graph_candidates, max_variables)
        if extra_candidates > 0
            outside = setdiff(collect(axes(X, 1)), graph_candidates)
            extras = screen_variables(X, derivative, outside,
                                      min(extra_candidates, length(outside)))
            append!(graph_candidates, extras)
        end
    elseif length(graph_candidates) > max_variables
        graph_candidates = graph_candidates[1:max_variables]
    end
    sort!(unique!(graph_candidates))

    numerator = MonomialTerm[
        MonomialTerm(Int[], Int[], "1"),
    ]
    for variable in graph_candidates, power in 1:degree
        push!(numerator, _monomial(variable, power))
    end
    if include_interactions
        for i in eachindex(graph_candidates)
            for j in (i + 1):length(graph_candidates)
                push!(numerator,
                      _interaction(graph_candidates[i], graph_candidates[j]))
            end
        end
    end

    # D(0)=1 is the implicit normalization, so only non-constant denominator
    # terms are estimated.
    denominator = copy(numerator[2:end])
    return LocalBasisSpec(target, graph_candidates, numerator, denominator)
end

function evaluate_term(term::MonomialTerm, X::AbstractMatrix)
    isempty(term.variables) && return ones(eltype(X), size(X, 2))
    output = ones(eltype(X), size(X, 2))
    for (variable, power) in zip(term.variables, term.powers)
        output .*= @view(X[variable, :]) .^ power
    end
    return output
end

function evaluate_term_range!(output::AbstractVector, term::MonomialTerm,
                              X::AbstractMatrix, sample_range)
    length(output) == length(sample_range) ||
        throw(DimensionMismatch("output length must match sample_range"))
    if isempty(term.variables)
        fill!(output, one(eltype(X)))
        return output
    end
    fill!(output, one(eltype(X)))
    @inbounds for (variable, power) in zip(term.variables, term.powers)
        for (row, sample) in enumerate(sample_range)
            output[row] *= X[variable, sample]^power
        end
    end
    return output
end

function evaluate_library!(output::AbstractMatrix, terms::Vector{MonomialTerm},
                           X::AbstractMatrix)
    size(output, 1) == size(X, 2) ||
        throw(DimensionMismatch("library rows must match sample count"))
    size(output, 2) == length(terms) ||
        throw(DimensionMismatch("library columns must match term count"))
    for (column, term) in pairs(terms)
        output[:, column] .= evaluate_term(term, X)
    end
    return output
end

"""
    evaluate_library_range!(output, terms, X, sample_range)

Fill `output` (`length(sample_range) × n_terms`) for a contiguous or arbitrary
sample index range without allocating per-term vectors.
"""
function evaluate_library_range!(output::AbstractMatrix, terms::Vector{MonomialTerm},
                                 X::AbstractMatrix, sample_range)
    size(output, 1) == length(sample_range) ||
        throw(DimensionMismatch("output rows must match sample_range"))
    size(output, 2) == length(terms) ||
        throw(DimensionMismatch("output columns must match term count"))
    for (column, term) in pairs(terms)
        evaluate_term_range!(@view(output[:, column]), term, X, sample_range)
    end
    return output
end

function evaluate_library(terms::Vector{MonomialTerm}, X::AbstractMatrix)
    output = Matrix{eltype(X)}(undef, size(X, 2), length(terms))
    return evaluate_library!(output, terms, X)
end

"""Chunked view over a monomial library for streaming evaluation."""
struct LibraryChunks{T<:AbstractFloat,M<:AbstractMatrix{T}}
    terms::Vector{MonomialTerm}
    X::M
    chunk_size::Int
end

"""
    each_library_chunk(terms, X; chunk_size=256)

Iterate `(chunk_matrix, sample_range)` pairs without materializing the full
`n_samples × n_terms` library at once.
"""
function each_library_chunk(terms::Vector{MonomialTerm}, X::AbstractMatrix;
                            chunk_size::Int = 256)
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    return LibraryChunks{eltype(X),typeof(X)}(terms, X, chunk_size)
end

Base.eltype(::Type{<:LibraryChunks{T}}) where {T} =
    Tuple{Matrix{T},UnitRange{Int}}
Base.IteratorSize(::Type{<:LibraryChunks}) = Base.SizeUnknown()

function Base.iterate(chunks::LibraryChunks, start::Int = 1)
    n = size(chunks.X, 2)
    start > n && return nothing
    stop = min(start + chunks.chunk_size - 1, n)
    sample_range = start:stop
    buffer = Matrix{eltype(chunks.X)}(undef, length(sample_range),
                                      length(chunks.terms))
    evaluate_library_range!(buffer, chunks.terms, chunks.X, sample_range)
    return (buffer, sample_range), stop + 1
end

candidate_count(spec::LocalBasisSpec) =
    length(spec.numerator) + length(spec.denominator)

function enforce_hierarchy!(coefficients, terms, threshold)
    main_effects = Dict{Int,Int}()
    for (index, term) in pairs(terms)
        length(term.variables) == 1 && term.powers == [1] &&
            (main_effects[only(term.variables)] = index)
    end
    for (index, term) in pairs(terms)
        length(term.variables) > 1 || continue
        parent_indices = [get(main_effects, variable, 0)
                          for variable in term.variables]
        supported = all(parent_index > 0 &&
                        abs(coefficients[parent_index]) ≥ threshold
                        for parent_index in parent_indices)
        supported || (coefficients[index] = zero(eltype(coefficients)))
    end
    return coefficients
end
