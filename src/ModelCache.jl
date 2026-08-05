"""Preallocated workspace for allocation-free UDE RHS evaluation."""
mutable struct UDEModelCache{T<:AbstractFloat,M}
    production::Vector{T}
    destruction::Vector{T}
    du::Vector{T}
    nn_inputs::M
end

function allocate_cache(model::UDEModel, ::Type{T}) where {T<:AbstractFloat}
    n = model.compiled.nstates
    nn_count = count(term -> term isa NeuralDestructionTerm,
                     model.compiled.destruction_terms)
    nn_inputs = nn_count > 0 ? Matrix{T}(undef, 1, nn_count) : Matrix{T}(undef, 0, 0)
    return UDEModelCache(
        zeros(T, n), zeros(T, n), zeros(T, n), nn_inputs)
end
