"""Preallocated workspace for allocation-free UDE RHS evaluation."""
mutable struct UDEModelCache{T<:AbstractFloat,M}
    production::Vector{T}
    destruction::Vector{T}
    du::Vector{T}
    nn_inputs::M
end

"""
    allocate_cache(model, T=Float64) -> UDEModelCache

Preallocate production, destruction, and `du` buffers for `ude_rhs!`.
"""
function allocate_cache(model::UDEModel, ::Type{T}) where {T<:AbstractFloat}
    n = model.compiled.nstates
    nn_terms = [term for term in model.compiled.destruction_terms
                if term isa NeuralDestructionTerm]
    nn_count = length(nn_terms)
    max_in = nn_count == 0 ? 0 :
        maximum(length(term.regulators) for term in nn_terms)
    nn_inputs = nn_count > 0 ? Matrix{T}(undef, max_in, nn_count) :
        Matrix{T}(undef, 0, 0)
    return UDEModelCache(
        zeros(T, n), zeros(T, n), zeros(T, n), nn_inputs)
end
