module BioDynaXDataDrivenSparseExt

using BioDynaX
using DataDrivenSparse
using LinearAlgebra

"""Fit `y ≈ A ξ` with DataDrivenSparse `STLSQ` (A is samples × features)."""
function sparse_coefficients(A::AbstractMatrix, y::AbstractVector,
                             backend::BioDynaX.DataDrivenSparseSTLSQ)
    alg = DataDrivenSparse.STLSQ(backend.threshold, backend.ridge)
    features_by_samples = permutedims(A)
    cache = DataDrivenSparse.init_cache(alg, features_by_samples, y)
    λ = backend.threshold
    for _ in 1:20
        DataDrivenSparse.step!(cache, λ)
    end
    return vec(cache.X)
end

end
