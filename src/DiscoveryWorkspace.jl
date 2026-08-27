###############################################################################
# Reusable discovery buffers (not exported).
#
# Blocked STLSQ, implicit design, and library chunks used to allocate a new
# Gram / design / per-term vector on every bootstrap draw. This file owns the
# grow-only workspaces those hot paths reuse. Numerics stay the blocked-STLSQ
# contract already used by `_fit_implicit`; the dense QR `_stlsq` path is the
# agreement oracle, not the production factorisation.
#
# validate_network is unchanged. Combined F1 is still a skeleton floor.
###############################################################################

"""Source strings that prove the streaming / workspace path stays wired."""
const DISCOVERY_WORKSPACE_MUST_CONTAIN = (
    "mutable struct STLSQWorkspace",
    "function _stlsq_blocked!",
    "function stlsq_from_chunk_filler!",
    "function _fit_implicit_stream",
    "function evaluate_candidate!",
    "function each_reusable_library_chunk",
    "function _backend_chunk_size")

const DISCOVERY_WORKSPACE_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "assert_single_unknown_destruction(network)",
    "function validate_network")

const DISCOVERY_STREAMING_DATAGEN_MUST_NOT_CONTAIN = (
    "dummy_nn",
    "Lux.Dense(1 => 1")

"""English contract sentences for the discovery-streaming page."""
function discovery_streaming_locked_sentences()
    return (;
        workspace = "Blocked STLSQ reuses one grow-only Gram workspace across bootstrap draws.",
        library = "evaluate_library! writes monomials in place; it does not allocate a per-term vector.",
        stream = "Implicit design can be filled in row chunks without materialising the full n × p matrix.",
        chunk = "Reusable library chunks overwrite one buffer; they do not allocate a new block per iterate.",
        backend = "ImplicitSINDyPI.chunk_size is the blocked-STLSQ row width; ExplicitSTLSQ uses the same helper default.")
end

function discovery_workspace_source_path()
    joinpath(pkgdir(BioDynaX), "src", "DiscoveryWorkspace.jl")
end

function discovery_jl_source_path()
    joinpath(pkgdir(BioDynaX), "src", "Discovery.jl")
end

function basis_factory_source_path()
    joinpath(pkgdir(BioDynaX), "src", "BasisFactory.jl")
end

# -- Workspaces ---------------------------------------------------------------

"""
    STLSQWorkspace

Grow-only buffers for blocked / streamed STLSQ. `scaled` is the last
coefficient vector (`coefficients ./ scales`) and is invalidated by the
next `_stlsq_blocked!` call. `resize_count` increments only when a buffer
is reallocated.
"""
mutable struct STLSQWorkspace{T<:AbstractFloat}
    scales::Vector{T}
    coefficients::Vector{T}
    scaled::Vector{T}
    active::BitVector
    gram::Matrix{T}
    rhs::Vector{T}
    norm_chunk::Matrix{T}
    design_chunk::Matrix{T}
    y_chunk::Vector{T}
    indices::Vector{Int}
    n_active::Int
    n::Int
    p::Int
    chunk_size::Int
    resize_count::Int
end

"""
    ImplicitLibraryWorkspace

Full-width implicit design buffers (`n × n_num`, `n × n_den`, `n × p`)
plus an STLSQ workspace. Used when the caller already materialised `X`.
Streaming fits prefer `StreamingImplicitWorkspace`.
"""
mutable struct ImplicitLibraryWorkspace{T<:AbstractFloat}
    num_buf::Matrix{T}
    den_buf::Matrix{T}
    design::Matrix{T}
    y::Vector{T}
    pred::Vector{T}
    denvals::Vector{T}
    stlsq::STLSQWorkspace{T}
    n::Int
    n_num::Int
    n_den::Int
    resize_count::Int
end

"""
    StreamingImplicitWorkspace

Chunk-sized implicit design only. The full `n × p` design is never stored.
"""
mutable struct StreamingImplicitWorkspace{T<:AbstractFloat}
    num_chunk::Matrix{T}
    den_chunk::Matrix{T}
    design_chunk::Matrix{T}
    y_chunk::Vector{T}
    pred::Vector{T}
    denvals::Vector{T}
    num_full::Matrix{T}
    den_full::Matrix{T}
    stlsq::STLSQWorkspace{T}
    n::Int
    n_num::Int
    n_den::Int
    chunk_size::Int
    resize_count::Int
end

"""
    LibraryChunkWorkspace

One reusable `chunk_size × n_terms` block for `each_reusable_library_chunk`.
"""
mutable struct LibraryChunkWorkspace{T<:AbstractFloat}
    buffer::Matrix{T}
    chunk_size::Int
    n_terms::Int
    resize_count::Int
end

struct ReusableLibraryChunks{T<:AbstractFloat,M<:AbstractMatrix{T}}
    terms::Vector{MonomialTerm}
    X::M
    chunk_size::Int
    workspace::LibraryChunkWorkspace{T}
end

# -- Allocation / grow --------------------------------------------------------

function _grow_vector!(store::Ref, current::Vector{T}, n::Int) where {T}
    length(current) ≥ n && return current
    grown = Vector{T}(undef, n)
    @inbounds for i in eachindex(current)
        grown[i] = current[i]
    end
    store[] = grown
    return grown
end

function allocate_stlsq_workspace(::Type{T}, n::Integer, p::Integer,
        chunk_size::Integer) where {T<:AbstractFloat}
    n ≥ 0 || throw(ArgumentError("n must be non-negative"))
    p ≥ 0 || throw(ArgumentError("p must be non-negative"))
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    cs = min(Int(chunk_size), max(Int(n), 1))
    pp = max(Int(p), 1)
    return STLSQWorkspace{T}(
        zeros(T, Int(p)),
        zeros(T, Int(p)),
        zeros(T, Int(p)),
        trues(Int(p)),
        zeros(T, pp, pp),
        zeros(T, pp),
        zeros(T, cs, pp),
        zeros(T, cs, pp),
        zeros(T, cs),
        Vector{Int}(undef, pp),
        0,
        Int(n),
        Int(p),
        Int(chunk_size),
        1)
end

function ensure_stlsq_workspace!(ws::STLSQWorkspace{T}, n::Integer, p::Integer,
        chunk_size::Integer) where {T}
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    need_p = Int(p) > length(ws.scales)
    cs = min(Int(chunk_size), max(Int(n), 1))
    need_chunk = cs > size(ws.design_chunk, 1) || Int(p) > size(ws.design_chunk, 2)
    need_gram = Int(p) > size(ws.gram, 1)
    if need_p || need_chunk || need_gram
        ws.scales = zeros(T, Int(p))
        ws.coefficients = zeros(T, Int(p))
        ws.scaled = zeros(T, Int(p))
        ws.active = trues(Int(p))
        pp = max(Int(p), 1)
        ws.gram = zeros(T, pp, pp)
        ws.rhs = zeros(T, pp)
        ws.norm_chunk = zeros(T, cs, pp)
        ws.design_chunk = zeros(T, cs, pp)
        ws.y_chunk = zeros(T, cs)
        ws.indices = Vector{Int}(undef, pp)
        ws.resize_count += 1
    elseif Int(n) > ws.n || Int(chunk_size) != ws.chunk_size
        # Row count is metadata; chunk height is already large enough.
    end
    ws.n = Int(n)
    ws.p = Int(p)
    ws.chunk_size = Int(chunk_size)
    return ws
end

function allocate_implicit_workspace(::Type{T}, n::Integer, n_num::Integer,
        n_den::Integer, chunk_size::Integer) where {T<:AbstractFloat}
    p = Int(n_num) + Int(n_den)
    return ImplicitLibraryWorkspace{T}(
        Matrix{T}(undef, Int(n), Int(n_num)),
        Matrix{T}(undef, Int(n), Int(n_den)),
        Matrix{T}(undef, Int(n), p),
        Vector{T}(undef, Int(n)),
        Vector{T}(undef, Int(n)),
        Vector{T}(undef, Int(n)),
        allocate_stlsq_workspace(T, n, p, chunk_size),
        Int(n),
        Int(n_num),
        Int(n_den),
        1)
end

function ensure_implicit_workspace!(ws::ImplicitLibraryWorkspace{T}, n::Integer,
        n_num::Integer, n_den::Integer, chunk_size::Integer) where {T}
    p = Int(n_num) + Int(n_den)
    need = Int(n) > size(ws.design, 1) || p > size(ws.design, 2) ||
           Int(n_num) > size(ws.num_buf, 2) || Int(n_den) > size(ws.den_buf, 2)
    if need
        ws.num_buf = Matrix{T}(undef, Int(n), Int(n_num))
        ws.den_buf = Matrix{T}(undef, Int(n), Int(n_den))
        ws.design = Matrix{T}(undef, Int(n), p)
        ws.y = Vector{T}(undef, Int(n))
        ws.pred = Vector{T}(undef, Int(n))
        ws.denvals = Vector{T}(undef, Int(n))
        ws.resize_count += 1
    end
    ensure_stlsq_workspace!(ws.stlsq, n, p, chunk_size)
    ws.n = Int(n)
    ws.n_num = Int(n_num)
    ws.n_den = Int(n_den)
    return ws
end

function allocate_streaming_implicit_workspace(::Type{T}, n::Integer,
        n_num::Integer, n_den::Integer, chunk_size::Integer) where {T<:AbstractFloat}
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    cs = min(Int(chunk_size), max(Int(n), 1))
    p = Int(n_num) + Int(n_den)
    return StreamingImplicitWorkspace{T}(
        Matrix{T}(undef, cs, Int(n_num)),
        Matrix{T}(undef, cs, Int(n_den)),
        Matrix{T}(undef, cs, p),
        Vector{T}(undef, cs),
        Vector{T}(undef, Int(n)),
        Vector{T}(undef, Int(n)),
        Matrix{T}(undef, Int(n), Int(n_num)),
        Matrix{T}(undef, Int(n), Int(n_den)),
        allocate_stlsq_workspace(T, n, p, chunk_size),
        Int(n),
        Int(n_num),
        Int(n_den),
        Int(chunk_size),
        1)
end

function ensure_streaming_implicit_workspace!(ws::StreamingImplicitWorkspace{T},
        n::Integer, n_num::Integer, n_den::Integer, chunk_size::Integer) where {T}
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    cs = min(Int(chunk_size), max(Int(n), 1))
    p = Int(n_num) + Int(n_den)
    need = cs > size(ws.design_chunk, 1) || p > size(ws.design_chunk, 2) ||
           Int(n) > length(ws.pred) || Int(n_num) > size(ws.num_full, 2) ||
           Int(n_den) > size(ws.den_full, 2)
    if need
        ws.num_chunk = Matrix{T}(undef, cs, Int(n_num))
        ws.den_chunk = Matrix{T}(undef, cs, Int(n_den))
        ws.design_chunk = Matrix{T}(undef, cs, p)
        ws.y_chunk = Vector{T}(undef, cs)
        ws.pred = Vector{T}(undef, Int(n))
        ws.denvals = Vector{T}(undef, Int(n))
        ws.num_full = Matrix{T}(undef, Int(n), Int(n_num))
        ws.den_full = Matrix{T}(undef, Int(n), Int(n_den))
        ws.resize_count += 1
    end
    ensure_stlsq_workspace!(ws.stlsq, n, p, chunk_size)
    ws.n = Int(n)
    ws.n_num = Int(n_num)
    ws.n_den = Int(n_den)
    ws.chunk_size = Int(chunk_size)
    return ws
end

function allocate_library_chunk_workspace(::Type{T}, chunk_size::Integer,
        n_terms::Integer) where {T<:AbstractFloat}
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    n_terms ≥ 0 || throw(ArgumentError("n_terms must be non-negative"))
    return LibraryChunkWorkspace{T}(
        Matrix{T}(undef, Int(chunk_size), max(Int(n_terms), 1)),
        Int(chunk_size),
        Int(n_terms),
        1)
end

function ensure_library_chunk_workspace!(ws::LibraryChunkWorkspace{T},
        chunk_size::Integer, n_terms::Integer) where {T}
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    if Int(chunk_size) > size(ws.buffer, 1) || Int(n_terms) > size(ws.buffer, 2)
        ws.buffer = Matrix{T}(undef, Int(chunk_size), max(Int(n_terms), 1))
        ws.resize_count += 1
    end
    ws.chunk_size = max(ws.chunk_size, Int(chunk_size))
    ws.n_terms = Int(n_terms)
    return ws
end

# -- Active-set collection (no findall) ---------------------------------------

function collect_active_indices!(indices::Vector{Int}, active::AbstractVector{Bool})
    empty!(indices)
    @inbounds for i in eachindex(active)
        active[i] && push!(indices, i)
    end
    return indices
end

# -- Backend chunk width ------------------------------------------------------

_backend_chunk_size(backend::ImplicitSINDyPI) = backend.chunk_size

function _backend_chunk_size(backend::ExplicitSTLSQ)
    return 256
end

function _backend_chunk_size(backend::DataDrivenSparseSTLSQ)
    return 256
end

_backend_chunk_size(::AbstractDiscoveryBackend) = 256

function _backend_chunk_size(config::DiscoveryConfig)
    return _backend_chunk_size(config.backend)
end

# -- In-place candidate evaluation --------------------------------------------

"""
    evaluate_candidate!(pred, denvals, spec, num_coef, den_coef, X, num_buf, den_buf)

Write `N(z)/D(z)` into `pred` and the denominator into `denvals`. Library
buffers must already be sized `n × n_terms`.
"""
function evaluate_candidate!(pred::AbstractVector, denvals::AbstractVector,
        spec::LocalBasisSpec, num_coef::AbstractVector, den_coef::AbstractVector,
        X::AbstractMatrix, num_buf::AbstractMatrix, den_buf::AbstractMatrix)
    n = size(X, 2)
    length(pred) == n || throw(DimensionMismatch("pred length must match sample count"))
    length(denvals) == n ||
        throw(DimensionMismatch("denvals length must match sample count"))
    evaluate_library!(num_buf, spec.numerator, X)
    evaluate_library!(den_buf, spec.denominator, X)
    mul!(pred, num_buf, num_coef)
    mul!(denvals, den_buf, den_coef)
    @inbounds for i in 1:n
        denvals[i] += one(eltype(denvals))
        pred[i] /= denvals[i]
    end
    return pred, denvals
end

function evaluate_candidate!(ws::ImplicitLibraryWorkspace, spec::LocalBasisSpec,
        num_coef::AbstractVector, den_coef::AbstractVector, X::AbstractMatrix)
    n = size(X, 2)
    pred = @view ws.pred[1:n]
    denvals = @view ws.denvals[1:n]
    num_buf = @view ws.num_buf[1:n, 1:length(spec.numerator)]
    den_buf = @view ws.den_buf[1:n, 1:length(spec.denominator)]
    return evaluate_candidate!(
        pred, denvals, spec, num_coef, den_coef, X, num_buf, den_buf)
end

function evaluate_candidate!(ws::StreamingImplicitWorkspace, spec::LocalBasisSpec,
        num_coef::AbstractVector, den_coef::AbstractVector, X::AbstractMatrix)
    n = size(X, 2)
    pred = @view ws.pred[1:n]
    denvals = @view ws.denvals[1:n]
    num_buf = @view ws.num_full[1:n, 1:length(spec.numerator)]
    den_buf = @view ws.den_full[1:n, 1:length(spec.denominator)]
    return evaluate_candidate!(
        pred, denvals, spec, num_coef, den_coef, X, num_buf, den_buf)
end

# -- Implicit design chunks ---------------------------------------------------

"""
    implicit_design_chunk!(design, num, den, spec, X, y, sample_range)

Fill one row-block of the implicit identity `N(z) - D(z) ẏ` (constant-free
denominator columns already omit the `1`). `y` is the *chunk* of the
target (derivative or previous prediction).
"""
function implicit_design_chunk!(design::AbstractMatrix, num::AbstractMatrix,
        den::AbstractMatrix, spec::LocalBasisSpec, X::AbstractMatrix,
        y::AbstractVector, sample_range)
    length(y) == length(sample_range) ||
        throw(DimensionMismatch("y chunk must match sample_range"))
    size(design, 1) == length(sample_range) ||
        throw(DimensionMismatch("design rows must match sample_range"))
    evaluate_library_range!(num, spec.numerator, X, sample_range)
    evaluate_library_range!(den, spec.denominator, X, sample_range)
    n_num = size(num, 2)
    @inbounds for j in 1:n_num
        for i in axes(num, 1)
            design[i, j] = num[i, j]
        end
    end
    @inbounds for j in axes(den, 2)
        col = n_num + j
        for i in axes(den, 1)
            design[i, col] = -den[i, j] * y[i]
        end
    end
    return design
end

function implicit_design!(ws::ImplicitLibraryWorkspace, spec::LocalBasisSpec,
        X::AbstractMatrix, y::AbstractVector)
    n = length(y)
    n == size(X, 2) || throw(DimensionMismatch("y must match X columns"))
    num = @view ws.num_buf[1:n, 1:ws.n_num]
    den = @view ws.den_buf[1:n, 1:ws.n_den]
    design = @view ws.design[1:n, 1:(ws.n_num + ws.n_den)]
    evaluate_library!(num, spec.numerator, X)
    evaluate_library!(den, spec.denominator, X)
    @inbounds for j in 1:ws.n_num
        for i in 1:n
            design[i, j] = num[i, j]
        end
    end
    @inbounds for j in 1:ws.n_den
        col = ws.n_num + j
        for i in 1:n
            design[i, col] = -den[i, j] * y[i]
        end
    end
    return design
end

# -- Blocked STLSQ on a materialised design -----------------------------------

function _accumulate_scales!(scales::AbstractVector, A::AbstractMatrix,
        chunk_size::Integer)
    n, p = size(A)
    fill!(scales, zero(eltype(scales)))
    @inbounds for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        for j in 1:p
            acc = zero(eltype(scales))
            for i in start:stop
                acc += abs2(A[i, j])
            end
            scales[j] += acc
        end
    end
    @inbounds for j in 1:p
        scales[j] = sqrt(scales[j])
        scales[j] < eps(eltype(scales)) && (scales[j] = eps(eltype(scales)))
    end
    return scales
end

function _accumulate_gram_chunk!(G::AbstractMatrix, b::AbstractVector,
        Anorm::AbstractMatrix, y::AbstractVector, k::Integer)
    # G += A'A and b += A'y for the first k columns of Anorm.
    rows = size(Anorm, 1)
    @inbounds for j in 1:k
        accb = zero(eltype(b))
        for i in 1:rows
            accb += Anorm[i, j] * y[i]
        end
        b[j] += accb
        for i in 1:j
            acc = zero(eltype(G))
            for r in 1:rows
                acc += Anorm[r, i] * Anorm[r, j]
            end
            G[i, j] += acc
            i != j && (G[j, i] += acc)
        end
    end
    return G, b
end

"""
    _stlsq_blocked!(ws, A, y, threshold; ...)

Blocked ridge STLSQ that reuses `ws`. Returns `ws.scaled` (valid until the
next call). Thresholding semantics match `_stlsq_blocked`.
"""
function _stlsq_blocked!(ws::STLSQWorkspace{T}, A::AbstractMatrix,
        y::AbstractVector, threshold; max_iterations::Int = 20,
        ridge = 1e-10) where {T}
    n, p = size(A)
    length(y) == n || throw(DimensionMismatch("y must match A rows"))
    ensure_stlsq_workspace!(ws, n, p, ws.chunk_size)
    chunk_size = min(ws.chunk_size, max(n, 1))
    scales = @view ws.scales[1:p]
    _accumulate_scales!(scales, A, chunk_size)
    fill!(ws.active, true)
    fill!(ws.coefficients, zero(T))
    for _ in 1:max_iterations
        indices = collect_active_indices!(ws.indices, ws.active)
        k = length(indices)
        ws.n_active = k
        k == 0 && break
        G = @view ws.gram[1:k, 1:k]
        b = @view ws.rhs[1:k]
        fill!(G, zero(T))
        fill!(b, zero(T))
        for start in 1:chunk_size:n
            stop = min(start + chunk_size - 1, n)
            rows = stop - start + 1
            Anorm = @view ws.norm_chunk[1:rows, 1:k]
            ychunk = @view ws.y_chunk[1:rows]
            @inbounds for i in 1:rows
                ychunk[i] = y[start + i - 1]
                for jj in 1:k
                    j = indices[jj]
                    Anorm[i, jj] = A[start + i - 1, j] / scales[j]
                end
            end
            _accumulate_gram_chunk!(G, b, Anorm, ychunk, k)
        end
        @inbounds for i in 1:k
            G[i, i] += T(ridge)
        end
        local_coefficients = G \ b
        fill!(ws.coefficients, zero(T))
        @inbounds for jj in 1:k
            ws.coefficients[indices[jj]] = local_coefficients[jj]
        end
        changed = false
        @inbounds for j in 1:p
            next = abs(ws.coefficients[j]) ≥ threshold
            if next != ws.active[j]
                ws.active[j] = next
                changed = true
            end
        end
        changed || break
    end
    @inbounds for j in 1:p
        ws.scaled[j] = ws.coefficients[j] / scales[j]
    end
    return @view ws.scaled[1:p]
end

# -- Streamed STLSQ from a row-filler -----------------------------------------

"""
    stlsq_from_chunk_filler!(ws, n, p, y_rhs, threshold, fill_design!; ...)

`fill_design!(design_view, range)` writes the current implicit / explicit
design rows. `y_rhs` is the STLSQ target (the observed derivative) and is
never replaced by a rebuilt prediction. The full `n × p` matrix is never
stored. Returns `ws.scaled`.
"""
function stlsq_from_chunk_filler!(ws::STLSQWorkspace{T}, n::Integer, p::Integer,
        y_rhs::AbstractVector, threshold, fill_design!;
        max_iterations::Int = 20, ridge = 1e-10) where {T}
    n = Int(n)
    p = Int(p)
    length(y_rhs) == n || throw(DimensionMismatch("y_rhs must have n samples"))
    ensure_stlsq_workspace!(ws, n, p, ws.chunk_size)
    chunk_size = min(ws.chunk_size, max(n, 1))
    scales = @view ws.scales[1:p]
    fill!(scales, zero(T))
    for start in 1:chunk_size:n
        stop = min(start + chunk_size - 1, n)
        rows = stop - start + 1
        design = @view ws.design_chunk[1:rows, 1:p]
        fill_design!(design, start:stop)
        @inbounds for j in 1:p
            acc = zero(T)
            for i in 1:rows
                acc += abs2(design[i, j])
            end
            scales[j] += acc
        end
    end
    @inbounds for j in 1:p
        scales[j] = sqrt(scales[j])
        scales[j] < eps(T) && (scales[j] = eps(T))
    end
    fill!(ws.active, true)
    fill!(ws.coefficients, zero(T))
    for _ in 1:max_iterations
        indices = collect_active_indices!(ws.indices, ws.active)
        k = length(indices)
        ws.n_active = k
        k == 0 && break
        G = @view ws.gram[1:k, 1:k]
        b = @view ws.rhs[1:k]
        fill!(G, zero(T))
        fill!(b, zero(T))
        for start in 1:chunk_size:n
            stop = min(start + chunk_size - 1, n)
            rows = stop - start + 1
            design = @view ws.design_chunk[1:rows, 1:p]
            ychunk = @view ws.y_chunk[1:rows]
            fill_design!(design, start:stop)
            _copy_target_chunk!(ychunk, y_rhs, start:stop)
            Anorm = @view ws.norm_chunk[1:rows, 1:k]
            @inbounds for jj in 1:k
                j = indices[jj]
                invs = inv(scales[j])
                for i in 1:rows
                    Anorm[i, jj] = design[i, j] * invs
                end
            end
            _accumulate_gram_chunk!(G, b, Anorm, ychunk, k)
        end
        @inbounds for i in 1:k
            G[i, i] += T(ridge)
        end
        local_coefficients = G \ b
        fill!(ws.coefficients, zero(T))
        @inbounds for jj in 1:k
            ws.coefficients[indices[jj]] = local_coefficients[jj]
        end
        changed = false
        @inbounds for j in 1:p
            next = abs(ws.coefficients[j]) ≥ threshold
            if next != ws.active[j]
                ws.active[j] = next
                changed = true
            end
        end
        changed || break
    end
    @inbounds for j in 1:p
        ws.scaled[j] = ws.coefficients[j] / scales[j]
    end
    return @view ws.scaled[1:p]
end

# -- Streaming implicit fit ---------------------------------------------------

function _copy_target_chunk!(dest::AbstractVector, src::AbstractVector,
        sample_range)
    @inbounds for (i, sample) in enumerate(sample_range)
        dest[i] = src[sample]
    end
    return dest
end

"""
    _fit_implicit_stream(spec, X, derivative, indices, threshold; ...)

Two-stage implicit STLSQ (target, then prediction) that never stores the
full design. Hierarchy pruning still uses `prune_nested_implicit`.
"""
function _fit_implicit_stream(spec::LocalBasisSpec, X, derivative, indices,
        threshold; chunk_size::Int = 256,
        workspace::Union{Nothing,StreamingImplicitWorkspace} = nothing)
    local_X = @view X[:, indices]
    n = length(indices)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    T = eltype(X)
    y = Vector{T}(undef, n)
    @inbounds for i in 1:n
        y[i] = derivative[indices[i]]
    end
    ws = workspace === nothing ?
        allocate_streaming_implicit_workspace(T, n, n_num, n_den, chunk_size) :
        ensure_streaming_implicit_workspace!(
            workspace, n, n_num, n_den, chunk_size)
    function fill_from_target!(design, sample_range)
        rows = length(sample_range)
        num = @view ws.num_chunk[1:rows, 1:n_num]
        den = @view ws.den_chunk[1:rows, 1:n_den]
        rate = @view ws.y_chunk[1:rows]
        _copy_target_chunk!(rate, y, sample_range)
        implicit_design_chunk!(
            design, num, den, spec, local_X, rate, sample_range)
        return nothing
    end
    coefficients = stlsq_from_chunk_filler!(
        ws.stlsq, n, n_num + n_den, y, threshold, fill_from_target!;
        max_iterations = 20, ridge = 1e-10)
    numerator = copy(coefficients[1:n_num])
    denominator = copy(coefficients[(n_num + 1):end])
    enforce_hierarchy!(numerator, spec.numerator, threshold)
    enforce_hierarchy!(denominator, spec.denominator, threshold)
    pred, _ = evaluate_candidate!(ws, spec, numerator, denominator, local_X)
    if all(isfinite, pred)
        pred_copy = copy(pred)
        function fill_from_pred!(design, sample_range)
            rows = length(sample_range)
            num = @view ws.num_chunk[1:rows, 1:n_num]
            den = @view ws.den_chunk[1:rows, 1:n_den]
            rate = @view ws.y_chunk[1:rows]
            _copy_target_chunk!(rate, pred_copy, sample_range)
            implicit_design_chunk!(
                design, num, den, spec, local_X, rate, sample_range)
            return nothing
        end
        coefficients = stlsq_from_chunk_filler!(
            ws.stlsq, n, n_num + n_den, y, threshold, fill_from_pred!;
            max_iterations = 20, ridge = 1e-10)
        numerator = copy(coefficients[1:n_num])
        denominator = copy(coefficients[(n_num + 1):end])
        enforce_hierarchy!(numerator, spec.numerator, threshold)
        enforce_hierarchy!(denominator, spec.denominator, threshold)
    end
    return prune_nested_implicit(
        spec, numerator, denominator, local_X, y, threshold)
end

function _fit_implicit_workspace(spec::LocalBasisSpec, X, derivative, indices,
        threshold; chunk_size::Int = 256,
        workspace::Union{Nothing,ImplicitLibraryWorkspace} = nothing)
    local_X = @view X[:, indices]
    n = length(indices)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    T = eltype(X)
    ws = workspace === nothing ?
        allocate_implicit_workspace(T, n, n_num, n_den, chunk_size) :
        ensure_implicit_workspace!(workspace, n, n_num, n_den, chunk_size)
    y = @view ws.y[1:n]
    @inbounds for i in 1:n
        y[i] = derivative[indices[i]]
    end
    design = implicit_design!(ws, spec, local_X, y)
    coefficients = _stlsq_blocked!(ws.stlsq, design, y, threshold)
    numerator = copy(coefficients[1:n_num])
    denominator = copy(coefficients[(n_num + 1):end])
    enforce_hierarchy!(numerator, spec.numerator, threshold)
    enforce_hierarchy!(denominator, spec.denominator, threshold)
    pred, _ = evaluate_candidate!(ws, spec, numerator, denominator, local_X)
    if all(isfinite, pred)
        pred_vec = @view ws.pred[1:n]
        design = implicit_design!(ws, spec, local_X, pred_vec)
        coefficients = _stlsq_blocked!(ws.stlsq, design, y, threshold)
        numerator = copy(coefficients[1:n_num])
        denominator = copy(coefficients[(n_num + 1):end])
        enforce_hierarchy!(numerator, spec.numerator, threshold)
        enforce_hierarchy!(denominator, spec.denominator, threshold)
    end
    return prune_nested_implicit(
        spec, numerator, denominator, local_X, Vector(y), threshold)
end

# -- Reusable library chunks --------------------------------------------------

"""
    each_reusable_library_chunk(terms, X, workspace; chunk_size=256)

Iterate `(chunk_view, sample_range)` pairs, overwriting `workspace.buffer`.
The view is invalidated by the next iterate.
"""
function each_reusable_library_chunk(terms::Vector{MonomialTerm},
        X::AbstractMatrix, workspace::LibraryChunkWorkspace;
        chunk_size::Int = 256)
    chunk_size > 0 || throw(ArgumentError("chunk_size must be positive"))
    ensure_library_chunk_workspace!(workspace, chunk_size, length(terms))
    return ReusableLibraryChunks{eltype(X),typeof(X)}(
        terms, X, chunk_size, workspace)
end

function each_reusable_library_chunk(terms::Vector{MonomialTerm},
        X::AbstractMatrix; chunk_size::Int = 256)
    ws = allocate_library_chunk_workspace(eltype(X), chunk_size, length(terms))
    return each_reusable_library_chunk(terms, X, ws; chunk_size = chunk_size)
end

Base.eltype(::Type{<:ReusableLibraryChunks{T}}) where {T} =
    Tuple{SubArray{T,2},UnitRange{Int}}
Base.IteratorSize(::Type{<:ReusableLibraryChunks}) = Base.SizeUnknown()

function Base.iterate(chunks::ReusableLibraryChunks, start::Int = 1)
    n = size(chunks.X, 2)
    start > n && return nothing
    stop = min(start + chunks.chunk_size - 1, n)
    sample_range = start:stop
    rows = length(sample_range)
    ensure_library_chunk_workspace!(
        chunks.workspace, rows, length(chunks.terms))
    buffer = @view chunks.workspace.buffer[1:rows, 1:length(chunks.terms)]
    evaluate_library_range!(buffer, chunks.terms, chunks.X, sample_range)
    return (buffer, sample_range), stop + 1
end

function reusable_library_chunk_row_count(terms, X; chunk_size::Int = 256)
    n = 0
    for _ in each_reusable_library_chunk(terms, X; chunk_size = chunk_size)
        n += 1
    end
    return n
end

function materialise_library_via_chunks(terms, X; chunk_size::Int = 256)
    output = Matrix{eltype(X)}(undef, size(X, 2), length(terms))
    ws = allocate_library_chunk_workspace(eltype(X), chunk_size, length(terms))
    for (chunk, sample_range) in each_reusable_library_chunk(
            terms, X, ws; chunk_size = chunk_size)
        output[sample_range, :] .= chunk
    end
    return output
end

# -- Agreement / reports ------------------------------------------------------

"""
    stlsq_path_agreement(A, y, threshold; chunk_size)

Dense QR `_stlsq`, allocating blocked `_stlsq_blocked`, and workspace
`_stlsq_blocked!`. Blocked paths must match each other; dense QR is the
oracle within a looser tolerance (normal equations vs augmented QR).
"""
function stlsq_path_agreement(A::AbstractMatrix, y::AbstractVector, threshold;
        chunk_size::Int = 32, ridge = 1e-10)
    dense = _stlsq(A, y, threshold; ridge = ridge)
    blocked = _stlsq_blocked(A, y, threshold; chunk_size = chunk_size, ridge = ridge)
    ws = allocate_stlsq_workspace(eltype(A), size(A, 1), size(A, 2), chunk_size)
    reused = collect(_stlsq_blocked!(ws, A, y, threshold; ridge = ridge))
    reused2 = collect(_stlsq_blocked!(ws, A, y, threshold; ridge = ridge))
    return (;
        dense,
        blocked,
        reused,
        reused2,
        resize_count = ws.resize_count,
        blocked_matches_workspace = blocked ≈ reused,
        workspace_stable = reused ≈ reused2,
        dense_matches_blocked = dense ≈ blocked,
        holds = blocked ≈ reused && reused ≈ reused2)
end

function implicit_stream_agreement(spec::LocalBasisSpec, X, derivative, indices,
        threshold; chunk_size::Int = 32)
    classic = _fit_implicit_materialised(
        spec, X, derivative, indices, threshold; chunk_size = chunk_size)
    streamed = _fit_implicit_stream(
        spec, X, derivative, indices, threshold; chunk_size = chunk_size)
    worked = _fit_implicit_workspace(
        spec, X, derivative, indices, threshold; chunk_size = chunk_size)
    return (;
        classic,
        streamed,
        worked,
        num_stream = streamed[1] ≈ classic[1],
        den_stream = streamed[2] ≈ classic[2],
        num_workspace = worked[1] ≈ classic[1],
        den_workspace = worked[2] ≈ classic[2],
        holds = streamed[1] ≈ classic[1] && streamed[2] ≈ classic[2] &&
                worked[1] ≈ classic[1] && worked[2] ≈ classic[2])
end

"""Classic materialised implicit fit used as the agreement oracle."""
function _fit_implicit_materialised(spec::LocalBasisSpec, X, derivative, indices,
        threshold; chunk_size::Int = 256)
    local_X = @view X[:, indices]
    y = collect(@view derivative[indices])
    n = length(indices)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    num_buf = Matrix{eltype(X)}(undef, n, n_num)
    den_buf = Matrix{eltype(X)}(undef, n, n_den)
    design = Matrix{eltype(X)}(undef, n, n_num + n_den)
    _implicit_design!(design, num_buf, den_buf, spec, local_X, y)
    coefficients = _stlsq_blocked(design, y, threshold; chunk_size = chunk_size)
    numerator = coefficients[1:n_num]
    denominator = coefficients[(n_num + 1):end]
    enforce_hierarchy!(numerator, spec.numerator, threshold)
    enforce_hierarchy!(denominator, spec.denominator, threshold)
    pred, _ = _evaluate_candidate(spec, numerator, denominator, local_X)
    if all(isfinite, pred)
        _implicit_design!(design, num_buf, den_buf, spec, local_X, pred)
        coefficients = _stlsq_blocked(design, y, threshold; chunk_size = chunk_size)
        numerator = coefficients[1:n_num]
        denominator = coefficients[(n_num + 1):end]
        enforce_hierarchy!(numerator, spec.numerator, threshold)
        enforce_hierarchy!(denominator, spec.denominator, threshold)
    end
    return prune_nested_implicit(
        spec, numerator, denominator, local_X, y, threshold)
end

function library_chunk_agreement(terms, X; chunk_size::Int = 17)
    full = evaluate_library(terms, X)
    chunked = materialise_library_via_chunks(terms, X; chunk_size = chunk_size)
    allocating = Matrix{eltype(X)}(undef, 0, 0)
    for (chunk, sample_range) in each_library_chunk(terms, X; chunk_size = chunk_size)
        if isempty(allocating)
            allocating = zeros(eltype(X), size(X, 2), length(terms))
        end
        allocating[sample_range, :] .= chunk
    end
    return (;
        full,
        chunked,
        allocating,
        matches_full = full ≈ chunked,
        matches_allocating = chunked ≈ allocating,
        holds = full ≈ chunked && chunked ≈ allocating)
end

function evaluate_candidate_agreement(spec, num, den, X)
    pred, denvals = _evaluate_candidate(spec, num, den, X)
    n = size(X, 2)
    pred2 = similar(pred)
    den2 = similar(denvals)
    num_buf = Matrix{eltype(X)}(undef, n, length(spec.numerator))
    den_buf = Matrix{eltype(X)}(undef, n, length(spec.denominator))
    evaluate_candidate!(pred2, den2, spec, num, den, X, num_buf, den_buf)
    return (;
        pred,
        denvals,
        pred2,
        den2,
        holds = pred ≈ pred2 && denvals ≈ den2)
end

function bootstrap_workspace_reuse_report(spec, X, derivative, train_indices,
        threshold; bootstrap_samples::Int = 6, chunk_size::Int = 32,
        seed::Integer = 7)
    rng = MersenneTwister(seed)
    n = length(train_indices)
    n_num = length(spec.numerator)
    n_den = length(spec.denominator)
    ws = allocate_streaming_implicit_workspace(
        eltype(X), n, n_num, n_den, chunk_size)
    start_resize = ws.resize_count
    start_stlsq = ws.stlsq.resize_count
    selected = zeros(Float64, n_num + n_den)
    for _ in 1:bootstrap_samples
        block_length = max(2, round(Int, sqrt(n)))
        indices = _block_bootstrap_indices(rng, train_indices, block_length)
        numerator, denominator = _fit_implicit_stream(
            spec, X, derivative, indices, threshold;
            chunk_size = chunk_size, workspace = ws)
        selected .+= .!iszero.(vcat(numerator, denominator))
    end
    return (;
        selected = selected ./ bootstrap_samples,
        workspace_resizes = ws.resize_count - start_resize,
        stlsq_resizes = ws.stlsq.resize_count - start_stlsq,
        reused = (ws.resize_count - start_resize) == 0 &&
                 (ws.stlsq.resize_count - start_stlsq) == 0)
end

function discovery_workspace_alloc_report(A, y, threshold; chunk_size::Int = 32)
    ws = allocate_stlsq_workspace(eltype(A), size(A, 1), size(A, 2), chunk_size)
    _stlsq_blocked!(ws, A, y, threshold)
    gram_ptr = pointer(ws.gram)
    chunk_ptr = pointer(ws.design_chunk)
    _stlsq_blocked!(ws, A, y, threshold)
    naive = @allocated _stlsq_blocked(A, y, threshold; chunk_size = chunk_size)
    reused = @allocated _stlsq_blocked!(ws, A, y, threshold)
    return (;
        gram_stable = pointer(ws.gram) == gram_ptr,
        chunk_stable = pointer(ws.design_chunk) == chunk_ptr,
        naive_bytes = naive,
        reused_bytes = reused,
        reuse_smaller = reused < naive,
        resize_count = ws.resize_count,
        holds = pointer(ws.gram) == gram_ptr &&
                pointer(ws.design_chunk) == chunk_ptr &&
                reused < naive)
end

# -- Source / docs locks ------------------------------------------------------

function discovery_workspace_source_holds()
    src = read(discovery_workspace_source_path(), String)
    impl = read(discovery_jl_source_path(), String) *
           read(basis_factory_source_path(), String)
    docs = isfile(discovery_streaming_docs_path()) ?
        read(discovery_streaming_docs_path(), String) : ""
    return all(occursin(needle, src) for needle in DISCOVERY_WORKSPACE_MUST_CONTAIN) &&
           !any(occursin(needle, impl) || occursin(needle, docs)
                for needle in DISCOVERY_WORKSPACE_MUST_NOT_CONTAIN)
end

function discovery_jl_uses_workspace()
    src = read(discovery_jl_source_path(), String)
    return occursin("_stlsq_blocked!", src) &&
           occursin("_fit_implicit_stream", src) &&
           occursin("_backend_chunk_size", src) &&
           occursin("StreamingImplicitWorkspace", src)
end

function basis_factory_evaluates_in_place()
    src = read(basis_factory_source_path(), String)
    start = findfirst("function evaluate_library!(output::AbstractMatrix", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("evaluate_term!", body) &&
           !occursin("evaluate_term(term, X)", body)
end

function discovery_streaming_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "discovery-streaming.md")
end

function discovery_streaming_docs_hold()
    path = discovery_streaming_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(discovery_streaming_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("discovery-streaming.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function discovery_streaming_landing_docs_hold()
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sentences = discovery_streaming_locked_sentences()
    return occursin("discovery-streaming", sciml) &&
           occursin("_stlsq_blocked!", howto) &&
           occursin(sentences.workspace, sciml)
end

function discovery_workspace_contract_holds()
    return discovery_workspace_source_holds() &&
           discovery_jl_uses_workspace() &&
           basis_factory_evaluates_in_place() &&
           discovery_streaming_docs_hold() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source()
end

function discovery_workspace_source_violations()
    src = read(discovery_workspace_source_path(), String)
    impl = read(discovery_jl_source_path(), String) *
           read(basis_factory_source_path(), String)
    docs = isfile(discovery_streaming_docs_path()) ?
        read(discovery_streaming_docs_path(), String) : ""
    missing = [s for s in DISCOVERY_WORKSPACE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in DISCOVERY_WORKSPACE_MUST_NOT_CONTAIN
                 if occursin(s, impl) || occursin(s, docs)]
    return (; missing, forbidden)
end
