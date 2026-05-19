###############################################################################
# Discovery.jl — multivariate symbolic regression on the trained UDE.
#
# Pipeline (canonical UDE → SINDy-style sparse regression, multivariate):
#
#   1. Simulate the *trained* UDE on a dense time grid → state matrix
#      X ∈ ℝ^{2 × N} where row 1 = p53(t), row 2 = Mdm2(t).
#
#   2. Evaluate the trained UDE right-hand side at each sample point to
#      obtain the "UDE-inferred" derivative matrix Ẋ ∈ ℝ^{2 × N}.  This is
#      cleaner than collocation-based numerical differentiation because the
#      UDE RHS itself is closed-form (NN + analytic kinetics).
#
#   3. Build a *multivariate* polynomial basis in (u₁, u₂) = (p53, Mdm2)
#      that includes interaction terms (u₁·u₂, u₁²·u₂, …) — these are what
#      let STLSQ recover mass-action kinetics.
#
#   4. Solve a `DirectDataDrivenProblem(X, Ẋ)` with `STLSQ(threshold)`.
#      Each row of Ẋ is regressed independently, so the discovered system
#      may use different sparse terms for dp53/dt and dMdm2/dt.
#
#   5. Every potentially-throwing step is wrapped in try/catch and folded
#      into a structured NamedTuple return — this routine never crashes
#      the host pipeline.
###############################################################################

"""
    sample_learned_function(p_trained, nn, st;
                            mdm2_range = range(1e-3, 2.5; length = 400))
        -> (X::Matrix, Y::Matrix)

Probe the trained NN on a 1-D sweep of Mdm2 values.  Kept for callers
that want to inspect / plot the learned scalar function directly.
"""
function sample_learned_function(p_trained, nn, st;
                                 mdm2_range = range(1e-3, 2.5; length = 400))
    xs = collect(mdm2_range)
    ys = map(xs) do m
        out, _ = nn([m], p_trained.nn, st)
        out[1]
    end
    keep = isfinite.(ys)
    X = reshape(xs[keep], 1, :)
    Y = reshape(ys[keep], 1, :)
    return X, Y
end

# ---------------------------------------------------------------------------
# Multivariate basis construction.
# ---------------------------------------------------------------------------
"""
    _build_multivariate_basis(degree, include_interactions, include_constant)
        -> (basis, used_terms)

Construct a `Basis` in two state variables `u[1], u[2]` containing all
monomials up to `degree`, plus optional interaction monomials and an
optional constant.  Returns both the `Basis` and the human-readable list
of symbolic terms for diagnostics.
"""
function _build_multivariate_basis(degree::Int,
                                   include_interactions::Bool,
                                   include_constant::Bool)
    @variables u[1:2]
    u1, u2 = u[1], u[2]

    terms = Any[]
    include_constant && push!(terms, one(u1))   # constant production
    push!(terms, u1, u2)

    if degree ≥ 2
        push!(terms, u1^2, u2^2)
        include_interactions && push!(terms, u1 * u2)
    end
    if degree ≥ 3
        push!(terms, u1^3, u2^3)
        if include_interactions
            push!(terms, u1^2 * u2, u1 * u2^2)
        end
    end

    return Basis(terms, [u1, u2]), terms
end

# ---------------------------------------------------------------------------
# Build (states, derivatives) regression data from the trained UDE.
# ---------------------------------------------------------------------------
"""
    _collect_trajectory_data(p_trained, nn, st, u0, tspan, n_samples)
        -> (X::Matrix, Ẋ::Matrix, t_grid::Vector)

Simulate the trained UDE and return the state matrix `X` plus the
matrix of UDE-inferred derivatives `Ẋ` at the same time points.

Drops any column containing a non-finite value (numerical hygiene).
"""
function _collect_trajectory_data(p_trained, nn, st,
                                  u0::Vector{Float64},
                                  tspan::Tuple{Float64,Float64},
                                  n_samples::Int)

    t_grid = collect(range(tspan[1], tspan[2]; length = n_samples))
    X = predict_ude(p_trained, u0, tspan, t_grid, nn, st)

    # Number of points the solver actually returned.
    N = size(X, 2)
    Ẋ = Matrix{Float64}(undef, 2, N)
    @inbounds for i in 1:N
        d = ude_system(X[:, i], p_trained, t_grid[i], nn, st)
        Ẋ[1, i] = d[1]
        Ẋ[2, i] = d[2]
    end

    keep = vec(all(isfinite, X; dims = 1) .& all(isfinite, Ẋ; dims = 1))
    return X[:, keep], Ẋ[:, keep], t_grid[1:N][keep]
end

# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
"""
    discover_equations(p_trained, nn, st;
                       u0                   = [0.2, 0.1],
                       tspan                = (0.0, 20.0),
                       n_samples            = 200,
                       polynomial_degree    = 3,
                       include_interactions = true,
                       include_constant     = true,
                       sparsity_threshold   = 1e-2,
                       verbose              = true)
        -> NamedTuple

Multivariate sparse symbolic regression on the trained UDE.

The trained UDE is simulated on a dense grid, its right-hand side is
evaluated to produce an `(X, Ẋ)` dataset, and a `DirectDataDrivenProblem`
is solved with `STLSQ(sparsity_threshold)` against a *multivariate*
polynomial basis in `(u[1], u[2]) = (p53, Mdm2)` — including the
mass-action interaction `u[1] * u[2]`.

# Returns
NamedTuple with at least:
* `success::Bool`
* `message::String`
* `equation::Union{Nothing, String}` — pretty-printed equations on success
* `basis_used::String`              — list of basis terms tried (diagnostic)
* `basis_result`, `solution`        — raw DataDrivenDiffEq objects on success

# Robustness
Every potentially-throwing step (basis construction, trajectory sim,
problem instantiation, sparse solve, basis extraction) is wrapped in
`try/catch`.  Failures are logged with `@warn` and folded into the
returned NamedTuple — this routine never propagates an exception.
"""
function discover_equations(p_trained, nn, st;
                            u0::Vector{Float64} = [0.2, 0.1],
                            tspan::Tuple{Float64,Float64} = (0.0, 20.0),
                            n_samples::Int = 200,
                            polynomial_degree::Int = 3,
                            include_interactions::Bool = true,
                            include_constant::Bool = true,
                            sparsity_threshold = 1e-2,
                            verbose::Bool = true)

    polynomial_degree ≥ 1 ||
        throw(ArgumentError("polynomial_degree must be ≥ 1 (got $polynomial_degree)"))
    n_samples ≥ 10 ||
        throw(ArgumentError("n_samples must be ≥ 10 (got $n_samples)"))

    # 1) Simulate trained UDE + compute derivatives ------------------------
    X, Ẋ = try
        _X, _Ẋ, _ = _collect_trajectory_data(p_trained, nn, st,
                                             u0, tspan, n_samples)
        _X, _Ẋ
    catch err
        msg = "Could not simulate trained UDE for discovery: " *
              sprint(showerror, err)
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = "")
    end

    if size(X, 2) < 10
        msg = "Too few finite trajectory samples ($(size(X,2))) — " *
              "discovery aborted."
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = "")
    end

    # 2) Multivariate symbolic basis ---------------------------------------
    basis, terms_used = try
        _build_multivariate_basis(polynomial_degree,
                                  include_interactions,
                                  include_constant)
    catch err
        msg = "Failed to construct multivariate Basis: " *
              sprint(showerror, err)
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = "")
    end
    basis_used = string(terms_used)

    # 3) DataDrivenProblem  X (states) → Ẋ (derivatives) -------------------
    problem = try
        DirectDataDrivenProblem(X, Ẋ; name = :biodynax_full_system)
    catch err
        msg = "Failed to build DirectDataDrivenProblem: " *
              sprint(showerror, err)
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = basis_used)
    end

    # 4) Sparse solve ------------------------------------------------------
    opt = STLSQ(sparsity_threshold)
    sol = try
        solve(problem, basis, opt)
    catch err
        msg = "Sparse regression failed: " * sprint(showerror, err)
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = basis_used)
    end

    # 5) Extract result basis ---------------------------------------------
    basis_result = try
        get_basis(sol)
    catch err
        msg = "Could not extract result basis: " * sprint(showerror, err)
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = basis_used, solution = sol)
    end

    eqs = try
        equations(basis_result)
    catch err
        msg = "Could not read equations from result basis: " *
              sprint(showerror, err)
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = basis_used, solution = sol,
                  basis_result = basis_result)
    end

    if isempty(eqs)
        msg = "Sparse regression returned empty basis " *
              "(threshold = $sparsity_threshold)."
        @warn msg
        return (; success = false, message = msg, equation = nothing,
                  basis_used = basis_used, solution = sol,
                  basis_result = basis_result)
    end

    eq_string = sprint(io -> show(io, "text/plain", basis_result))
    if verbose
        println("\n[Discovery] Multivariate symbolic system recovered:")
        println(eq_string)
    end

    return (; success      = true,
              message      = "ok",
              equation     = eq_string,
              basis_used   = basis_used,
              basis_result = basis_result,
              solution     = sol)
end
