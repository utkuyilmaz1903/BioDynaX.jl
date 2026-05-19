###############################################################################
# DataGen.jl — synthetic-data generation from the true (Hill-kinetics) model.
#
# This file is *not* in the differentiable training path, so in-place mutation
# inside the ground-truth RHS is fine and slightly faster.
###############################################################################

"""
    ground_truth!(dx, x, p, t)

Mechanistic ground-truth p53 / Mdm2 model with Hill-saturation kinetics on
the Mdm2 → p53 degradation term.

The Hill function is

    deg(Mdm2) = γ_p53 · Mdm2ⁿ / (Kⁿ + Mdm2ⁿ)

— a saturating non-linearity that the UDE / NN has to rediscover, and that
`Discovery.jl` will try to approximate with a sparse polynomial expansion.

Used *only* to fabricate synthetic data; never differentiated through.
"""
function ground_truth!(dx, x, p, t)
    p53, mdm2 = x[1], x[2]
    deg = p.γ_p53 * mdm2^p.n / (p.K^p.n + mdm2^p.n)
    dx[1] = p.α_p53 * p.signal - deg     * p53
    dx[2] = p.β_mdm2 * p53     - p.γ_mdm2 * mdm2
    return nothing
end

"""
    default_truth_params() -> NamedTuple

Reasonable default constants for the Hill-kinetics ground truth.
The "physical" subset (`α_p53`, `β_mdm2`, `γ_mdm2`, `signal`) is what the
UDE is *told*; the Hill-specific subset (`γ_p53`, `K`, `n`) is what it
must learn to imitate.
"""
default_truth_params() = (
    α_p53  = 0.9,
    β_mdm2 = 1.1,
    γ_mdm2 = 1.5,
    signal = 1.0,
    γ_p53  = 2.0,
    K      = 0.5,
    n      = 4,
)

"""
    generate_data(rng;
                  u0           = [0.2, 0.1],
                  tspan        = (0.0, 20.0),
                  n_points     = 40,
                  noise_σ      = 0.05,
                  truth_params = default_truth_params())
        -> (t_data, clean_data, noisy_data, truth_params)

Solve the ground-truth ODE on a uniform time grid and contaminate the
solution with i.i.d. Gaussian noise `𝒩(0, σ²)`.

* `clean_data` — noise-free trajectory (for visualisation / metrics).
* `noisy_data` — what the UDE is trained against.

Throws `ArgumentError` for malformed `n_points` / `noise_σ` arguments.
"""
function generate_data(rng::AbstractRNG;
                       u0::Vector{Float64} = [0.2, 0.1],
                       tspan::Tuple{Float64,Float64} = (0.0, 20.0),
                       n_points::Int = 40,
                       noise_σ::Float64 = 0.05,
                       truth_params::NamedTuple = default_truth_params())

    n_points ≥ 2 || throw(ArgumentError("n_points must be ≥ 2 (got $n_points)"))
    noise_σ ≥ 0  || throw(ArgumentError("noise_σ must be ≥ 0 (got $noise_σ)"))
    tspan[2] > tspan[1] ||
        throw(ArgumentError("tspan must be increasing (got $tspan)"))

    t_data = collect(range(tspan[1], tspan[2]; length = n_points))
    prob   = ODEProblem(ground_truth!, u0, tspan, truth_params)
    sol    = solve(prob, Tsit5(); saveat = t_data,
                   abstol = 1e-9, reltol = 1e-9)

    clean_data = Array(sol)
    noisy_data = clean_data .+ noise_σ .* randn(rng, size(clean_data))
    return t_data, clean_data, noisy_data, truth_params
end
