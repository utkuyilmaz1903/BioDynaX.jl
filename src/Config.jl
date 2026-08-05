abstract type AbstractConstraintStrategy end
abstract type AbstractDiscoveryBackend end
abstract type AbstractADPolicy end
struct StructuralPositivity <: AbstractConstraintStrategy end

"""
    AugmentedLagrangianConfig

Smooth inequality-constraint policy. `ρ` is updated automatically from
primal progress; users do not select a fixed penalty multiplier.
"""
Base.@kwdef struct AugmentedLagrangianConfig <: AbstractConstraintStrategy
    initial_ρ::Float64 = 1.0
    growth::Float64 = 5.0
    max_ρ::Float64 = 1e6
    tolerance::Float64 = 1e-6
    smoothness::Float64 = 1e-3
    outer_iterations::Int = 4
    progress_ratio::Float64 = 0.5
end


"""Compatibility AD backend using Zygote out-of-place sensitivities."""
struct ZygoteAD <: AbstractADPolicy end

"""Production AD backend; uses in-place RHS with SciMLSensitivity adjoints."""
struct ProductionAD <: AbstractADPolicy end

function sensealg(policy::ZygoteAD)
    return InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)
end

function sensealg(::ProductionAD)
    return InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)
end

struct SolverConfig{A,S,T<:AbstractFloat,P<:AbstractADPolicy}
    algorithm::A
    sensealg::S
    abstol::T
    reltol::T
    maxiters::Int
    ad_policy::P
end

function SolverConfig(; algorithm = Tsit5(),
                      ad_policy::AbstractADPolicy = ZygoteAD(),
                      sensealg = sensealg(ad_policy),
                      abstol::Real = 1e-6, reltol::Real = 1e-6,
                      maxiters::Int = 1_000_000)
    tolerance_type = promote_type(typeof(float(abstol)), typeof(float(reltol)))
    return SolverConfig(
        algorithm, sensealg, tolerance_type(abstol), tolerance_type(reltol),
        maxiters, ad_policy)
end

struct TrainingConfig{T<:AbstractFloat,C<:AbstractConstraintStrategy,S}
    adam_iterations::Int
    adam_learning_rate::T
    bfgs_iterations::Int
    gradient_clip::T
    log_every::Int
    constraint::C
    solver::S
    horizon_schedule::Vector{T}
end

function TrainingConfig(; adam_iterations::Int = 300,
                        adam_learning_rate::Real = 1e-2,
                        bfgs_iterations::Int = 100,
                        gradient_clip::Real = 10,
                        log_every::Int = 20,
                        constraint::AbstractConstraintStrategy =
                            StructuralPositivity(),
                        solver::SolverConfig = SolverConfig(),
                        horizon_schedule = [0.25, 0.5, 1.0])
    T = promote_type(typeof(float(adam_learning_rate)),
                     typeof(float(gradient_clip)),
                     eltype(float.(horizon_schedule)))
    return TrainingConfig(
        adam_iterations, T(adam_learning_rate), bfgs_iterations,
        T(gradient_clip), log_every, constraint, solver,
        T.(horizon_schedule))
end

"""Current explicit STLSQ backend retained for compatibility."""
Base.@kwdef struct ExplicitSTLSQ <: AbstractDiscoveryBackend
    threshold::Float64 = 1e-2
end

"""
    ImplicitSINDyPI

Configuration for graph-local rational discovery through the implicit identity
`D(z)ẋ - N(z) = 0`.
"""
Base.@kwdef struct ImplicitSINDyPI <: AbstractDiscoveryBackend
    threshold::Float64 = 1e-2
    max_degree::Int = 3
    max_hill_degree::Int = 4
    max_parents::Int = 8
    extra_candidates::Int = 0
    bootstrap_samples::Int = 32
    validation_fraction::Float64 = 0.2
    denominator_floor::Float64 = 1e-8
end

struct DiscoveryConfig{B<:AbstractDiscoveryBackend}
    backend::B
    include_constant::Bool
    include_interactions::Bool
    max_interaction_order::Int
    seed::UInt64
end

function DiscoveryConfig(; backend::AbstractDiscoveryBackend = ImplicitSINDyPI(),
                         include_constant::Bool = true,
                         include_interactions::Bool = true,
                         max_interaction_order::Int = 2,
                         seed::Integer = 42)
    return DiscoveryConfig(
        backend, include_constant, include_interactions,
        max_interaction_order, UInt64(seed))
end

Base.@kwdef struct ExecutionConfig
    backend::Symbol = :serial
    batch_size::Int = 1
    deterministic::Bool = true
    checkpoint_every::Int = 50
end
