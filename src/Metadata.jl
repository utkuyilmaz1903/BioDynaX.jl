"""
    KineticMetadata

Typed compile-time metadata for edges and reactions. Replaces ad-hoc
`Dict{Symbol,Any}` fields with SciML-style explicit structs while retaining
backward-compatible dict coercion during the 0.x migration.
"""
abstract type KineticMetadata end

"""Empty metadata placeholder."""
struct EmptyMetadata <: KineticMetadata end

"""Input-driven production (e.g. damage signal drives p53)."""
Base.@kwdef struct InputDriveMetadata <: KineticMetadata
    drive::Symbol = :input
    input_node::Union{Int, Nothing} = nothing
    rate_param::Symbol
    input_param::Symbol = :signal
end

"""Mass-action production or regulation."""
Base.@kwdef struct MassActionMetadata <: KineticMetadata
    rate_param::Symbol
    order::Int = 1
    input_param::Union{Symbol, Nothing} = nothing
end

"""Hill-type destruction kinetics."""
Base.@kwdef struct HillMetadata <: KineticMetadata
    vmax_param::Symbol = :vmax
    k_param::Symbol = :K
    hill_order::Int = 4
end

"""Competitive inhibition destruction kinetics."""
Base.@kwdef struct CompetitiveMetadata <: KineticMetadata
    vmax_param::Symbol = :vmax
    km_param::Symbol = :km
    ki_param::Symbol = :ki
    substrate::Union{Int, Nothing} = nothing
    inhibitor::Union{Int, Nothing} = nothing
    regulators::Vector{Int} = Int[]
end

"""First-order linear decay."""
Base.@kwdef struct LinearDecayMetadata <: KineticMetadata
    rate_param::Symbol
end

"""Michaelis–Menten saturation kinetics (single regulator)."""
Base.@kwdef struct SaturationMetadata <: KineticMetadata
    vmax_param::Symbol = :vmax
    km_param::Symbol = :km
end

"""
    CustomKineticMetadata

User-defined or preset destruction/production kinetics. When `evaluator` is set,
it must be a pure function `(x, p, regulators) -> non-negative rate coefficient`.
"""
Base.@kwdef struct CustomKineticMetadata <: KineticMetadata
    rate_param::Symbol = :vmax
    evaluator::Union{Function, Nothing} = nothing
    preset::Symbol = :none
end

"""
    MetadataLike

Accepted metadata on edges and reactions: a `KineticMetadata` subtype or a
legacy `Dict{Symbol}`.
"""
const MetadataLike = Union{KineticMetadata, AbstractDict{Symbol}}

@inline function _coerce_symbol(value, default::Symbol)
    value isa Symbol && return value
    value isa AbstractString && return Symbol(value)
    return default
end

@inline function _meta_symbol(meta::AbstractDict{Symbol}, key::Symbol, default::Symbol)
    haskey(meta, key) || return default
    return _coerce_symbol(get(meta, key, default), default)
end

@inline function _meta_int(meta::AbstractDict{Symbol}, key::Symbol, default::Int)
    haskey(meta, key) || return default
    value = get(meta, key, default)
    return value isa Integer ? Int(value) : default
end

@inline _meta_symbol(meta::InputDriveMetadata, key::Symbol, default::Symbol) = key ===
                                                                               :drive ?
                                                                               meta.drive :
                                                                               key ===
                                                                               :rate_param ?
                                                                               meta.rate_param :
                                                                               key ===
                                                                               :input_param ?
                                                                               meta.input_param :
                                                                               default

@inline _meta_symbol(meta::MassActionMetadata, key::Symbol, default::Symbol) = key ===
                                                                               :rate_param ?
                                                                               meta.rate_param :
                                                                               key ===
                                                                               :input_param ?
                                                                               something(
    meta.input_param, default) : default

@inline _meta_int(meta::MassActionMetadata, key::Symbol, default::Int) = key === :order ?
                                                                         meta.order :
                                                                         default

@inline _meta_symbol(meta::HillMetadata, key::Symbol, default::Symbol) = key ===
                                                                         :vmax_param ?
                                                                         meta.vmax_param :
                                                                         key === :k_param ?
                                                                         meta.k_param :
                                                                         default

@inline _meta_int(meta::HillMetadata, key::Symbol, default::Int) = key === :hill_order ?
                                                                   meta.hill_order : default

@inline _meta_symbol(meta::CompetitiveMetadata, key::Symbol, default::Symbol) = key ===
                                                                                :vmax_param ?
                                                                                meta.vmax_param :
                                                                                key ===
                                                                                :km_param ?
                                                                                meta.km_param :
                                                                                key ===
                                                                                :ki_param ?
                                                                                meta.ki_param :
                                                                                default

@inline function _meta_get(meta::CompetitiveMetadata, key::Symbol)
    key === :substrate && return meta.substrate
    key === :inhibitor && return meta.inhibitor
    key === :regulators && return meta.regulators
    return nothing
end

@inline _meta_symbol(meta::LinearDecayMetadata, key::Symbol, default::Symbol) = key ===
                                                                                :rate_param ?
                                                                                meta.rate_param :
                                                                                default

@inline _meta_symbol(meta::SaturationMetadata, key::Symbol, default::Symbol) = key ===
                                                                               :vmax_param ?
                                                                               meta.vmax_param :
                                                                               key ===
                                                                               :km_param ?
                                                                               meta.km_param :
                                                                               default

@inline _meta_symbol(meta::CustomKineticMetadata, key::Symbol, default::Symbol) = key ===
                                                                                  :rate_param ?
                                                                                  meta.rate_param :
                                                                                  default

@inline function _meta_get(meta::CustomKineticMetadata, key::Symbol)
    key === :evaluator && return meta.evaluator
    key === :preset && return meta.preset
    return nothing
end

@inline _meta_symbol(meta::EmptyMetadata, key::Symbol, default::Symbol) = default
@inline _meta_int(meta::EmptyMetadata, key::Symbol, default::Int) = default
@inline _meta_symbol(meta::KineticMetadata, key::Symbol, default::Symbol) = default
@inline _meta_int(meta::KineticMetadata, key::Symbol, default::Int) = default
@inline _meta_symbol(meta::KineticMetadata, key::Symbol, default::Integer) = _meta_symbol(
    meta, key, Symbol("k_", default))
@inline _meta_symbol(meta::AbstractDict{Symbol}, key::Symbol, default::Integer) = _meta_symbol(
    meta, key, Symbol("k_", default))

@inline function _meta_get(meta::AbstractDict{Symbol}, key::Symbol)
    haskey(meta, key) || return nothing
    return get(meta, key, nothing)
end

@inline _meta_get(::KineticMetadata, ::Symbol) = nothing

@inline _meta_haskey(meta::AbstractDict{Symbol}, key::Symbol) = haskey(meta, key)
@inline _meta_haskey(meta::InputDriveMetadata, key::Symbol) = key === :drive ||
                                                              key === :rate_param ||
                                                              key === :input_param ||
                                                              key === :input_node
@inline _meta_haskey(meta::MassActionMetadata, key::Symbol) = key === :rate_param ||
                                                              key === :order ||
                                                              key === :input_param
@inline _meta_haskey(meta::HillMetadata, key::Symbol) = key === :vmax_param ||
                                                        key === :k_param ||
                                                        key === :hill_order
@inline _meta_haskey(meta::CompetitiveMetadata, key::Symbol) = key === :vmax_param ||
                                                               key === :km_param ||
                                                               key === :ki_param ||
                                                               key === :substrate ||
                                                               key === :inhibitor ||
                                                               key === :regulators
@inline _meta_haskey(meta::LinearDecayMetadata, key::Symbol) = key === :rate_param
@inline _meta_haskey(meta::SaturationMetadata, key::Symbol) = key === :vmax_param ||
                                                              key === :km_param
@inline _meta_haskey(meta::CustomKineticMetadata, key::Symbol) = key === :rate_param ||
                                                                 key === :evaluator ||
                                                                 key === :preset
@inline _meta_haskey(::KineticMetadata, ::Symbol) = false

@inline _is_input_drive(meta::InputDriveMetadata) = meta.drive === :input
@inline function _is_input_drive(meta::AbstractDict{Symbol})
    haskey(meta, :drive) || return false
    return _meta_symbol(meta, :drive, :none) === :input
end
@inline _is_input_drive(::KineticMetadata) = false

"""
    metadata_summary(meta) -> String

Compact diagnostic label for typed kinetic metadata or a `Dict{Symbol}` payload.
"""
metadata_summary(meta::AbstractDict{Symbol}) = "Dict{Symbol}($(length(meta)) keys)"
metadata_summary(meta::KineticMetadata) = string(typeof(meta))
