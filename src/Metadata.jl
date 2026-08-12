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
    input_node::Union{Int,Nothing} = nothing
    rate_param::Symbol
    input_param::Symbol = :signal
end

"""Mass-action production or regulation."""
Base.@kwdef struct MassActionMetadata <: KineticMetadata
    rate_param::Symbol
    order::Int = 1
    input_param::Union{Symbol,Nothing} = nothing
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
    substrate::Union{Int,Nothing} = nothing
    inhibitor::Union{Int,Nothing} = nothing
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
    evaluator::Union{Function,Nothing} = nothing
    preset::Symbol = :none
end

const MetadataLike = Union{KineticMetadata,AbstractDict{Symbol}}

@inline _meta_symbol(meta::AbstractDict{Symbol}, key::Symbol, default::Symbol) =
    Symbol(get(meta, key, default))

@inline function _meta_int(meta::AbstractDict{Symbol}, key::Symbol, default::Int)
    value = get(meta, key, default)
    return value isa Integer ? Int(value) : default
end

@inline _meta_symbol(meta::Dict{Symbol,Any}, key::Symbol, default::Symbol) =
    get(meta, key, default)

@inline function _meta_int(meta::Dict{Symbol,Any}, key::Symbol, default::Int)
    value = get(meta, key, default)
    return value isa Integer ? Int(value) : default
end

@inline _meta_symbol(meta::InputDriveMetadata, key::Symbol, default::Symbol) =
    key === :drive ? meta.drive :
    key === :rate_param ? meta.rate_param :
    key === :input_param ? meta.input_param : default

@inline _meta_symbol(meta::MassActionMetadata, key::Symbol, default::Symbol) =
    key === :rate_param ? meta.rate_param :
    key === :input_param ? something(meta.input_param, default) : default

@inline _meta_int(meta::MassActionMetadata, key::Symbol, default::Int) =
    key === :order ? meta.order : default

@inline _meta_symbol(meta::HillMetadata, key::Symbol, default::Symbol) =
    key === :vmax_param ? meta.vmax_param :
    key === :k_param ? meta.k_param : default

@inline _meta_int(meta::HillMetadata, key::Symbol, default::Int) =
    key === :hill_order ? meta.hill_order : default

@inline _meta_symbol(meta::CompetitiveMetadata, key::Symbol, default::Symbol) =
    key === :vmax_param ? meta.vmax_param :
    key === :km_param ? meta.km_param :
    key === :ki_param ? meta.ki_param : default

@inline function _meta_get(meta::CompetitiveMetadata, key::Symbol)
    key === :substrate && return meta.substrate
    key === :inhibitor && return meta.inhibitor
    key === :regulators && return meta.regulators
    return nothing
end

@inline _meta_symbol(meta::LinearDecayMetadata, key::Symbol, default::Symbol) =
    key === :rate_param ? meta.rate_param : default

@inline _meta_symbol(meta::SaturationMetadata, key::Symbol, default::Symbol) =
    key === :vmax_param ? meta.vmax_param :
    key === :km_param ? meta.km_param : default

@inline _meta_symbol(meta::CustomKineticMetadata, key::Symbol, default::Symbol) =
    key === :rate_param ? meta.rate_param : default

@inline function _meta_get(meta::CustomKineticMetadata, key::Symbol)
    key === :evaluator && return meta.evaluator
    key === :preset && return meta.preset
    return nothing
end

@inline _meta_symbol(meta::EmptyMetadata, key::Symbol, default::Symbol) = default
@inline _meta_int(meta::EmptyMetadata, key::Symbol, default::Int) = default

@inline function _meta_get(meta::MetadataLike, key::Symbol)
    meta isa AbstractDict{Symbol} && return get(meta, key, nothing)
    meta isa CompetitiveMetadata && return _meta_get(meta, key)
    return nothing
end

@inline function _meta_haskey(meta::MetadataLike, key::Symbol)
    meta isa AbstractDict{Symbol} && return haskey(meta, key)
    meta isa InputDriveMetadata &&
        return key in (:drive, :rate_param, :input_param, :input_node)
    meta isa MassActionMetadata &&
        return key in (:rate_param, :order, :input_param)
    meta isa HillMetadata &&
        return key in (:vmax_param, :k_param, :hill_order)
    meta isa CompetitiveMetadata &&
        return key in (:vmax_param, :km_param, :ki_param,
                       :substrate, :inhibitor, :regulators)
    meta isa LinearDecayMetadata && return key === :rate_param
    meta isa SaturationMetadata &&
        return key in (:vmax_param, :km_param)
    meta isa CustomKineticMetadata &&
        return key in (:rate_param, :evaluator, :preset)
    return false
end

function metadata_summary(meta::MetadataLike)
    meta isa AbstractDict{Symbol} && return "Dict{Symbol}($(length(meta)) keys)"
    return string(typeof(meta))
end
