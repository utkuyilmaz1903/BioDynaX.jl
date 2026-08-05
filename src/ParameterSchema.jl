"""Compile-time parameter names required by a compiled mechanism."""
struct ParameterSchema
    phys_names::Vector{Symbol}
    nn_heads::Int
end

function parameter_schema(model::UDEModel)
    names = Symbol[]
    seen = Set{Symbol}()
    for term in model.compiled.production_terms
        if term isa InputProductionTerm
            for sym in (term.rate_param, term.input_param)
                sym ∈ seen || (push!(names, sym); push!(seen, sym))
            end
        elseif term isa MassActionProductionTerm
            term.param ∈ seen || (push!(names, term.param); push!(seen, term.param))
        end
    end
    for term in model.compiled.destruction_terms
        if term isa LinearDestructionTerm
            term.param ∈ seen || (push!(names, term.param); push!(seen, term.param))
        elseif term isa HillDestructionTerm
            for sym in (term.vmax_param, term.k_param)
                sym ∈ seen || (push!(names, sym); push!(seen, sym))
            end
        elseif term isa CompetitiveDestructionTerm
            for sym in (term.vmax_param, term.km_param, term.ki_param)
                sym ∈ seen || (push!(names, sym); push!(seen, sym))
            end
        end
    end
    nn_heads = count(term -> term isa NeuralDestructionTerm,
                     model.compiled.destruction_terms)
    return ParameterSchema(names, nn_heads)
end

function validate_phys_parameters(phys::NamedTuple, schema::ParameterSchema)
    for name in schema.phys_names
        hasproperty(phys, name) ||
            throw(ArgumentError("missing physical parameter :$name"))
        value = getproperty(phys, name)
        value > zero(value) ||
            throw(ArgumentError("physical parameter :$name must be positive"))
    end
    return nothing
end

"""Default positive physical parameters for a compiled model."""
function default_phys_parameters(schema::ParameterSchema)
    defaults = Dict{Symbol,Float64}(
        :α_p53 => 0.9, :β_mdm2 => 1.1, :γ_mdm2 => 1.5, :signal => 1.0,
        :vmax => 1.0, :km => 0.5, :ki => 0.5, :n => 4.0, :K => 0.5,
        :γ => 1.0, :k => 0.5)
    return (; (name => get(defaults, name, 1.0) for name in schema.phys_names)...)
end

function default_parameters(network::BiologicalNetwork, model::UDEModel;
                            rng::AbstractRNG = Random.default_rng())
    schema = parameter_schema(model)
    nn, nn_ps, _ = build_ude_nn(rng)
    validate_phys_parameters(default_phys_parameters(schema), schema)
    return pack_parameters(default_phys_parameters(schema), nn_ps)
end

function default_parameters(model::UDEModel; kwargs...)
    return default_parameters(model.network, model; kwargs...)
end
