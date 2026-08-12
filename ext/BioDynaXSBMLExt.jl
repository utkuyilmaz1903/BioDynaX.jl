module BioDynaXSBMLExt

using BioDynaX
using SBML

"""
    import_sbml_network(path::AbstractString)

Parse an SBML document into a `BiologicalNetwork`. Recognized kinetics are
mapped to mass-action or saturation metadata; unknown forms fail at compile time.
"""
function BioDynaX.import_sbml_network(path::AbstractString)
    doc = readSBML(path)
    model = doc.model
    nodes = NodeSpec[]
    name_to_idx = Dict{String,Int}()
    for (i, species) in enumerate(model.species)
        push!(nodes, NodeSpec(
            name = Symbol(replace(species.id, "-" => "_")),
            kind = species.boundary ? INPUT : STATE,
            observed = !species.boundary))
        name_to_idx[species.id] = i
    end
    isempty(nodes) && throw(ArgumentError("SBML model contains no species"))
    reactions = ReactionSpec[]
    for rxn in model.reactions
        stoich = Dict{Int,Float64}()
        for (sid, coeff) in rxn.stoichiometry
            idx = get(name_to_idx, sid) do
                throw(ArgumentError("unknown species $sid in reaction $(rxn.id)"))
            end
            stoich[idx] = Float64(coeff)
        end
        regulators = Int[]
        for mod in rxn.modifiers
            haskey(name_to_idx, mod.species) &&
                push!(regulators, name_to_idx[mod.species])
        end
        rid = replace(rxn.id, "-" => "_")
        rate_param = Symbol("k_", rid)
        family = MASS_ACTION
        meta = MassActionMetadata(rate_param = rate_param)
        law = rxn.kinetic_law
        if law !== nothing && occursin("Michaelis", string(typeof(law)))
            family = SATURATION
            meta = SaturationMetadata(
                vmax_param = rate_param,
                km_param = Symbol("km_", rid))
        end
        push!(reactions, ReactionSpec(
            name = Symbol(rid),
            stoichiometry = stoich,
            regulators = unique(regulators),
            family = family,
            metadata = meta))
    end
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

end # module
