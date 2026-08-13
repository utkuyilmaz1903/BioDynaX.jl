module BioDynaXSBMLExt

using BioDynaX
using SBML

"""
    import_sbml_network(path)

Map SBML species and reaction stoichiometry into a `BiologicalNetwork`.
Kinetic laws are **not** guessed: mass-action-looking reactions stay
`MASS_ACTION`; everything else is `known=false` so the compiler uses a neural
unknown. Prefer `import_sbmltoolkit_network` for Catalyst/MTK lowering.
"""
function BioDynaX.import_sbml_network(path::AbstractString)
    model = _sbml_model(readSBML(path))
    species = _sbml_species(model)
    nodes = NodeSpec[]
    name_to_idx = Dict{String,Int}()
    for (i, (sid, spec)) in enumerate(species)
        push!(nodes, NodeSpec(
            name = Symbol(replace(sid, "-" => "_")),
            kind = _sbml_boundary(spec) ? INPUT : STATE,
            observed = !_sbml_boundary(spec)))
        name_to_idx[sid] = i
    end
    isempty(nodes) && throw(ArgumentError("SBML model contains no species"))
    reactions = ReactionSpec[]
    for (rid, rxn) in _sbml_reactions(model)
        stoich = Dict{Int,Float64}()
        for (sid, coeff) in _sbml_stoichiometry(rxn)
            idx = get(name_to_idx, sid) do
                throw(ArgumentError("unknown species $sid in reaction $rid"))
            end
            stoich[idx] = Float64(coeff)
        end
        regulators = Int[]
        for sid in _sbml_modifiers(rxn)
            haskey(name_to_idx, sid) && push!(regulators, name_to_idx[sid])
        end
        if isempty(regulators)
            for (idx, coeff) in stoich
                coeff < 0 && push!(regulators, idx)
            end
            unique!(regulators)
        end
        safe_id = replace(string(rid), "-" => "_")
        known, family, meta = _honest_kinetic_metadata(rxn, safe_id)
        push!(reactions, ReactionSpec(
            name = Symbol(safe_id),
            stoichiometry = stoich,
            regulators = regulators,
            known = known,
            family = family,
            metadata = meta))
    end
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

_sbml_model(doc) = hasproperty(doc, :model) ? doc.model : doc

function _sbml_species(model)
    raw = model.species
    raw isa AbstractDict ? collect(raw) : collect(pairs(raw))
end

function _sbml_reactions(model)
    raw = model.reactions
    raw isa AbstractDict ? collect(raw) : collect(pairs(raw))
end

_sbml_boundary(spec) = hasproperty(spec, :boundary_condition) ?
    spec.boundary_condition :
    (hasproperty(spec, :boundary) ? spec.boundary : false)

function _sbml_stoichiometry(rxn)
    if hasproperty(rxn, :stoichiometry)
        return rxn.stoichiometry
    end
    stoich = Dict{String,Float64}()
    if hasproperty(rxn, :reactants)
        for ref in rxn.reactants
            stoich[string(ref.species)] = -Float64(ref.stoichiometry)
        end
    end
    if hasproperty(rxn, :products)
        for ref in rxn.products
            sid = string(ref.species)
            stoich[sid] = get(stoich, sid, 0.0) + Float64(ref.stoichiometry)
        end
    end
    return stoich
end

function _sbml_modifiers(rxn)
    hasproperty(rxn, :modifiers) || return String[]
    return [hasproperty(mod, :species) ? string(mod.species) : string(mod)
            for mod in rxn.modifiers]
end

function _honest_kinetic_metadata(rxn, rid::AbstractString)
    law = hasproperty(rxn, :kinetic_law) ? rxn.kinetic_law : nothing
    rate_param = Symbol("k_", rid)
    if law === nothing
        return true, MASS_ACTION, MassActionMetadata(rate_param = rate_param)
    end
    # MathML is not parsed here. Unknown laws compile as neural edges.
    return false, HILL, HillMetadata()
end

end
