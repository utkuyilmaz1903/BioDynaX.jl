module BioDynaXSBMLToolkitExt

using BioDynaX
using SBMLToolkit
using Catalyst
using ModelingToolkit

"""
    import_sbmltoolkit_network(path)

Lower an SBML file through SBMLToolkit to a Catalyst `ReactionSystem`, then
map species and stoichiometry onto a `BiologicalNetwork`. Rate laws are not
re-parsed into Hill/MM metadata; reactions become mass-action when Catalyst
reports a mass-action rate, otherwise `known=false`.
"""
function BioDynaX.import_sbmltoolkit_network(path::AbstractString)
    rs = SBMLToolkit.readSBML(path)
    sps = Catalyst.species(rs)
    isempty(sps) && throw(ArgumentError("SBMLToolkit model contains no species"))
    nodes = NodeSpec[]
    name_to_idx = Dict{String,Int}()
    for (i, sp) in enumerate(sps)
        sid = _species_id(sp)
        push!(nodes, NodeSpec(name = Symbol(replace(sid, "-" => "_"))))
        name_to_idx[sid] = i
        name_to_idx[string(sp)] = i
    end
    reactions = ReactionSpec[]
    for (k, rxn) in enumerate(Catalyst.reactions(rs))
        stoich = Dict{Int,Float64}()
        for (sp, coeff) in _reaction_stoich(rxn)
            idx = _lookup_species(name_to_idx, sp)
            stoich[idx] = get(stoich, idx, 0.0) + Float64(coeff)
        end
        filter!(pair -> pair.second != 0, stoich)
        regulators = [idx for (idx, coeff) in stoich if coeff < 0]
        rid = Symbol("r", k)
        known = _is_massaction(rxn)
        family = known ? MASS_ACTION : HILL
        meta = known ?
            MassActionMetadata(rate_param = Symbol("k_", rid)) :
            HillMetadata()
        push!(reactions, ReactionSpec(
            name = rid,
            stoichiometry = stoich,
            regulators = isempty(regulators) ? collect(keys(stoich)) : regulators,
            known = known,
            family = family,
            metadata = meta))
    end
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

_species_id(sp) = string(ModelingToolkit.tosymbol(sp; escape = false))

function _lookup_species(name_to_idx, sp)
    for key in (string(sp), _species_id(sp))
        haskey(name_to_idx, key) && return name_to_idx[key]
    end
    throw(ArgumentError("unmapped species $sp"))
end

function _reaction_stoich(rxn)
    pairs = Pair[]
    if hasproperty(rxn, :substrates)
        for (sp, c) in zip(rxn.substrates, rxn.substoich)
            push!(pairs, sp => -c)
        end
        for (sp, c) in zip(rxn.products, rxn.prodstoich)
            push!(pairs, sp => c)
        end
        return pairs
    end
    return pairs
end

function _is_massaction(rxn)
    hasproperty(rxn, :issimplified) && return getproperty(rxn, :only_use_rate) == false
    rate = hasproperty(rxn, :rate) ? rxn.rate : nothing
    rate === nothing && return true
    text = string(rate)
    return !occursin('/', text)
end

end
