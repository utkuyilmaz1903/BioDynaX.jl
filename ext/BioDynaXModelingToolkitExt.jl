module BioDynaXModelingToolkitExt

using BioDynaX
using ModelingToolkit
using BioDynaX: UDEModel, InputProductionTerm, MassActionProductionTerm,
                LinearDestructionTerm, HillDestructionTerm,
                SaturationDestructionTerm, SaturationProductionTerm,
                CompetitiveDestructionTerm, NeuralDestructionTerm,
                ImplicitCandidate, ExplicitCandidate, DiscoveryResult, UnknownTermResult

"""
    export_mtk_system(model::UDEModel; name=:BioDynaXNetwork, discovered=nothing)

Build a symbolic `ModelingToolkit.ODESystem` from a compiled UDE. States are
named after the network's dynamic nodes. Neural terms appear as placeholder
variables `nn_i(t)` unless `discovered` gives the discovered rate: a
`DiscoveryResult`, an `ImplicitCandidate` or `ExplicitCandidate` (in the
regulator variables of the single unknown term, in order), or an
`UnknownTermResult`; the rational rate then replaces the placeholder and the
system is complete. Called through `BioDynaX.export_mtk_system`.
"""
function export_mtk_system(model::UDEModel; name::Symbol = :BioDynaXNetwork,
        discovered = nothing)
    cm = model.compiled
    n = cm.nstates
    t = ModelingToolkit.t_nounits
    D = ModelingToolkit.D_nounits
    state_syms = [model.network.nodes[i].name for i in BioDynaX.state_nodes(model.network)]
    length(state_syms) == n || (state_syms = [Symbol("x$i") for i in 1:n])
    sts = [first(@variables($sym(t))) for sym in state_syms]
    state_map = Dict(i => sts[i] for i in 1:n)

    param_syms = Symbol[]
    for term in (cm.production_terms..., cm.destruction_terms...)
        term isa InputProductionTerm &&
            (push!(param_syms, term.rate_param, term.input_param))
        term isa MassActionProductionTerm && push!(param_syms, term.param)
        term isa LinearDestructionTerm && push!(param_syms, term.param)
        term isa HillDestructionTerm && (push!(param_syms, term.vmax_param, term.k_param))
        term isa SaturationDestructionTerm &&
            (push!(param_syms, term.vmax_param, term.km_param))
        term isa SaturationProductionTerm &&
            (push!(param_syms, term.vmax_param, term.km_param))
        term isa CompetitiveDestructionTerm &&
            (push!(param_syms, term.vmax_param, term.km_param, term.ki_param))
    end
    unique!(param_syms)
    params = [first(@parameters($sym)) for sym in param_syms]
    pmap = Dict{Symbol, Num}(sym => p for (sym, p) in zip(param_syms, params))

    nn_map = Dict{Int, Num}()
    neural = [term for term in cm.destruction_terms if term isa NeuralDestructionTerm]
    for term in neural
        haskey(nn_map, term.nn_index) && continue
        sym = Symbol("nn_", term.nn_index)
        nn_map[term.nn_index] = first(@variables($sym(t)))
    end
    if discovered !== nothing
        length(neural) == 1 || throw(ArgumentError(
            "discovered rates can replace the placeholder of exactly one unknown term; the model has $(length(neural))"))
        term = only(neural)
        regulators = [sts[r] for r in term.regulators]
        nn_map[term.nn_index] = _discovered_rate(discovered, regulators)
    end

    eqs = Equation[]
    for row in 1:n
        sym = state_map[row]
        prod = 0
        dest = 0
        for term in cm.production_terms
            term.target == row || continue
            prod = prod + _mtk_prod(term, state_map, pmap)
        end
        for term in cm.destruction_terms
            term.target == row || continue
            dest = dest + _mtk_dest(term, state_map, pmap, nn_map)
        end
        push!(eqs, D(sym) ~ prod - dest * sym)
    end
    constructor = isdefined(ModelingToolkit, :System) ? ModelingToolkit.System :
                  ModelingToolkit.ODESystem
    return constructor(eqs, t, sts, params; name = name)
end

function _discovered_rate(candidate::Union{ImplicitCandidate, ExplicitCandidate},
        regulators)
    return _rate_expression(candidate, regulators)
end

function _discovered_rate(result::DiscoveryResult, regulators)
    result.success || throw(ArgumentError(
        "the discovery did not succeed ($(result.retcode)); there is no rate to export"))
    return _rate_expression(first(result.candidates), regulators)
end

function _discovered_rate(result::UnknownTermResult, regulators)
    _discovered_rate(result.discovery,
        regulators)
end

function _rate_expression(candidate::ImplicitCandidate, variables)
    spec = candidate.specification
    numerator = _polynomial(candidate.numerator_coefficients, spec.numerator, variables)
    denominator = 1 + _polynomial(candidate.denominator_coefficients, spec.denominator,
        variables)
    return numerator / denominator
end

function _rate_expression(candidate::ExplicitCandidate, variables)
    return _polynomial(candidate.coefficients, candidate.specification.numerator, variables)
end

function _polynomial(coefficients, terms, variables)
    expression = Num(0)
    for (coefficient, term) in zip(coefficients, terms)
        iszero(coefficient) && continue
        monomial = Num(1)
        for (variable, power) in zip(term.variables, term.powers)
            variable <= length(variables) || throw(ArgumentError(
                "the candidate uses variable $(variable) but the unknown term has $(length(variables)) regulators"))
            monomial *= variables[variable]^power
        end
        expression += coefficient * monomial
    end
    return expression
end

function _mtk_prod(term::InputProductionTerm, smap, pmap)
    return term.scale * pmap[term.rate_param] * pmap[term.input_param]
end

function _mtk_prod(term::MassActionProductionTerm, smap, pmap)
    reg = smap[term.regulator]
    return term.scale * pmap[term.param] * reg^term.order
end

function _mtk_prod(term::SaturationProductionTerm, smap, pmap)
    reg = smap[term.regulator]
    return term.scale * pmap[term.vmax_param] * reg / (pmap[term.km_param] + reg)
end

function _mtk_prod(term, smap, pmap)
    return 0
end

function _mtk_dest(term::LinearDestructionTerm, smap, pmap, nn_map)
    return term.scale * pmap[term.param]
end

function _mtk_dest(term::HillDestructionTerm, smap, pmap, nn_map)
    reg = smap[term.regulator]
    kn = pmap[term.k_param]^term.hill_order
    regn = reg^term.hill_order
    return term.scale * pmap[term.vmax_param] * regn / (kn + regn)
end

function _mtk_dest(term::SaturationDestructionTerm, smap, pmap, nn_map)
    reg = smap[term.regulator]
    return term.scale * pmap[term.vmax_param] * reg / (pmap[term.km_param] + reg)
end

function _mtk_dest(term::CompetitiveDestructionTerm, smap, pmap, nn_map)
    sub, inh = smap[term.substrate], smap[term.inhibitor]
    km, ki, vmax = pmap[term.km_param], pmap[term.ki_param], pmap[term.vmax_param]
    return term.scale * vmax * sub / (km * (1 + inh / ki) + sub)
end

function _mtk_dest(term::NeuralDestructionTerm, smap, pmap, nn_map)
    reg = smap[term.regulator]
    return term.scale * nn_map[term.nn_index]
end

function _mtk_dest(term, smap, pmap, nn_map)
    return 0
end

end # module
