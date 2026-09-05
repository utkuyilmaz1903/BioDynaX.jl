module BioDynaXModelingToolkitExt

using BioDynaX
using ModelingToolkit
using Symbolics

"""
    export_mtk_system(model::UDEModel; name=:BioDynaXNetwork)

Build a symbolic `ModelingToolkit.ODESystem` preview from a compiled UDE.
"""
function BioDynaX.export_mtk_system(model::UDEModel; name::Symbol = :BioDynaXNetwork)
    cm = model.compiled
    n = cm.nstates
    @parameters t
    state_syms = [Symbol("x$i") for i in 1:n]
    D = Differential(t)
    @variables (state_syms...)(t)
    state_map = Dict(i => state_syms[i] for i in 1:n)

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
    @parameters (param_syms...)
    pmap = Dict{Symbol, Num}(s => s for s in param_syms)

    nn_map = Dict{Int, Num}()
    for term in cm.destruction_terms
        term isa NeuralDestructionTerm || continue
        sym = Symbol("nn_", term.nn_index)
        haskey(nn_map, term.nn_index) && continue
        @variables $sym(t)
        nn_map[term.nn_index] = sym
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
    sts = [state_map[i] for i in 1:n]
    return ODESystem(eqs, t, sts, param_syms; name = name)
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
