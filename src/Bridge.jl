# Optional extension entry points. The methods live in the weak-dependency
# extensions; these wrappers look the extension up at call time so that the
# extension modules never redefine a method of this module.
function _optional_extension(name::Symbol, hint::AbstractString)
    extension = Base.get_extension(@__MODULE__, name)
    extension === nothing && throw(ErrorException(hint))
    return extension
end

"""
    export_mtk_system(model::UDEModel; name = :BioDynaXNetwork)

Convert the compiled known terms of `model` to a ModelingToolkit `ODESystem`.
Neural terms appear as placeholder variables `nn_i(t)`. Requires
`using ModelingToolkit` (extension `BioDynaXModelingToolkitExt`). Not exported.
"""
function export_mtk_system(model::UDEModel; kwargs...)
    extension = _optional_extension(:BioDynaXModelingToolkitExt,
        "export_mtk_system requires `using ModelingToolkit` (BioDynaXModelingToolkitExt)")
    return extension.export_mtk_system(model; kwargs...)
end

"""
    import_sbml_network(path::AbstractString)

Build a `BiologicalNetwork` from the species, reactions, and stoichiometry of
an SBML file. Kinetic laws are not parsed; reactions with an explicit kinetic
law compile as unknown neural terms. Requires `using SBML` (extension
`BioDynaXSBMLExt`). Not exported.
"""
function import_sbml_network(path::AbstractString)
    extension = _optional_extension(:BioDynaXSBMLExt,
        "import_sbml_network requires `using SBML` (BioDynaXSBMLExt). " *
        "The importer maps species and stoichiometry; unrecognized kinetic " *
        "laws become unknown reactions rather than guessed Michaelis forms.")
    return extension.import_sbml_network(path)
end

"""
    import_sbmltoolkit_network(path::AbstractString)

Build a `BiologicalNetwork` from an SBML file through SBMLToolkit and Catalyst.
Mass-action reactions are recognized; other rate laws become unknown neural
terms. Requires `using SBMLToolkit` and `using Catalyst` (extension
`BioDynaXSBMLToolkitExt`). Not exported.
"""
function import_sbmltoolkit_network(path::AbstractString)
    extension = _optional_extension(:BioDynaXSBMLToolkitExt,
        "import_sbmltoolkit_network requires `using SBMLToolkit` and `using Catalyst` " *
        "(BioDynaXSBMLToolkitExt)")
    return extension.import_sbmltoolkit_network(path)
end

"""Two-state network with two independent neural unknowns (multi-head test fixture)."""
function build_dual_unknown_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :A),
        NodeSpec(name = :B),
        NodeSpec(name = :C)
    ]
    reactions = [
        ReactionSpec(name = :drive_a,
            stoichiometry = Dict(1 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_ca)),
        ReactionSpec(name = :drive_b,
            stoichiometry = Dict(2 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_cb)),
        ReactionSpec(name = :unknown_a_decay,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = false, family = HILL,
            metadata = HillMetadata()),
        ReactionSpec(name = :unknown_b_decay,
            stoichiometry = Dict(2 => -1.0), regulators = [1],
            known = false, family = HILL,
            metadata = HillMetadata()),
        ReactionSpec(name = :c_decay,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_c))
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""Network exercising SATURATION and CUSTOM_KINETIC IR."""
function build_kinetic_generalization_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :E),
        NodeSpec(name = :S)
    ]
    custom_eval = (x, p, regulators) -> begin
        reg = max(zero(eltype(x)), x[only(regulators)])
        return positive_parameter(p.phys.k_custom) * reg^2
    end
    reactions = [
        ReactionSpec(name = :sat_prod,
            stoichiometry = Dict(1 => 1.0), regulators = [2],
            known = true, family = SATURATION,
            metadata = SaturationMetadata(
                vmax_param = :vmax, km_param = :km)),
        ReactionSpec(name = :custom_decay,
            stoichiometry = Dict(1 => -2.0), regulators = [2],
            known = true, family = CUSTOM_KINETIC,
            metadata = CustomKineticMetadata(
                rate_param = :k_custom, evaluator = custom_eval)),
        ReactionSpec(name = :s_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_s))
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end
