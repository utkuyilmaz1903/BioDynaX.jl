# Optional extension entry points (implemented in weakdep extensions).
function export_mtk_system(model::UDEModel; kwargs...)
    throw(ErrorException(
        "export_mtk_system requires `using ModelingToolkit` " *
        "(BioDynaXModelingToolkitExt)"))
end

function import_sbml_network(path::AbstractString)
    throw(ErrorException(
        "import_sbml_network requires `using SBML` (BioDynaXSBMLExt). " *
        "The importer maps species and stoichiometry; unrecognized kinetic " *
        "laws become unknown reactions rather than guessed Michaelis forms."))
end

function import_sbmltoolkit_network(path::AbstractString)
    throw(ErrorException(
        "import_sbmltoolkit_network requires `using SBMLToolkit` " *
        "(BioDynaXSBMLToolkitExt)"))
end

"""Two-state network with two independent neural unknowns (multi-head test fixture)."""
function build_dual_unknown_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :A),
        NodeSpec(name = :B),
        NodeSpec(name = :C),
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
                     metadata = LinearDecayMetadata(rate_param = :k_c)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

"""Network exercising SATURATION and CUSTOM_KINETIC IR."""
function build_kinetic_generalization_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :E),
        NodeSpec(name = :S),
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
                     metadata = LinearDecayMetadata(rate_param = :k_s)),
    ]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end
