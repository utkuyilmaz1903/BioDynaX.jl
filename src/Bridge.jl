# Optional extension entry points. The methods live in the weak-dependency
# extensions; these wrappers look the extension up at call time so that the
# extension modules never redefine a method of this module.
function _optional_extension(name::Symbol, hint::AbstractString)
    extension = Base.get_extension(@__MODULE__, name)
    extension === nothing && throw(ErrorException(hint))
    return extension
end

"""
    export_mtk_system(model::UDEModel; name = :BioDynaXNetwork, discovered = nothing)

Convert the compiled known terms of `model` to a ModelingToolkit `ODESystem`
whose states carry the network's node names. Neural terms appear as
placeholder variables `nn_i(t)`; `discovered` (a `DiscoveryResult`, a
candidate, or an `UnknownTermResult`) replaces the placeholder of the single
unknown term with the discovered rational rate, so the completed model can
be handed to ModelingToolkit and OrdinaryDiffEq. Requires
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

"""
    network_from_reactionsystem(rs; unknown) -> BiologicalNetwork

Build a `BiologicalNetwork` from a Catalyst `ReactionSystem`: the same species
in Catalyst's order, the known kinetics compiled from the rate laws Catalyst
exposes, the interaction graph derived from the reactions, and exactly one
reaction, `unknown`, marked as the unknown destruction term. `unknown` is the
reaction's index in `Catalyst.reactions(rs)` or the string of its
`description` metadata (`[description = "..."]` in the DSL); `unknown =
nothing` compiles every reaction as known kinetics (a ground-truth model).

Each Catalyst reaction is split into one BioDynaX term per species it
changes. Supported rates (the factor Catalyst multiplies by the mass-action
term): a parameter `k` (first-order loss of a single substrate; constant
production `k, 0 --> X`, which adds the parameter `input` that multiplies it
and is meant to be fixed at 1; mass action from one substrate `k, Y --> X`);
`k * Y` for a species `Y` that is not a substrate (`k * Y, 0 --> X`);
`hill(Y, v, K, n)` with parameters `v`, `K` and a literal integer `n` on a
single substrate of stoichiometry 1 (Hill destruction); `mm(Y, v, K)` on a
single substrate (Michaelis-Menten destruction) or with no substrate
(Michaelis-Menten production). Any other rate, a full rate law written with
`=>`, or a shape the compiler has no term for raises an `ArgumentError`
naming the reaction and the rate. The unknown reaction must consume exactly
one species with stoichiometry 1 and produce nothing; its regulators are the
species of its rate (the consumed species itself when the rate has none).
Requires `using Catalyst` (extension `BioDynaXCatalystExt`).
"""
function network_from_reactionsystem(rs; unknown)
    extension = _optional_extension(:BioDynaXCatalystExt,
        "network_from_reactionsystem requires `using Catalyst` (BioDynaXCatalystExt)")
    return extension.network_from_reactionsystem(rs; unknown = unknown)
end

"""
    symbolic(candidate, names) -> Num
    symbolic(result::DiscoveryResult, names; index = 1) -> Num
    symbolic(result::UnknownTermResult; index = 1) -> Num

The discovered rational rate as a `Symbolics.Num` in the named variables:
`names` gives one symbol per variable of the candidate's library (for
`discover_unknown_rate`, the regulators in order; for `discover_equations`,
the network's states); an `UnknownTermResult` uses its network's state names.
Requires `using Symbolics` (extension `BioDynaXSymbolicsExt`).
"""
function symbolic(args...; kwargs...)
    extension = _optional_extension(:BioDynaXSymbolicsExt,
        "symbolic requires `using Symbolics` (BioDynaXSymbolicsExt)")
    return extension.symbolic(args...; kwargs...)
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
