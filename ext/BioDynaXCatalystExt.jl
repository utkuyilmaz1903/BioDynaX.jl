module BioDynaXCatalystExt

using BioDynaX
using Catalyst
using ModelingToolkit
using Symbolics

const _MTK = ModelingToolkit

"""
    network_from_reactionsystem(rs::ReactionSystem; unknown) -> BiologicalNetwork

Convert a Catalyst `ReactionSystem` into a `BiologicalNetwork` with the same
species (in Catalyst's order), the known kinetics compiled from the rate laws
Catalyst exposes, and exactly one reaction, `unknown`, marked as the unknown
destruction term. Called through `BioDynaX.network_from_reactionsystem`; see
its docstring.
"""
function network_from_reactionsystem(rs::ReactionSystem; unknown)
    species = Catalyst.species(rs)
    isempty(species) && throw(ArgumentError("the reaction system has no species"))
    names = [_symbol(sp) for sp in species]
    allunique(names) || throw(ArgumentError("species names must be unique: $(names)"))
    rxs = collect(Catalyst.reactions(rs))
    isempty(rxs) && throw(ArgumentError("the reaction system has no reactions"))
    unknown_index = _unknown_reaction_index(rxs, unknown)
    specs = ReactionSpec[]
    for (k, rx) in enumerate(rxs)
        label = _reaction_label(rx, k)
        rx.only_use_rate && throw(ArgumentError(
            "reaction $(label): rate laws written as a full rate (`=>`, only_use_rate = true) are not supported; write the rate and let Catalyst add the mass-action factor"))
        substrates = [(_species_index(species, sp, label), Int(c))
                      for (sp, c) in zip(rx.substrates, rx.substoich)]
        products = [(_species_index(species, sp, label), Int(c))
                    for (sp, c) in zip(rx.products, rx.prodstoich)]
        rate = _classify_rate(rx.rate, species, label)
        if k == unknown_index
            append!(specs, _unknown_specs(rate, substrates, products, label, k))
        else
            append!(specs, _known_specs(rate, substrates, products, label, k))
        end
    end
    nodes = [NodeSpec(name = name) for name in names]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = specs)
end

_symbol(x) = _MTK.tosymbol(x; escape = false)

function _species_index(species, sp, label)
    index = findfirst(s -> isequal(s, sp), species)
    index === nothing && throw(ArgumentError(
        "reaction $(label): $(sp) is not a species of the reaction system"))
    return index
end

function _reaction_label(rx, k)
    return Catalyst.hasdescription(rx) ? string(k, " (", Catalyst.getdescription(rx), ")") :
           string(k)
end

"""
The unknown reaction: an integer index into `reactions(rs)`, a string matched
against the reactions' `description` metadata (exactly one match), or
`nothing` for a fully known network (every reaction compiled as known
kinetics).
"""
function _unknown_reaction_index(rxs, unknown)
    unknown === nothing && return 0
    if unknown isa Integer
        1 <= unknown <= length(rxs) || throw(ArgumentError(
            "unknown = $(unknown) is not a reaction index (the system has $(length(rxs)) reactions)"))
        return Int(unknown)
    elseif unknown isa AbstractString
        matches = [k
                   for (k, rx) in enumerate(rxs)
                   if Catalyst.hasdescription(rx) && Catalyst.getdescription(rx) == unknown]
        length(matches) == 1 || throw(ArgumentError(
            "unknown = \"$(unknown)\" must match the description metadata of exactly one reaction; matched $(length(matches))"))
        return only(matches)
    end
    throw(ArgumentError(
        "unknown must be a reaction index or the description string of one reaction; got $(unknown)"))
end

# -- Rate classification ------------------------------------------------------

"""
Classify a Catalyst rate expression (the factor Catalyst multiplies by the
mass-action term) into the forms the BioDynaX compiler has a term for:

- `(:constant, p)`: a single parameter `p`;
- `(:parameter_times_species, p, X)`: a parameter times one species;
- `(:hill, X, v, K, n)`: `hill(X, v, K, n)` with `v` and `K` parameters and an
  integer `n`;
- `(:mm, X, v, K)`: `mm(X, v, K)` with `v` and `K` parameters.

Anything else raises an `ArgumentError` that names the reaction and the rate.
"""
function _classify_rate(rate, species, label)
    expr = Symbolics.unwrap(rate)
    if _is_parameter(expr, species)
        return (; kind = :constant, param = _symbol(expr), regulator = nothing, order = 1)
    end
    if Symbolics.iscall(expr)
        op = Symbolics.operation(expr)
        args = Symbolics.arguments(expr)
        if op === Catalyst.hill && length(args) == 4
            X = _species_or_error(args[1], species, label, rate)
            v, K = _param_or_error(args[2], species, label, rate),
            _param_or_error(args[3], species, label, rate)
            n = _integer_or_error(args[4], label, rate)
            return (; kind = :hill, regulator = X, vmax = v, k = K, order = n)
        elseif op === Catalyst.mm && length(args) == 3
            X = _species_or_error(args[1], species, label, rate)
            v, K = _param_or_error(args[2], species, label, rate),
            _param_or_error(args[3], species, label, rate)
            return (; kind = :mm, regulator = X, vmax = v, k = K)
        elseif op === (*) && length(args) == 2
            a, b = args
            if _is_parameter(a, species) &&
               _species_index_or_nothing(b, species) !== nothing
                return (; kind = :parameter_times_species, param = _symbol(a),
                    regulator = _species_index_or_nothing(b, species), order = 1)
            elseif _is_parameter(b, species) &&
                   _species_index_or_nothing(a, species) !== nothing
                return (; kind = :parameter_times_species, param = _symbol(b),
                    regulator = _species_index_or_nothing(a, species), order = 1)
            end
        end
    end
    throw(ArgumentError(string("reaction ", label, ": the rate ", rate,
        " is not a form BioDynaX can compile; supported rates are a parameter k, ",
        "k * X for a species X that is not a substrate, hill(X, v, K, n) with ",
        "parameters v and K and an integer n, and mm(X, v, K)")))
end

function _species_index_or_nothing(x, species)
    findfirst(s -> isequal(s, Symbolics.wrap(x)), species)
end

function _is_parameter(x, species)
    w = Symbolics.wrap(x)
    Symbolics.issym(Symbolics.unwrap(x)) || return false
    return _species_index_or_nothing(x, species) === nothing && _MTK.isparameter(w)
end

function _species_or_error(x, species, label, rate)
    index = _species_index_or_nothing(x, species)
    index === nothing && throw(ArgumentError(
        "reaction $(label): the first argument of the rate $(rate) must be a species"))
    return index
end

function _param_or_error(x, species, label, rate)
    _is_parameter(x, species) || throw(ArgumentError(
        "reaction $(label): $(x) in the rate $(rate) must be a parameter"))
    return _symbol(x)
end

function _integer_or_error(x, label, rate)
    v = Symbolics.unwrap(x)
    if !(v isa Real)
        # Catalyst stores literal exponents as symbolic constants; `value` unwraps them.
        try
            v = Symbolics.value(x)
        catch
            v = nothing
        end
    end
    v isa Integer && return Int(v)
    v isa Real && isinteger(v) && return Int(v)
    throw(ArgumentError(
        "reaction $(label): the Hill exponent in the rate $(rate) must be a literal integer"))
end

# -- Term construction --------------------------------------------------------

function _known_specs(rate, substrates, products, label, k)
    specs = ReactionSpec[]
    for (index, stoich) in substrates
        name = Symbol("r", k, "_destroys_", index)
        stoichiometry = Dict(index => -Float64(stoich))
        if rate.kind === :constant && length(substrates) == 1 && stoich == 1
            push!(specs,
                ReactionSpec(name = name, stoichiometry = stoichiometry,
                    regulators = Int[], metadata = LinearDecayMetadata(rate_param = rate.param)))
        elseif rate.kind === :hill && length(substrates) == 1 && stoich == 1
            push!(specs,
                ReactionSpec(name = name, stoichiometry = stoichiometry,
                    regulators = [rate.regulator], family = HILL,
                    metadata = HillMetadata(vmax_param = rate.vmax, k_param = rate.k,
                        hill_order = rate.order)))
        elseif rate.kind === :mm && length(substrates) == 1 && stoich == 1
            push!(specs,
                ReactionSpec(name = name, stoichiometry = stoichiometry,
                    regulators = [rate.regulator], family = SATURATION,
                    metadata = SaturationMetadata(
                        vmax_param = rate.vmax, km_param = rate.k)))
        else
            throw(ArgumentError(string("reaction ", label,
                ": BioDynaX compiles a substrate's loss only as first-order decay (k, X --> ...), ",
                "Hill (hill(Y, v, K, n), X --> ...), or Michaelis-Menten (mm(Y, v, K), X --> ...) ",
                "with one substrate of stoichiometry 1; this reaction has ",
                length(substrates), " substrate(s) with stoichiometry ", stoich)))
        end
    end
    for (index, stoich) in products
        name = Symbol("r", k, "_makes_", index)
        stoichiometry = Dict(index => Float64(stoich))
        if rate.kind === :constant && isempty(substrates)
            push!(specs,
                ReactionSpec(name = name, stoichiometry = stoichiometry,
                    regulators = Int[],
                    metadata = InputDriveMetadata(
                        rate_param = rate.param, input_param = :input)))
        elseif rate.kind === :constant && length(substrates) == 1
            source, order = only(substrates)
            push!(specs,
                ReactionSpec(name = name, stoichiometry = stoichiometry,
                    regulators = [source],
                    metadata = MassActionMetadata(rate_param = rate.param, order = order)))
        elseif rate.kind === :parameter_times_species && isempty(substrates)
            push!(specs,
                ReactionSpec(name = name, stoichiometry = stoichiometry,
                    regulators = [rate.regulator],
                    metadata = MassActionMetadata(rate_param = rate.param, order = 1)))
        elseif rate.kind === :mm && isempty(substrates)
            push!(specs,
                ReactionSpec(name = name, stoichiometry = stoichiometry,
                    regulators = [rate.regulator], family = SATURATION,
                    metadata = SaturationMetadata(
                        vmax_param = rate.vmax, km_param = rate.k)))
        else
            throw(ArgumentError(string("reaction ", label,
                ": BioDynaX compiles a product's formation only as constant production ",
                "(k, 0 --> X), mass action from one substrate (k, Y --> X), mass action ",
                "regulated by one species (k * Y, 0 --> X), or Michaelis-Menten ",
                "(mm(Y, v, K), 0 --> X); this reaction's rate is ", rate.kind, " with ",
                length(substrates), " substrate(s)")))
        end
    end
    return specs
end

function _unknown_specs(rate, substrates, products, label, k)
    length(substrates) == 1 && only(substrates)[2] == 1 || throw(ArgumentError(
        "reaction $(label): the unknown reaction must consume exactly one species with stoichiometry 1 (it becomes the unknown destruction term of that species)"))
    isempty(products) || throw(ArgumentError(
        "reaction $(label): the unknown reaction must not produce anything; declare the products' formation as separate known reactions"))
    target = only(substrates)[1]
    regulators = if rate.kind in (:hill, :mm, :parameter_times_species)
        [rate.regulator]
    else
        [target]
    end
    metadata = rate.kind === :hill ?
               HillMetadata(
        vmax_param = rate.vmax, k_param = rate.k, hill_order = rate.order) :
               HillMetadata()
    return [ReactionSpec(name = Symbol("r", k, "_unknown_destroys_", target),
        stoichiometry = Dict(target => -1.0), regulators = regulators,
        known = false, family = HILL, metadata = metadata)]
end

end # module
