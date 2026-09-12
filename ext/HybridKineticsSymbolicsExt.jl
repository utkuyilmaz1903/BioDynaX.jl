module HybridKineticsSymbolicsExt

using HybridKinetics
using HybridKinetics: ImplicitCandidate, ExplicitCandidate, MonomialTerm, DiscoveryResult,
                      UnknownTermResult, DiscoveryRun
using Symbolics

"""
    symbolic(candidate, names) -> Num

The candidate's rate as a `Symbolics.Num`: for an `ImplicitCandidate` the
rational function numerator / (1 + denominator), for an `ExplicitCandidate`
the polynomial, in symbolic variables named by `names` (one per library
variable index). Called through `HybridKinetics.symbolic`.
"""
function symbolic(candidate::ImplicitCandidate, names::AbstractVector{Symbol})
    variables = _variables(names)
    spec = candidate.specification
    numerator = _polynomial(candidate.numerator_coefficients, spec.numerator, variables)
    denominator = 1 + _polynomial(candidate.denominator_coefficients, spec.denominator,
        variables)
    return numerator / denominator
end

function symbolic(candidate::ExplicitCandidate, names::AbstractVector{Symbol})
    variables = _variables(names)
    return _polynomial(candidate.coefficients, candidate.specification.numerator,
        variables)
end

function symbolic(
        result::DiscoveryResult, names::AbstractVector{Symbol}; index::Integer = 1)
    result.success || throw(ArgumentError(
        "the discovery did not succeed ($(result.retcode)); there is no candidate to convert"))
    1 <= index <= length(result.candidates) || throw(ArgumentError(
        "index must be between 1 and $(length(result.candidates))"))
    return symbolic(result.candidates[index], names)
end

function symbolic(result::UnknownTermResult; index::Integer = 1)
    names = [node.name for node in result.network.nodes]
    regulators = HybridKinetics.state_nodes(result.network)[result.term.regulators]
    return symbolic(result.discovery, names[regulators]; index = index)
end

"""
    symbolic(result::DiscoveryRun; node = nothing, index = 1) -> Num

The discovered rate of one unknown term as a `Symbolics.Num` in the names
of its regulators. With one unknown term `node` may be omitted; with
several it names the term (`result[:S]` works too).
"""
function symbolic(result::DiscoveryRun; node = nothing, index::Integer = 1)
    if node === nothing
        length(result.terms) == 1 || throw(ArgumentError(string(
            "the result has $(length(result.terms)) unknown terms (",
            join(string.(keys(result)), ", "), "); pass node = :name or use symbolic(result[:name])")))
        return symbolic(only(result.terms); index = index)
    end
    return symbolic(result[node]; index = index)
end

function _variables(names::AbstractVector{Symbol})
    allunique(names) || throw(ArgumentError("variable names must be unique: $(names)"))
    return [Symbolics.variable(name) for name in names]
end

function _polynomial(coefficients, terms::Vector{MonomialTerm}, variables)
    expression = Num(0)
    for (coefficient, term) in zip(coefficients, terms)
        iszero(coefficient) && continue
        monomial = Num(1)
        for (variable, power) in zip(term.variables, term.powers)
            variable <= length(variables) || throw(ArgumentError(
                "the candidate uses variable $(variable) but only $(length(variables)) names were given"))
            monomial *= variables[variable]^power
        end
        expression += coefficient * monomial
    end
    return expression
end

end # module
