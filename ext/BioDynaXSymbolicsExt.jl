module BioDynaXSymbolicsExt

using BioDynaX
using BioDynaX: ImplicitCandidate, ExplicitCandidate, MonomialTerm, DiscoveryResult,
                UnknownTermResult
using Symbolics

"""
    symbolic(candidate, names) -> Num

The candidate's rate as a `Symbolics.Num`: for an `ImplicitCandidate` the
rational function numerator / (1 + denominator), for an `ExplicitCandidate`
the polynomial, in symbolic variables named by `names` (one per library
variable index). Called through `BioDynaX.symbolic`.
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
    regulators = result.term.regulators
    return symbolic(result.discovery, names[regulators]; index = index)
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
