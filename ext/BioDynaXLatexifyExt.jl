module BioDynaXLatexifyExt

using BioDynaX
using BioDynaX: ImplicitCandidate, ExplicitCandidate, DiscoveryResult, UnknownTermResult
using Latexify
using Symbolics

# `latexify` of a discovered rate goes through the symbolic expression of
# BioDynaXSymbolicsExt, so `latexify(symbolic(result, names))` and these
# recipes agree.

Latexify.@latexrecipe function _(candidate::Union{ImplicitCandidate, ExplicitCandidate},
        names::AbstractVector{Symbol})
    env --> :equation
    return BioDynaX.symbolic(candidate, names)
end

Latexify.@latexrecipe function _(result::DiscoveryResult, names::AbstractVector{Symbol})
    env --> :equation
    return BioDynaX.symbolic(result, names)
end

Latexify.@latexrecipe function _(result::UnknownTermResult)
    env --> :equation
    return BioDynaX.symbolic(result)
end

end # module
