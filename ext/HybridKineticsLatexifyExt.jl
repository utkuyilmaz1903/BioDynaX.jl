module HybridKineticsLatexifyExt

using HybridKinetics
using HybridKinetics: ImplicitCandidate, ExplicitCandidate, DiscoveryResult,
                      UnknownTermResult, UnknownTermsResult
using Latexify
using Symbolics

# `latexify` of a discovered rate goes through the symbolic expression of
# HybridKineticsSymbolicsExt, so `latexify(symbolic(result, names))` and these
# recipes agree.

Latexify.@latexrecipe function _(candidate::Union{ImplicitCandidate, ExplicitCandidate},
        names::AbstractVector{Symbol})
    env --> :equation
    return HybridKinetics.symbolic(candidate, names)
end

Latexify.@latexrecipe function _(result::DiscoveryResult, names::AbstractVector{Symbol})
    env --> :equation
    return HybridKinetics.symbolic(result, names)
end

Latexify.@latexrecipe function _(result::UnknownTermResult)
    env --> :equation
    return HybridKinetics.symbolic(result)
end

# One unknown term renders as before; with several, latexify a per-term
# result (`latexify(result[:S])`) so it is clear which rate is shown.
Latexify.@latexrecipe function _(result::UnknownTermsResult)
    env --> :equation
    return HybridKinetics.symbolic(result)
end

end # module
