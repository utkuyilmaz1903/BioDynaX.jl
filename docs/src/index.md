# BioDynaX.jl

BioDynaX is a one-hole **research preview, not v1.0**: known graph, compiled
known kinetics, exactly one unknown destruction `D(z)`. The product is
practical identifiability (`unidentifiable_edge`; coefficients are not
biological constants) together with a gated hybrid residual versus data and
true-monomial recall. Canonical Hill from a trained NN is closed. This is not
a general CRN or SINDy replacement. There is no licensed experimental series
on this protocol.

```@contents
Pages = ["tutorial.md", "howto.md", "unique-claim.md", "sciml.md",
         "metadata.md", "architecture.md", "benchmarks.md", "api.md",
         "experimental.md", "stability.md"]
Depth = 2
```

Start with the [unknown-inhibition tutorial](tutorial.md). The
[unique-claim](unique-claim.md) page is the product block
(IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION). Recipes are in
[How-to](howto.md). SciML `ODEProblem` construction is in
[SciML Integration](sciml.md). Recovery gates are in
[Recovery benchmarks](benchmarks.md).
