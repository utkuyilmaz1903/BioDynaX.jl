# BioDynaX.jl

BioDynaX compiles a biological interaction graph into a positivity-preserving
UDE and recovers graph-local rational kinetics on unknown edges. This is a
**research preview**: the unique path is gated unknown-edge `D(z)` recovery
(true-monomial recall + hybrid residual versus data on Hill-class edges), not
canonical Hill from every trained NN, and not a general CRN or SINDy
replacement. Practical `k_prod`↔`D` collinearity is reported, not solved.

```@contents
Pages = ["tutorial.md", "howto.md", "sciml.md", "metadata.md",
         "architecture.md", "benchmarks.md", "api.md",
         "experimental.md", "stability.md"]
Depth = 2
```

Start with the [unknown-inhibition tutorial](tutorial.md). Recipes are in
[How-to](howto.md). SciML `ODEProblem` construction is in
[SciML Integration](sciml.md). Recovery gates are in
[Recovery benchmarks](benchmarks.md).
