# BioDynaX.jl

BioDynaX is a one-hole **research preview, not v1.0**: known graph, compiled
known kinetics, exactly one unknown destruction `D(z)`. The current
unique-claim hold is the Q3 practical scale warning (`unidentifiable_edge`;
coefficients are not biological constants) together with a gated Q1 hybrid
residual versus observed data and Q5 true-monomial recall. Q3 is a local
practical warning, not the whole product. Trajectory residual is not
mechanistic recovery. Canonical Hill from a trained NN is closed. This is
not a general CRN or SINDy replacement. There is no licensed experimental
series on this protocol.

The compiled dynamics are
\(\dot u_i = P_i - D_i u_i\) (Q1–Q7 kept conceptually separate; Q4
is a practical functional-identifiability diagnostic, not a gate and
not a formal identifiability certificate; Q7 is reported held-out
generalization evidence, not an additional success gate). Nine ICs
are generated once; ICs 1–7 are used for training and ICs 8–9 are
held out. M4 (robustness / trajectory-context validation) remains
pending future work.

```@contents
Pages = ["tutorial.md", "howto.md", "unique-claim.md",
         "compiled-path.md",
         "sciml.md", "metadata.md", "architecture.md", "benchmarks.md",
         "api.md", "experimental.md"]
Depth = 2
```

Start with the [unknown-inhibition tutorial](tutorial.md). The
[unique-claim](unique-claim.md) page is the product block
(IDENTIFIABILITY → FIT → DISCOVERY → REPRODUCTION). The
[compiled experiment path](compiled-path.md) is multi-IC generate-once,
SciML `ODEProblem(model, ...)`, and the suite hole-policy matrix.
Recipes are in [How-to](howto.md). SciML `ODEProblem` construction is in
[SciML Integration](sciml.md). Recovery gates are in
[Recovery benchmarks](benchmarks.md).
