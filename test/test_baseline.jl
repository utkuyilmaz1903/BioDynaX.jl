# Frozen regression baseline: named testsets that must remain green across remediation.
const BASELINE_TESTSETS = (
    "typed biological network",
    "positive production-destruction UDE",
    "augmented lagrangian mechanics",
    "augmented lagrangian end-to-end training",
    "deterministic resume equivalence",
    "augmented lagrangian resume equivalence",
    "implicit rational recovery",
    "Hill and competitive inhibition recovery",
    "graph-local basis scales with indegree",
    "experiment contracts",
    "execution backends and generated replicates",
    "versioned checkpoint",
    "compiler kinetic families",
    "general pipeline second topology",
    "type stability and allocation gates",
    "scientific release qualification",
    "multi-topology E2E",
)

@testset "baseline gate" begin
    @test !isempty(BASELINE_TESTSETS)
    @test all(!isempty, BASELINE_TESTSETS)
end
