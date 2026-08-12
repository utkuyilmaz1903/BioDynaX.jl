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
    "SciMLBase ODEProblem contract",
    "SciML forward and adjoint contracts",
    "build_ude_function and SciML solve",
    "typed kinetic metadata",
    "typed metadata compiles p53 network",
    "dict metadata backward compatibility",
    "explicit STLSQ discovery backend",
    "Fisher identifiability report",
    "training retcode and gradient diagnostics",
    "discovery uncertainty reports",
    "multi-trajectory discovery",
    "scientific benchmark suite",
    "ground truth generator separation",
)

@testset "baseline gate" begin
    @test !isempty(BASELINE_TESTSETS)
    @test all(!isempty, BASELINE_TESTSETS)
end
