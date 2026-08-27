using Aqua
using BioDynaX
using JET
using Test

# SciMLBase is a transitive dep; Julia 1.12 requires importing it via BioDynaX.
using BioDynaX: SciMLBase

# Previous (lenient) Aqua.test_all kwargs on 0.9.2:
#   ambiguities = false
#   unbound_args = false
#   persistent_tasks = false
#   piracies = (treat_as_own = [SciMLBase.ODEProblem, SciMLBase.solve],)
# Newly strict relative to 0.9.2. Measured on this tree: all of these
# currently pass (including piracies without treat_as_own, because
# ODEProblem/solve methods take UDEModel). Keep them on so a regression
# fails the quality job. The failing industry bar lives in
# test/run_standards.jl, not here.
#   ambiguities          (was skipped)
#   unbound_args         (was skipped)
#   persistent_tasks     (was skipped)
#   piracies             (waiver for ODEProblem/solve removed)
#   undocumented_names   (Aqua default is false; now on)
# Already-on and unchanged:
#   undefined_exports, stale_deps, deps_compat, project_extras
@testset "Aqua quality" begin
    Aqua.test_all(BioDynaX;
        ambiguities = true,
        unbound_args = true,
        undefined_exports = true,
        piracies = true,
        stale_deps = true,
        deps_compat = true,
        persistent_tasks = true,
        project_extras = true,
        undocumented_names = true)
end

@testset "public API ambiguities" begin
    ambs = Test.detect_ambiguities(BioDynaX)
    @test isempty(ambs)
end

@testset "JET typos" begin
    JET.test_package(BioDynaX; target_modules = (BioDynaX,), mode = :typo)
end
