using Aqua
using BioDynaX
using JET
using Test

# SciMLBase is a transitive dep; Julia 1.12 requires importing it via BioDynaX.
using BioDynaX: SciMLBase

@testset "Aqua quality" begin
    Aqua.test_all(BioDynaX;
        ambiguities = false,
        unbound_args = false,
        piracies = (treat_as_own = [SciMLBase.ODEProblem, SciMLBase.solve],),
        persistent_tasks = false)
end

@testset "public API ambiguities" begin
    ambs = Test.detect_ambiguities(BioDynaX)
    @test isempty(ambs)
end

@testset "JET typos" begin
    JET.test_package(BioDynaX; target_modules = (BioDynaX,), mode = :typo)
end
