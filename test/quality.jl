using Aqua
using BioDynaX
using JET
using SciMLBase
using Test

@testset "Aqua quality" begin
    Aqua.test_all(BioDynaX;
        ambiguities = false,
        unbound_args = false,
        piracies = (treat_as_own = [SciMLBase.ODEProblem, SciMLBase.solve],),
        persistent_tasks = false)
end

@testset "JET typos" begin
    JET.test_package(BioDynaX; target_modules = (BioDynaX,), mode = :typo)
end
