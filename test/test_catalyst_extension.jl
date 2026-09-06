# Catalyst input, symbolic output, and the ModelingToolkit round trip.
# The tutorial network written in Catalyst must convert to a BiologicalNetwork
# whose compiled model simulates and discovers exactly as the existing
# fixture does, and whose ModelingToolkit export agrees with Catalyst's own
# ODE system.

using Catalyst
using Latexify
using ModelingToolkit
using Symbolics

const _CAT_TRUTH = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
const _CAT_ICS = [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10]]
const _CAT_CONFIG = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0,
    log_every = 10^6)

const _CAT_TUTORIAL = @reaction_network tutorial begin
    k_prod * R, 0 --> S
    hill(R, vmax, K, 2), S --> 0, [description = "unknown"]
    k_rs * S, 0 --> R
    k_r, R --> 0
end

# ModelingToolkit 10 changed the problem constructor to a single merged map.
function _cat_problem(sys, u0, tspan, p)
    pkgversion(ModelingToolkit) >= v"10" ?
    ODEProblem(sys, merge(Dict(u0), Dict(p)), tspan) :
    ODEProblem(sys, u0, tspan, p)
end
function _cat_evaluate(expression, assignments)
    return Float64(Symbolics.value(Symbolics.substitute(expression, assignments)))
end

@testset "Catalyst input" begin
    @testset "conversion matches the tutorial fixture" begin
        by_index = network_from_reactionsystem(_CAT_TUTORIAL; unknown = 2)
        by_name = network_from_reactionsystem(_CAT_TUTORIAL; unknown = "unknown")
        fixture = BioDynaX.build_hill_recovery_network(; known = false, hill_order = 2)
        for net in (by_index, by_name)
            @test [node.name for node in net.nodes] == [:S, :R]
            @test length(net.reactions) == 4
            @test BioDynaX.count_unknown_destructions(net) == 1
            @test candidate_parents(net, 1) == [2]
            unknown = only(r for r in net.reactions if !r.known)
            @test unknown.regulators == [2]
            @test unknown.stoichiometry == Dict(1 => -1.0)
            model, _ = build_ude_model(MersenneTwister(1), net)
            reference, _ = build_ude_model(MersenneTwister(1), fixture)
            @test parameter_schema(model).phys_names ==
                  parameter_schema(reference).phys_names
        end
        @test_throws ArgumentError network_from_reactionsystem(_CAT_TUTORIAL; unknown = 5)
        @test_throws ArgumentError network_from_reactionsystem(_CAT_TUTORIAL;
            unknown = "no such reaction")
        @test_throws ArgumentError network_from_reactionsystem(_CAT_TUTORIAL;
            unknown = 1)
    end

    @testset "known kinetics simulate as the fixture and as Catalyst" begin
        truth = network_from_reactionsystem(_CAT_TUTORIAL; unknown = nothing)
        fixture = BioDynaX.build_hill_recovery_network(; known = true, hill_order = 2)
        @test BioDynaX.count_unknown_destructions(truth) == 0
        converted = BioDynaX.reference_protocol_experiment_set(
            MersenneTwister(103), truth; smoke = true, truth_params = _CAT_TRUTH,
            initial_conditions = _CAT_ICS)
        expected = BioDynaX.reference_protocol_experiment_set(
            MersenneTwister(103), fixture; smoke = true, truth_params = _CAT_TRUTH,
            initial_conditions = _CAT_ICS)
        for (a, b) in zip(converted.experiments, expected.experiments)
            @test a.observations == b.observations
            @test a.times == b.times
        end
        # Catalyst's own ODE system, same parameters and initial condition.
        u0 = [_CAT_TUTORIAL.S => _CAT_ICS[1][1], _CAT_TUTORIAL.R => _CAT_ICS[1][2]]
        p = [_CAT_TUTORIAL.k_prod => _CAT_TRUTH.k_prod,
            _CAT_TUTORIAL.vmax => _CAT_TRUTH.vmax,
            _CAT_TUTORIAL.K => _CAT_TRUTH.K, _CAT_TUTORIAL.k_rs => _CAT_TRUTH.k_rs,
            _CAT_TUTORIAL.k_r => _CAT_TRUTH.k_r]
        times = expected.experiments[1].times
        problem = _cat_problem(_CAT_TUTORIAL, u0, (first(times), last(times)), p)
        solution = solve(problem, Tsit5(); saveat = times, abstol = 1e-10, reltol = 1e-10)
        catalyst_states = Array(solution)
        model, p0 = build_ude_model(MersenneTwister(1), truth)
        params = pack_parameters(_CAT_TRUTH, p0.nn)
        ours = predict_ude(params, _CAT_ICS[1], (first(times), last(times)), times, model;
            solver_config = SolverConfig(abstol = 1e-10, reltol = 1e-10))
        @test maximum(abs, ours .- catalyst_states) < 1e-6
    end

    @testset "discovery output is identical to the fixture" begin
        converted = network_from_reactionsystem(_CAT_TUTORIAL; unknown = "unknown")
        fixture = BioDynaX.build_hill_recovery_network(; known = false, hill_order = 2)
        truth_net = BioDynaX.build_hill_recovery_network(; known = true, hill_order = 2)
        set = BioDynaX.reference_protocol_experiment_set(
            MersenneTwister(103), truth_net; smoke = true, truth_params = _CAT_TRUTH,
            initial_conditions = _CAT_ICS)
        a = discover_unknown_term(converted, set; training = _CAT_CONFIG, holdout = 0,
            rng = MersenneTwister(7), verbose = false)
        b = discover_unknown_term(fixture, set; training = _CAT_CONFIG, holdout = 0,
            rng = MersenneTwister(7), verbose = false)
        @test BioDynaX.nn_parameter_fingerprint(a.params.nn) ==
              BioDynaX.nn_parameter_fingerprint(b.params.nn)
        @test collect(a.params.phys) == collect(b.params.phys)
        @test a.samples.R == b.samples.R
        @test a.samples.D == b.samples.D
        @test a.discovery.success == b.discovery.success
        @test a.discovery.equations == b.discovery.equations
        if a.discovery.success
            @test a.discovery.candidates[1].numerator_coefficients ==
                  b.discovery.candidates[1].numerator_coefficients
            @test a.discovery.candidates[1].denominator_coefficients ==
                  b.discovery.candidates[1].denominator_coefficients
        end
    end

    @testset "unsupported rates are refused by name" begin
        bimolecular = @reaction_network bimolecular begin
            k, S + R --> 0
            k2, 0 --> S
        end
        err = try
            network_from_reactionsystem(bimolecular; unknown = 2)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("reaction 1", sprint(showerror, err))
        repressive = @reaction_network repressive begin
            hillr(R, v, K, 2), S --> 0
            k, 0 --> S
        end
        err = try
            network_from_reactionsystem(repressive; unknown = 2)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("hillr", sprint(showerror, err))
        full_rate = @reaction_network full_rate begin
            k * S, S => 0
            d, S --> 0
        end
        @test_throws ArgumentError network_from_reactionsystem(full_rate; unknown = 2)
    end

    @testset "ModelingToolkit round trip" begin
        truth = network_from_reactionsystem(_CAT_TUTORIAL; unknown = nothing)
        model, p0 = build_ude_model(MersenneTwister(1), truth)
        ours = BioDynaX.export_mtk_system(model)
        # Catalyst 16 replaced `convert(ODESystem, rs)` with `ode_model(rs)`.
        theirs = isdefined(Catalyst, :ode_model) ? Catalyst.ode_model(_CAT_TUTORIAL) :
                 convert(ODESystem, _CAT_TUTORIAL)
        our_states = ModelingToolkit.unknowns(ours)
        their_states = ModelingToolkit.unknowns(theirs)
        @test [ModelingToolkit.tosymbol(s; escape = false) for s in our_states] == [:S, :R]
        @test [ModelingToolkit.tosymbol(s; escape = false) for s in their_states] ==
              [:S, :R]
        @test Set(ModelingToolkit.tosymbol.(ModelingToolkit.parameters(ours))) ==
              Set(ModelingToolkit.tosymbol.(ModelingToolkit.parameters(theirs)))
        rhs_ours = [eq.rhs for eq in ModelingToolkit.equations(ours)]
        rhs_theirs = [eq.rhs for eq in ModelingToolkit.equations(theirs)]
        # Compare the right-hand sides numerically on random states and the truth parameters.
        rng = MersenneTwister(11)
        for _ in 1:20
            s, r = rand(rng, 2) .* 2
            ours_map = Dict(our_states[1] => s, our_states[2] => r,
                (pa => getproperty(_CAT_TRUTH, ModelingToolkit.tosymbol(pa))
                for pa in ModelingToolkit.parameters(ours))...)
            theirs_map = Dict(their_states[1] => s, their_states[2] => r,
                (pa => getproperty(_CAT_TRUTH, ModelingToolkit.tosymbol(pa))
                for pa in ModelingToolkit.parameters(theirs))...)
            for i in 1:2
                @test isapprox(_cat_evaluate(rhs_ours[i], ours_map),
                    _cat_evaluate(rhs_theirs[i], theirs_map); rtol = 1e-12, atol = 1e-12)
            end
        end
        # Catalyst -> BioDynaX -> ModelingToolkit -> ODEProblem solves like Catalyst.
        problem = _cat_problem(complete(ours),
            [our_states[1] => 0.25, our_states[2] => 0.20],
            (0.0, 4.0),
            [pa => getproperty(_CAT_TRUTH, ModelingToolkit.tosymbol(pa))
             for pa in ModelingToolkit.parameters(ours)])
        reference = _cat_problem(_CAT_TUTORIAL,
            [_CAT_TUTORIAL.S => 0.25, _CAT_TUTORIAL.R => 0.20], (0.0, 4.0),
            [_CAT_TUTORIAL.k_prod => 0.9,
                _CAT_TUTORIAL.vmax => 1.8, _CAT_TUTORIAL.K => 0.55,
                _CAT_TUTORIAL.k_rs => 1.0, _CAT_TUTORIAL.k_r => 0.6])
        saveat = 0.0:0.5:4.0
        a = Array(solve(problem, Tsit5(); saveat, abstol = 1e-10, reltol = 1e-10))
        b = Array(solve(reference, Tsit5(); saveat, abstol = 1e-10, reltol = 1e-10))
        @test maximum(abs, a .- b) < 1e-6
    end
end

@testset "symbolic output" begin
    fixture = BioDynaX.build_hill_recovery_network(; known = false, hill_order = 2)
    truth_net = BioDynaX.build_hill_recovery_network(; known = true, hill_order = 2)
    set = BioDynaX.reference_protocol_experiment_set(
        MersenneTwister(103), truth_net; smoke = true, truth_params = _CAT_TRUTH,
        initial_conditions = _CAT_ICS)
    result = discover_unknown_term(fixture, set; training = _CAT_CONFIG, holdout = 0,
        rng = MersenneTwister(7), verbose = false)
    if result.discovery.success
        candidate = result.discovery.candidates[1]
        expression = symbolic(result)
        @test expression isa Num
        by_names = symbolic(result.discovery, [:R])
        @test isequal(expression, by_names)
        variable = only(Symbolics.get_variables(expression))
        fn = equation_to_function(candidate)
        for r in vec(result.samples.R)
            @test isapprox(_cat_evaluate(expression, Dict(variable => r)), fn([r]);
                rtol = 1e-12, atol = 1e-12)
        end
        @test_throws ArgumentError symbolic(result.discovery, [:R]; index = 5)
        @test_throws ArgumentError symbolic(candidate, Symbol[])
        # Latexify agrees with the symbolic expression.
        @test latexify(result) == latexify(expression)
        @test latexify(result.discovery, [:R]) == latexify(expression)
        @test occursin("R", string(latexify(candidate, [:R])))
        # The completed ModelingToolkit system evaluates the discovered rate.
        completed = BioDynaX.export_mtk_system(result.model; discovered = result)
        states = ModelingToolkit.unknowns(completed)
        @test !any(occursin("nn_", string(s)) for s in states)
        rhs = [eq.rhs for eq in ModelingToolkit.equations(completed)]
        phys = BioDynaX.unpack_parameters(result.params).phys
        rng = MersenneTwister(5)
        for _ in 1:10
            s, r = rand(rng, 2) .* 1.5 .+ 0.05
            assignments = Dict(states[1] => s, states[2] => r,
                (pa => getproperty(phys, ModelingToolkit.tosymbol(pa))
                for pa in ModelingToolkit.parameters(completed))...)
            hybrid = compose_hybrid_rhs(result.model, result.params, result.term, fn)
            expected = hybrid([s, r], nothing, 0.0)
            @test isapprox(_cat_evaluate(rhs[1], assignments), expected[1];
                rtol = 1e-8, atol = 1e-8)
            @test isapprox(_cat_evaluate(rhs[2], assignments), expected[2];
                rtol = 1e-8, atol = 1e-8)
        end
    else
        @test_throws ArgumentError symbolic(result.discovery, [:R])
    end
    # With the extension loaded, an unsupported argument is a MethodError.
    @test_throws MethodError symbolic(1)
end
