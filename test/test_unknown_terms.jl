# Several unknown destruction terms (0.16): the UnknownTerm spec, the network
# guard rails, the plural entry point, its result, report and cross-term
# diagnostic, and the loud errors of the removed 0.15 names.

const _UT_CONFIG = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0,
    log_every = 10^6)

function _ut_two_term_truth()
    nodes = [NodeSpec(name = :A), NodeSpec(name = :B), NodeSpec(name = :C)]
    function hill(name, target, reg, vmax, K)
        ReactionSpec(name = name,
            stoichiometry = Dict(target => -1.0), regulators = [reg], known = true,
            family = HILL,
            metadata = HillMetadata(vmax_param = vmax, k_param = K, hill_order = 2))
    end
    reactions = [
        ReactionSpec(name = :drive_a, stoichiometry = Dict(1 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_ca)),
        ReactionSpec(name = :drive_b, stoichiometry = Dict(2 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_cb)),
        hill(:a_decay, 1, 2, :vmax_a, :K_a),
        hill(:b_decay, 2, 1, :vmax_b, :K_b),
        ReactionSpec(name = :c_decay, stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_c))]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

function _ut_two_term_set()
    return generate_experiment_set(MersenneTwister(103); network = _ut_two_term_truth(),
        initial_conditions = [[0.3, 0.2, 0.5], [0.6, 0.4, 0.8], [0.2, 0.7, 0.3]],
        tspan = (0.0, 6.0), n_points = 12, noise_σ = 0.0,
        truth_params = (k_ca = 0.8, k_cb = 0.9, vmax_a = 1.5, K_a = 0.5,
            vmax_b = 1.2, K_b = 0.6, k_c = 0.5))
end

@testset "UnknownTerm specs and the network" begin
    spec = UnknownTerm(:S)
    @test spec.node == :S && spec.regulators === nothing && spec.library === nothing
    @test UnknownTerm(:S; regulators = [:R]).regulators == [:R]
    @test UnknownTerm(:S; regulators = (:R, :S)).regulators == [:R, :S]
    @test_throws ArgumentError UnknownTerm(:S; regulators = Symbol[])
    @test_throws ArgumentError UnknownTerm(:S; regulators = [:A, :B, :C])
    @test sprint(show, UnknownTerm(:S; regulators = [:R])) ==
          "UnknownTerm(:S; regulators = [:R])"

    # unknown_terms reads the marks of existing fixtures
    tutorial = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    @test unknown_terms(tutorial) == [UnknownTerm(:S; regulators = [:R])]
    @test isempty(unknown_terms(HybridKinetics.build_hill_recovery_network(; known = true)))
    @test unknown_terms(HybridKinetics.build_dual_unknown_network()) ==
          [UnknownTerm(:A; regulators = [:B]), UnknownTerm(:B; regulators = [:A])]

    # marking through the constructor gives the same network as ReactionSpec(known = false)
    truth = _ut_two_term_truth()
    marked = BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = truth.reactions,
        unknown = [UnknownTerm(:A), UnknownTerm(:B)])
    @test unknown_terms(marked) == [UnknownTerm(:A; regulators = [:B]),
        UnknownTerm(:B; regulators = [:A])]
    @test HybridKinetics.neural_head_count(marked) == 2
    @test [r.known for r in marked.reactions] == [true, true, false, false, true]
    # regulators override
    override = BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = truth.reactions,
        unknown = [UnknownTerm(:A; regulators = [:C])])
    @test unknown_terms(override) == [UnknownTerm(:A; regulators = [:C])]
    @test HybridKinetics.candidate_parents(override, 1) == [3]
    # the one-term tutorial network marked through the constructor compiles identically
    known_tutorial = HybridKinetics.build_hill_recovery_network(;
        known = true, hill_order = 2)
    via_spec = BiologicalNetwork(known_tutorial.nodes, EdgeSpec[];
        reactions = known_tutorial.reactions, unknown = [UnknownTerm(:S)])
    @test parameter_schema(build_ude_model(MersenneTwister(1), via_spec)[1]).phys_names ==
          parameter_schema(build_ude_model(MersenneTwister(1), tutorial)[1]).phys_names

    # guard rails, each naming what is wrong
    @test_throws ArgumentError BiologicalNetwork(truth.nodes, EdgeSpec[];
        reactions = truth.reactions, unknown = [UnknownTerm(:A), UnknownTerm(:A)])
    err = try
        BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = truth.reactions,
            unknown = [UnknownTerm(:A), UnknownTerm(:A)])
    catch e
        e
    end
    @test occursin("same node A", sprint(showerror, err))
    @test_throws ArgumentError BiologicalNetwork(truth.nodes, EdgeSpec[];
        reactions = truth.reactions, unknown = [UnknownTerm(:Z)])
    @test_throws ArgumentError BiologicalNetwork(truth.nodes, EdgeSpec[];
        reactions = truth.reactions, unknown = [UnknownTerm(:A; regulators = [:Z])])
    no_decay = BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = truth.reactions[1:2])
    @test_throws ArgumentError BiologicalNetwork(truth.nodes, EdgeSpec[];
        reactions = no_decay.reactions, unknown = [UnknownTerm(:A)])
    # two unknown destruction reactions on one node, marked by hand
    twice = vcat(truth.reactions,
        [ReactionSpec(name = :a_decay_2,
            stoichiometry = Dict(1 => -1.0), regulators = [3], known = false,
            family = HILL, metadata = HillMetadata())])
    twice[3] = ReactionSpec(name = :a_decay, stoichiometry = Dict(1 => -1.0),
        regulators = [2], known = false, family = HILL, metadata = HillMetadata())
    err = try
        BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = twice)
    catch e
        e
    end
    @test err isa ArgumentError && occursin("same node A", sprint(showerror, err))
    # an unknown production term is refused, naming the scope
    production = copy(truth.reactions)
    production[1] = ReactionSpec(name = :drive_a, stoichiometry = Dict(1 => 1.0),
        regulators = [3], known = false, metadata = MassActionMetadata(rate_param = :k_ca))
    err = try
        BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = production)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("unknown production term of A", sprint(showerror, err))
    @test occursin("out of scope", sprint(showerror, err))
    # and the compiler refuses it too, should a network reach it another way
    @test_throws ArgumentError HybridKinetics._reaction_production_term(
        production[1], 1, Dict(1 => 1, 2 => 2, 3 => 3), 1.0)
end

@testset "the removed 0.15 names fail loudly" begin
    tutorial = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    err = try
        discover_unknown_term(tutorial, nothing)
    catch e
        e
    end
    @test err isa ErrorException
    msg = sprint(showerror, err)
    @test occursin("discover_unknown_terms", msg)
    @test occursin(HybridKinetics.MIGRATION_SECTION, msg)
    err = try
        report_unknown_term(nothing)
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("report_unknown_terms", sprint(showerror, err))
    @test occursin(HybridKinetics.MIGRATION_SECTION, sprint(showerror, err))
    @test_throws ErrorException discover_unknown_term()
    @test_throws ErrorException report_unknown_term()
end

@testset "two unknown terms" begin
    net = HybridKinetics.build_dual_unknown_network()
    set = _ut_two_term_set()
    @test_throws ArgumentError discover_unknown_terms(net, set; training = _UT_CONFIG,
        holdout = 1, verbose = false, known_support = HybridKinetics.hill_rate_support(2))
    @test_throws ArgumentError discover_unknown_terms(net, set; training = _UT_CONFIG,
        holdout = 1, verbose = false, regulator_grid = range(0.1, 1.0; length = 8))
    @test_throws ArgumentError discover_unknown_terms(net, set; training = _UT_CONFIG,
        holdout = 1, verbose = false, terms = [UnknownTerm(:Z)])
    result = discover_unknown_terms(net, set; training = _UT_CONFIG, holdout = 1,
        rng = MersenneTwister(7), verbose = false,
        known_support = Dict(:A => HybridKinetics.hill_rate_support(2),
            :B => HybridKinetics.hill_rate_support(2)),
        production_param = Dict(:A => :k_ca, :B => :k_cb))
    @test result isa DiscoveryRun
    @test length(result) == 2
    @test keys(result) == [:A, :B]
    @test result[:A] === result[1] && result[:B] === result[2]
    @test haskey(result, :A) && !haskey(result, :C)
    @test_throws KeyError result[:C]
    @test [t.node for t in result] == [:A, :B]
    @test unknown_terms(result) === result.terms
    @test result[:A].term.target == 1 && result[:B].term.target == 2
    @test result[:A].term.regulators == [2] && result[:B].term.regulators == [1]
    @test result[:A].identifiability.production_param == :k_ca
    @test result[:B].identifiability.production_param == :k_cb
    @test result[:A].params === result.params
    @test result[:A].training === result.training
    @test HybridKinetics.neural_head_count(result.model) == 2
    @test haskey(result.params.nn, :head_1) && haskey(result.params.nn, :head_2)
    # the first term's initial weights are the single-term draw from the same seed
    _, p_two = build_ude_model(MersenneTwister(7), net)
    _, p_one = build_ude_model(MersenneTwister(7),
        HybridKinetics.build_two_term_coupled_network(; unknown = (:A,)))
    @test collect(p_two.nn.head_1) == collect(p_one.nn)
    @test keys(p_one.nn) == (:layer_1, :layer_2, :layer_3)
    @test result.settings.unknown_holes == 2
    @test result.training_indices == 1:2 && result.holdout_indices == [3]
    # each term has its own samples and its own discovery
    @test size(result[:A].samples.R, 1) == 1 && size(result[:B].samples.R, 1) == 1
    @test result[:A].samples.D != result[:B].samples.D
    for term in result
        @test term.discovery isa DiscoveryResult
        term.discovery.success && @test term.extras isa Vector{String}
        term.discovery.success && @test term.selection_frequency isa Vector{Float64}
    end
    # aggregate residuals at the top level
    @test all(isfinite,
        (result.residuals.data_residual, result.residuals.data_residual_train,
            result.residuals.data_residual_holdout)) ||
          !all(t -> t.discovery.success, result.terms)
    # the cross-term diagnostic: one pair, a cosine in [0, 1]
    @test length(result.cross_term) == 1
    pair = only(result.cross_term)
    @test pair.nodes == (:A, :B) && pair.terms == (1, 2)
    @test 0.0 <= pair.collinearity <= 1.0
    direct = cross_term_collinearity(result.model, result.params,
        set.experiments[1].u0, (0.0, 6.0), set.experiments[1].times;
        mask = set.experiments[1].mask)
    @test only(direct).collinearity == pair.collinearity
    # report: one fit section, one block per term, a cross-term section
    text = report_unknown_terms(result)
    @test count("\nFIT\n", "\n" * text) == 1
    @test occursin("unknown_terms: A, B", text)
    @test occursin("TERM A (unknown destruction of A, regulators B)", text)
    @test occursin("TERM B (unknown destruction of B, regulators A)", text)
    @test count("IDENTIFIABILITY", text) == 2 && count("DISCOVERY", text) == 2
    @test occursin("production_param: k_ca", text) &&
          occursin("production_param: k_cb", text)
    @test occursin("\nCROSS-TERM\n", text) && occursin("A, B: collinearity", text)
    @test occursin("\nREPRODUCTION\n", text) && occursin("unknown_terms: 2", text)
    @test sprint(show, MIME("text/plain"), result) == text
    @test startswith(sprint(show, result), "DiscoveryRun(unknown terms = [:A, :B]")
    @test startswith(sprint(show, result[:A]), "UnknownTermResult(:A")
    @test startswith(sprint(show, MIME("text/plain"), result[:B]), "TERM B")
    # the joint fit is the same whichever way the network was marked unknown
    truth = _ut_two_term_truth()
    marked = BiologicalNetwork(truth.nodes, EdgeSpec[]; reactions = truth.reactions,
        unknown = [UnknownTerm(:A), UnknownTerm(:B)])
    again = discover_unknown_terms(marked, set; training = _UT_CONFIG, holdout = 1,
        rng = MersenneTwister(7), verbose = false,
        production_param = Dict(:A => :k_ca, :B => :k_cb))
    @test collect(again.params) == collect(result.params)
    @test again[:A].samples.D == result[:A].samples.D
    # the multi-term hybrid residual with one pair is the single-term function
    if result[:A].discovery.success
        fn = equation_to_function(result[:A].discovery.candidates[1])
        e = set.experiments[1]
        @test hybrid_data_residual(result.model, result.params, [result[:A].term => fn],
            e.u0, (0.0, 6.0), e.times, e.observations; mask = e.mask) ==
              hybrid_data_residual(result.model, result.params, result[:A].term, fn,
            e.u0, (0.0, 6.0), e.times, e.observations; mask = e.mask)
    end
    # production_param as one symbol applies to every term; :auto finds each node's own
    auto = discover_unknown_terms(net, set; training = _UT_CONFIG, holdout = 1,
        rng = MersenneTwister(7), verbose = false)
    @test auto[:A].identifiability.production_param == :k_ca
    @test auto[:B].identifiability.production_param == :k_cb
    one_symbol = discover_unknown_terms(net, set; training = _UT_CONFIG, holdout = 1,
        rng = MersenneTwister(7), verbose = false, production_param = :k_c)
    @test one_symbol[:A].identifiability.production_param == :k_c
end

@testset "one unknown term keeps the 0.15 surface" begin
    tutorial = HybridKinetics.build_hill_recovery_network(; known = false, hill_order = 2)
    truth = HybridKinetics.build_hill_recovery_network(; known = true, hill_order = 2)
    set = HybridKinetics.reference_protocol_experiment_set(MersenneTwister(103), truth;
        smoke = true, truth_params = (
            k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6),
        initial_conditions = [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10]])
    result = discover_unknown_terms(tutorial, set; training = _UT_CONFIG, holdout = 1,
        rng = MersenneTwister(7), verbose = false, seed = 103)
    @test isempty(result.cross_term)
    @test isempty(cross_term_collinearity(result.model, result.params,
        set.experiments[1].u0, (0.0, 1.0), set.experiments[1].times))
    text = report_unknown_terms(result)
    @test !occursin("CROSS-TERM", text) && !occursin("TERM S", text)
    @test text == HybridKinetics.format_protocol_result(result[1].identifiability;
        residual = result.residuals.data_residual,
        residual_train = result.residuals.data_residual_train,
        residual_holdout = result.residuals.data_residual_holdout,
        equations = result[1].discovery.equations, extras = result[1].extras,
        unknown_holes = 1, seed = 103, n_ics = 3, n_points = result.settings.n_points,
        adam_iters = 2, bfgs_iters = 0, bootstrap = result.settings.bootstrap,
        discovery_seed = result.settings.discovery_seed)
    @test result[1].identifiability.production_param == :k_prod
    @test haskey(result.params.nn, :layer_1)
    # a bare support and a bare grid are accepted with one term
    given = discover_unknown_terms(tutorial, set; training = _UT_CONFIG, holdout = 1,
        rng = MersenneTwister(7), verbose = false,
        known_support = HybridKinetics.hill_rate_support(2),
        regulator_grid = range(0.2, 1.4; length = 24))
    @test given[1].extras isa Vector{String} || !given[1].discovery.success
    @test length(vec(given[1].samples.R)) == 24
end
