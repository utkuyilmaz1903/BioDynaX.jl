@testset "phase 2 saturation and custom kinetic IR" begin
    rng = MersenneTwister(42)
    network = build_kinetic_generalization_network()
    compiled = compile_mechanism(network)
    @test any(t -> t isa BioDynaX.SaturationProductionTerm, compiled.production_terms)
    @test any(t -> t isa BioDynaX.CustomDestructionTerm, compiled.destruction_terms)
    custom = only(filter(t -> t isa BioDynaX.CustomDestructionTerm,
        compiled.destruction_terms))
    @test custom.scale ≈ 2.0

    nn, nn_ps, st = build_ude_nn(rng)
    model = compile_network(network, nn, st)
    params = pack_parameters((vmax = 1.5, km = 0.4, k_custom = 0.8, k_s = 0.6), nn_ps)
    x = [0.3, 0.5]
    dx = ude_system(x, params, 0.0, model)
    @test all(isfinite, dx)

    s = x[2]
    expected_prod = 1.5 * s / (0.4 + s)
    reg = max(0.0, s)
    expected_dest = 2.0 * 0.8 * reg^2
    @test dx[1] ≈ expected_prod - expected_dest * x[1]
end

@testset "phase 2 multi-head neural unknowns" begin
    rng = MersenneTwister(7)
    network = build_dual_unknown_network()
    compiled = compile_mechanism(network)
    nn_terms = filter(t -> t isa BioDynaX.NeuralDestructionTerm,
        compiled.destruction_terms)
    @test length(nn_terms) == 2
    @test nn_terms[1].nn_index == 1
    @test nn_terms[2].nn_index == 2

    model, params = build_ude_model(rng, network)
    @test model.nn isa MultiHeadNetwork
    @test length(model.nn.heads) == 2
    schema = parameter_schema(model)
    @test schema.nn_heads == 2

    dx = ude_system([0.2, 0.3, 0.4], params, 0.0, model)
    @test all(isfinite, dx)
end

@testset "skipped duplicate unknown edge does not gap nn_index" begin
    # Same dual-declaration as DEFAULT_EXAMPLE_NETWORK (unknown reaction plus
    # matching UNKNOWN_NN edge) plus a second unknown edge compiled later.
    nodes = [NodeSpec(name = :A), NodeSpec(name = :B), NodeSpec(name = :C)]
    reactions = [
        ReactionSpec(name = :drive_a,
            stoichiometry = Dict(1 => 1.0), regulators = [3],
            metadata = MassActionMetadata(rate_param = :k_ca)),
        ReactionSpec(name = :unknown_ab,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = false, family = HILL, metadata = HillMetadata()),
        ReactionSpec(name = :b_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_b)),
        ReactionSpec(name = :c_decay,
            stoichiometry = Dict(3 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_c))
    ]
    edges = [
        EdgeSpec(source = 2, target = 1, kind = UNKNOWN_NN, known = false,
            family = HILL),
        EdgeSpec(source = 3, target = 1, kind = UNKNOWN_NN, known = false,
            family = HILL)
    ]
    network = BiologicalNetwork(nodes, edges; reactions = reactions)
    compiled = compile_mechanism(network)
    nn_terms = [t
                for t in compiled.destruction_terms
                if t isa BioDynaX.NeuralDestructionTerm]
    @test length(nn_terms) == 2
    @test sort(getfield.(nn_terms, :nn_index)) == [1, 2]

    rng = MersenneTwister(13)
    model, params = build_ude_model(rng, network)
    @test model.nn isa MultiHeadNetwork
    @test length(model.nn.heads) == 2
    x = [0.2, 0.3, 0.4]
    dx = ude_system(x, params, 0.0, model)
    @test all(isfinite, dx)
    cache = allocate_cache(model, Float64)
    ude_rhs!(cache.du, x, params, 0.0, model, cache)
    @test all(isfinite, cache.du)
    @test Vector(cache.du) ≈ dx
end

@testset "generate_data and default_parameters match compiled NN architecture" begin
    rng = MersenneTwister(17)
    dual = build_dual_unknown_network()
    dual_model, dual_p = build_ude_model(rng, dual)
    times, clean, _, packed_cv = generate_data(
        rng; network = dual, u0 = [0.2, 0.3, 0.4], tspan = (0.0, 1.0),
        n_points = 8, noise_σ = 0.0, truth_params = dual_p)
    @test length(times) == 8
    @test size(clean) == (3, 8)
    @test all(isfinite, clean)
    @test packed_cv.nn == dual_p.nn

    times2, clean2, _, packed_nt = generate_data(
        rng; network = dual, u0 = [0.2, 0.3, 0.4], tspan = (0.0, 1.0),
        n_points = 8, noise_σ = 0.0,
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
    @test size(clean2) == (3, 8)
    @test all(isfinite, clean2)
    @test hasproperty(packed_nt.nn, :head_1)
    @test hasproperty(packed_nt.nn, :head_2)

    set = generate_experiment_set(
        rng; network = dual, initial_conditions = [[0.2, 0.3, 0.4]],
        tspan = (0.0, 1.0), n_points = 8, noise_σ = 0.0,
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
    @test size(first(set.experiments).observations) == (3, 8)
    @test all(isfinite, first(set.experiments).observations)

    defaults = default_parameters(dual_model; rng = MersenneTwister(18))
    @test hasproperty(defaults.nn, :head_1)
    @test hasproperty(defaults.nn, :head_2)
    @test all(isfinite, ude_system([0.2, 0.3, 0.4], defaults, 0.0, dual_model))

    comp_net = build_competitive_test_network(; known = false)
    times3, clean3, _, packed_comp = generate_data(
        rng; network = comp_net, u0 = [0.25, 0.45, 0.2], tspan = (0.0, 1.0),
        n_points = 8, noise_σ = 0.0,
        truth_params = (k_in = 0.9, k_s = 0.8, k_i = 0.5))
    @test size(clean3) == (3, 8)
    @test all(isfinite, clean3)
    @test size(packed_comp.nn.layer_1.weight, 2) == 2

    comp_model, _ = build_ude_model(rng, comp_net)
    comp_defaults = default_parameters(comp_model; rng = MersenneTwister(19))
    @test size(comp_defaults.nn.layer_1.weight, 2) == 2
    @test all(isfinite, ude_system([0.25, 0.45, 0.2], comp_defaults, 0.0, comp_model))
end

@testset "phase 2 static specialization parity" begin
    rng = MersenneTwister(11)
    network = build_linear_test_network()
    model, params = build_ude_model(rng, network)
    x = [0.25, 0.15]
    dx_vec = ude_system(x, params, 0.0, model)
    dx_static = Vector(ude_system(
        StaticArrays.SVector{2}(x[1], x[2]), params, 0.0, model))
    @test dx_vec ≈ dx_static
    dx_explicit = Vector(BioDynaX._ude_system_static(
        StaticArrays.SVector{2}(x[1], x[2]), params, 0.0, model))
    @test dx_vec ≈ dx_explicit
end

@testset "phase 2 network validation" begin
    @test_throws ArgumentError BiologicalNetwork(
        [NodeSpec(name = :a), NodeSpec(name = :b)], EdgeSpec[];
        reactions = [ReactionSpec(name = :bad_sat,
            stoichiometry = Dict(1 => -1.0),
            regulators = Int[], known = true,
            family = SATURATION,
            metadata = EmptyMetadata())])

    @test_throws ArgumentError BiologicalNetwork(
        [NodeSpec(name = :a)], EdgeSpec[];
        reactions = [ReactionSpec(name = :bad_custom,
            stoichiometry = Dict(1 => -1.0),
            regulators = [1], known = true,
            family = CUSTOM_KINETIC,
            metadata = EmptyMetadata())])

    @test_throws ArgumentError BiologicalNetwork(
        [NodeSpec(name = :a)], EdgeSpec[];
        reactions = [ReactionSpec(name = :zero_stoich,
            stoichiometry = Dict(1 => 0.0),
            regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k))])
end

@testset "phase 2 optional MTK export" begin
    if !isdefined(Base, :get_extension) ||
       Base.get_extension(BioDynaX, :BioDynaXModelingToolkitExt) === nothing
        @test_throws ErrorException export_mtk_system(build_ude_model(MersenneTwister(0))[1])
    else
        using ModelingToolkit
        model, params = build_ude_model(MersenneTwister(0), build_linear_test_network())
        sys = export_mtk_system(model)
        @test sys isa ODESystem
        nstates = try
            length(unknowns(sys))
        catch
            length(states(sys))
        end
        @test nstates == 2
        param_text = join(string.(parameters(sys)), " ")
        @test occursin("k_ba", param_text)
        @test occursin("k_a", param_text)
        @test occursin("k_b", param_text)
        eq_text = join(string.(equations(sys)), " ")
        @test occursin("k_ba", eq_text)
        hill_model, _ = build_ude_model(
            MersenneTwister(1), build_hill_recovery_network(; known = true))
        hill_sys = export_mtk_system(hill_model)
        hill_eqs = join(string.(equations(hill_sys)), " ")
        @test occursin("vmax", hill_eqs)
        @test occursin("K", hill_eqs) || occursin("k_param", lowercase(hill_eqs))
        nn_model, _ = build_ude_model(
            MersenneTwister(2), build_hill_recovery_network(; known = false))
        nn_sys = export_mtk_system(nn_model)
        @test occursin("nn_", join(string.(equations(nn_sys)), " "))
    end
end

@testset "phase 2 optional SBML import" begin
    if !isdefined(Base, :get_extension) ||
       Base.get_extension(BioDynaX, :BioDynaXSBMLExt) === nothing
        @test_throws ErrorException import_sbml_network("missing.xml")
        @test_throws ErrorException import_sbmltoolkit_network("missing.xml")
    end
end
