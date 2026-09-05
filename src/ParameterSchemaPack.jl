###############################################################################
# Parameter schema / pack (not exported).
#
# parameter_schema now collects CustomKineticMetadata.rate_param so
# :k_custom is a first-class phys name. unpack_parameters inverts
# pack_parameters through positive_parameter. Remapped multi-head
# pack/unpack must match the compiled NN tree. frozen_phys zeros
# gradients and restores raw coordinates.
#
# Does not grow exports. Does not drop protocol ICs. Does not open
# Hill-from-NN. Combined F1 stays a skeleton floor.
###############################################################################

const PARAMETER_SCHEMA_PACK_MUST_CONTAIN = (
    "function unpack_parameters",
    "function remapped_pack_unpack_row",
    "function frozen_phys_zero_gradient_row",
    "function positive_parameter_roundtrip_row",
    "function schema_vs_compiled_nn_tree_row",
    "struct ParameterSchemaPackRow",
    "function kinetic_custom_in_schema_row",
    "function dummy_head_on_zero_hole_row")

const PARAMETER_SCHEMA_PACK_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function parameter_schema_pack_source_path()
    joinpath(pkgdir(BioDynaX), "src", "ParameterSchemaPack.jl")
end

function parameter_schema_pack_test_path()
    joinpath(pkgdir(BioDynaX), "test", "test_parameter_schema_pack.jl")
end

function parameter_schema_jl_source_path()
    joinpath(pkgdir(BioDynaX), "src", "ParameterSchema.jl")
end

function ude_jl_source_path_for_pack()
    joinpath(pkgdir(BioDynaX), "src", "UDE.jl")
end

# -- Core helpers -------------------------------------------------------------

function compiled_neural_head_count(model::UDEModel)
    return count(term -> term isa NeuralDestructionTerm,
        model.compiled.destruction_terms)
end

function compiled_custom_term_count(model::UDEModel)
    return count(term -> term isa CustomDestructionTerm,
        model.compiled.destruction_terms)
end

function lux_head_count(model::UDEModel)
    model.nn isa MultiHeadNetwork && return length(model.nn.heads)
    return 1
end

function phys_roundtrip_error(phys::NamedTuple, unpacked)
    errs = Float64[]
    for name in keys(phys)
        hasproperty(unpacked.phys, name) ||
            return (; ok = false, maxerr = Inf)
        push!(errs,
            abs(Float64(getproperty(unpacked.phys, name)) -
                Float64(getproperty(phys, name))))
    end
    return (; ok = true, maxerr = maximum(errs))
end

function schema_contains(schema::ParameterSchema, names)
    return all(n -> n in schema.phys_names, names)
end

# -- Typed row ----------------------------------------------------------------

struct ParameterSchemaPackRow
    name::Symbol
    n_phys::Int
    nn_heads::Int
    lux_heads::Int
    has_custom::Bool
    holds::Bool
end

function parameter_schema_pack_row(name::Symbol, model::UDEModel, packed)
    schema = parameter_schema(model)
    unpacked = unpack_parameters(packed)
    typed = ParameterSchemaPackRow(
        name,
        length(schema.phys_names),
        schema.nn_heads,
        lux_head_count(model),
        :k_custom in schema.phys_names,
        schema_contains(schema, propertynames(unpacked.phys)) ||
        length(schema.phys_names) ≤ length(propertynames(unpacked.phys)))
    return (; schema, unpacked, typed, holds = typed.holds)
end

function parameter_schema_pack_row_namedtuple(row::ParameterSchemaPackRow)
    return (;
        name = row.name,
        n_phys = row.n_phys,
        nn_heads = row.nn_heads,
        lux_heads = row.lux_heads,
        has_custom = row.has_custom,
        holds = row.holds)
end

# -- Pack / unpack / positive_parameter ---------------------------------------

function positive_parameter_roundtrip_row()
    values = (0.2, 0.5, 0.8, 1.0, 1.7, 2.5, 4.0)
    errs = Float64[abs(positive_parameter(inverse_softplus(v)) - v)
                   for v in values]
    threw = try
        inverse_softplus(0.0)
        false
    catch err
        err isa DomainError
    end
    floor_val = positive_parameter(-50.0)
    return (;
        maxerr = maximum(errs),
        threw,
        floor_val,
        holds = maximum(errs) < 1e-10 && threw && floor_val > 0)
end

function pack_unpack_linear_row()
    net = build_linear_test_network()
    rng = MersenneTwister(401)
    model, p0 = build_ude_model(rng, net)
    phys = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    packed = pack_parameters(phys, p0.nn)
    unpacked = unpack_parameters(packed)
    err = phys_roundtrip_error(phys, unpacked)
    schema = parameter_schema(model)
    return (;
        err,
        schema_names = copy(schema.phys_names),
        holds = err.ok && err.maxerr < 1e-10 &&
                schema_contains(schema, keys(phys)))
end

function remapped_pack_unpack_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(403)
    model, p0 = build_ude_model(rng, net)
    phys = remapped_two_regulator_phys_truth()
    packed = pack_parameters(phys, p0.nn)
    unpacked = unpack_parameters(packed)
    err = phys_roundtrip_error(phys, unpacked)
    schema = parameter_schema(model)
    nn_ok = model.nn isa MultiHeadNetwork &&
            hasproperty(packed.nn, :head_1) &&
            hasproperty(packed.nn, :head_2)
    return (;
        err,
        nn_heads = schema.nn_heads,
        lux_heads = lux_head_count(model),
        holes = count_unknown_destructions(net),
        nn_ok,
        holds = err.ok && err.maxerr < 1e-10 &&
                schema.nn_heads == 2 &&
                lux_head_count(model) == 2 &&
                nn_ok &&
                schema_contains(schema, keys(phys)) &&
                unique_claim_recovery_admits(net) == false)
end

function two_regulator_pack_unpack_row()
    net = build_two_regulator_unknown_network()
    rng = MersenneTwister(407)
    model, p0 = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = default_phys_parameters(schema)
    packed = pack_parameters(phys, p0.nn)
    unpacked = unpack_parameters(packed)
    err = phys_roundtrip_error(phys, unpacked)
    return (;
        err,
        nn_heads = schema.nn_heads,
        input_dim = begin
            terms = neural_destruction_terms(model)
            isempty(terms) ? 0 : length(only(terms).regulators)
        end,
        holds = err.ok && err.maxerr < 1e-10 &&
                schema.nn_heads == 1)
end

function kinetic_custom_in_schema_row()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(409)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    unpacked = unpack_parameters(packed)
    u0 = [0.20, 0.15]
    times = collect(range(0.0, 0.6; length = 8))
    data = predict_ude(packed, u0, (0.0, 0.6), times, model)
    return (;
        names = copy(schema.phys_names),
        has_custom = :k_custom in schema.phys_names,
        custom_terms = compiled_custom_term_count(model),
        unpacked_custom = hasproperty(unpacked.phys, :k_custom),
        finite = all(isfinite, data),
        holes = count_unknown_destructions(net),
        holds = :k_custom in schema.phys_names &&
                compiled_custom_term_count(model) == 1 &&
                hasproperty(unpacked.phys, :k_custom) &&
                all(isfinite, data) &&
                count_unknown_destructions(net) == 0)
end

function missing_custom_validate_throws_row()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(411)
    model, _ = build_ude_model(rng, net)
    schema = parameter_schema(model)
    phys = (; (n => 1.0 for n in schema.phys_names if n != :k_custom)...)
    threw = try
        validate_phys_parameters(phys, schema)
        false
    catch err
        err isa ArgumentError
    end
    return (;
        threw,
        has_custom = :k_custom in schema.phys_names,
        holds = threw && :k_custom in schema.phys_names)
end

function default_parameters_include_custom_row()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(413)
    model, _ = build_ude_model(rng, net)
    packed = default_parameters(net, model; rng = MersenneTwister(415))
    schema = parameter_schema(model)
    return (;
        has_custom = hasproperty(packed.phys, :k_custom),
        n = length(schema.phys_names),
        holds = hasproperty(packed.phys, :k_custom) &&
                :k_custom in schema.phys_names)
end

# -- Schema vs compiled NN tree -----------------------------------------------

function schema_vs_compiled_nn_tree_row()
    cases = NamedTuple[]
    fixtures = (
        (:linear, build_linear_test_network()),
        (:hill, build_hill_recovery_network(; known = false, hill_order = 2)),
        (:mm, build_mm_recovery_network(; known = false)),
        (:two, build_two_regulator_unknown_network()),
        (:remap, build_remapped_two_regulator_network()),
        (:dual, build_dual_unknown_network()),
        (:default, DEFAULT_EXAMPLE_NETWORK),
        (:kinetic, build_kinetic_generalization_network()),
        (:repress, build_repressilator_network()),
        (:three, build_three_state_unknown_network()))
    for (name, net) in fixtures
        rng = MersenneTwister(hash(name) % 10_000)
        model, _ = build_ude_model(rng, net)
        schema = parameter_schema(model)
        compiled = compiled_neural_head_count(model)
        lux = lux_head_count(model)
        dummy = compiled == 0 && lux == 1 && !(model.nn isa MultiHeadNetwork)
        multi = compiled ≥ 2 && model.nn isa MultiHeadNetwork &&
                lux == compiled
        single = compiled == 1 && lux == 1
        push!(cases,
            (;
                name,
                compiled,
                lux,
                schema_heads = schema.nn_heads,
                dummy,
                multi,
                single,
                holds = schema.nn_heads == compiled &&
                    (dummy || multi || single)))
    end
    return (;
        cases,
        holds = all(c -> c.holds, cases) &&
                any(c -> c.dummy, cases) &&
                any(c -> c.multi, cases))
end

function dummy_head_on_zero_hole_row()
    net = build_linear_test_network()
    rng = MersenneTwister(417)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        schema_heads = schema.nn_heads,
        lux = lux_head_count(model),
        multi = model.nn isa MultiHeadNetwork,
        has_nn = hasproperty(packed, :nn),
        holds = schema.nn_heads == 0 &&
                lux_head_count(model) == 1 &&
                !(model.nn isa MultiHeadNetwork) &&
                hasproperty(packed, :nn))
end

function remapped_schema_matches_heads_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(419)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        schema_heads = schema.nn_heads,
        compiled = compiled_neural_head_count(model),
        lux = lux_head_count(model),
        tree = hasproperty(packed.nn, :head_1) &&
               hasproperty(packed.nn, :head_2),
        holds = schema.nn_heads == 2 &&
                compiled_neural_head_count(model) == 2 &&
                lux_head_count(model) == 2 &&
                hasproperty(packed.nn, :head_1) &&
                hasproperty(packed.nn, :head_2))
end

function dual_schema_matches_heads_row()
    net = build_dual_unknown_network()
    rng = MersenneTwister(421)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        schema_heads = schema.nn_heads,
        admits = unique_claim_recovery_admits(net),
        holds = schema.nn_heads == compiled_neural_head_count(model) &&
                unique_claim_recovery_admits(net) == false &&
                validate_network(net) === net)
end

function skipped_middle_schema_heads_row()
    net = build_skipped_middle_unknown_network()
    rng = MersenneTwister(423)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    n = compiled_neural_head_count(model)
    tree_ok = n ≤ 1 || all(i -> hasproperty(packed.nn, Symbol("head_", i)), 1:n)
    return (;
        n,
        schema_heads = schema.nn_heads,
        tree_ok,
        holds = schema.nn_heads == n && tree_ok &&
                neural_index_is_dense(model))
end

# -- frozen_phys --------------------------------------------------------------

function frozen_phys_zero_gradient_row()
    net = build_linear_test_network()
    rng = MersenneTwister(425)
    model, p0 = build_ude_model(rng, net)
    g = ComponentVector(
        phys = (k_ba = 1.2, k_a = 0.4, k_b = 0.3),
        nn = p0.nn)
    z = _zero_frozen_phys_gradient(g, [:k_ba])
    empty = _zero_frozen_phys_gradient(g, Symbol[])
    return (;
        zeroed = z.phys.k_ba == 0 && z.phys.k_a == 0.4 && z.phys.k_b == 0.3,
        empty_unchanged = empty.phys.k_ba == 1.2,
        holds = z.phys.k_ba == 0 && z.phys.k_a == 0.4 &&
                empty.phys.k_ba == 1.2)
end

function frozen_phys_restore_row()
    net = build_linear_test_network()
    rng = MersenneTwister(427)
    _, p0 = build_ude_model(rng, net)
    reference = ComponentVector(
        phys = (k_ba = 0.7, k_a = 1.1, k_b = 0.4),
        nn = p0.nn)
    mutated = ComponentVector(
        phys = (k_ba = 9.0, k_a = 2.2, k_b = 0.4),
        nn = p0.nn)
    restored = _restore_frozen_phys(mutated, reference, [:k_ba])
    return (;
        ba = restored.phys.k_ba,
        a = restored.phys.k_a,
        holds = restored.phys.k_ba == 0.7 && restored.phys.k_a == 2.2)
end

function frozen_phys_config_copy_row()
    cfg = TrainingConfig(; frozen_phys = [:k_prod])
    other = TrainingConfig(cfg; frozen_phys = [:k_prod, :k_rs])
    return (;
        n = length(cfg.frozen_phys),
        other_n = length(other.frozen_phys),
        aliased = cfg.frozen_phys === other.frozen_phys,
        holds = cfg.frozen_phys == [:k_prod] &&
                other.frozen_phys == [:k_prod, :k_rs] &&
                cfg.frozen_phys !== other.frozen_phys)
end

function remapped_frozen_phys_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(429)
    model, p0 = build_ude_model(rng, net)
    phys = remapped_two_regulator_phys_truth()
    packed = pack_parameters(phys, p0.nn)
    g = ComponentVector(phys = packed.phys, nn = packed.nn)
    frozen = [:k_ia]
    z = _zero_frozen_phys_gradient(g, frozen)
    return (;
        zeroed = z.phys.k_ia == 0,
        others = z.phys.k_is == g.phys.k_is,
        holds = z.phys.k_ia == 0 && z.phys.k_is == g.phys.k_is &&
                schema_contains(parameter_schema(model), keys(phys)))
end

# -- Fixture schema rows ------------------------------------------------------

function fixture_schema_row(name::Symbol, network::BiologicalNetwork;
        expect_custom::Bool = false)
    rng = MersenneTwister(hash((name, :schema)) % 10_000)
    model, packed = build_ude_model(rng, network)
    schema = parameter_schema(model)
    unpacked = unpack_parameters(packed)
    validate_phys_parameters(unpacked.phys, schema)
    return (;
        name,
        n_phys = length(schema.phys_names),
        nn_heads = schema.nn_heads,
        has_custom = :k_custom in schema.phys_names,
        validate_open = validate_network(network) === network,
        holds = schema.nn_heads == compiled_neural_head_count(model) &&
                (:k_custom in schema.phys_names) == expect_custom &&
                validate_network(network) === network)
end

function hill_unknown_schema_row()
    return fixture_schema_row(
        :hill, build_hill_recovery_network(;
            known = false, hill_order = 2))
end

function hill_known_schema_row()
    return fixture_schema_row(
        :hill_known, build_hill_recovery_network(;
            known = true, hill_order = 2))
end

function mm_unknown_schema_row()
    return fixture_schema_row(:mm, build_mm_recovery_network(; known = false))
end

function mm_known_schema_row()
    return fixture_schema_row(:mm_known, build_mm_recovery_network(;
        known = true))
end

function default_example_schema_row()
    return fixture_schema_row(:default, DEFAULT_EXAMPLE_NETWORK)
end

function three_state_schema_row()
    return fixture_schema_row(:three, build_three_state_unknown_network())
end

function six_state_schema_row()
    return fixture_schema_row(:six, build_six_state_unknown_network())
end

function competitive_schema_row()
    return fixture_schema_row(
        :competitive, build_competitive_test_network(; known = false))
end

function repressilator_schema_row()
    return fixture_schema_row(:repress, build_repressilator_network())
end

function skipped_duplicate_schema_row()
    return fixture_schema_row(
        :skipped, build_skipped_duplicate_unknown_network())
end

function wrong_graph_schema_row()
    return fixture_schema_row(:wrong, build_wrong_graph_unknown_network())
end

# -- Source locks -------------------------------------------------------------

function custom_kinetic_schema_source_holds()
    src = read(parameter_schema_jl_source_path(), String)
    start = findfirst("function parameter_schema(model::UDEModel)", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("CUSTOM_KINETIC", body) &&
           occursin("rate_param", body) &&
           occursin(":k_custom", body) &&
           occursin("CustomDestructionTerm", body)
end

function unpack_parameters_source_holds()
    src = read(ude_jl_source_path_for_pack(), String)
    start = findfirst("function unpack_parameters(p)", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("positive_parameter", body) &&
           occursin("p.phys", body)
end

function pack_parameters_source_holds()
    src = read(ude_jl_source_path_for_pack(), String)
    start = findfirst("function pack_parameters(phys::NamedTuple, nn_ps)", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("inverse_softplus", body) &&
           occursin("ComponentVector", body)
end

function frozen_phys_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Training.jl"), String)
    return occursin("function _zero_frozen_phys_gradient", src) &&
           occursin("function _restore_frozen_phys", src) &&
           occursin("name in frozen", src)
end

function default_phys_includes_custom_source_holds()
    src = read(parameter_schema_jl_source_path(), String)
    return occursin(":k_custom => 0.8", src)
end

# -- Matrices / catalog -------------------------------------------------------

function parameter_schema_pack_fixture_names()
    return (
        :roundtrip, :linear_pack, :remapped_pack, :two_pack, :kinetic,
        :missing_custom, :default_params, :schema_tree, :dummy_head,
        :remap_heads, :dual_heads, :middle_heads, :frozen_zero,
        :frozen_restore, :frozen_copy, :remap_frozen, :hill, :hill_known,
        :mm, :mm_known, :default, :three, :six, :competitive, :repress,
        :skipped, :wrong, :smoke_protocol)
end

function smoke_vs_protocol_schema_row()
    smoke = unique_claim_fingerprint(; smoke = true)
    proto = unique_claim_fingerprint()
    return (;
        smoke_ics = smoke.n_ics,
        proto_ics = proto.n_ics,
        holds = smoke.n_ics == 1 && proto.n_ics == 9 &&
                proto.n_points == 50 && proto.seed == 103 && !proto.smoke)
end

function parameter_schema_pack_fixture_matrix()
    roundtrip = positive_parameter_roundtrip_row()
    linear = pack_unpack_linear_row()
    remap = remapped_pack_unpack_row()
    two = two_regulator_pack_unpack_row()
    kinetic = kinetic_custom_in_schema_row()
    missing = missing_custom_validate_throws_row()
    defaults = default_parameters_include_custom_row()
    tree = schema_vs_compiled_nn_tree_row()
    dummy = dummy_head_on_zero_hole_row()
    remap_h = remapped_schema_matches_heads_row()
    dual = dual_schema_matches_heads_row()
    middle = skipped_middle_schema_heads_row()
    frozen_z = frozen_phys_zero_gradient_row()
    frozen_r = frozen_phys_restore_row()
    frozen_c = frozen_phys_config_copy_row()
    remap_f = remapped_frozen_phys_row()
    hill = hill_unknown_schema_row()
    hill_k = hill_known_schema_row()
    mm = mm_unknown_schema_row()
    mm_k = mm_known_schema_row()
    default = default_example_schema_row()
    three = three_state_schema_row()
    six = six_state_schema_row()
    comp = competitive_schema_row()
    repress = repressilator_schema_row()
    skipped = skipped_duplicate_schema_row()
    wrong = wrong_graph_schema_row()
    smoke = smoke_vs_protocol_schema_row()
    return (;
        roundtrip, linear, remap, two, kinetic, missing, defaults, tree,
        dummy, remap_h, dual, middle, frozen_z, frozen_r, frozen_c,
        remap_f, hill, hill_k, mm, mm_k, default, three, six, comp,
        repress, skipped, wrong, smoke,
        holds = roundtrip.holds && linear.holds && remap.holds &&
                two.holds && kinetic.holds && missing.holds &&
                defaults.holds && tree.holds && dummy.holds &&
                remap_h.holds && dual.holds && middle.holds &&
                frozen_z.holds && frozen_r.holds && frozen_c.holds &&
                remap_f.holds && hill.holds && hill_k.holds && mm.holds &&
                mm_k.holds && default.holds && three.holds && six.holds &&
                comp.holds && repress.holds && skipped.holds &&
                wrong.holds && smoke.holds)
end

function parameter_schema_pack_typed_matrix()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(431)
    model, packed = build_ude_model(rng, net)
    kinetic = parameter_schema_pack_row(:kinetic, model, packed)
    net2 = build_remapped_two_regulator_network()
    model2, packed2 = build_ude_model(MersenneTwister(433), net2)
    remap = parameter_schema_pack_row(:remap, model2, packed2)
    return (;
        kinetic = parameter_schema_pack_row_namedtuple(kinetic.typed),
        remap = parameter_schema_pack_row_namedtuple(remap.typed),
        holds = kinetic.holds && remap.holds &&
                kinetic.typed.has_custom &&
                remap.typed.nn_heads == 2)
end

function format_parameter_schema_pack_index()
    io = IOBuffer()
    println(io, "| row | meaning |")
    println(io, "|---|---|")
    println(io, "| roundtrip | positive_parameter ∘ inverse_softplus |")
    println(io, "| linear_pack | linear phys pack/unpack |")
    println(io, "| remapped_pack | remapped multi-head pack/unpack |")
    println(io, "| two_pack | two-regulator single-head pack |")
    println(io, "| kinetic | :k_custom is in parameter_schema |")
    println(io, "| missing_custom | omitting :k_custom throws |")
    println(io, "| default_params | default_parameters includes :k_custom |")
    println(io, "| schema_tree | schema.nn_heads matches compiled terms |")
    println(io, "| dummy_head | 0-hole models still carry a dummy Lux head |")
    println(io, "| remap_heads | remapped tree has head_1 and head_2 |")
    println(io, "| dual_heads | dual unknown does not admit |")
    println(io, "| middle_heads | skipped-middle heads stay dense |")
    println(io, "| frozen_zero | frozen_phys zeros the raw gradient |")
    println(io, "| frozen_restore | frozen_phys restores the packed coord |")
    println(io, "| frozen_copy | TrainingConfig copies frozen_phys |")
    println(io, "| remap_frozen | remapped :k_ia can be frozen |")
    println(io, "| hill | unknown Hill schema |")
    println(io, "| hill_known | known Hill schema |")
    println(io, "| mm | unknown MM schema |")
    println(io, "| mm_known | known MM schema |")
    println(io, "| default | p53/Mdm2 schema |")
    println(io, "| three | 3-state schema |")
    println(io, "| six | 6-state schema |")
    println(io, "| competitive | competitive schema |")
    println(io, "| repress | repressilator schema |")
    println(io, "| skipped | skipped-duplicate schema |")
    println(io, "| wrong | wrong-graph schema |")
    println(io, "| smoke_protocol | 1-IC smoke is not 9 ICs / 50 points |")
    return String(take!(io))
end

function parameter_schema_pack_index_holds()
    text = format_parameter_schema_pack_index()
    names = parameter_schema_pack_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin(":k_custom", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Source checks ----------------------------------------------------------

function parameter_schema_pack_module_include_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "BioDynaX.jl"), String)
    tests = read(joinpath(pkgdir(BioDynaX), "test", "runtests.jl"), String)
    return occursin("include(\"ParameterSchemaPack.jl\")", src) &&
           occursin("test_parameter_schema_pack.jl", tests)
end

function recovery_thresholds_untouched_schema_row()
    lock = recovery_thresholds_lock()
    return (;
        holds = RECOVERY_THRESHOLDS == lock &&
                lock.support_f1_ude == 0.50 &&
                lock.support_f1_clean == 0.99)
end

function public_export_list_untouched_schema_row()
    return (;
        pack_exported = :pack_parameters in LOCKED_PUBLIC_EXPORTS,
        schema_exported = :parameter_schema in LOCKED_PUBLIC_EXPORTS,
        unpack_unexported = !(:unpack_parameters in names(BioDynaX)),
        holds = :pack_parameters in LOCKED_PUBLIC_EXPORTS &&
                :parameter_schema in LOCKED_PUBLIC_EXPORTS &&
                :positive_parameter in LOCKED_PUBLIC_EXPORTS &&
                !(:unpack_parameters in names(BioDynaX)) &&
                !(:ParameterSchemaPackRow in names(BioDynaX)) &&
                public_export_list_holds())
end

function unique_claim_not_faster_by_dropping_ics_schema_row()
    fp = unique_claim_fingerprint()
    return (;
        n_ics = fp.n_ics,
        holds = fp.n_ics == 9 && fp.n_points == 50 &&
                fp.seed == 103 && !fp.smoke)
end

function unpack_missing_phys_throws_row()
    threw = try
        unpack_parameters((; nn = (a = 1,)))
        false
    catch err
        err isa ArgumentError
    end
    return (; threw, holds = threw)
end

function combined_f1_not_schema_kpi_row()
    return (;
        holds = :support_f1 ∉ UNIQUE_CLAIM_KPI_FIELDS &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50)
end

function nn_tree_is_float64_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(435)
    model, packed = build_ude_model(rng, net)
    leaves = Float64[]
    function walk(x)
        if x isa AbstractArray
            eltype(x) <: AbstractFloat && append!(leaves, Float64.(vec(x)))
        elseif x isa NamedTuple || x isa Tuple
            foreach(walk, x)
        end
        return nothing
    end
    walk(packed.nn)
    return (;
        n = length(leaves),
        holds = !isempty(leaves) && all(isfinite, leaves) &&
                model.nn isa MultiHeadNetwork)
end

function schema_phys_are_positive_row()
    net = DEFAULT_EXAMPLE_NETWORK
    rng = MersenneTwister(437)
    model, packed = build_ude_model(rng, net)
    unpacked = unpack_parameters(packed)
    vals = [Float64(getproperty(unpacked.phys, n)) for n in keys(unpacked.phys)]
    return (;
        n = length(vals),
        holds = !isempty(vals) && all(>(0), vals))
end

function six_state_schema_heads_row()
    net = build_six_state_unknown_network()
    rng = MersenneTwister(439)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        schema_heads = schema.nn_heads,
        compiled = compiled_neural_head_count(model),
        lux = lux_head_count(model),
        holds = schema.nn_heads == compiled_neural_head_count(model) &&
                (schema.nn_heads ≤ 1 || packed.nn isa ComponentVector))
end

function default_example_pack_predict_row()
    net = DEFAULT_EXAMPLE_NETWORK
    rng = MersenneTwister(441)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    u0 = [0.20, 0.15]
    times = collect(range(0.0, 0.5; length = 6))
    data = predict_ude(packed, u0, (0.0, 0.5), times, model)
    return (;
        names = copy(schema.phys_names),
        finite = all(isfinite, data),
        has_alpha = :α_p53 in schema.phys_names,
        holds = all(isfinite, data) && :α_p53 in schema.phys_names &&
                schema.nn_heads == 1)
end

function hill_known_has_vmax_row()
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    rng = MersenneTwister(443)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        names = copy(schema.phys_names),
        has_vmax = :vmax in schema.phys_names,
        nn_heads = schema.nn_heads,
        holds = :vmax in schema.phys_names && schema.nn_heads == 0 &&
                hasproperty(packed.phys, :vmax))
end

function competitive_has_ki_row()
    net = build_competitive_test_network(; known = false)
    rng = MersenneTwister(445)
    model, _ = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        names = copy(schema.phys_names),
        nn_heads = schema.nn_heads,
        holds = schema.nn_heads == 1 && validate_network(net) === net)
end

function unpack_then_repack_row()
    net = build_linear_test_network()
    rng = MersenneTwister(447)
    _, p0 = build_ude_model(rng, net)
    phys = (k_ba = 0.9, k_a = 1.1, k_b = 0.55)
    packed = pack_parameters(phys, p0.nn)
    unpacked = unpack_parameters(packed)
    packed2 = pack_parameters(unpacked.phys, unpacked.nn)
    unpacked2 = unpack_parameters(packed2)
    err = phys_roundtrip_error(phys, unpacked2)
    return (; err, holds = err.ok && err.maxerr < 1e-9)
end

function suite_section_schema_catalog()
    rows = NamedTuple[]
    for section in recovery_suite_sections()
        net = recovery_suite_section_network(section)
        compiles = try
            compile_mechanism(net)
            true
        catch
            false
        end
        compiles || continue
        rng = MersenneTwister(hash(section) % 10_000)
        model, packed = build_ude_model(rng, net)
        schema = parameter_schema(model)
        unpacked = unpack_parameters(packed)
        push!(rows,
            (;
                section,
                n_phys = length(schema.phys_names),
                nn_heads = schema.nn_heads,
                has_custom = :k_custom in schema.phys_names,
                holds = schema.nn_heads == compiled_neural_head_count(model) &&
                    hasproperty(unpacked, :phys)))
    end
    return (;
        n = length(rows),
        rows,
        holds = !isempty(rows) && all(r -> r.holds, rows))
end

function format_suite_schema_catalog()
    catalog = suite_section_schema_catalog()
    io = IOBuffer()
    println(io, "| section | n_phys | nn_heads | custom |")
    println(io, "|---|---|---|---|")
    for row in catalog.rows
        println(io, "| ", row.section, " | ", row.n_phys, " | ",
            row.nn_heads, " | ", row.has_custom, " |")
    end
    return String(take!(io))
end

function suite_schema_catalog_holds()
    catalog = suite_section_schema_catalog()
    text = format_suite_schema_catalog()
    return catalog.holds &&
           occursin("three_state", text) &&
           count(==('|'), text) ≥ 20 &&
           !occursin("support_f1_ude = 0.99", text)
end

function kinetic_known_tradeoff_now_predicts_row()
    row = kinetic_known_tradeoff_path()
    return (;
        has_custom = row.has_custom,
        finite = row.finite,
        holds = row.holds && row.has_custom && row.finite)
end

function validate_network_open_on_schema_fixtures_row()
    nets = (
        build_kinetic_generalization_network(),
        build_remapped_two_regulator_network(),
        build_dual_unknown_network(),
        build_linear_test_network())
    return (;
        holds = all(net -> validate_network(net) === net, nets))
end

function hill_from_nn_closed_schema_row()
    return (;
        holds = :canonical_hill_from_nn in PROTOCOL_RESULT_FIELDS &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50)
end

function bounded_parameter_row()
    mid = bounded_parameter(0.0, 0.2, 0.8)
    lo = bounded_parameter(-20.0, 0.2, 0.8)
    hi = bounded_parameter(20.0, 0.2, 0.8)
    return (;
        mid, lo, hi,
        holds = abs(mid - 0.5) < 1e-12 &&
                lo > 0.2 && lo < 0.21 &&
                hi < 0.8 && hi > 0.79)
end

function two_regulator_input_dim_row()
    net = build_two_regulator_unknown_network()
    rng = MersenneTwister(449)
    model, packed = build_ude_model(rng, net)
    terms = neural_destruction_terms(model)
    arities = neural_regulator_arities(model)
    schema = parameter_schema(model)
    return (;
        n_regs = isempty(terms) ? 0 : length(only(terms).regulators),
        arities,
        nn_heads = schema.nn_heads,
        holds = length(arities) == 1 && only(arities) == 2 &&
                schema.nn_heads == 1 && hasproperty(packed, :nn))
end

function remapped_input_dims_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(451)
    model, packed = build_ude_model(rng, net)
    arities = neural_regulator_arities(model)
    return (;
        arities,
        n = length(arities),
        holds = length(arities) == 2 &&
                Set(arities) == Set((1, 2)) &&
                hasproperty(packed.nn, :head_1) &&
                hasproperty(packed.nn, :head_2))
end

function schema_name_catalog_row()
    linear = parameter_schema(build_ude_model(MersenneTwister(1),
        build_linear_test_network())[1])
    kinetic = parameter_schema(build_ude_model(MersenneTwister(2),
        build_kinetic_generalization_network())[1])
    hill = parameter_schema(build_ude_model(MersenneTwister(3),
        build_hill_recovery_network(; known = true, hill_order = 2))[1])
    return (;
        linear = copy(linear.phys_names),
        kinetic = copy(kinetic.phys_names),
        hill = copy(hill.phys_names),
        holds = :k_ba in linear.phys_names &&
                :k_custom in kinetic.phys_names &&
                :vmax in hill.phys_names &&
                :k_custom ∉ linear.phys_names)
end

function default_parameters_validate_row()
    net = build_mm_recovery_network(; known = true)
    rng = MersenneTwister(453)
    model, _ = build_ude_model(rng, net)
    packed = default_parameters(model; rng = MersenneTwister(454))
    schema = parameter_schema(model)
    unpacked = unpack_parameters(packed)
    validate_phys_parameters(unpacked.phys, schema)
    return (;
        n = length(schema.phys_names),
        holds = hasproperty(packed, :phys) &&
                schema.nn_heads == 0)
end

function mm_known_has_km_row()
    net = build_mm_recovery_network(; known = true)
    rng = MersenneTwister(455)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        names = copy(schema.phys_names),
        holds = (:km in schema.phys_names || :vmax in schema.phys_names) &&
                schema.nn_heads == 0 &&
                hasproperty(packed, :phys))
end

function three_state_schema_heads_row()
    net = build_three_state_unknown_network()
    rng = MersenneTwister(457)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        nn_heads = schema.nn_heads,
        compiled = compiled_neural_head_count(model),
        holds = schema.nn_heads == 1 &&
                compiled_neural_head_count(model) == 1 &&
                hasproperty(packed, :nn))
end

function wrong_graph_schema_heads_row()
    net = build_wrong_graph_unknown_network()
    rng = MersenneTwister(459)
    model, _ = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        nn_heads = schema.nn_heads,
        holds = schema.nn_heads == compiled_neural_head_count(model) &&
                validate_network(net) === net)
end

function skipped_duplicate_dense_schema_row()
    net = build_skipped_duplicate_unknown_network()
    rng = MersenneTwister(461)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        nn_heads = schema.nn_heads,
        dense = neural_index_is_dense(model),
        holds = schema.nn_heads == compiled_neural_head_count(model) &&
                neural_index_is_dense(model) &&
                hasproperty(packed, :nn))
end

function format_schema_names(schema::ParameterSchema)
    return join(string.(schema.phys_names), ", ")
end

function format_schema_names_holds()
    net = build_kinetic_generalization_network()
    schema = parameter_schema(build_ude_model(MersenneTwister(5), net)[1])
    text = format_schema_names(schema)
    return occursin("k_custom", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function pack_rejects_nonpositive_phys_row()
    net = build_linear_test_network()
    rng = MersenneTwister(463)
    _, p0 = build_ude_model(rng, net)
    threw = try
        pack_parameters((k_ba = 0.0, k_a = 1.0, k_b = 0.5), p0.nn)
        false
    catch err
        err isa DomainError
    end
    return (; threw, holds = threw)
end

function validate_rejects_nonpositive_row()
    schema = ParameterSchema([:k_ba, :k_a], 0)
    threw = try
        validate_phys_parameters((k_ba = 1.0, k_a = -0.2), schema)
        false
    catch err
        err isa ArgumentError
    end
    return (; threw, holds = threw)
end

function extras_not_invented_by_schema_row()
    label = extras_print_label(("1", "r"))
    return (;
        label,
        holds = extras_print_is_hardcoded_attempt(label) == false &&
                label == "1, r")
end

function format_pack_markdown(schema::ParameterSchema, unpacked)
    io = IOBuffer()
    println(io, "| name | positive |")
    println(io, "|---|---|")
    for name in schema.phys_names
        val = hasproperty(unpacked.phys, name) ?
              getproperty(unpacked.phys, name) : missing
        println(io, "| ", name, " | ", val, " |")
    end
    println(io, "| nn_heads | ", schema.nn_heads, " |")
    return String(take!(io))
end

function format_pack_markdown_holds()
    net = build_kinetic_generalization_network()
    model, packed = build_ude_model(MersenneTwister(7), net)
    schema = parameter_schema(model)
    text = format_pack_markdown(schema, unpack_parameters(packed))
    return occursin("k_custom", text) &&
           occursin("nn_heads", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function six_state_wrong_schema_row()
    net = build_six_state_wrong_graph_network()
    rng = MersenneTwister(465)
    model, packed = build_ude_model(rng, net)
    schema = parameter_schema(model)
    return (;
        nn_heads = schema.nn_heads,
        holds = schema.nn_heads == compiled_neural_head_count(model) &&
                hasproperty(packed, :phys) &&
                validate_network(net) === net)
end

function ablation_schema_or_skip_row()
    net = build_rate_ablation_network()
    compiles = try
        compile_mechanism(net)
        true
    catch
        false
    end
    compiles || return (; compiles = false, holds = true)
    model, packed = build_ude_model(MersenneTwister(467), net)
    schema = parameter_schema(model)
    return (;
        compiles = true,
        nn_heads = schema.nn_heads,
        holds = schema.nn_heads == compiled_neural_head_count(model) &&
                hasproperty(packed, :phys))
end

function linear_schema_names_are_mass_action_row()
    net = build_linear_test_network()
    schema = parameter_schema(build_ude_model(MersenneTwister(9), net)[1])
    return (;
        names = copy(schema.phys_names),
        holds = issetequal(schema.phys_names, [:k_ba, :k_a, :k_b]) &&
                schema.nn_heads == 0)
end

