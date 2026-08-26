###############################################################################
# Allocation / type-stability gates (not exported).
#
# Extends the quality-gate surface to the new unexported workspaces:
# DiscoveryWorkspace, TrainingReuse, HybridResidual, GraphLocalLibrary,
# DenominatorDomain, ParameterSchemaPack, DocsExecutable. Thresholds
# are measured hot-path byte counts that can fail. Combined F1 stays
# a skeleton floor.
#
# Does not grow exports. Does not drop protocol ICs. Does not open
# Hill-from-NN. Does not loosen RECOVERY_THRESHOLDS.
###############################################################################

const ALLOCATION_GATES_MUST_CONTAIN = (
    "function allocation_hot",
    "function pack_parameters_allocation_row",
    "function stlsq_workspace_reuse_row",
    "function unpack_parameters_allocation_row",
    "function denominator_split_allocation_row",
    "struct AllocationGateRow",
    "function positive_parameter_allocation_row",
    "function schema_type_row")

const ALLOCATION_GATES_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

# Measured hot-path ceilings (bytes). Slack is intentional so a
# regression still fails; these are not "zero allocation" claims.
const ALLOCATION_GATE_LIMITS = (
    pack_parameters = 12288,
    unpack_parameters = 4096,
    parameter_schema = 2048,
    positive_parameter = 0,
    inverse_softplus = 0,
    denominator_violation = 4096,
    denominator_split = 8192,
    local_basis = 65536,
    graph_vs_global = 131072,
    ude_rhs_linear = 512,
    ude_rhs_p53 = 4096)

function allocation_gates_locked_sentences()
    return (;
        measured = "Allocation gates use measured hot-path byte ceilings that can fail.",
        workspace = "STLSQWorkspace reuse must not increment resize_count on a same-shape ensure.",
        types = "unpack_parameters and parameter_schema keep concrete return types.",
        protocol = "Allocation smoke is not the seed-103 / 9-IC protocol.")
end

allocation_gates_contract() = allocation_gates_locked_sentences().measured

function allocation_gates_source_path()
    joinpath(pkgdir(BioDynaX), "src", "AllocationGates.jl")
end

function allocation_gates_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "allocation-gates.md")
end

function allocation_gates_test_path()
    joinpath(pkgdir(BioDynaX), "test", "test_allocation_gates.jl")
end

# -- Measurement --------------------------------------------------------------

"""
    allocation_hot(f; warmup=2)

Call `f` `warmup` times, then return `(warm, hot)` byte counts from
two subsequent `@allocated` calls. Thresholds compare against `hot`.
"""
function allocation_hot(f; warmup::Int = 2)
    for _ in 1:warmup
        f()
    end
    warm = @allocated f()
    hot = @allocated f()
    return (; warm, hot)
end

function allocation_under(hot::Integer, limit::Integer)
    return Int(hot) ≤ Int(limit)
end

struct AllocationGateRow
    name::Symbol
    warm::Int
    hot::Int
    limit::Int
    holds::Bool
end

function allocation_gate_row(name::Symbol, f, limit::Integer; warmup::Int = 2)
    bytes = allocation_hot(f; warmup = warmup)
    holds = allocation_under(bytes.hot, limit)
    typed = AllocationGateRow(name, bytes.warm, bytes.hot, Int(limit), holds)
    return (; bytes, typed, holds)
end

function allocation_gate_row_namedtuple(row::AllocationGateRow)
    return (;
        name = row.name,
        warm = row.warm,
        hot = row.hot,
        limit = row.limit,
        holds = row.holds)
end

# -- Pack / schema / positive_parameter ---------------------------------------

function _linear_pack_fixture()
    net = build_linear_test_network()
    rng = MersenneTwister(701)
    model, p0 = build_ude_model(rng, net)
    phys = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    return (; net, model, p0, phys)
end

function pack_parameters_allocation_row()
    fx = _linear_pack_fixture()
    row = allocation_gate_row(:pack, () -> pack_parameters(fx.phys, fx.p0.nn),
        ALLOCATION_GATE_LIMITS.pack_parameters)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.pack_parameters))
end

function unpack_parameters_allocation_row()
    fx = _linear_pack_fixture()
    packed = pack_parameters(fx.phys, fx.p0.nn)
    row = allocation_gate_row(:unpack, () -> unpack_parameters(packed),
        ALLOCATION_GATE_LIMITS.unpack_parameters)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.unpack_parameters))
end

function parameter_schema_allocation_row()
    fx = _linear_pack_fixture()
    row = allocation_gate_row(:schema, () -> parameter_schema(fx.model),
        ALLOCATION_GATE_LIMITS.parameter_schema)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.parameter_schema))
end

function positive_parameter_allocation_row()
    raw = 0.5
    row = allocation_gate_row(:positive, () -> positive_parameter(raw),
        ALLOCATION_GATE_LIMITS.positive_parameter)
    return merge(row, (;
        value = positive_parameter(raw),
        limit = ALLOCATION_GATE_LIMITS.positive_parameter))
end

function inverse_softplus_allocation_row()
    row = allocation_gate_row(:inverse, () -> BioDynaX.inverse_softplus(1.7),
        ALLOCATION_GATE_LIMITS.inverse_softplus)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.inverse_softplus))
end

function schema_type_row()
    fx = _linear_pack_fixture()
    schema = parameter_schema(fx.model)
    unpacked = unpack_parameters(fx.p0)
    return (;
        schema_type = typeof(schema),
        unpacked_type = typeof(unpacked),
        holds = schema isa ParameterSchema &&
                unpacked isa NamedTuple &&
                hasproperty(unpacked, :phys) &&
                hasproperty(unpacked, :nn))
end

function remapped_pack_allocation_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(703)
    model, p0 = build_ude_model(rng, net)
    phys = remapped_two_regulator_phys_truth()
    row = allocation_gate_row(:remap_pack,
        () -> pack_parameters(phys, p0.nn),
        ALLOCATION_GATE_LIMITS.pack_parameters)
    schema = parameter_schema(model)
    return merge(row, (;
        nn_heads = schema.nn_heads,
        holds = row.holds && schema.nn_heads == 2))
end

function kinetic_schema_allocation_row()
    net = build_kinetic_generalization_network()
    rng = MersenneTwister(705)
    model, packed = build_ude_model(rng, net)
    row = allocation_gate_row(:kinetic_schema,
        () -> parameter_schema(model),
        ALLOCATION_GATE_LIMITS.parameter_schema)
    schema = parameter_schema(model)
    return merge(row, (;
        has_custom = :k_custom in schema.phys_names,
        holds = row.holds && :k_custom in schema.phys_names))
end

# -- Denominator / library ----------------------------------------------------

function denominator_violation_allocation_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(40)
    row = allocation_gate_row(:den,
        () -> denominator_violation_count(cand, X),
        ALLOCATION_GATE_LIMITS.denominator_violation)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.denominator_violation))
end

function denominator_split_allocation_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(40)
    train, val = split_train_val(X)
    domain = _denominator_domain_grid(X; n = 8)
    row = allocation_gate_row(:split,
        () -> denominator_split_counts(cand, train, val, domain),
        ALLOCATION_GATE_LIMITS.denominator_split)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.denominator_split))
end

function local_basis_allocation_row()
    net = build_linear_test_network()
    row = allocation_gate_row(:basis,
        () -> local_basis(net, 1; degree = 2,
            include_interactions = false, scope = :graph),
        ALLOCATION_GATE_LIMITS.local_basis)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.local_basis))
end

function graph_vs_global_allocation_row()
    net = build_three_state_unknown_network()
    row = allocation_gate_row(:graph,
        () -> graph_vs_global_library_row(net, 1),
        ALLOCATION_GATE_LIMITS.graph_vs_global)
    return merge(row, (; limit = ALLOCATION_GATE_LIMITS.graph_vs_global))
end

function extras_denominator_allocation_row()
    cand = synthetic_safe_implicit_candidate()
    X = regulator_grid(24)
    row = allocation_gate_row(:extras_den,
        () -> ude_extras_denominator_row(cand, X; extras = String[],
            domain_samples = 8),
        65536)
    return row
end

# -- Discovery workspace reuse ------------------------------------------------

function stlsq_workspace_reuse_row()
    ws = allocate_stlsq_workspace(Float64, 32, 8, 16)
    start = ws.resize_count
    ensure_stlsq_workspace!(ws, 32, 8, 16)
    ensure_stlsq_workspace!(ws, 32, 8, 16)
    same_shape = ws.resize_count == start
    grew = allocate_stlsq_workspace(Float64, 64, 16, 16)
    ensure_stlsq_workspace!(ws, 64, 16, 16)
    return (;
        start,
        same_shape,
        grew = grew.resize_count == 1,
        after_grow = ws.resize_count > start,
        holds = same_shape &&
                grew.resize_count == 1 &&
                ws.resize_count > start &&
                grew isa STLSQWorkspace{Float64})
end

function implicit_workspace_reuse_row()
    ws = allocate_implicit_workspace(Float64, 40, 4, 3, 16)
    start = ws.resize_count
    ensure_implicit_workspace!(ws, 40, 4, 3, 16)
    same = ws.resize_count == start
    ensure_implicit_workspace!(ws, 80, 6, 5, 16)
    return (;
        start,
        same,
        after = ws.resize_count,
        holds = same && ws.resize_count > start &&
                ws isa ImplicitLibraryWorkspace{Float64})
end

function ude_rhs_linear_allocation_row()
    fx = _linear_pack_fixture()
    packed = pack_parameters(fx.phys, fx.p0.nn)
    u = [0.2, 0.1]
    cache = allocate_cache(fx.model, Float64)
    for _ in 1:20
        ude_rhs!(cache.du, u, packed, 0.0, fx.model, cache)
    end
    row = allocation_gate_row(:rhs_linear,
        () -> ude_rhs!(cache.du, u, packed, 0.0, fx.model, cache),
        ALLOCATION_GATE_LIMITS.ude_rhs_linear)
    return merge(row, (;
        finite = all(isfinite, cache.du),
        holds = row.holds && all(isfinite, cache.du)))
end

function ude_rhs_default_allocation_row()
    rng = MersenneTwister(707)
    model, packed = build_ude_model(rng, DEFAULT_EXAMPLE_NETWORK)
    u = [0.2, 0.1]
    cache = allocate_cache(model, Float64)
    for _ in 1:20
        ude_rhs!(cache.du, u, packed, 0.0, model, cache)
    end
    row = allocation_gate_row(:rhs_p53,
        () -> ude_rhs!(cache.du, u, packed, 0.0, model, cache),
        ALLOCATION_GATE_LIMITS.ude_rhs_p53)
    return merge(row, (;
        finite = all(isfinite, cache.du),
        holds = row.holds && all(isfinite, cache.du)))
end

# -- Type rows ----------------------------------------------------------------

function pack_type_row()
    fx = _linear_pack_fixture()
    packed = pack_parameters(fx.phys, fx.p0.nn)
    return (;
        type = typeof(packed),
        holds = packed isa ComponentVector &&
                hasproperty(packed, :phys) &&
                hasproperty(packed, :nn))
end

function schema_head_type_row()
    net = build_remapped_two_regulator_network()
    model, packed = build_ude_model(MersenneTwister(709), net)
    schema = parameter_schema(model)
    return (;
        schema_heads = schema.nn_heads,
        multi = model.nn isa MultiHeadNetwork,
        holds = schema isa ParameterSchema &&
                schema.nn_heads == 2 &&
                model.nn isa MultiHeadNetwork &&
                packed isa ComponentVector)
end

function candidate_type_row()
    safe = synthetic_safe_implicit_candidate()
    expl = synthetic_explicit_candidate()
    return (;
        implicit = safe isa ImplicitCandidate,
        explicit = expl isa ExplicitCandidate,
        holds = safe isa ImplicitCandidate &&
                expl isa ExplicitCandidate)
end

function fingerprint_type_row()
    proto = unique_claim_fingerprint()
    smoke = unique_claim_fingerprint(; smoke = true)
    return (;
        proto_type = typeof(proto),
        smoke_type = typeof(smoke),
        holds = proto isa UniqueClaimFingerprint &&
                smoke isa UniqueClaimFingerprint &&
                proto.n_ics == 9 && smoke.n_ics == 1)
end

function contract_string_type_row()
    texts = docs_hl_contract_strings()
    return (;
        n = length(texts),
        holds = all(t -> t isa String, texts) &&
                length(texts) == 5 &&
                length(unique(texts)) == 5)
end

# -- Surface allocation rows --------------------------------------------------

function hybrid_residual_contract_allocation_row()
    return allocation_gate_row(:residual_contract,
        () -> hybrid_residual_contract(), 256)
end

function identifiability_contract_allocation_row()
    return allocation_gate_row(:ident_contract,
        () -> identifiability_product_contract(), 256)
end

function graph_local_contract_allocation_row()
    return allocation_gate_row(:library_contract,
        () -> graph_local_library_contract(), 256)
end

function denominator_contract_allocation_row()
    return allocation_gate_row(:denom_contract,
        () -> denominator_domain_contract(), 256)
end

function schema_pack_contract_allocation_row()
    return allocation_gate_row(:schema_contract,
        () -> parameter_schema_pack_contract(), 256)
end

function docs_join_allocation_row()
    return allocation_gate_row(:docs_join,
        () -> docs_executable_join_row(), 4096)
end

function leftover_hits_allocation_row()
    return allocation_gate_row(:leftover,
        () -> leftover_contradiction_hits(), 65536)
end

function library_contains_allocation_row()
    net = build_linear_test_network()
    spec = local_basis(net, 1; degree = 2, include_interactions = false)
    return allocation_gate_row(:contains,
        () -> library_contains_variable(spec, 1), 64)
end

function graph_parent_set_allocation_row()
    net = build_three_state_unknown_network()
    return allocation_gate_row(:parents,
        () -> graph_parent_set(net, 1), 2048)
end

function default_phys_allocation_row()
    fx = _linear_pack_fixture()
    schema = parameter_schema(fx.model)
    return allocation_gate_row(:defaults,
        () -> default_phys_parameters(schema), 2048)
end

function frozen_zero_allocation_row()
    fx = _linear_pack_fixture()
    g = ComponentVector(phys = (k_ba = 1.2, k_a = 0.4, k_b = 0.3),
        nn = fx.p0.nn)
    return allocation_gate_row(:frozen,
        () -> _zero_frozen_phys_gradient(g, [:k_ba]), 4096)
end

# -- Extra workspace reuse (DiscoveryWorkspace grow-only) ---------------------

function streaming_implicit_reuse_row()
    ws = allocate_streaming_implicit_workspace(Float64, 40, 4, 3, 16)
    start = ws.resize_count
    ensure_streaming_implicit_workspace!(ws, 40, 4, 3, 16)
    same = ws.resize_count == start
    ensure_streaming_implicit_workspace!(ws, 80, 6, 5, 16)
    return (;
        start,
        same,
        after = ws.resize_count,
        holds = same && ws.resize_count > start &&
                ws isa StreamingImplicitWorkspace{Float64})
end

function library_chunk_reuse_row()
    ws = allocate_library_chunk_workspace(Float64, 16, 8)
    start = ws.resize_count
    ensure_library_chunk_workspace!(ws, 16, 8)
    same = ws.resize_count == start
    ensure_library_chunk_workspace!(ws, 32, 12)
    return (;
        start,
        same,
        after = ws.resize_count,
        holds = same && ws.resize_count > start &&
                ws isa LibraryChunkWorkspace{Float64})
end

function collect_active_allocation_row()
    active = trues(16)
    indices = sizehint!(Int[], 16)
    collect_active_indices!(indices, active)
    row = allocation_gate_row(:active,
        () -> collect_active_indices!(indices, active), 256)
    return merge(row, (;
        n = length(indices),
        holds = row.holds && length(indices) == 16))
end

function discovery_stlsq_reuse_row()
    rng = MersenneTwister(711)
    A = randn(rng, 48, 6)
    y = A * [1.0, 0.0, 0.4, 0.0, 0.0, 0.25] .+ 0.01 .* randn(rng, 48)
    report = discovery_workspace_alloc_report(A, y, 0.08; chunk_size = 16)
    return (;
        gram_stable = report.gram_stable,
        chunk_stable = report.chunk_stable,
        naive_bytes = report.naive_bytes,
        reused_bytes = report.reused_bytes,
        reuse_smaller = report.reuse_smaller,
        resize_count = report.resize_count,
        holds = report.holds && report.reused_bytes < report.naive_bytes)
end

function pack_does_not_compile_row()
    fx = _linear_pack_fixture()
    n = with_compile_network_counter() do counter
        pack_parameters(fx.phys, fx.p0.nn)
        unpack_parameters(fx.p0)
        parameter_schema(fx.model)
        counter[]
    end
    return (; compiles = n, holds = n == 0)
end

function allocate_cache_type_row()
    fx = _linear_pack_fixture()
    cache = allocate_cache(fx.model, Float64)
    return (;
        type = typeof(cache),
        n = length(cache.du),
        holds = cache isa UDEModelCache &&
                cache.du isa Vector{Float64} &&
                length(cache.du) == fx.model.compiled.nstates)
end

function unpack_type_row()
    fx = _linear_pack_fixture()
    unpacked = unpack_parameters(fx.p0)
    return (;
        type = typeof(unpacked),
        holds = unpacked isa NamedTuple &&
                unpacked.phys isa NamedTuple &&
                hasproperty(unpacked, :nn))
end

function positive_parameter_type_row()
    value = positive_parameter(0.5)
    inverse = BioDynaX.inverse_softplus(1.7)
    return (;
        value_type = typeof(value),
        inverse_type = typeof(inverse),
        holds = value isa Float64 && inverse isa Float64 && value > 0)
end

function training_solver_lock_type_row()
    fx = _linear_pack_fixture()
    solver = lock_training_solver(fx.model)
    return (;
        type = typeof(solver),
        holds = solver isa SolverConfig)
end

function hybrid_identity_type_row()
    built = hybrid_linear_unknown_model(719)
    term = hybrid_identity_term(built.model)
    return (;
        type = typeof(term),
        holds = term isa NeuralDestructionTerm)
end

function leftover_empty_type_row()
    hits = leftover_contradiction_hits()
    return (;
        n = length(hits),
        type = typeof(hits),
        holds = hits isa Vector{String} && isempty(hits))
end

function docs_executable_join_type_row()
    row = docs_executable_join_row()
    return (;
        n = row.n,
        holds = row.holds && row.n == 5)
end

function local_parent_gate_allocation_row()
    closed = local_has_true_parent_gate(nothing; variable = 1)
    row = allocation_gate_row(:parent_gate,
        () -> local_has_true_parent_gate(nothing; variable = 1), 256)
    return merge(row, (;
        closed,
        holds = row.holds && closed == false))
end

function identifiability_product_type_row()
    text = identifiability_product_contract()
    return (;
        type = typeof(text),
        holds = text isa String &&
                text == identifiability_product_locked_sentences().join)
end

function graph_local_type_row()
    text = graph_local_library_contract()
    return (;
        type = typeof(text),
        holds = text isa String &&
                text == graph_local_library_locked_sentences().prior)
end

function denominator_domain_type_row()
    text = denominator_domain_contract()
    return (;
        type = typeof(text),
        holds = text isa String &&
                text == denominator_domain_locked_sentences().split)
end

function schema_pack_type_row()
    text = parameter_schema_pack_contract()
    return (;
        type = typeof(text),
        holds = text isa String &&
                text == parameter_schema_pack_locked_sentences().custom)
end

function hybrid_residual_type_row()
    text = hybrid_residual_contract()
    return (;
        type = typeof(text),
        holds = text isa String &&
                text == hybrid_residual_locked_sentences().solve)
end

function docs_executable_type_row()
    text = docs_executable_contract()
    return (;
        type = typeof(text),
        holds = text isa String &&
                text == docs_executable_locked_sentences().join)
end

function allocation_gates_hl_contract_strings_row()
    texts = docs_hl_contract_strings()
    return (;
        n = length(texts),
        holds = length(texts) == 5 && length(unique(texts)) == 5)
end

function allocation_gates_ag_helpers_exist_row()
    return (;
        holds = isdefined(BioDynaX, :allocate_stlsq_workspace) &&
                isdefined(BioDynaX, :ensure_stlsq_workspace!) &&
                isdefined(BioDynaX, :lock_training_solver) &&
                isdefined(BioDynaX, :hybrid_residual_contract) &&
                isdefined(BioDynaX, :docs_executable_join_row) &&
                isdefined(BioDynaX, :parameter_schema) &&
                isdefined(BioDynaX, :unpack_parameters))
end

function workspace_not_exported_row()
    return (;
        holds = !(:STLSQWorkspace in names(BioDynaX)) &&
                !(:allocation_hot in names(BioDynaX)) &&
                !(:AllocationGateRow in names(BioDynaX)) &&
                !(:DiscoveryWorkspace in names(BioDynaX)) &&
                !(:TrainingReuse in names(BioDynaX)) &&
                public_export_list_holds())
end

function assert_allocation_under(hot::Integer, limit::Integer, name::Symbol)
    allocation_under(hot, limit) || throw(ErrorException(
        "allocation gate $name exceeded: $hot > $limit"))
    return true
end

function quality_gates_ude_rhs_live_row()
    linear = ude_rhs_linear_allocation_row()
    p53 = ude_rhs_default_allocation_row()
    return (;
        linear_hot = linear.typed.hot,
        p53_hot = p53.typed.hot,
        holds = linear.holds && p53.holds &&
                linear.typed.hot ≤ ALLOCATION_GATE_LIMITS.ude_rhs_linear &&
                p53.typed.hot ≤ ALLOCATION_GATE_LIMITS.ude_rhs_p53)
end

# -- Source locks -------------------------------------------------------------

function allocation_hot_source_holds()
    src = read(allocation_gates_source_path(), String)
    start = findfirst("function allocation_hot(f; warmup::Int = 2)", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("@allocated", body) && occursin("hot", body)
end

function quality_gates_test_still_holds()
    path = joinpath(pkgdir(BioDynaX), "test", "test_quality_gates.jl")
    isfile(path) || return false
    text = read(path, String)
    return occursin("@allocated", text) &&
           occursin("@inferred", text) &&
           occursin("ude_rhs!", text)
end

function quality_gates_extended_source_holds()
    path = joinpath(pkgdir(BioDynaX), "test", "test_quality_gates.jl")
    isfile(path) || return false
    text = read(path, String)
    return occursin("allocation_gates_contract_holds", text) &&
           occursin("pack_parameters_allocation_row", text) &&
           occursin("stlsq_workspace_reuse_row", text) &&
           occursin("schema_type_row", text)
end

function stlsq_reuse_source_holds()
    src = read(allocation_gates_source_path(), String)
    start = findfirst("function stlsq_workspace_reuse_row()", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("same_shape", body) &&
           occursin("ensure_stlsq_workspace!", body) &&
           occursin("resize_count", body)
end

function discovery_report_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "DiscoveryWorkspace.jl"), String)
    start = findfirst("function discovery_workspace_alloc_report", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("@allocated", body) &&
           occursin("_stlsq_blocked!", body) &&
           occursin("reused < naive", body)
end

# -- Catalog ------------------------------------------------------------------

function allocation_gates_fixture_names()
    return (
        :pack, :unpack, :schema, :positive, :inverse, :schema_type,
        :remap_pack, :kinetic, :den, :split, :basis, :graph,
        :extras_den, :stlsq, :implicit_ws, :rhs_linear, :rhs_p53,
        :pack_type, :heads, :cand_type, :fingerprint, :contracts,
        :residual_c, :ident_c, :library_c, :denom_c, :schema_c,
        :docs_join, :leftover, :contains, :parents, :defaults,
        :frozen, :smoke, :streaming, :chunk, :active, :stlsq_reuse,
        :no_compile, :cache, :unpack_t, :positive_t, :solver,
        :hybrid_id, :leftover_t, :docs_t, :parent_gate, :ident_t,
        :library_t, :denom_t, :schema_t, :residual_t, :docs_join_t,
        :hl_strings, :ag_helpers, :not_exported, :rhs_live)
end

function smoke_vs_protocol_allocation_row()
    smoke = unique_claim_fingerprint(; smoke = true)
    proto = unique_claim_fingerprint()
    return (;
        smoke_ics = smoke.n_ics,
        proto_ics = proto.n_ics,
        holds = smoke.n_ics == 1 && proto.n_ics == 9 &&
                proto.n_points == 50 && proto.seed == 103 && !proto.smoke)
end

function allocation_gates_fixture_matrix()
    pack = pack_parameters_allocation_row()
    unpack = unpack_parameters_allocation_row()
    schema = parameter_schema_allocation_row()
    pos = positive_parameter_allocation_row()
    inv = inverse_softplus_allocation_row()
    stype = schema_type_row()
    remap = remapped_pack_allocation_row()
    kinetic = kinetic_schema_allocation_row()
    den = denominator_violation_allocation_row()
    split = denominator_split_allocation_row()
    basis = local_basis_allocation_row()
    graph = graph_vs_global_allocation_row()
    extras = extras_denominator_allocation_row()
    stlsq = stlsq_workspace_reuse_row()
    implicit = implicit_workspace_reuse_row()
    rhs_l = ude_rhs_linear_allocation_row()
    rhs_p = ude_rhs_default_allocation_row()
    ptype = pack_type_row()
    heads = schema_head_type_row()
    ctype = candidate_type_row()
    ftype = fingerprint_type_row()
    cstr = contract_string_type_row()
    residual = hybrid_residual_contract_allocation_row()
    ident = identifiability_contract_allocation_row()
    library = graph_local_contract_allocation_row()
    denom = denominator_contract_allocation_row()
    schema_c = schema_pack_contract_allocation_row()
    docs = docs_join_allocation_row()
    leftover = leftover_hits_allocation_row()
    contains = library_contains_allocation_row()
    parents = graph_parent_set_allocation_row()
    defaults = default_phys_allocation_row()
    frozen = frozen_zero_allocation_row()
    smoke = smoke_vs_protocol_allocation_row()
    streaming = streaming_implicit_reuse_row()
    chunk = library_chunk_reuse_row()
    active = collect_active_allocation_row()
    stlsq_reuse = discovery_stlsq_reuse_row()
    no_compile = pack_does_not_compile_row()
    cache = allocate_cache_type_row()
    unpack_t = unpack_type_row()
    positive_t = positive_parameter_type_row()
    solver = training_solver_lock_type_row()
    hybrid_id = hybrid_identity_type_row()
    leftover_t = leftover_empty_type_row()
    docs_t = docs_executable_join_type_row()
    parent_gate = local_parent_gate_allocation_row()
    ident_t = identifiability_product_type_row()
    library_t = graph_local_type_row()
    denom_t = denominator_domain_type_row()
    schema_t = schema_pack_type_row()
    residual_t = hybrid_residual_type_row()
    docs_join_t = docs_executable_type_row()
    hl_strings = allocation_gates_hl_contract_strings_row()
    ag_helpers = allocation_gates_ag_helpers_exist_row()
    not_exported = workspace_not_exported_row()
    rhs_live = quality_gates_ude_rhs_live_row()
    return (;
        pack, unpack, schema, pos, inv, stype, remap, kinetic, den,
        split, basis, graph, extras, stlsq, implicit, rhs_l, rhs_p,
        ptype, heads, ctype, ftype, cstr, residual, ident, library,
        denom, schema_c, docs, leftover, contains, parents, defaults,
        frozen, smoke, streaming, chunk, active, stlsq_reuse,
        no_compile, cache, unpack_t, positive_t, solver, hybrid_id,
        leftover_t, docs_t, parent_gate, ident_t, library_t, denom_t,
        schema_t, residual_t, docs_join_t, hl_strings, ag_helpers,
        not_exported, rhs_live,
        holds = pack.holds && unpack.holds && schema.holds && pos.holds &&
                inv.holds && stype.holds && remap.holds && kinetic.holds &&
                den.holds && split.holds && basis.holds && graph.holds &&
                extras.holds && stlsq.holds && implicit.holds &&
                rhs_l.holds && rhs_p.holds && ptype.holds && heads.holds &&
                ctype.holds && ftype.holds && cstr.holds && residual.holds &&
                ident.holds && library.holds && denom.holds &&
                schema_c.holds && docs.holds && leftover.holds &&
                contains.holds && parents.holds && defaults.holds &&
                frozen.holds && smoke.holds && streaming.holds &&
                chunk.holds && active.holds && stlsq_reuse.holds &&
                no_compile.holds && cache.holds && unpack_t.holds &&
                positive_t.holds && solver.holds && hybrid_id.holds &&
                leftover_t.holds && docs_t.holds && parent_gate.holds &&
                ident_t.holds && library_t.holds && denom_t.holds &&
                schema_t.holds && residual_t.holds && docs_join_t.holds &&
                hl_strings.holds && ag_helpers.holds &&
                not_exported.holds && rhs_live.holds)
end

function allocation_gates_typed_matrix()
    pack = pack_parameters_allocation_row()
    pos = positive_parameter_allocation_row()
    unpack = unpack_parameters_allocation_row()
    schema = parameter_schema_allocation_row()
    return (;
        pack = allocation_gate_row_namedtuple(pack.typed),
        pos = allocation_gate_row_namedtuple(pos.typed),
        unpack = allocation_gate_row_namedtuple(unpack.typed),
        schema = allocation_gate_row_namedtuple(schema.typed),
        holds = pack.holds && pos.holds && unpack.holds && schema.holds &&
                pos.typed.hot == 0 &&
                pack.typed.limit == ALLOCATION_GATE_LIMITS.pack_parameters &&
                unpack.typed.limit == ALLOCATION_GATE_LIMITS.unpack_parameters &&
                schema.typed.limit == ALLOCATION_GATE_LIMITS.parameter_schema)
end

function format_allocation_gates_index()
    io = IOBuffer()
    println(io, "| row | meaning |")
    println(io, "|---|---|")
    println(io, "| pack | pack_parameters hot bytes |")
    println(io, "| unpack | unpack_parameters hot bytes |")
    println(io, "| schema | parameter_schema hot bytes |")
    println(io, "| positive | positive_parameter is 0 bytes |")
    println(io, "| inverse | inverse_softplus is 0 bytes |")
    println(io, "| schema_type | ParameterSchema + NamedTuple unpack |")
    println(io, "| remap_pack | remapped multi-head pack |")
    println(io, "| kinetic | :k_custom schema allocation |")
    println(io, "| den | denominator_violation_count |")
    println(io, "| split | denominator_split_counts |")
    println(io, "| basis | local_basis |")
    println(io, "| graph | graph_vs_global_library_row |")
    println(io, "| extras_den | ude_extras_denominator_row |")
    println(io, "| stlsq | STLSQWorkspace reuse |")
    println(io, "| implicit_ws | ImplicitLibraryWorkspace reuse |")
    println(io, "| rhs_linear | ude_rhs! linear cache |")
    println(io, "| rhs_p53 | ude_rhs! default example |")
    println(io, "| pack_type | ComponentVector pack |")
    println(io, "| heads | remapped MultiHeadNetwork |")
    println(io, "| cand_type | Implicit / Explicit candidates |")
    println(io, "| fingerprint | UniqueClaimFingerprint |")
    println(io, "| contracts | H–L contract strings |")
    println(io, "| residual_c | hybrid residual contract bytes |")
    println(io, "| ident_c | identifiability contract bytes |")
    println(io, "| library_c | graph-local contract bytes |")
    println(io, "| denom_c | denominator contract bytes |")
    println(io, "| schema_c | schema-pack contract bytes |")
    println(io, "| docs_join | docs_executable_join_row |")
    println(io, "| leftover | leftover_contradiction_hits |")
    println(io, "| contains | library_contains_variable |")
    println(io, "| parents | graph_parent_set |")
    println(io, "| defaults | default_phys_parameters |")
    println(io, "| frozen | _zero_frozen_phys_gradient |")
    println(io, "| smoke | 1 IC is not 9 ICs / 50 points |")
    println(io, "| streaming | StreamingImplicitWorkspace reuse |")
    println(io, "| chunk | LibraryChunkWorkspace reuse |")
    println(io, "| active | collect_active_indices! |")
    println(io, "| stlsq_reuse | discovery_workspace_alloc_report |")
    println(io, "| no_compile | pack/unpack do not compile_network |")
    println(io, "| cache | allocate_cache type |")
    println(io, "| unpack_t | unpack_parameters NamedTuple |")
    println(io, "| positive_t | positive_parameter Float64 |")
    println(io, "| solver | lock_training_solver SolverConfig |")
    println(io, "| hybrid_id | hybrid_identity_term |")
    println(io, "| leftover_t | leftover hits stay empty |")
    println(io, "| docs_t | docs_executable_join_row |")
    println(io, "| parent_gate | local_has_true_parent_gate nothing |")
    println(io, "| ident_t | identifiability product contract |")
    println(io, "| library_t | graph-local contract |")
    println(io, "| denom_t | denominator domain contract |")
    println(io, "| schema_t | schema-pack contract |")
    println(io, "| residual_t | hybrid residual contract |")
    println(io, "| docs_join_t | docs executable contract |")
    println(io, "| hl_strings | five H–L contract strings |")
    println(io, "| ag_helpers | A–G / H–L helpers exist |")
    println(io, "| not_exported | workspaces stay unexported |")
    println(io, "| rhs_live | ude_rhs! 512 / 4096 ceilings |")
    return String(take!(io))
end

function allocation_gates_index_holds()
    text = format_allocation_gates_index()
    names = allocation_gates_fixture_names()
    return length(unique(names)) == length(names) &&
           length(names) ≥ 50 &&
           occursin("hot bytes", text) &&
           occursin("9 ICs", text) &&
           occursin("StreamingImplicitWorkspace", text) &&
           occursin("ude_rhs! 512", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function format_allocation_limits()
    io = IOBuffer()
    println(io, "| helper | limit |")
    println(io, "|---|---|")
    for name in propertynames(ALLOCATION_GATE_LIMITS)
        println(io, "| ", name, " | ", getproperty(ALLOCATION_GATE_LIMITS, name), " |")
    end
    return String(take!(io))
end

function allocation_limits_hold()
    text = format_allocation_limits()
    return ALLOCATION_GATE_LIMITS.positive_parameter == 0 &&
           ALLOCATION_GATE_LIMITS.pack_parameters == 12288 &&
           ALLOCATION_GATE_LIMITS.ude_rhs_linear == 512 &&
           occursin("pack_parameters", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Docs / contract ----------------------------------------------------------

function allocation_gates_source_holds()
    src = read(allocation_gates_source_path(), String)
    docs = isfile(allocation_gates_docs_path()) ?
        read(allocation_gates_docs_path(), String) : ""
    return all(occursin(needle, src) for needle in ALLOCATION_GATES_MUST_CONTAIN) &&
           !occursin("support_f1_ude = 0.99", docs) &&
           !occursin("function validate_network", docs)
end

function allocation_gates_source_violations()
    src = read(allocation_gates_source_path(), String)
    docs = isfile(allocation_gates_docs_path()) ?
        read(allocation_gates_docs_path(), String) : ""
    missing = [s for s in ALLOCATION_GATES_MUST_CONTAIN if !occursin(s, src)]
    forbidden = String[]
    occursin("support_f1_ude = 0.99", docs) &&
        push!(forbidden, "docs: support_f1_ude = 0.99")
    occursin("function validate_network", docs) &&
        push!(forbidden, "docs: function validate_network")
    return (; missing, forbidden)
end

function allocation_gates_docs_hold()
    path = allocation_gates_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(allocation_gates_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("allocation-gates.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function allocation_gates_landing_docs_hold()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    sentences = allocation_gates_locked_sentences()
    return occursin("allocation-gates", howto) &&
           occursin("allocation_hot", howto) &&
           occursin(sentences.measured, sciml)
end

function allocation_gates_example_source_holds()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    docs = read(allocation_gates_docs_path(), String)
    return occursin("STLSQWorkspace", howto) &&
           occursin("@allocated", docs) &&
           occursin("resize_count", docs) &&
           occursin("1 IC", docs)
end

function allocation_gates_docs_mention_helpers()
    path = allocation_gates_docs_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("allocation_hot", text) &&
           occursin("pack_parameters_allocation_row", text) &&
           occursin("stlsq_workspace_reuse_row", text) &&
           occursin("positive_parameter_allocation_row", text)
end

function allocation_gates_test_file_holds()
    path = allocation_gates_test_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("allocation_gates_contract_holds", text) &&
           occursin("public_export_list_holds", text) &&
           occursin("RECOVERY_THRESHOLDS.support_f1_ude == 0.50", text)
end

function allocation_gates_module_include_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "BioDynaX.jl"), String)
    tests = read(joinpath(pkgdir(BioDynaX), "test", "runtests.jl"), String)
    return occursin("include(\"AllocationGates.jl\")", src) &&
           occursin("test_allocation_gates.jl", tests)
end

function recovery_thresholds_untouched_allocation_row()
    lock = recovery_thresholds_lock()
    return (;
        holds = RECOVERY_THRESHOLDS == lock &&
                lock.support_f1_ude == 0.50 &&
                lock.support_f1_clean == 0.99)
end

function public_export_untouched_allocation_row()
    return (;
        holds = !(:allocation_hot in names(BioDynaX)) &&
                !(:AllocationGateRow in names(BioDynaX)) &&
                !(:pack_parameters_allocation_row in names(BioDynaX)) &&
                public_export_list_holds())
end

function unique_claim_not_faster_allocation_row()
    fp = unique_claim_fingerprint()
    return (;
        n_ics = fp.n_ics,
        holds = fp.n_ics == 9 && fp.n_points == 50 &&
                fp.seed == 103 && !fp.smoke)
end

function combined_f1_not_allocation_kpi_row()
    return (;
        holds = :support_f1 ∉ UNIQUE_CLAIM_KPI_FIELDS &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50)
end

function hill_from_nn_closed_allocation_row()
    return (;
        holds = :canonical_hill_from_nn in PROTOCOL_RESULT_FIELDS &&
                RECOVERY_THRESHOLDS.support_f1_ude == 0.50)
end

function validate_open_allocation_row()
    net = build_linear_test_network()
    dual = build_dual_unknown_network()
    return (;
        holds = validate_network_stays_open_source() &&
                validate_network(net) === net &&
                validate_network(dual) === dual)
end

function allocation_gates_live_reuse_row()
    stlsq = stlsq_workspace_reuse_row()
    implicit = implicit_workspace_reuse_row()
    streaming = streaming_implicit_reuse_row()
    chunk = library_chunk_reuse_row()
    report = discovery_stlsq_reuse_row()
    return (;
        stlsq = stlsq.holds,
        implicit = implicit.holds,
        streaming = streaming.holds,
        chunk = chunk.holds,
        report = report.holds,
        holds = stlsq.holds && implicit.holds && streaming.holds &&
                chunk.holds && report.holds && stlsq.same_shape)
end

function allocation_gates_live_type_row()
    schema = schema_type_row()
    pack = pack_type_row()
    cache = allocate_cache_type_row()
    unpack = unpack_type_row()
    pos = positive_parameter_type_row()
    solver = training_solver_lock_type_row()
    hybrid = hybrid_identity_type_row()
    return (;
        holds = schema.holds && pack.holds && cache.holds &&
                unpack.holds && pos.holds && solver.holds && hybrid.holds)
end

function allocation_gates_live_surface_row()
    smoke = smoke_vs_protocol_allocation_row()
    leftover = leftover_empty_type_row()
    docs = docs_executable_join_type_row()
    hl = allocation_gates_hl_contract_strings_row()
    helpers = allocation_gates_ag_helpers_exist_row()
    exported = workspace_not_exported_row()
    compile = pack_does_not_compile_row()
    return (;
        holds = smoke.holds && leftover.holds && docs.holds &&
                hl.holds && helpers.holds && exported.holds &&
                compile.holds)
end

function format_allocation_workspace_catalog()
    io = IOBuffer()
    println(io, "| workspace | gate |")
    println(io, "|---|---|")
    println(io, "| STLSQWorkspace | stlsq_workspace_reuse_row |")
    println(io, "| ImplicitLibraryWorkspace | implicit_workspace_reuse_row |")
    println(io, "| StreamingImplicitWorkspace | streaming_implicit_reuse_row |")
    println(io, "| LibraryChunkWorkspace | library_chunk_reuse_row |")
    println(io, "| UDEModelCache | allocate_cache_type_row |")
    println(io, "| ParameterSchema | schema_type_row |")
    println(io, "| TrainingReuse | lock_training_solver |")
    println(io, "| HybridResidual | hybrid_identity_term |")
    println(io, "| GraphLocalLibrary | graph_parent_set |")
    println(io, "| DenominatorDomain | denominator_split_counts |")
    println(io, "| ParameterSchemaPack | pack_parameters / unpack_parameters |")
    println(io, "| DocsExecutable | docs_executable_join_row |")
    return String(take!(io))
end

function allocation_workspace_catalog_holds()
    text = format_allocation_workspace_catalog()
    return occursin("STLSQWorkspace", text) &&
           occursin("StreamingImplicitWorkspace", text) &&
           occursin("ParameterSchemaPack", text) &&
           occursin("DocsExecutable", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function format_allocation_gates_report()
    typed = allocation_gates_typed_matrix()
    reuse = allocation_gates_live_reuse_row()
    io = IOBuffer()
    println(io, "# Allocation gates report")
    println(io, "")
    println(io, "Measured hot-path ceilings can fail.")
    println(io, "pack hot: ", typed.pack.hot, " / ", typed.pack.limit)
    println(io, "positive hot: ", typed.pos.hot, " / ", typed.pos.limit)
    println(io, "STLSQ reuse holds: ", reuse.stlsq)
    println(io, "Discovery report holds: ", reuse.report)
    println(io, "")
    println(io, format_allocation_limits())
    println(io, "")
    println(io, format_allocation_gates_index())
    println(io, "")
    println(io, format_allocation_workspace_catalog())
    return String(take!(io))
end

function allocation_gates_report_holds()
    text = format_allocation_gates_report()
    return occursin("Allocation gates report", text) &&
           occursin("pack hot:", text) &&
           occursin("STLSQ reuse holds:", text) &&
           occursin("STLSQWorkspace", text) &&
           !occursin("support_f1_ude = 0.99", text) &&
           sizeof(text) > 400
end

function allocation_gates_unique_claim_path_holds()
    joined = join(unique_claim_user_doc_paths(), " ")
    return occursin("allocation-gates.md", joined)
end

function allocation_gates_make_listed_holds()
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    return occursin("allocation-gates.md", make)
end

function quality_gates_ude_rhs_source_holds()
    path = joinpath(pkgdir(BioDynaX), "test", "test_quality_gates.jl")
    isfile(path) || return false
    text = read(path, String)
    return occursin("hot ≤ 512", text) &&
           occursin("p53_hot ≤ 4096", text) &&
           occursin("@inferred", text)
end

function allocation_gates_changelog_holds()
    path = joinpath(pkgdir(BioDynaX), "CHANGELOG.md")
    isfile(path) || return false
    text = read(path, String)
    return occursin("AllocationGates.jl", text) &&
           occursin("allocation-gates", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function allocation_gates_news_holds()
    path = joinpath(pkgdir(BioDynaX), "NEWS.md")
    isfile(path) || return false
    text = read(path, String)
    return occursin("AllocationGates.jl", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function allocation_gates_architecture_holds()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "architecture.md"),
        String)
    sentences = allocation_gates_locked_sentences()
    return occursin("allocation-gates", text) &&
           occursin(sentences.measured, text)
end

function allocation_gates_tutorial_holds()
    text = read(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"), String)
    return occursin("allocation-gates", text) &&
           occursin("allocation_hot", text)
end

function allocation_gates_internals_hold()
    path = joinpath(pkgdir(BioDynaX), "test", "internals.jl")
    isfile(path) || return false
    text = read(path, String)
    return occursin("allocation_gates_contract_holds", text) &&
           occursin("pack_parameters_allocation_row", text) &&
           occursin("stlsq_workspace_reuse_row", text)
end

function default_example_cache_type_row()
    rng = MersenneTwister(713)
    model, packed = build_ude_model(rng, DEFAULT_EXAMPLE_NETWORK)
    cache = allocate_cache(model, Float64)
    return (;
        n = length(cache.du),
        holds = cache isa UDEModelCache &&
                length(cache.du) == model.compiled.nstates &&
                packed isa ComponentVector)
end

function remapped_cache_type_row()
    net = build_remapped_two_regulator_network()
    model, packed = build_ude_model(MersenneTwister(715), net)
    cache = allocate_cache(model, Float64)
    schema = parameter_schema(model)
    return (;
        heads = schema.nn_heads,
        holds = cache isa UDEModelCache &&
                schema.nn_heads == 2 &&
                packed isa ComponentVector)
end

function kinetic_cache_type_row()
    net = build_kinetic_generalization_network()
    model, packed = build_ude_model(MersenneTwister(717), net)
    schema = parameter_schema(model)
    return (;
        has_custom = :k_custom in schema.phys_names,
        holds = :k_custom in schema.phys_names &&
                packed isa ComponentVector)
end

function zero_hole_dummy_head_not_schema_row()
    dummy = dummy_head_on_zero_hole_row()
    return (;
        schema_heads = dummy.schema_heads,
        holds = dummy.holds && dummy.schema_heads == 0)
end

function allocation_gates_contract_holds()
    return allocation_gates_source_holds() &&
           allocation_hot_source_holds() &&
           stlsq_reuse_source_holds() &&
           discovery_report_source_holds() &&
           quality_gates_test_still_holds() &&
           quality_gates_extended_source_holds() &&
           quality_gates_ude_rhs_source_holds() &&
           allocation_gates_docs_hold() &&
           allocation_gates_landing_docs_hold() &&
           allocation_gates_example_source_holds() &&
           allocation_gates_docs_mention_helpers() &&
           allocation_gates_index_holds() &&
           allocation_gates_test_file_holds() &&
           allocation_gates_module_include_holds() &&
           allocation_gates_unique_claim_path_holds() &&
           allocation_gates_make_listed_holds() &&
           allocation_gates_changelog_holds() &&
           allocation_gates_news_holds() &&
           allocation_gates_architecture_holds() &&
           allocation_gates_tutorial_holds() &&
           allocation_gates_internals_hold() &&
           allocation_workspace_catalog_holds() &&
           allocation_gates_report_holds() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source() &&
           recovery_thresholds_untouched_allocation_row().holds &&
           public_export_untouched_allocation_row().holds &&
           unique_claim_not_faster_allocation_row().holds &&
           combined_f1_not_allocation_kpi_row().holds &&
           hill_from_nn_closed_allocation_row().holds &&
           validate_open_allocation_row().holds &&
           allocation_limits_hold() &&
           allocation_gates_typed_matrix().holds &&
           allocation_gates_live_reuse_row().holds &&
           allocation_gates_live_type_row().holds &&
           allocation_gates_live_surface_row().holds &&
           zero_hole_dummy_head_not_schema_row().holds
end

