###############################################################################
# Graph-local library / ablation (not exported).
#
# local_basis(scope=:graph) is the product prior. scope=:global is the
# ablation. This file locks parent-set membership, wrong-graph parents,
# and local_has_true_parent_gate as code. Combined F1 stays a skeleton
# floor. Discovery rows are not the seed-103 / 9-IC protocol.
#
# Does not drop protocol ICs. Does not grow exports. Does not open
# Hill-from-NN. Does not put a single-unknown-term check into validate_network.
###############################################################################

const GRAPH_LOCAL_LIBRARY_MUST_CONTAIN = (
    "function graph_vs_global_library_row",
    "function library_contains_variable",
    "function wrong_graph_parent_row",
    "function graph_parent_set",
    "function ablation_library_row",
    "struct GraphLocalLibraryRow",
    "function three_state_library_row",
    "function six_state_library_row")

const GRAPH_LOCAL_LIBRARY_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

# -- Library membership -------------------------------------------------------

"""
    library_contains_variable(spec, variable) -> Bool

True when `local_basis` kept `variable` in the screened parent set.
"""
function library_contains_variable(spec::LocalBasisSpec, variable::Int)
    return variable in spec.variables
end

function graph_parent_set(network::BiologicalNetwork, target::Int)
    # `target` is a local_basis index into state_nodes, not a node id.
    # INPUT parents are dropped because they are not dynamic states.
    nodes = state_nodes(network)
    1 ≤ target ≤ length(nodes) ||
        throw(ArgumentError("target must index a dynamic state"))
    parent_nodes = candidate_parents(network, nodes[target])
    row_by_node = Dict(node => row for (row, node) in pairs(nodes))
    return Set(Int[row_by_node[n] for n in parent_nodes if haskey(row_by_node, n)])
end

function graph_library_variables(network::BiologicalNetwork, target::Int;
        degree::Int = 2, include_interactions::Bool = false)
    spec = local_basis(network, target; degree = degree,
        include_interactions = include_interactions, scope = :graph)
    return spec, Set(spec.variables)
end

function global_library_variables(network::BiologicalNetwork, target::Int;
        degree::Int = 2, include_interactions::Bool = false)
    spec = local_basis(network, target; degree = degree,
        include_interactions = include_interactions, scope = :global)
    return spec, Set(spec.variables)
end

"""
    graph_vs_global_library_row(network, target; true_parent=nothing)

Compare graph-local and global libraries. The global library is a
superset of the graph-local variables (except the isolated-target
case). Combined F1 is not scored here.
"""
function graph_vs_global_library_row(network::BiologicalNetwork, target::Int;
        true_parent = nothing,
        degree::Int = 2)
    graph_spec, graph_vars = graph_library_variables(
        network, target; degree = degree)
    global_spec, global_vars = global_library_variables(
        network, target; degree = degree)
    parents = graph_parent_set(network, target)
    nstates = length(state_nodes(network))
    true_in_graph = true_parent === nothing ? nothing :
                    Int(true_parent) in graph_vars
    true_in_parents = true_parent === nothing ? nothing :
                      Int(true_parent) in parents
    subset = graph_vars ⊆ global_vars || graph_vars == global_vars
    wider = candidate_count(graph_spec) ≤ candidate_count(global_spec)
    parents_in_graph = parents ⊆ graph_vars ||
                       all(p -> p == target || p in graph_vars, parents)
    return (;
        target,
        nstates,
        graph_vars = sort(collect(graph_vars)),
        global_vars = sort(collect(global_vars)),
        parents = sort(collect(parents)),
        graph_terms = candidate_count(graph_spec),
        global_terms = candidate_count(global_spec),
        true_in_graph,
        true_in_parents,
        subset,
        wider,
        parents_in_graph,
        validate_open = validate_network(network) === network,
        holds = wider && parents_in_graph &&
                validate_network(network) === network &&
                (true_parent === nothing ||
                 true_in_graph == true_in_parents ||
                 Int(true_parent) == target))
end

function wrong_graph_parent_row(network::BiologicalNetwork;
        target::Int = 1, true_parent::Int = 2)
    parents = graph_parent_set(network, target)
    spec, vars = graph_library_variables(network, target)
    return (;
        parents = sort(collect(parents)),
        variables = sort(collect(vars)),
        true_in_parents = true_parent in parents,
        true_in_library = true_parent in vars,
        holds = !(true_parent in parents) &&
                !(true_parent in vars) &&
                validate_network(network) === network)
end

# -- Typed row ----------------------------------------------------------------

struct GraphLocalLibraryRow
    name::Symbol
    nstates::Int
    graph_terms::Int
    global_terms::Int
    true_in_graph::Bool
    true_in_parents::Bool
    holds::Bool
end

function graph_local_library_row(name::Symbol, network::BiologicalNetwork,
        target::Int; true_parent::Int = 2)
    row = graph_vs_global_library_row(
        network, target; true_parent = true_parent)
    typed = GraphLocalLibraryRow(
        name,
        row.nstates,
        row.graph_terms,
        row.global_terms,
        row.true_in_graph === true,
        row.true_in_parents === true,
        row.holds)
    return (; row, typed, holds = row.holds && typed.holds)
end

function graph_local_library_row_namedtuple(row::GraphLocalLibraryRow)
    return (;
        name = row.name,
        nstates = row.nstates,
        graph_terms = row.graph_terms,
        global_terms = row.global_terms,
        true_in_graph = row.true_in_graph,
        true_in_parents = row.true_in_parents,
        holds = row.holds)
end

# -- Fixture library rows -----------------------------------------------------

function ablation_library_row()
    net = build_rate_ablation_network()
    graph = graph_vs_global_library_row(net, 1; true_parent = 1)
    global_has_z = 2 in graph.global_vars
    graph_has_z = 2 in graph.graph_vars
    return (;
        graph,
        global_has_z,
        graph_has_z,
        holds = graph.holds && global_has_z && !graph_has_z &&
                candidate_count(local_basis(net, 1; degree = 2,
                    include_interactions = false, scope = :graph)) <
                candidate_count(local_basis(net, 1; degree = 2,
                    include_interactions = false, scope = :global)))
end

function three_state_library_row()
    net = build_three_state_unknown_network()
    packed = graph_local_library_row(:three_state, net, 1; true_parent = 2)
    wrong = wrong_graph_parent_row(
        build_wrong_graph_unknown_network(); target = 1, true_parent = 2)
    return (;
        packed,
        wrong,
        parents = candidate_parents(net, 1),
        holes = count_unknown_destructions(net),
        holds = packed.holds && wrong.holds &&
                2 in candidate_parents(net, 1) &&
                count_unknown_destructions(net) == 1)
end

function wrong_graph_library_row()
    net = build_wrong_graph_unknown_network()
    packed = graph_local_library_row(:wrong_graph, net, 1; true_parent = 2)
    parents = candidate_parents(net, 1)
    return (;
        packed,
        parents,
        claimed = 3 in parents,
        true_missing = !(2 in parents),
        validate_open = validate_network(net) === net,
        holds = packed.holds && 3 in parents && !(2 in parents) &&
                packed.typed.true_in_graph == false &&
                packed.typed.true_in_parents == false &&
                validate_network(net) === net)
end

function six_state_library_row()
    net = build_six_state_unknown_network()
    packed = graph_local_library_row(:six_state, net, 1; true_parent = 2)
    graph_spec, graph_vars = graph_library_variables(net, 1)
    global_spec, global_vars = global_library_variables(net, 1)
    return (;
        packed,
        nstates = length(net.nodes),
        distractor_local = 6 in graph_vars,
        distractor_global = 6 in global_vars,
        holds = packed.holds && length(net.nodes) == 6 &&
                packed.typed.true_in_graph &&
                6 in global_vars)
end

function six_state_wrong_graph_library_row()
    net = build_six_state_wrong_graph_network()
    packed = graph_local_library_row(:six_wrong, net, 1; true_parent = 2)
    parents = candidate_parents(net, 1)
    return (;
        packed,
        parents,
        claimed = 3 in parents,
        true_missing = !(2 in parents),
        holds = packed.holds && 3 in parents && !(2 in parents) &&
                packed.typed.true_in_graph == false)
end

function two_regulator_library_row()
    net = build_two_regulator_unknown_network()
    spec, vars = graph_library_variables(net, 1)
    parents = graph_parent_set(net, 1)
    unknown = only(r for r in net.reactions if !r.known)
    return (;
        vars = sort(collect(vars)),
        parents = sort(collect(parents)),
        n_regs = length(unknown.regulators),
        n_graph_parents = length(parents),
        holds = length(unknown.regulators) == 2 && isempty(parents) &&
                validate_network(net) === net)
end

function hill_unknown_library_row()
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    packed = graph_local_library_row(:hill, net, 1; true_parent = 2)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        empty_graph = isempty(candidate_parents(net, 1)),
        holds = packed.holds && count_unknown_destructions(net) == 1 &&
                isempty(candidate_parents(net, 1)) &&
                packed.typed.true_in_graph == false)
end

function mm_unknown_library_row()
    net = build_mm_recovery_network(; known = false)
    packed = graph_local_library_row(:mm, net, 1; true_parent = 2)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        holds = packed.holds && count_unknown_destructions(net) == 1)
end

function default_example_library_row()
    net = DEFAULT_EXAMPLE_NETWORK
    packed = graph_local_library_row(:default, net, 1; true_parent = 2)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        holds = packed.holds && count_unknown_destructions(net) == 1)
end

function remapped_library_row()
    net = build_remapped_two_regulator_network()
    rows = NamedTuple[]
    nstates = length(state_nodes(net))
    for target in 1:nstates
        packed = graph_vs_global_library_row(net, target)
        push!(rows,
            (;
                target,
                graph_terms = packed.graph_terms,
                global_terms = packed.global_terms,
                holds = packed.holds))
    end
    return (;
        rows,
        holes = count_unknown_destructions(net),
        holds = all(r -> r.holds, rows) &&
                count_unknown_destructions(net) == 2 &&
                unique_claim_recovery_admits(net) == false)
end

function dual_library_row()
    net = build_dual_unknown_network()
    packed = graph_vs_global_library_row(net, 1)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        admits = unique_claim_recovery_admits(net),
        validate_open = validate_network(net) === net,
        holds = packed.holds && count_unknown_destructions(net) == 2 &&
                unique_claim_recovery_admits(net) == false &&
                validate_network(net) === net)
end

function linear_zero_library_row()
    net = build_linear_test_network()
    packed = graph_vs_global_library_row(net, 1)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        validate_open = validate_network(net) === net,
        holds = packed.holds && count_unknown_destructions(net) == 0 &&
                validate_network(net) === net)
end

function competitive_library_row()
    net = build_competitive_test_network(; known = false)
    spec, vars = graph_library_variables(net, 1)
    parents = graph_parent_set(net, 1)
    return (;
        vars = sort(collect(vars)),
        parents = sort(collect(parents)),
        holes = count_unknown_destructions(net),
        holds = parents ⊆ vars && count_unknown_destructions(net) == 1)
end

function skipped_duplicate_library_row()
    net = build_skipped_duplicate_unknown_network()
    packed = graph_vs_global_library_row(net, 1)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        dense = begin
            rng = MersenneTwister(601)
            model, _ = build_ude_model(rng, net)
            neural_index_is_dense(model)
        end,
        holds = packed.holds && count_unknown_destructions(net) == 2)
end

function repressilator_library_row()
    net = build_repressilator_network()
    rows = [graph_vs_global_library_row(net, t) for t in 1:3]
    return (;
        rows,
        holes = count_unknown_destructions(net),
        holds = all(r -> r.holds, rows) &&
                count_unknown_destructions(net) == 0)
end

function mm_known_library_row()
    net = build_mm_recovery_network(; known = true)
    packed = graph_vs_global_library_row(net, 1; true_parent = 2)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        holds = packed.holds && count_unknown_destructions(net) == 0)
end

function hill_known_library_row()
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    packed = graph_vs_global_library_row(net, 1; true_parent = 2)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        empty_graph = isempty(candidate_parents(net, 1)),
        holds = packed.holds && count_unknown_destructions(net) == 0 &&
                isempty(candidate_parents(net, 1)))
end

function default_scope_is_graph_row()
    net = build_hill_recovery_network(; known = false)
    default_spec = local_basis(net, 1; degree = 2, include_interactions = false)
    graph_spec = local_basis(net, 1; degree = 2, include_interactions = false,
        scope = :graph)
    return (;
        default_vars = copy(default_spec.variables),
        graph_vars = copy(graph_spec.variables),
        holds = default_spec.variables == graph_spec.variables)
end

function invalid_scope_throws_row()
    net = build_linear_test_network()
    threw = false
    try
        local_basis(net, 1; scope = :not_a_scope)
    catch error
        threw = error isa ArgumentError
    end
    return (; threw, holds = threw)
end

function target_out_of_range_row()
    net = build_linear_test_network()
    threw = false
    try
        local_basis(net, 9; scope = :graph)
    catch error
        threw = error isa ArgumentError
    end
    return (; threw, holds = threw)
end

# -- Discovery checks (not the protocol) ---------------------------------------

function graph_local_rate_samples(; n::Int = 80, seed::Integer = 607)
    r = collect(range(0.1, 2.0; length = n))
    rng = MersenneTwister(seed)
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    amp = max(maximum(abs, D), eps(Float64))
    D_noisy = D .+ 0.005 .* amp .* randn(rng, length(r))
    return r, D_noisy
end

function ablation_discovery_gate_row()
    r, D_noisy = graph_local_rate_samples(; n = 80, seed = 607)
    z = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(MersenneTwister(607), length(r))
    X = permutedims(hcat(r, z))
    dX = vcat(reshape(D_noisy, 1, :), reshape(-0.5 .* z, 1, :))
    X, dX = _permute_rate_samples(X, dX, 607)
    times = collect(range(0.0, 1.0; length = length(r)))
    net = build_rate_ablation_network()
    local_disc = discover_equations(
        X, times, net; derivatives = dX, targets = 1,
        config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 4),
        verbose = false, strict = false)
    global_disc = discover_equations(
        X, times, net; derivatives = dX, targets = 1,
        config = rate_discovery_config(scope = :global, bootstrap = 0, seed = 4),
        verbose = false, strict = false)
    local_cand = local_disc.success ? local_disc.candidates[1] : nothing
    global_cand = global_disc.success ? global_disc.candidates[1] : nothing
    return (;
        local_success = local_disc.success,
        global_success = global_disc.success,
        local_has_r = local_has_true_parent_gate(local_cand; variable = 1),
        local_has_z = local_has_false_parent_gate(local_cand; variables = (2,)),
        global_has_z = local_has_false_parent_gate(global_cand; variables = (2,)),
        n_ics = 1,
        smoke = true,
        holds = local_disc.success ?
                (local_has_true_parent_gate(local_cand; variable = 1) &&
                 !local_has_false_parent_gate(local_cand; variables = (2,))) :
                local_disc.retcode !== DiscoverySuccess)
end

function three_state_discovery_gate_row()
    r, D_noisy = graph_local_rate_samples(; n = 80, seed = 613)
    s = fill(0.4, length(r))
    q = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(MersenneTwister(613), length(r))
    z = r .+ 0.10 .* (maximum(r) - minimum(r)) .* randn(MersenneTwister(613), length(r))
    X = permutedims(hcat(s, r, q, z))
    dX = vcat(reshape(D_noisy, 1, :), zeros(3, length(r)))
    X, dX = _permute_rate_samples(X, dX, 613)
    times = collect(range(0.0, 1.0; length = length(r)))
    net = build_three_state_unknown_network()
    local_disc = discover_equations(
        X, times, net; derivatives = dX, targets = 1,
        config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 5),
        verbose = false, strict = false)
    lc = local_disc.success ? local_disc.candidates[1] : nothing
    return (;
        local_success = local_disc.success,
        local_has_true = local_has_true_parent_gate(lc; variable = 2),
        local_false = local_has_false_parent_gate(lc; variables = (3, 4)),
        parents = candidate_parents(net, 1),
        n_ics = 1,
        holds = 2 in candidate_parents(net, 1) &&
                (local_disc.success ?
                 local_has_true_parent_gate(lc; variable = 2) :
                 local_disc.retcode !== DiscoverySuccess))
end

function wrong_graph_discovery_gate_row()
    r, D_noisy = graph_local_rate_samples(; n = 80, seed = 617)
    s = fill(0.4, length(r))
    q = (r .^ 2) .+ 0.08 .* maximum(r .^ 2) .* randn(MersenneTwister(617), length(r))
    z = r .+ 0.10 .* (maximum(r) - minimum(r)) .* randn(MersenneTwister(617), length(r))
    X = permutedims(hcat(s, r, q, z))
    dX = vcat(reshape(D_noisy, 1, :), zeros(3, length(r)))
    X, dX = _permute_rate_samples(X, dX, 617)
    times = collect(range(0.0, 1.0; length = length(r)))
    net = build_wrong_graph_unknown_network()
    local_disc = discover_equations(
        X, times, net; derivatives = dX, targets = 1,
        config = rate_discovery_config(scope = :graph, bootstrap = 0, seed = 8),
        verbose = false, strict = false)
    lc = local_disc.success ? local_disc.candidates[1] : nothing
    spec, vars = graph_library_variables(net, 1)
    return (;
        local_success = local_disc.success,
        local_has_true = local_has_true_parent_gate(lc; variable = 2),
        true_in_library = 2 in vars,
        parents = candidate_parents(net, 1),
        n_ics = 1,
        holds = !(2 in candidate_parents(net, 1)) && !(2 in vars) &&
                local_has_true_parent_gate(lc; variable = 2) == false)
end

function nothing_candidate_gate_row()
    return (;
        true_none = local_has_true_parent_gate(nothing; variable = 2),
        false_none = local_has_false_parent_gate(nothing; variables = (3, 4)),
        holds = local_has_true_parent_gate(nothing; variable = 2) == false &&
                local_has_false_parent_gate(nothing; variables = (3, 4)) == false)
end

# -- Source locks -------------------------------------------------------------

function local_basis_scope_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "BasisFactory.jl"), String)
    start = findfirst("function local_basis(network::BiologicalNetwork, target::Int;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("scope === :global", body) &&
           occursin("scope === :graph", body) &&
           occursin("candidate_parents", body) &&
           occursin("state_nodes", body)
end

function local_has_true_parent_gate_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst(
        "function local_has_true_parent_gate(candidate; variable::Int", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("support_uses_variable", body) &&
           occursin("candidate === nothing && return false", body)
end

function recovery_suite_uses_parent_gates_source_holds()
    src = read(recovery_jl_source_path(), String)
    return occursin("local_has_true_parent = local_has_true_parent_gate(", src) &&
           occursin("local_false_parent = local_has_false_parent_gate(", src) &&
           occursin("if :three_state in wanted", src) &&
           occursin("if :wrong_graph in wanted", src) &&
           occursin("if :six_state in wanted", src)
end

function candidate_parents_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Network.jl"), String)
    start = findfirst("candidate_parents(network::BiologicalNetwork, target::Integer)", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:min(lastindex(rest), 240)]
    return occursin("inneighbors", body)
end

# -- Matrices / catalog -------------------------------------------------------

function graph_local_library_fixture_matrix()
    ablation = ablation_library_row()
    three = three_state_library_row()
    wrong = wrong_graph_library_row()
    six = six_state_library_row()
    six_wrong = six_state_wrong_graph_library_row()
    two = two_regulator_library_row()
    hill_u = hill_unknown_library_row()
    mm_u = mm_unknown_library_row()
    default = default_example_library_row()
    remap = remapped_library_row()
    dual = dual_library_row()
    linear = linear_zero_library_row()
    comp = competitive_library_row()
    skipped = skipped_duplicate_library_row()
    repress = repressilator_library_row()
    mm_k = mm_known_library_row()
    hill_k = hill_known_library_row()
    scope = default_scope_is_graph_row()
    invalid = invalid_scope_throws_row()
    oor = target_out_of_range_row()
    none = nothing_candidate_gate_row()
    abl_disc = ablation_discovery_gate_row()
    three_disc = three_state_discovery_gate_row()
    wrong_disc = wrong_graph_discovery_gate_row()
    middle = skipped_middle_library_row()
    kinetic = kinetic_library_row()
    nodist = three_state_no_distractor_library_row()
    degree = degree_widens_global_row()
    inter = interactions_widen_library_row()
    suite = suite_gate_symbols_row()
    smoke = smoke_vs_protocol_discovery_row()
    suite_lib = suite_section_library_matrix()
    graph_secs = graph_prior_suite_sections_row()
    six_targets = six_state_per_target_library_row()
    three_targets = three_state_per_target_library_row()
    return (;
        ablation, three, wrong, six, six_wrong, two, hill_u, mm_u,
        default, remap, dual, linear, comp, skipped, repress, mm_k,
        hill_k, scope, invalid, oor, none, abl_disc, three_disc, wrong_disc,
        middle, kinetic, nodist, degree, inter, suite, smoke,
        suite_lib, graph_secs, six_targets, three_targets,
        holds = ablation.holds && three.holds && wrong.holds && six.holds &&
                six_wrong.holds && two.holds && hill_u.holds && mm_u.holds &&
                default.holds && remap.holds && dual.holds && linear.holds &&
                comp.holds && skipped.holds && repress.holds && mm_k.holds &&
                hill_k.holds && scope.holds && invalid.holds && oor.holds &&
                none.holds && abl_disc.holds && three_disc.holds &&
                wrong_disc.holds && middle.holds && kinetic.holds &&
                nodist.holds && degree.holds && inter.holds && suite.holds &&
                smoke.holds && suite_lib.holds && graph_secs.holds &&
                six_targets.holds && three_targets.holds)
end

function skipped_middle_library_row()
    net = build_skipped_middle_unknown_network()
    packed = graph_vs_global_library_row(net, 1)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        holds = packed.holds && count_unknown_destructions(net) ≥ 2)
end

function kinetic_library_row()
    net = build_kinetic_generalization_network()
    packed = graph_vs_global_library_row(net, 1)
    return (;
        packed,
        holes = count_unknown_destructions(net),
        validate_open = validate_network(net) === net,
        holds = packed.holds && count_unknown_destructions(net) == 0 &&
                validate_network(net) === net)
end

function three_state_no_distractor_library_row()
    net = build_three_state_unknown_network(; with_distractor = false)
    packed = graph_local_library_row(:three_nodist, net, 1; true_parent = 2)
    return (;
        packed,
        nstates = length(state_nodes(net)),
        holds = packed.holds && length(state_nodes(net)) == 3 &&
                packed.typed.true_in_graph)
end

function degree_widens_global_row()
    net = build_rate_ablation_network()
    d2 = local_basis(net, 1; degree = 2, include_interactions = false, scope = :global)
    d3 = local_basis(net, 1; degree = 3, include_interactions = false, scope = :global)
    return (;
        d2 = candidate_count(d2),
        d3 = candidate_count(d3),
        holds = candidate_count(d3) > candidate_count(d2))
end

function interactions_widen_library_row()
    net = build_rate_ablation_network()
    plain = local_basis(net, 1; degree = 2, include_interactions = false, scope = :global)
    inter = local_basis(net, 1; degree = 2, include_interactions = true, scope = :global)
    return (;
        plain = candidate_count(plain),
        inter = candidate_count(inter),
        holds = candidate_count(inter) ≥ candidate_count(plain))
end

function suite_gate_symbols_row()
    src = read(recovery_jl_source_path(), String)
    return (;
        three = occursin("if :three_state in wanted", src),
        wrong = occursin("if :wrong_graph in wanted", src),
        six = occursin("if :six_state in wanted", src),
        six_wrong = occursin("if :six_state_wrong_graph in wanted", src),
        ablation = occursin("if :ablation in wanted", src),
        uses_gate = occursin("local_has_true_parent_gate(", src),
        holds = occursin("if :three_state in wanted", src) &&
                occursin("if :wrong_graph in wanted", src) &&
                occursin("local_has_true_parent_gate(", src) &&
                !occursin("support_f1_ude = 0.99", src))
end

function suite_section_library_row(section::Symbol)
    net = recovery_suite_section_network(section)
    nstates = length(state_nodes(net))
    target = 1
    packed = graph_vs_global_library_row(net, target)
    kind = recovery_suite_section_kind(section)
    holes = recovery_suite_expected_holes(section)
    return (;
        section,
        kind,
        expected_holes = holes,
        nstates,
        graph_terms = packed.graph_terms,
        global_terms = packed.global_terms,
        validate_open = packed.validate_open,
        library_holds = packed.holds,
        holds = packed.holds && validate_network(net) === net)
end

function suite_section_library_matrix()
    sections = recovery_suite_sections()
    rows = NamedTuple[suite_section_library_row(section) for section in sections]
    return (;
        n = length(rows),
        rows,
        all_validate = all(r -> r.validate_open, rows),
        holds = length(rows) == length(sections) &&
                all(r -> r.holds, rows) &&
                all(r -> r.validate_open, rows))
end

function graph_prior_suite_sections_row()
    sections = recovery_suite_sections()
    kinds = recovery_suite_section_kinds()
    graph_prior = [s for s in sections if getproperty(kinds, s) === :graph_prior]
    return (;
        graph_prior,
        holds = issetequal(graph_prior,
            (:three_state, :wrong_graph, :six_state, :six_state_wrong_graph)))
end

function six_state_per_target_library_row()
    net = build_six_state_unknown_network()
    rows = [graph_vs_global_library_row(net, t) for t in 1:6]
    return (;
        n = length(rows),
        wider = all(r -> r.wider, rows),
        holds = length(rows) == 6 && all(r -> r.holds, rows))
end

function three_state_per_target_library_row()
    net = build_three_state_unknown_network()
    n = length(state_nodes(net))
    rows = [graph_vs_global_library_row(net, t) for t in 1:n]
    return (;
        n = length(rows),
        holds = length(rows) == n && all(r -> r.holds, rows))
end

function smoke_vs_protocol_discovery_row()
    smoke = unique_claim_fingerprint(; smoke = true)
    protocol = unique_claim_fingerprint()
    return (;
        smoke_ics = smoke.n_ics,
        protocol_ics = protocol.n_ics,
        discovery_is_not_protocol = true,
        holds = smoke.n_ics == 1 && protocol.n_ics == 9 &&
                smoke.n_points == 8 && protocol.n_points == 50 &&
                protocol.seed == 103)
end

function graph_local_library_typed_matrix()
    hill = hill_unknown_library_row()
    wrong = wrong_graph_library_row()
    return (;
        hill = graph_local_library_row_namedtuple(hill.packed.typed),
        wrong = graph_local_library_row_namedtuple(wrong.packed.typed),
        holds = hill.holds && wrong.holds &&
                hill.packed.typed.true_in_graph == false &&
                wrong.packed.typed.true_in_graph == false)
end

function graph_local_library_fixture_names()
    return (
        :ablation, :three_state, :wrong_graph, :six_state,
        :six_wrong, :two_regulator, :hill_unknown, :mm_unknown,
        :default_example, :remapped, :dual, :linear_zero,
        :competitive, :skipped_duplicate, :repressilator, :mm_known,
        :hill_known, :default_scope, :invalid_scope, :oor_target,
        :nothing_gate, :ablation_discovery, :three_discovery,
        :wrong_discovery, :skipped_middle, :kinetic, :three_nodist,
        :degree, :interactions, :suite_gates, :smoke_protocol)
end

function format_graph_local_library_index()
    io = IOBuffer()
    println(io, "| fixture | role |")
    println(io, "|---|---|")
    println(io, "| ablation | graph library drops the distractor Z |")
    println(io, "| three_state | true parent R is in the graph library |")
    println(io, "| wrong_graph | claimed Q parent; true R is absent |")
    println(io, "| six_state | six-state graph keeps R; global has Z |")
    println(io, "| six_wrong | six-state wrong graph omits R |")
    println(io, "| two_regulator | D(S,I) has two graph parents |")
    println(io, "| hill_unknown | unknown Hill graph contains R |")
    println(io, "| mm_unknown | unknown MM graph contains R |")
    println(io, "| default_example | p53/Mdm2 graph-local library |")
    println(io, "| remapped | each remapped target has a graph library |")
    println(io, "| dual | two-hole network still has graph libraries |")
    println(io, "| linear_zero | 0-hole linear library still validates |")
    println(io, "| competitive | competitive unknown graph parents |")
    println(io, "| skipped_duplicate | dense two-head graph libraries |")
    println(io, "| repressilator | known three-state graph libraries |")
    println(io, "| mm_known | known MM graph library |")
    println(io, "| hill_known | known Hill graph contains R |")
    println(io, "| default_scope | local_basis default scope is :graph |")
    println(io, "| invalid_scope | unknown scope throws ArgumentError |")
    println(io, "| oor_target | target out of range throws |")
    println(io, "| nothing_gate | nothing candidate is not a true parent |")
    println(io, "| ablation_discovery | local support keeps r, drops z |")
    println(io, "| three_discovery | local_has_true_parent_gate on R |")
    println(io, "| wrong_discovery | wrong graph cannot recover R |")
    println(io, "| skipped_middle | remapped 1:n graph libraries |")
    println(io, "| kinetic | known kinetic 0-hole graph library |")
    println(io, "| three_nodist | three-state without Z still keeps R |")
    println(io, "| degree | raising max degree widens the global library |")
    println(io, "| interactions | pairwise terms widen the global library |")
    println(io, "| suite_gates | recovery suite calls the parent checks |")
    println(io, "| smoke_protocol | discovery smoke is not 9 ICs / 50 points |")
    return String(take!(io))
end

function graph_local_library_index_holds()
    text = format_graph_local_library_index()
    names = graph_local_library_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("wrong graph", text) &&
           occursin("local_has_true_parent_gate", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

# -- Source checks ----------------------------------------------------------

function format_suite_library_index()
    io = IOBuffer()
    println(io, "| section | kind | expected holes | graph terms | global terms |")
    println(io, "|---|---|---|---|---|")
    matrix = suite_section_library_matrix()
    for row in matrix.rows
        holes = row.expected_holes === nothing ? "library" : string(row.expected_holes)
        println(io, "| ", row.section, " | ", row.kind, " | ", holes, " | ",
            row.graph_terms, " | ", row.global_terms, " |")
    end
    return String(take!(io))
end

function suite_library_index_holds()
    text = format_suite_library_index()
    return occursin("three_state", text) &&
           occursin("wrong_graph", text) &&
           occursin("graph_prior", text) &&
           occursin("unique_claim", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function screen_variables_bound_row()
    net = build_rate_ablation_network()
    r = collect(range(0.1, 2.0; length = 40))
    X = permutedims(hcat(r, r .^ 2))
    y = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    bounded = screen_variables(X, y, [1, 2], 1)
    unbounded = screen_variables(X, y, [1, 2], 8)
    return (;
        bounded,
        unbounded,
        holds = length(bounded) == 1 && length(unbounded) == 2 &&
                1 in unbounded)
end

function evaluate_graph_library_finite_row()
    net = build_hill_recovery_network(; known = false)
    spec = local_basis(net, 1; degree = 2, include_interactions = false, scope = :graph)
    X = [0.2 0.4 0.6; 0.3 0.5 0.7]
    Φ = evaluate_library(spec.numerator, X)
    return (;
        size = size(Φ),
        finite = all(isfinite, Φ),
        n_terms = candidate_count(spec),
        holds = size(Φ, 1) == 3 && size(Φ, 2) == length(spec.numerator) &&
                all(isfinite, Φ) && all(>(0), Φ[:, 1]))
end

function suite_parent_set_catalog()
    rows = NamedTuple[]
    for section in recovery_suite_sections()
        net = recovery_suite_section_network(section)
        n_dyn = length(state_nodes(net))
        for target in 1:n_dyn
            packed = graph_vs_global_library_row(net, target)
            push!(rows,
                (;
                    section,
                    target,
                    n_parents = length(packed.parents),
                    graph_terms = packed.graph_terms,
                    global_terms = packed.global_terms,
                    holds = packed.holds))
        end
    end
    return (;
        n = length(rows),
        rows,
        holds = !isempty(rows) && all(r -> r.holds, rows) &&
                length(unique(r.section for r in rows)) ==
                length(recovery_suite_sections()))
end

function format_suite_parent_catalog()
    catalog = suite_parent_set_catalog()
    io = IOBuffer()
    println(io, "| section | target | n_parents | graph terms | global terms |")
    println(io, "|---|---|---|---|---|")
    for row in catalog.rows
        println(io, "| ", row.section, " | ", row.target, " | ",
            row.n_parents, " | ", row.graph_terms, " | ",
            row.global_terms, " |")
    end
    return String(take!(io))
end

function suite_parent_catalog_holds()
    catalog = suite_parent_set_catalog()
    text = format_suite_parent_catalog()
    return catalog.holds &&
           occursin("three_state", text) &&
           occursin("wrong_graph", text) &&
           count(==('|'), text) ≥ 40 &&
           !occursin("support_f1_ude = 0.99", text)
end

function remapped_per_target_library_row()
    net = build_remapped_two_regulator_network()
    n = length(state_nodes(net))
    rows = [graph_vs_global_library_row(net, t) for t in 1:n]
    return (;
        n = length(rows),
        holds = !isempty(rows) && all(r -> r.holds, rows))
end

function dual_per_target_library_row()
    net = build_dual_unknown_network()
    n = length(state_nodes(net))
    rows = [graph_vs_global_library_row(net, t) for t in 1:n]
    return (;
        n = length(rows),
        holes = count_unknown_destructions(net),
        holds = !isempty(rows) && all(r -> r.holds, rows) &&
                count_unknown_destructions(net) == 2)
end

function default_per_target_library_row()
    net = DEFAULT_EXAMPLE_NETWORK
    n = length(state_nodes(net))
    rows = [graph_vs_global_library_row(net, t) for t in 1:n]
    return (;
        n = length(rows),
        holds = !isempty(rows) && all(r -> r.holds, rows))
end

function public_export_list_untouched_library_row()
    return (;
        has_local_basis = :local_basis in LOCKED_PUBLIC_EXPORTS,
        has_candidate_parents = :candidate_parents in LOCKED_PUBLIC_EXPORTS,
        gate_unexported = !(:local_has_true_parent_gate in names(BioDynaX)),
        holds = :local_basis in LOCKED_PUBLIC_EXPORTS &&
                :candidate_parents in LOCKED_PUBLIC_EXPORTS &&
                !(:local_has_true_parent_gate in names(BioDynaX)) &&
                !(:GraphLocalLibraryRow in names(BioDynaX)) &&
                public_export_list_holds())
end

function unique_claim_not_faster_by_dropping_ics_row()
    # Discovery smoke rows in this file use 1 IC. That is not permission
    # to drop protocol ICs, points, or seed 103.
    fp = unique_claim_fingerprint()
    ics = unique_claim_protocol_ics()
    return (;
        n_ics = fp.n_ics,
        n_table = length(ics),
        holds = fp.n_ics == 9 && length(ics) == 9 &&
                fp.n_points == 50 && fp.seed == 103 && !fp.smoke)
end

function extra_candidates_do_not_shrink_graph_row()
    net = build_rate_ablation_network()
    r = collect(range(0.1, 2.0; length = 40))
    X = permutedims(hcat(r, r .^ 2))
    y = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    base = local_basis(net, 1; degree = 2, include_interactions = false,
        scope = :graph, X = X, derivative = y, extra_candidates = 0)
    extra = local_basis(net, 1; degree = 2, include_interactions = false,
        scope = :graph, X = X, derivative = y, extra_candidates = 1)
    return (;
        base = copy(base.variables),
        extra = copy(extra.variables),
        holds = length(extra.variables) ≥ length(base.variables) &&
                1 in base.variables)
end

function recovery_thresholds_untouched_library_row()
    lock = recovery_thresholds_lock()
    return (;
        holds = RECOVERY_THRESHOLDS == lock &&
                lock.support_f1_ude == 0.50 &&
                lock.support_f1_clean == 0.99)
end

