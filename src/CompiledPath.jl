###############################################################################
# Joint compiled experiment path (not exported).
#
# generate_experiment_set compiles once. generate_from_compiled_model uses
# SciMLBase.ODEProblem(model, ...). Remapped multi-head / two-regulator
# generation, suite admission, and UniqueClaimProtocolRow are one path.
# validate_network stays a topology checker. Unique-claim recovery still
# admits exactly one unknown D(z).
###############################################################################

"""Source strings that prove the joint compiled path stays wired."""
const COMPILED_PATH_DATAGEN_MUST_CONTAIN = (
    "function compile_ground_truth_model",
    "function generate_experiment_set_from_compiled_model",
    "SciMLBase.ODEProblem(model")

const COMPILED_PATH_ADMISSION_MUST_CONTAIN = (
    "function recovery_suite_admission_matrix",
    "function recovery_suite_zero_dual_matrix",
    "RECOVERY_SUITE_HOLE_POLICY",
    "RECOVERY_SUITE_EXPECTED_HOLES")

function compiled_path_datagen_source_holds()
    src = read(datagen_source_path(), String)
    return all(occursin(needle, src) for needle in COMPILED_PATH_DATAGEN_MUST_CONTAIN)
end

function compiled_path_admission_source_holds()
    path = joinpath(pkgdir(BioDynaX), "src", "RecoveryAdmission.jl")
    src = read(path, String)
    return all(occursin(needle, src) for needle in COMPILED_PATH_ADMISSION_MUST_CONTAIN)
end

# -- SciML generate agreement -------------------------------------------------

"""
    sciml_compiled_generate_agreement(model, params, u0; tspan, n_points)

`generate_from_compiled_model` versus `ODEProblem(model, ...)` and
`SciMLBase.solve(model, ...)`. Does not train a UDE.
"""
function sciml_compiled_generate_agreement(model::UDEModel, params, u0;
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        rng::AbstractRNG = MersenneTwister(0))
    times, clean, _, used = generate_from_compiled_model(
        model, params, rng; u0 = Float64.(u0), tspan, n_points, noise_σ = 0.0)
    prob = SciMLBase.ODEProblem(model, Float64.(u0), tspan, params)
    sol = solve(prob, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9)
    solved = Array(sol)
    cache = allocate_cache(model, Float64)
    prob_ip = SciMLBase.ODEProblem(
        model, Float64.(u0), tspan, params; inplace = true, cache = cache)
    sol_ip = solve(prob_ip, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9)
    solved_ip = Array(sol_ip)
    remade = SciMLBase.remake(prob; p = params)
    sol_remade = solve(remade, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9)
    sciml_sol = SciMLBase.solve(
        model, Float64.(u0), tspan, params;
        saveat = times,
        solver_config = SolverConfig(algorithm = Tsit5(), sensealg = nothing))
    sciml_arr = Array(sciml_sol)
    arch = compiled_nn_architecture(model, params)
    return (;
        times,
        clean,
        solved,
        solved_ip,
        remade = Array(sol_remade),
        sciml = sciml_arr,
        used,
        arch,
        cache,
        finite = all(isfinite, clean) && all(isfinite, solved) &&
                 all(isfinite, solved_ip) && all(isfinite, sciml_arr),
        matches_odeproblem = clean ≈ solved,
        matches_inplace = clean ≈ solved_ip,
        matches_remake = clean ≈ Array(sol_remade),
        matches_sciml_solve = clean ≈ sciml_arr,
        cache_matches = neural_cache_matches_heads(model, cache),
        holds = all(isfinite, clean) && clean ≈ solved && clean ≈ solved_ip &&
                clean ≈ Array(sol_remade) && clean ≈ sciml_arr &&
                neural_cache_matches_heads(model, cache) && arch.dense)
end

function sciml_compiled_generate_agreement(network::BiologicalNetwork;
        rng::AbstractRNG = MersenneTwister(13),
        u0 = nothing,
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        truth_params = nothing)
    truth = compile_ground_truth_model(
        rng, network; truth_params = truth_params)
    n = truth.model.compiled.nstates
    state = u0 === nothing ? fill(0.25, n) : Float64.(u0)
    return sciml_compiled_generate_agreement(
        truth.model, truth.parameters, state; tspan, n_points,
        rng = MersenneTwister(0))
end

# -- Compiled experiment-set identity ----------------------------------------

function compiled_experiment_set_row(network::BiologicalNetwork;
        rng::AbstractRNG = MersenneTwister(17),
        initial_conditions,
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        noise_σ::Float64 = 0.0,
        truth_params = nothing)
    truth = compile_ground_truth_model(
        rng, network; truth_params = truth_params)
    set = generate_experiment_set_from_compiled_model(
        truth, rng;
        initial_conditions,
        tspan,
        n_points,
        noise_σ)
    rebuilt = generate_experiment_set(
        MersenneTwister(19);
        network,
        initial_conditions,
        tspan,
        n_points,
        noise_σ,
        truth_params = truth_params)
    arch = compiled_nn_architecture(truth.model, truth.parameters)
    nstates = truth.model.compiled.nstates
    finite = all(exp -> all(isfinite, exp.observations), set.experiments)
    widths = [size(exp.observations, 2) for exp in set.experiments]
    heights = [size(exp.observations, 1) for exp in set.experiments]
    return (;
        truth,
        set,
        rebuilt,
        arch,
        nstates,
        n_ics = length(set.experiments),
        finite,
        widths,
        heights,
        compiled_once = experiment_set_is_compiled_once(set),
        rebuilt_compiled_once = experiment_set_is_compiled_once(rebuilt),
        shared_params = experiment_set_shares_compiled_parameters(set),
        same_nstates = all(h -> h == nstates, heights),
        same_points = all(w -> w == n_points, widths),
        holds = experiment_set_is_compiled_once(set) &&
                experiment_set_is_compiled_once(rebuilt) &&
                finite &&
                length(set.experiments) == length(initial_conditions) &&
                all(h -> h == nstates, heights) &&
                all(w -> w == n_points, widths) &&
                arch.dense)
end

# -- Joint row ----------------------------------------------------------------

"""
    CompiledPathRow

Executable joint path: stored compiled model, multi-IC experiment set,
SciML generate agreement, suite admission, and a protocol print row.
Not exported. Combined F1 is stored on the protocol row only as context.
"""
struct CompiledPathRow
    network::BiologicalNetwork
    truth::GroundTruthModel
    set::ExperimentSet
    sciml
    admission
    protocol_row::UniqueClaimProtocolRow
    arch
    remapped_dense::Bool
    compiled_once::Bool
    recovery_admits::Bool
    validate_open::Bool
end

function compiled_path_row_namedtuple(row::CompiledPathRow)
    return (;
        nstates = row.truth.model.compiled.nstates,
        n_heads = row.arch.n_heads,
        arities = row.arch.arities,
        packed_dims = row.arch.packed_dims,
        remapped_dense = row.remapped_dense,
        compiled_once = row.compiled_once,
        n_ics = length(row.set.experiments),
        recovery_admits = row.recovery_admits,
        validate_open = row.validate_open,
        extras_label = row.protocol_row.extras_label,
        kpi_failures = Tuple(row.protocol_row.kpi_failures),
        is_protocol = unique_claim_fingerprint_is_protocol(row.protocol_row.fingerprint),
        sciml_holds = row.sciml.holds,
        admission_kind = hasproperty(row.admission, :kind) ? row.admission.kind : nothing)
end

"""
    joint_compiled_path(network; kwargs...)

Compile once, generate every IC from that model, compare SciML solves,
admit or reject a suite section, and attach a `UniqueClaimProtocolRow`.
Does not train a UDE.
"""
function joint_compiled_path(network::BiologicalNetwork;
        rng::AbstractRNG = MersenneTwister(13),
        initial_conditions = nothing,
        u0 = nothing,
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        truth_params = nothing,
        section::Symbol = :ude_discovery,
        smoke::Bool = true,
        protocol_fields...)
    truth = compile_ground_truth_model(
        rng, network; truth_params = truth_params)
    n = truth.model.compiled.nstates
    state = u0 === nothing ? fill(0.22, n) : Float64.(u0)
    ics = initial_conditions === nothing ? [state] : initial_conditions
    set = generate_experiment_set_from_compiled_model(
        truth, rng;
        initial_conditions = ics,
        tspan,
        n_points,
        noise_σ = 0.0)
    sciml = sciml_compiled_generate_agreement(
        truth.model, truth.parameters, first(ics); tspan, n_points,
        rng = MersenneTwister(0))
    admission = recovery_suite_admission_row(section, network)
    protocol = unique_claim_protocol_row_from_fields(;
        smoke = smoke, protocol_fields...)
    arch = compiled_nn_architecture(truth.model, truth.parameters)
    return CompiledPathRow(
        network,
        truth,
        set,
        sciml,
        admission,
        protocol,
        arch,
        arch.dense,
        experiment_set_is_compiled_once(set),
        admission.recovery_admits,
        admission.validate_open && sciml.holds)
end

function joint_compiled_path_holds(row::CompiledPathRow)
    return row.compiled_once &&
           row.remapped_dense &&
           row.sciml.holds &&
           row.validate_open &&
           experiment_set_shares_compiled_parameters(row.set) &&
           assert_unique_claim_protocol_row(row.protocol_row) === row.protocol_row &&
           row.arch.matches
end

function remapped_two_regulator_compiled_path()
    net = build_remapped_two_regulator_network()
    row = joint_compiled_path(
        net;
        rng = MersenneTwister(13),
        u0 = remapped_two_regulator_state(),
        initial_conditions = [
            remapped_two_regulator_state(),
            [0.18, 0.16, 0.22, 0.12]
        ],
        truth_params = remapped_two_regulator_phys_truth(),
        section = :ude_discovery,
        smoke = true)
    return (;
        row,
        holds = joint_compiled_path_holds(row) &&
                row.arch.n_heads == 2 &&
                row.arch.arities == [1, 2] &&
                row.arch.packed_dims == [1, 2] &&
                row.recovery_admits == false &&
                row.admission.admitted == false &&
                length(row.set.experiments) == 2)
end

function unique_claim_compiled_path(; smoke::Bool = true)
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude = build_hill_recovery_network(; known = false, hill_order = 2)
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    rng = MersenneTwister(103)
    set = unique_claim_experiment_set(
        rng, net; smoke = smoke, truth_params = truth_params)
    fp = unique_claim_fingerprint(; smoke)
    sciml = sciml_compiled_generate_agreement(
        net; rng = MersenneTwister(13), u0 = first(unique_claim_protocol_ics(; smoke)),
        tspan = fp.tspan, n_points = fp.n_points, truth_params = truth_params)
    admission = recovery_suite_admission_row(:ude_discovery, ude)
    protocol = unique_claim_protocol_row_from_fields(; smoke = smoke)
    return (;
        set,
        fingerprint = fp,
        sciml,
        admission,
        protocol,
        compiled_once = experiment_set_is_compiled_once(set),
        matches_fingerprint = unique_claim_experiment_set_matches_fingerprint(
            set; smoke = smoke),
        recovery_admits = admission.admitted && admission.recovery_admits,
        validate_open = unique_claim_compiler_stays_open(net) &&
                        unique_claim_compiler_stays_open(ude),
        holds = experiment_set_is_compiled_once(set) &&
                unique_claim_experiment_set_matches_fingerprint(set; smoke = smoke) &&
                sciml.holds &&
                admission.admitted &&
                unique_claim_compiler_stays_open(ude) &&
                assert_unique_claim_protocol_row(protocol) === protocol)
end

function dual_unknown_compiled_path()
    net = build_dual_unknown_network()
    row = joint_compiled_path(
        net;
        rng = MersenneTwister(21),
        u0 = [0.22, 0.18, 0.16],
        initial_conditions = [[0.22, 0.18, 0.16], [0.30, 0.24, 0.20]],
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5),
        section = :linear,
        smoke = true)
    closed = joint_compiled_path(
        net;
        rng = MersenneTwister(21),
        u0 = [0.22, 0.18, 0.16],
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5),
        section = :ude_discovery,
        smoke = true)
    return (;
        open = row,
        closed,
        holds = joint_compiled_path_holds(row) &&
                row.admission.admitted &&
                row.recovery_admits == false &&
                closed.admission.admitted == false &&
                row.arch.n_heads == 2 &&
                packed_nn_head_count(row.truth.parameters) == 2)
end

function two_regulator_sciml_path()
    net = build_two_regulator_unknown_network()
    agree = sciml_compiled_generate_agreement(
        net;
        rng = MersenneTwister(19),
        u0 = [0.25, 0.20, 0.15],
        truth_params = (k_es = 0.8, k_i = 0.5, k_e = 0.4))
    return (;
        agree,
        holds = agree.holds &&
                agree.arch.n_heads == 1 &&
                agree.arch.arities == [2] &&
                agree.arch.packed_dims == [2] &&
                unique_claim_recovery_admits(net) &&
                unique_claim_compiler_stays_open(net))
end

function skipped_duplicate_sciml_path()
    net = build_skipped_duplicate_unknown_network()
    agree = sciml_compiled_generate_agreement(
        net;
        rng = MersenneTwister(13),
        u0 = [0.2, 0.3, 0.4],
        truth_params = (k_ca = 0.8, k_b = 0.5, k_c = 0.4))
    return (;
        agree,
        holds = agree.holds &&
                agree.arch.n_heads == 2 &&
                agree.arch.dense &&
                unique_claim_recovery_admits(net) == false &&
                unique_claim_compiler_stays_open(net))
end

function zero_hole_compiled_path()
    net = build_zero_unknown_linear_network()
    row = joint_compiled_path(
        net;
        rng = MersenneTwister(7),
        u0 = [0.22, 0.14],
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5),
        section = :linear,
        smoke = true)
    closed = recovery_suite_admission_row(:ude_discovery, net)
    return (;
        row,
        closed,
        holds = joint_compiled_path_holds(row) &&
                row.arch.n_heads == 0 &&
                row.admission.admitted &&
                closed.admitted == false &&
                closed.validate_open)
end

function default_example_compiled_once_path()
    net = DEFAULT_EXAMPLE_NETWORK
    ics = [[0.20, 0.10], [0.30, 0.15]]
    row = compiled_experiment_set_row(
        net;
        rng = MersenneTwister(11),
        initial_conditions = ics,
        tspan = (0.0, 1.0),
        n_points = 6,
        noise_σ = 0.0)
    return (;
        row,
        holes = count_unknown_destructions(net),
        holds = row.holds &&
                default_example_has_duplicate_unknown_declaration() &&
                row.arch.n_heads == 1 &&
                row.arch.dense &&
                experiment_set_shares_compiled_parameters(row.set))
end

# -- Suite matrix + joint admission ------------------------------------------

function joint_admission_and_compiled_matrix()
    matrix = recovery_suite_admission_matrix()
    zero_dual = recovery_suite_zero_dual_matrix()
    remap = remapped_two_regulator_compiled_path()
    claim = unique_claim_compiled_path()
    dual = dual_unknown_compiled_path()
    zero = zero_hole_compiled_path()
    return (;
        matrix,
        zero_dual,
        remap,
        claim,
        dual,
        zero,
        holds = matrix.holds && zero_dual.holds && remap.holds &&
                claim.holds && dual.holds && zero.holds)
end

# -- Docs / example locks for this layer -------------------------------------

function compiled_path_user_doc_paths()
    root = pkgdir(BioDynaX)
    return (
        joinpath(root, "docs", "src", "compiled-path.md"),
        joinpath(root, "docs", "src", "unique-claim.md"),
        joinpath(root, "docs", "src", "sciml.md"),
        joinpath(root, "docs", "src", "tutorial.md"),
        joinpath(root, "docs", "src", "howto.md"),
        joinpath(root, "docs", "src", "architecture.md"),
        joinpath(root, "README.md"))
end

function compiled_path_locked_sentences()
    return (;
        compiled_once = "generate_experiment_set compiles the ground-truth model once and generates every IC from that stored model.",
        sciml_generate = "generate_from_compiled_model integrates SciMLBase.ODEProblem(model, u0, tspan, p).",
        admission_matrix = "Every recovery-suite section has a hole policy; only unique-claim sections reject 0/2 holes before training.",
        joint = "The joint compiled path is generate_from_compiled_model + remapped heads + admit_recovery_suite_network + UniqueClaimProtocolRow.")
end

function compiled_path_docs_hold()
    sentences = compiled_path_locked_sentences()
    page = joinpath(pkgdir(BioDynaX), "docs", "src", "compiled-path.md")
    isfile(page) || return false
    text = read(page, String)
    for sentence in values(sentences)
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("compiled-path.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text)
end

function compiled_path_landing_docs_hold()
    sentences = compiled_path_locked_sentences()
    tutorial = read(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"), String)
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    readme = read(joinpath(pkgdir(BioDynaX), "README.md"), String)
    unique_page = read(joinpath(pkgdir(BioDynaX), "docs", "src", "unique-claim.md"), String)
    architecture = read(joinpath(pkgdir(BioDynaX), "docs", "src", "architecture.md"), String)
    experimental = read(joinpath(pkgdir(BioDynaX), "docs", "src", "experimental.md"), String)
    return occursin("generate_experiment_set_from_compiled_model", tutorial) &&
           occursin("compile_ground_truth_model", howto) &&
           occursin("SciMLBase.ODEProblem(model", sciml) &&
           occursin("generate_from_compiled_model", readme) &&
           occursin(sentences.compiled_once, unique_page) &&
           occursin(sentences.admission_matrix, unique_page) &&
           occursin("admit_recovery_suite_network", architecture) &&
           occursin("UniqueClaimFingerprint", experimental)
end

function compiled_path_contract_holds()
    return compiled_path_datagen_source_holds() &&
           compiled_path_admission_source_holds() &&
           generate_experiment_set_uses_compiled_once() &&
           generate_from_compiled_model_uses_sciml_odeproblem() &&
           datagen_compiled_source_holds() &&
           remapped_two_regulator_compiled_path().holds &&
           unique_claim_compiled_path().holds &&
           recovery_suite_admission_matrix().holds &&
           recovery_suite_zero_dual_matrix().holds &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source()
end
