###############################################################################
# Data generation contract (not exported).
#
# generate_data uses the compiled NN tree (multi-head and multi-regulator).
# generate_data(::GroundTruthModel) integrates the stored model.
# Remapped nn_index and two-regulator D(S,I) are generated together.
# validate_network does not own this contract. Unique-claim recovery still
# admits exactly one unknown D(z).
###############################################################################

"""Source strings that prove generate_data builds the compiled NN tree."""
const DATAGEN_COMPILED_MUST_CONTAIN = (
    "function generate_data",
    "function generate_from_compiled_model",
    "function compile_ground_truth_model",
    "function generate_experiment_set_from_compiled_model",
    "build_ude_model",
    "pack_parameters",
    "generator === :compiled_mechanism",
    "SciMLBase.ODEProblem(model")

"""Source strings that prove generate_experiment_set compiles once."""
const DATAGEN_EXPERIMENT_SET_MUST_CONTAIN = (
    "function generate_experiment_set(",
    "compile_ground_truth_model",
    "generate_experiment_set_from_compiled_model",
    ":compiled_once")

"""Phrases that would restore the 1-input dummy hole."""
const DATAGEN_COMPILED_MUST_NOT_CONTAIN = (
    "Lux.Dense(1 => 1",
    "dummy_nn",
    "ones(1, 1) .* 0.0")

function datagen_source_path()
    joinpath(pkgdir(BioDynaX), "src", "DataGen.jl")
end

function datagen_compiled_source_holds()
    src = read(datagen_source_path(), String)
    return all(occursin(needle, src) for needle in DATAGEN_COMPILED_MUST_CONTAIN) &&
           !any(occursin(needle, src) for needle in DATAGEN_COMPILED_MUST_NOT_CONTAIN)
end

function datagen_source_violations()
    src = read(datagen_source_path(), String)
    missing = [s for s in DATAGEN_COMPILED_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in DATAGEN_COMPILED_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

function generate_data_uses_stored_ground_truth_model()
    src = read(datagen_source_path(), String)
    start = findfirst("function generate_data(truth::GroundTruthModel", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("generate_from_compiled_model", body) &&
           occursin("truth.model", body) &&
           !occursin("build_ude_model", body)
end

function generate_from_compiled_model_uses_sciml_odeproblem()
    src = read(datagen_source_path(), String)
    start = findfirst("function generate_from_compiled_model", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("SciMLBase.ODEProblem(model", body) &&
           !occursin("ODEProblem(rhs", body)
end

function generate_experiment_set_uses_compiled_once()
    src = read(datagen_source_path(), String)
    start = findfirst("function generate_experiment_set(rng", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("compile_ground_truth_model", body) &&
           occursin("generate_experiment_set_from_compiled_model", body) &&
           !occursin("generate_data(rng;", body)
end

function datagen_experiment_set_source_holds()
    src = read(datagen_source_path(), String)
    return all(occursin(needle, src) for needle in DATAGEN_EXPERIMENT_SET_MUST_CONTAIN) &&
           generate_experiment_set_uses_compiled_once() &&
           generate_from_compiled_model_uses_sciml_odeproblem()
end

function experiment_set_shares_compiled_parameters(set::ExperimentSet)
    isempty(set.experiments) && return false
    first_params = first(set.experiments).metadata[:truth_parameters]
    for exp in set.experiments
        haskey(exp.metadata, :truth_parameters) || return false
        exp.metadata[:truth_parameters] === first_params ||
            exp.metadata[:truth_parameters] == first_params || return false
    end
    return haskey(set.metadata, :truth_parameters) &&
           (set.metadata[:truth_parameters] === first_params ||
            set.metadata[:truth_parameters] == first_params)
end

function experiment_set_is_compiled_once(set::ExperimentSet)
    get(set.metadata, :compiled_once, false) === true &&
        experiment_set_shares_compiled_parameters(set)
end

# -- Architecture readers -----------------------------------------------------

"""Input width of a packed Lux `layer_1` weight (`out × in`)."""
function packed_nn_input_dim(p_nn)
    hasproperty(p_nn, :layer_1) || throw(ArgumentError(
        "packed NN tree has no layer_1; got $(propertynames(p_nn))"))
    return size(p_nn.layer_1.weight, 2)
end

"""Per-head input widths. Multi-head uses `head_i.layer_1`; single-head uses `layer_1`."""
function packed_nn_head_input_dims(params)
    nn = params.nn
    if hasproperty(nn, :head_1)
        dims = Int[]
        i = 1
        while hasproperty(nn, Symbol("head_$i"))
            push!(dims, packed_nn_input_dim(getproperty(nn, Symbol("head_$i"))))
            i += 1
        end
        return dims
    end
    return Int[packed_nn_input_dim(nn)]
end

function packed_nn_head_count(params)
    nn = params.nn
    hasproperty(nn, :head_1) || return 1
    i = 1
    while hasproperty(nn, Symbol("head_$i"))
        i += 1
    end
    return i - 1
end

"""
    compiled_nn_architecture(model, params) -> NamedTuple

Heads, dense `nn_index`, regulator arities, and packed input widths.
Used to lock generate_data / default_parameters against a 1-input dummy.
"""
function compiled_nn_architecture(model::UDEModel, params)
    compiled = model.compiled
    heads = neural_head_count(compiled)
    arities = neural_regulator_arities(compiled)
    packed_dims = packed_nn_head_input_dims(params)
    schema = parameter_schema(model)
    return (;
        n_heads = heads,
        indices = sort(neural_nn_indices(compiled)),
        arities,
        packed_dims,
        schema_heads = schema.nn_heads,
        dense = neural_index_is_dense(compiled),
        multihead = model.nn isa MultiHeadNetwork,
        packed_heads = packed_nn_head_count(params),
        matches = heads == schema.nn_heads &&
                  packed_dims == arities &&
                  (heads <= 1 || length(packed_dims) == heads))
end

function compiled_nn_architecture(network::BiologicalNetwork;
        rng::AbstractRNG = MersenneTwister(13))
    model, params = build_ude_model(rng, network)
    return compiled_nn_architecture(model, params)
end

function default_parameters_match_compiled(model::UDEModel, params)
    arch = compiled_nn_architecture(model, params)
    return arch.matches && arch.dense &&
           arch.n_heads == arch.schema_heads &&
           arch.packed_heads == max(arch.n_heads, 1)
end

# -- Joint remapped + two-regulator fixture -----------------------------------

"""
    build_remapped_two_regulator_network()

Skipped duplicate unknown (reaction + matching `UNKNOWN_NN` edge) together
with a later two-regulator `D(S, I)`. Remapping must keep heads `1:2` with
arities `[1, 2]`. This is not the unique-claim path.
"""
function build_remapped_two_regulator_network()::BiologicalNetwork
    nodes = [
        NodeSpec(name = :A),
        NodeSpec(name = :B),
        NodeSpec(name = :S),
        NodeSpec(name = :I)
    ]
    reactions = [
        ReactionSpec(name = :drive_a,
            stoichiometry = Dict(1 => 1.0), regulators = [4],
            metadata = MassActionMetadata(rate_param = :k_ia)),
        ReactionSpec(name = :drive_s,
            stoichiometry = Dict(3 => 1.0), regulators = [4],
            metadata = MassActionMetadata(rate_param = :k_is)),
        ReactionSpec(name = :unknown_ab,
            stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = false, family = HILL, metadata = HillMetadata()),
        ReactionSpec(name = :unknown_si,
            stoichiometry = Dict(3 => -1.0), regulators = [3, 4],
            known = false, family = COMPETITIVE, metadata = HillMetadata()),
        ReactionSpec(name = :b_decay,
            stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_b)),
        ReactionSpec(name = :i_decay,
            stoichiometry = Dict(4 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_i))
    ]
    edges = [
        EdgeSpec(source = 2, target = 1, kind = UNKNOWN_NN, known = false,
        family = HILL)
    ]
    return BiologicalNetwork(nodes, edges; reactions = reactions)
end

function remapped_two_regulator_phys_truth()
    return (k_ia = 0.8, k_is = 0.9, k_b = 0.5, k_i = 0.6)
end

function remapped_two_regulator_state(u0 = [0.25, 0.20, 0.30, 0.15])
    return Float64.(u0)
end

# -- Compiled snapshots -------------------------------------------------------

"""
    generate_compiled_snapshot(rng, network; kwargs...)

Build one `UDEModel`, integrate it with `generate_from_compiled_model`,
and compare against `ODEProblem(model, ...)`. Does not train a UDE.
"""
function generate_compiled_snapshot(rng::AbstractRNG, network::BiologicalNetwork;
        u0 = nothing,
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        noise_σ::Float64 = 0.0,
        truth_params = nothing)
    model, default_params = build_ude_model(rng, network)
    assert_dense_neural_index(model)
    params = if truth_params === nothing
        default_params
    elseif truth_params isa ComponentVector
        truth_params
    else
        pack_parameters(truth_params, default_params.nn)
    end
    n = model.compiled.nstates
    state = u0 === nothing ? fill(0.25, n) : Float64.(u0)
    times, clean, noisy, used = generate_from_compiled_model(
        model, params, rng; u0 = state, tspan, n_points, noise_σ)
    truth = GroundTruthModel(network, model, params, :compiled_mechanism)
    times_t, clean_t, _, used_t = generate_data(
        truth, MersenneTwister(0); u0 = state, tspan, n_points, noise_σ = 0.0)
    prob = ODEProblem(model, state, tspan, params)
    sol = solve(prob, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9)
    solved = Array(sol)
    rhs = evaluate_compiled_rhs(model, params, state)
    arch = compiled_nn_architecture(model, params)
    cache = allocate_cache(model, Float64)
    return (;
        network,
        model,
        params = used,
        truth,
        times,
        clean,
        noisy,
        solved,
        truth_clean = clean_t,
        truth_times = times_t,
        truth_params = used_t,
        rhs,
        arch,
        cache,
        nstates = n,
        finite = all(isfinite, clean) && all(isfinite, noisy) &&
                 all(isfinite, solved),
        matches_solve = clean ≈ solved,
        matches_stored_truth = clean_t ≈ solved,
        cache_matches = neural_cache_matches_heads(model, cache),
        validate_open = validate_network(network) === network)
end

function generate_data_namedtuple_snapshot(rng::AbstractRNG,
        network::BiologicalNetwork;
        u0,
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        noise_σ::Float64 = 0.0,
        truth_params)
    times, clean, noisy, packed = generate_data(
        rng; network, u0 = Float64.(u0), tspan, n_points, noise_σ,
        truth_params = truth_params)
    model, _ = build_ude_model(MersenneTwister(0), network)
    arch = compiled_nn_architecture(model, packed)
    return (;
        times, clean, noisy, packed, model, arch,
        finite = all(isfinite, clean) && all(isfinite, noisy),
        nstates = size(clean, 1),
        n_points = length(times))
end

function generate_experiment_set_snapshot(rng::AbstractRNG,
        network::BiologicalNetwork;
        initial_conditions,
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        noise_σ::Float64 = 0.0,
        truth_params = nothing)
    set = generate_experiment_set(
        rng; network, initial_conditions, tspan, n_points, noise_σ,
        truth_params = truth_params)
    observations = [all(isfinite, exp.observations) for exp in set.experiments]
    widths = [size(exp.observations, 2) for exp in set.experiments]
    heights = [size(exp.observations, 1) for exp in set.experiments]
    return (;
        set,
        n_experiments = length(set.experiments),
        finite = all(observations),
        n_points = widths,
        nstates = heights,
        requested_points = n_points,
        requested_ics = length(initial_conditions),
        compiled_once = experiment_set_is_compiled_once(set))
end

# -- Unique-claim experiment helper -------------------------------------------

"""
    unique_claim_experiment_set(rng, network; smoke=false, kwargs...)

`generate_experiment_set` on the unique-claim fingerprint: protocol ICs and
point count, fingerprint `tspan`, observation noise. Caller must pass
`initial_conditions` when the network is not 2-state.
"""
function unique_claim_experiment_set(rng::AbstractRNG, network::BiologicalNetwork;
        smoke::Bool = false,
        truth_params = nothing,
        noise_σ = nothing,
        initial_conditions = nothing)
    fp = unique_claim_fingerprint(; smoke)
    ics = initial_conditions === nothing ?
          unique_claim_protocol_ics(; smoke) : initial_conditions
    isempty(ics) && throw(ArgumentError("unique-claim experiment set needs ICs"))
    nstates = length(state_nodes(network))
    length(first(ics)) == nstates || throw(ArgumentError(
        "unique-claim experiment ICs are $(length(first(ics)))-state; network has $nstates"))
    σ = noise_σ === nothing ? fp.observation_noise : noise_σ
    truth = compile_ground_truth_model(rng, network; truth_params = truth_params)
    set = generate_experiment_set_from_compiled_model(
        truth, rng;
        initial_conditions = ics,
        tspan = fp.tspan,
        n_points = fp.n_points,
        noise_σ = σ)
    set.metadata[:unique_claim_fingerprint_kind] = fp.kind
    set.metadata[:unique_claim_n_ics] = fp.n_ics
    set.metadata[:unique_claim_n_points] = fp.n_points
    set.metadata[:unique_claim_smoke] = fp.smoke
    return set
end

function unique_claim_experiment_set_matches_fingerprint(set::ExperimentSet;
        smoke::Bool = false)
    fp = unique_claim_fingerprint(; smoke)
    n = length(set.experiments)
    n == fp.n_ics || return false
    get(set.metadata, :unique_claim_fingerprint_kind, nothing) === fp.kind ||
        return false
    experiment_set_is_compiled_once(set) || return false
    for exp in set.experiments
        size(exp.observations, 2) == fp.n_points || return false
        first(exp.times) == first(fp.tspan) || return false
        last(exp.times) == last(fp.tspan) || return false
    end
    return true
end

function unique_claim_example_uses_experiment_set()
    src = read(unique_claim_example_path(), String)
    return occursin("unique_claim_experiment_set", src) &&
           occursin("unique_claim_fingerprint", src)
end

# -- Joint contract row -------------------------------------------------------

"""
    joint_datagen_compiler_row(network; kwargs...)

Compile + generate_data + generate_experiment_set + default_parameters on
one network. Remapped multi-head and two-regulator rows must both be finite.
"""
function joint_datagen_compiler_row(network::BiologicalNetwork;
        rng::AbstractRNG = MersenneTwister(13),
        u0 = nothing,
        tspan::Tuple{Float64, Float64} = (0.0, 1.0),
        n_points::Int = 8,
        truth_params = nothing,
        extra_ics = nothing)
    snap = generate_compiled_snapshot(
        rng, network; u0, tspan, n_points, truth_params = truth_params)
    state0 = u0 === nothing ? fill(0.22, snap.nstates) : Float64.(u0)
    phys = if truth_params === nothing
        default_phys_parameters(parameter_schema(snap.model))
    elseif truth_params isa ComponentVector
        truth_params.phys
    else
        truth_params
    end
    named = generate_data_namedtuple_snapshot(
        MersenneTwister(19), network;
        u0 = state0, tspan, n_points, truth_params = phys)
    ics = extra_ics === nothing ? [state0] : extra_ics
    set_snap = generate_experiment_set_snapshot(
        MersenneTwister(23), network;
        initial_conditions = ics,
        tspan, n_points,
        truth_params = phys)
    defaults = default_parameters(snap.model; rng = MersenneTwister(29))
    admission = unique_claim_recovery_admission(network)
    return (;
        snap,
        named,
        set = set_snap,
        defaults,
        default_matches = default_parameters_match_compiled(snap.model, defaults),
        default_finite = all(isfinite,
            ude_system(state0, defaults, 0.0, snap.model)),
        admission,
        recovery_admits = admission.recovery_admits,
        validate_open = snap.validate_open && admission.validate_open,
        remapped_dense = snap.arch.dense,
        arities = snap.arch.arities,
        packed_dims = snap.arch.packed_dims,
        joint_holds = snap.finite && snap.matches_solve &&
                      snap.matches_stored_truth && snap.arch.matches &&
                      named.finite && set_snap.finite)
end

function remapped_two_regulator_contract_holds()
    row = joint_datagen_compiler_row(
        build_remapped_two_regulator_network();
        rng = MersenneTwister(13),
        u0 = remapped_two_regulator_state(),
        truth_params = remapped_two_regulator_phys_truth())
    return row.joint_holds &&
           row.snap.arch.n_heads == 2 &&
           row.arities == [1, 2] &&
           row.packed_dims == [1, 2] &&
           row.remapped_dense &&
           row.recovery_admits == false &&
           row.validate_open &&
           row.default_matches
end
