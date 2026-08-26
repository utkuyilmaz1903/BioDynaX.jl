###############################################################################
# Experiment fingerprints, batches, checkpoint resume, remapped generate+train.
#
# generate_experiment_set already compiles once. Training already reuses a
# compiled model. This file locks the remaining join: fingerprints stay
# stable, batches cover every IC, resume does not compile, and remapped
# multi-head generate + train_experiments stay on one compiled tree.
# Does not drop protocol ICs. Does not grow exports.
###############################################################################

const EXPERIMENT_CHECKPOINT_MUST_CONTAIN = (
    "function experiment_fingerprint_row",
    "function experiment_batch_row",
    "function checkpoint_resume_row",
    "function remapped_generate_train_row",
    "function unique_claim_fingerprint_set_row",
    "function remapped_warmup_generate_train_row",
    "function unique_claim_from_compiled_fingerprint_row",
    "function resume_equivalence_row",
    "function mm_unknown_generate_train_row",
    "function masked_fingerprint_train_row",
    "function frozen_phys_checkpoint_row",
    "function generated_data_fingerprint_row")

const EXPERIMENT_CHECKPOINT_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function experiment_checkpoint_locked_sentences()
    return (;
        fingerprint = "experiment_fingerprint hashes times, observations, mask, and u0; metadata is not part of the identity.",
        batches = "experiment_batches partitions every IC; shuffle does not drop or duplicate an experiment.",
        resume = "resume_training from a checkpoint reuses the compiled UDEModel and does not call compile_network.",
        remapped = "Remapped multi-head generate and train_experiments share one compiled tree; train_experiments does not compile per IC.",
        compiled = "generate_experiment_set_from_compiled_model fingerprints without calling compile_network.",
        warmup = "train_experiments_with_warmup on a remapped multi-head set does not call compile_network.")
end

"""Landing sentence used by `docs/src/sciml.md`."""
experiment_checkpoint_contract() =
    experiment_checkpoint_locked_sentences().remapped

function experiment_jl_source_path()
    joinpath(pkgdir(BioDynaX), "src", "Experiments.jl")
end

function experiment_checkpoint_source_path()
    joinpath(pkgdir(BioDynaX), "src", "ExperimentCheckpoint.jl")
end

function training_jl_checkpoint_source()
    src = read(training_jl_source_path(), String)
    start = findfirst("function resume_training", src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

# -- Fingerprints -------------------------------------------------------------

"""
    experiment_fingerprint_row(set)

Stability of `experiment_fingerprint`. A metadata-only change must not
change the hash. A mask or u0 change must.
"""
function experiment_fingerprint_row(set::ExperimentSet)
    first_hash = experiment_fingerprint(set)
    again = experiment_fingerprint(set)
    tagged = ExperimentSet(
        set.experiments, set.state_names;
        units = set.units,
        metadata = merge(copy(set.metadata), Dict{Symbol,Any}(:tag => :probe)))
    tagged_hash = experiment_fingerprint(tagged)
    experiments = map(set.experiments) do experiment
        Experiment(
            experiment.name, experiment.times, experiment.observations,
            experiment.u0; mask = experiment.mask,
            metadata = merge(copy(experiment.metadata),
                Dict{Symbol,Any}(:note => :ignored)))
    end
    meta_only = ExperimentSet(experiments, set.state_names; units = set.units)
    meta_hash = experiment_fingerprint(meta_only)
    masked = let
        exp = first(set.experiments)
        mask = copy(exp.mask)
        mask[1, 1] = !mask[1, 1]
        replaced = Experiment(
            exp.name, exp.times, exp.observations, exp.u0;
            mask = mask, metadata = exp.metadata)
        rest = set.experiments[2:end]
        ExperimentSet(vcat([replaced], rest), set.state_names; units = set.units)
    end
    mask_hash = experiment_fingerprint(masked)
    shifted = let
        exp = first(set.experiments)
        replaced = Experiment(
            exp.name, exp.times, exp.observations, exp.u0 .+ 0.01;
            mask = exp.mask, metadata = exp.metadata)
        rest = set.experiments[2:end]
        ExperimentSet(vcat([replaced], rest), set.state_names; units = set.units)
    end
    u0_hash = experiment_fingerprint(shifted)
    return (;
        first_hash,
        again,
        tagged_hash,
        meta_hash,
        mask_hash,
        u0_hash,
        stable = first_hash == again,
        metadata_ignored = first_hash == tagged_hash == meta_hash,
        mask_changes = first_hash != mask_hash,
        u0_changes = first_hash != u0_hash,
        hex64 = length(first_hash) == 64,
        holds = first_hash == again &&
                first_hash == tagged_hash == meta_hash &&
                first_hash != mask_hash &&
                first_hash != u0_hash &&
                length(first_hash) == 64)
end

function data_fingerprint_row(data, times, u0)
    a = data_fingerprint(data, times, u0)
    b = data_fingerprint(data, times, u0)
    c = data_fingerprint(data .+ 1e-12, times, u0)
    return (;
        a, b,
        stable = a == b,
        data_changes = a != c,
        hex64 = length(a) == 64,
        holds = a == b && a != c && length(a) == 64)
end

function unique_claim_fingerprint_set_row(; smoke::Bool = true)
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    set = unique_claim_experiment_set(
        MersenneTwister(103), net; smoke = smoke, truth_params = truth_params)
    again = unique_claim_experiment_set(
        MersenneTwister(103), net; smoke = smoke, truth_params = truth_params)
    fp = unique_claim_fingerprint(; smoke)
    row = experiment_fingerprint_row(set)
    return (;
        row,
        compiled_once = experiment_set_is_compiled_once(set),
        n_ics = length(set.experiments),
        n_points = size(first(set.experiments).observations, 2),
        fingerprint_n_ics = fp.n_ics,
        same_hash = experiment_fingerprint(set) == experiment_fingerprint(again),
        matches_kind = unique_claim_experiment_set_matches_fingerprint(
            set; smoke = smoke),
        holds = row.holds && experiment_set_is_compiled_once(set) &&
                experiment_fingerprint(set) == experiment_fingerprint(again) &&
                unique_claim_experiment_set_matches_fingerprint(set; smoke = smoke) &&
                length(set.experiments) == fp.n_ics)
end

# -- Batches ------------------------------------------------------------------

"""
    experiment_batch_row(set; batch_size)

Coverage and disjointness of `experiment_batches`. Shuffle uses an
explicit RNG and still covers every experiment once per pass.
"""
function experiment_batch_row(set::ExperimentSet; batch_size::Int = 2)
    names = [exp.name for exp in set.experiments]
    sequential = experiment_batches(set, batch_size; shuffle = false)
    shuffled = experiment_batches(
        set, batch_size; shuffle = true, rng = MersenneTwister(3))
    shuffled2 = experiment_batches(
        set, batch_size; shuffle = true, rng = MersenneTwister(3))
    seq_names = reduce(vcat, [[exp.name for exp in batch] for batch in sequential])
    shuf_names = reduce(vcat, [[exp.name for exp in batch] for batch in shuffled])
    shuf2_names = reduce(vcat, [[exp.name for exp in batch] for batch in shuffled2])
    widths = [length(batch) for batch in sequential]
    expected_n = cld(length(set), batch_size)
    return (;
        n_ics = length(set),
        batch_size,
        n_batches = length(sequential),
        expected_n,
        widths,
        sequential_covers = issetequal(seq_names, names),
        shuffled_covers = issetequal(shuf_names, names),
        shuffle_reproducible = shuf_names == shuf2_names,
        no_pad = sum(widths) == length(set),
        holds = length(sequential) == expected_n &&
                issetequal(seq_names, names) &&
                issetequal(shuf_names, names) &&
                shuf_names == shuf2_names &&
                sum(widths) == length(set))
end

function experiment_weight_row(set::ExperimentSet)
    weights = experiment_weight.(set.experiments)
    scales = experiment_noise_scale.(set.experiments)
    tagged = let
        exp = first(set.experiments)
        Experiment(
            exp.name, exp.times, exp.observations, exp.u0;
            mask = exp.mask,
            metadata = merge(copy(exp.metadata),
                Dict{Symbol,Any}(:weight => 2.0, :noise_σ => 0.25)))
    end
    return (;
        default_weight = all(==(1.0), weights),
        default_scale = all(==(1.0), scales),
        tagged_weight = experiment_weight(tagged),
        tagged_scale = experiment_noise_scale(tagged),
        holds = all(==(1.0), weights) && all(==(1.0), scales) &&
                experiment_weight(tagged) == 2.0 &&
                experiment_noise_scale(tagged) == 0.25)
end

# -- Checkpoint / resume ------------------------------------------------------

function checkpoint_schema_row()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Types.jl"), String)
    train = read(training_jl_source_path(), String)
    return (;
        version = CHECKPOINT_SCHEMA_VERSION,
        major = CHECKPOINT_SCHEMA_VERSION.major,
        source_has_version = occursin("CHECKPOINT_SCHEMA_VERSION = v\"1.0.0\"", src),
        save_uses_schema = occursin("CHECKPOINT_SCHEMA_VERSION", train),
        resume_passes_state = occursin("optimizer_state = checkpoint.optimizer_state", train),
        holds = CHECKPOINT_SCHEMA_VERSION == v"1.0.0" &&
                occursin("CHECKPOINT_SCHEMA_VERSION = v\"1.0.0\"", src) &&
                occursin("optimizer_state = checkpoint.optimizer_state", train))
end

"""
    checkpoint_resume_row(p_init, data, times, u0, tspan, model; dir)

Train two Adam steps with a checkpoint, resume one more step, and
require zero `compile_network` calls on resume.
"""
function checkpoint_resume_row(p_init, data, times, u0, tspan, model; dir)
    path = joinpath(dir, "joint.ckpt")
    config = TrainingConfig(
        adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6)
    trained = train_ude(
        p_init, data, times, u0, tspan, model;
        config = config, verbose = false, checkpoint_path = path,
        checkpoint_every = 1, seed = 5)
    isfile(path) || return (; holds = false, compiles = typemax(Int))
    ckpt = load_checkpoint(path)
    n = with_compile_network_counter() do counter
        resume_training(
            ckpt, data, times, u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    from_diag = optimizer_state_from_result(trained)
    n_diag = with_compile_network_counter() do counter
        train_ude(
            trained.params, data, times, u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            optimizer_state = from_diag, verbose = false)
        counter[]
    end
    return (;
        schema = ckpt.schema_version,
        has_state = ckpt.optimizer_state !== nothing,
        diag_state = from_diag !== nothing,
        compiles = n,
        diag_compiles = n_diag,
        iteration = ckpt.iteration,
        holds = n == 0 && n_diag == 0 &&
                ckpt.schema_version.major == CHECKPOINT_SCHEMA_VERSION.major &&
                ckpt.optimizer_state !== nothing &&
                from_diag !== nothing)
end

function artifact_roundtrip_row(result; dir)
    path = joinpath(dir, "result.jls")
    save_result(path, result)
    loaded = load_result(path)
    return (;
        kind_ok = loaded isa TrainingResult,
        retcode = loaded.retcode,
        history = length(loaded.history) == length(result.history),
        holds = loaded isa TrainingResult &&
                loaded.retcode == result.retcode &&
                length(loaded.history) == length(result.history))
end

function resume_source_holds()
    body = training_jl_checkpoint_source()
    return occursin("optimizer_state = checkpoint.optimizer_state", body) &&
           occursin("initial_iteration = checkpoint.iteration", body) &&
           occursin("train_ude(", body)
end

# -- Remapped generate + train ------------------------------------------------

"""
    remapped_generate_train_row()

Compile a remapped two-regulator network, generate two ICs from the
stored model, and `train_experiments` with that compiled model.
`compile_network` must stay at zero on the train call.
"""
function remapped_generate_train_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(13)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters(remapped_two_regulator_phys_truth(), p0.nn)
    set = generate_experiment_set(
        MersenneTwister(13); network = net,
        initial_conditions = [
            remapped_two_regulator_state(),
            [0.18, 0.16, 0.22, 0.12]
        ],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = remapped_two_regulator_phys_truth())
    init = pack_parameters(remapped_two_regulator_phys_truth(), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(
            init, set, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    fp = experiment_fingerprint_row(set)
    batches = experiment_batch_row(set; batch_size = 1)
    arch = compiled_nn_architecture(model, packed)
    return (;
        compiles = n,
        n_ics = length(set.experiments),
        compiled_once = experiment_set_is_compiled_once(set),
        fingerprint = fp,
        batches,
        n_heads = arch.n_heads,
        arities = arch.arities,
        dense = arch.dense,
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                fp.holds && batches.holds && arch.n_heads == 2 &&
                arch.arities == [1, 2] && arch.dense)
end

function two_regulator_generate_train_row()
    net = build_two_regulator_unknown_network()
    rng = MersenneTwister(19)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(19); network = net,
        initial_conditions = [[0.25, 0.20, 0.15], [0.30, 0.22, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_es = 0.8, k_i = 0.5, k_e = 0.4))
    init = pack_parameters((k_es = 1.0, k_i = 0.6, k_e = 0.5), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(
            init, set, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = neural_head_count(model),
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                neural_head_count(model) == 1)
end

function linear_generate_train_row()
    net = build_linear_test_network()
    rng = MersenneTwister(7)
    model, p0 = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(7); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18], [0.18, 0.12]],
        tspan = (0.0, 0.6), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    fp = experiment_fingerprint_row(set)
    batches = experiment_batch_row(set; batch_size = 2)
    weights = experiment_weight_row(set)
    n = with_compile_network_counter() do counter
        train_experiments(
            init, set, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        fp, batches, weights,
        compiles = n,
        n_ics = length(set),
        holds = fp.holds && batches.holds && weights.holds && n == 0)
end

function hill_ude_generate_train_row()
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    truth = build_hill_recovery_network(; known = true, hill_order = 2)
    rng = MersenneTwister(11)
    model, p0 = build_ude_model(rng, net)
    set = unique_claim_experiment_set(
        MersenneTwister(103), truth; smoke = true,
        truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6))
    init = pack_parameters((k_prod = 0.8, k_rs = 0.9, k_r = 0.7), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(
            init, set, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    fp = unique_claim_fingerprint_set_row()
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_ics = length(set),
        fingerprint = fp,
        holds = n == 0 && experiment_set_is_compiled_once(set) && fp.holds)
end

function dual_generate_train_row()
    net = build_dual_unknown_network()
    rng = MersenneTwister(21)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(21); network = net,
        initial_conditions = [[0.22, 0.18, 0.16], [0.30, 0.24, 0.20]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
    init = pack_parameters((k_ca = 1.0, k_cb = 1.0, k_c = 0.6), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(
            init, set, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        n_heads = neural_head_count(model),
        recovery_admits = unique_claim_recovery_admits(net),
        holds = n == 0 && neural_head_count(model) == 2 &&
                unique_claim_recovery_admits(net) == false)
end

function six_state_generate_train_row()
    net = build_six_state_unknown_network(; known = false)
    rng = MersenneTwister(41)
    model, p0 = build_ude_model(rng, net)
    ics = [[0.22, 0.18, 0.16, 0.14, 0.12, 0.10],
           [0.28, 0.20, 0.18, 0.16, 0.14, 0.12]]
    set = generate_experiment_set(
        MersenneTwister(41); network = net,
        initial_conditions = ics,
        tspan = (0.0, 0.4), n_points = 5, noise_σ = 0.0)
    n = with_compile_network_counter() do counter
        train_experiments(
            pack_parameters(
                NamedTuple{Tuple(parameter_schema(model).phys_names)}(
                    ntuple(_ -> 0.8, length(parameter_schema(model).phys_names))),
                p0.nn),
            set, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        nstates = model.compiled.nstates,
        n_heads = neural_head_count(model),
        holds = n == 0 && model.compiled.nstates == 6 &&
                neural_head_count(model) == 1)
end

function skipped_duplicate_generate_train_row()
    net = build_skipped_duplicate_unknown_network()
    rng = MersenneTwister(13)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(13); network = net,
        initial_conditions = [[0.2, 0.3, 0.4], [0.25, 0.28, 0.35]],
        tspan = (0.0, 0.4), n_points = 5, noise_σ = 0.0,
        truth_params = (k_ca = 0.8, k_b = 0.5, k_c = 0.4))
    init = pack_parameters((k_ca = 0.9, k_b = 0.6, k_c = 0.5), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(
            init, set, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        dense = neural_index_is_dense(model),
        n_heads = neural_head_count(model),
        holds = n == 0 && neural_index_is_dense(model) &&
                neural_head_count(model) == 2)
end

function linear_checkpoint_fixture_row(; dir)
    net = build_linear_test_network()
    rng = MersenneTwister(53)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(53); network = net,
        initial_conditions = [[0.22, 0.14]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    exp = first(set.experiments)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    resume = checkpoint_resume_row(
        init, exp.observations, exp.times, exp.u0,
        (first(exp.times), last(exp.times)), model; dir = dir)
    trained = train_ude(
        init, exp.observations, exp.times, exp.u0,
        (first(exp.times), last(exp.times)), model;
        config = TrainingConfig(
            adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
        verbose = false)
    artifact = artifact_roundtrip_row(trained; dir = dir)
    schema = checkpoint_schema_row()
    return (;
        resume, artifact, schema,
        holds = resume.holds && artifact.holds && schema.holds)
end

function experiment_checkpoint_fixture_matrix(; dir)
    linear = linear_generate_train_row()
    remap = remapped_generate_train_row()
    two = two_regulator_generate_train_row()
    hill = hill_ude_generate_train_row()
    dual = dual_generate_train_row()
    six = six_state_generate_train_row()
    skipped = skipped_duplicate_generate_train_row()
    claim = unique_claim_fingerprint_set_row()
    ckpt = linear_checkpoint_fixture_row(; dir = dir)
    extra = experiment_checkpoint_extended_matrix(; dir = dir)
    return (;
        linear, remap, two, hill, dual, six, skipped, claim, ckpt, extra,
        holds = linear.holds && remap.holds && two.holds && hill.holds &&
                dual.holds && six.holds && skipped.holds && claim.holds &&
                ckpt.holds && extra.holds)
end

# -- Source locks -------------------------------------------------------------

function experiment_fingerprint_source_holds()
    src = read(experiment_jl_source_path(), String)
    start = findfirst("function experiment_fingerprint", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("exp.times", body) &&
           occursin("exp.observations", body) &&
           occursin("exp.mask", body) &&
           occursin("exp.u0", body) &&
           !occursin("exp.metadata", body)
end

function experiment_batches_source_holds()
    src = read(experiment_jl_source_path(), String)
    start = findfirst("function experiment_batches", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("batch_size", body) &&
           occursin("shuffle", body) &&
           occursin("Random.shuffle!", body)
end

function experiment_checkpoint_source_holds()
    src = read(experiment_checkpoint_source_path(), String)
    docs = isfile(experiment_checkpoint_docs_path()) ?
        read(experiment_checkpoint_docs_path(), String) : ""
    impl = read(experiment_jl_source_path(), String)
    return all(occursin(needle, src) for needle in EXPERIMENT_CHECKPOINT_MUST_CONTAIN) &&
           !occursin("support_f1_ude = 0.99", impl) &&
           !occursin("support_f1_ude = 0.99", docs) &&
           !occursin("function validate_network", docs)
end

function experiment_checkpoint_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "experiment-checkpoint.md")
end

function experiment_checkpoint_docs_hold()
    path = experiment_checkpoint_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(experiment_checkpoint_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("experiment-checkpoint.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function experiment_checkpoint_landing_docs_hold()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    sentences = experiment_checkpoint_locked_sentences()
    return occursin("experiment-checkpoint", howto) &&
           occursin("experiment_fingerprint", howto) &&
           occursin(sentences.remapped, sciml)
end

function experiment_checkpoint_source_violations()
    src = read(experiment_checkpoint_source_path(), String)
    impl = read(experiment_jl_source_path(), String)
    docs = isfile(experiment_checkpoint_docs_path()) ?
        read(experiment_checkpoint_docs_path(), String) : ""
    missing = [s for s in EXPERIMENT_CHECKPOINT_MUST_CONTAIN if !occursin(s, src)]
    forbidden = String[]
    occursin("support_f1_ude = 0.99", impl) &&
        push!(forbidden, "Experiments.jl: support_f1_ude = 0.99")
    occursin("function validate_network", docs) &&
        push!(forbidden, "docs: function validate_network")
    return (; missing, forbidden)
end

function experiment_checkpoint_contract_holds()
    return experiment_checkpoint_source_holds() &&
           experiment_fingerprint_source_holds() &&
           experiment_batches_source_holds() &&
           resume_source_holds() &&
           save_checkpoint_source_holds() &&
           load_checkpoint_source_holds() &&
           experiment_checkpoint_docs_hold() &&
           experiment_checkpoint_landing_docs_hold() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source()
end

# -- Checkpoint serialize contract --------------------------------------------

function save_checkpoint_source_holds()
    src = read(training_jl_source_path(), String)
    start = findfirst("function save_checkpoint", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("serialize(io, checkpoint)", body) &&
           !occursin("JSON", body) &&
           !occursin("json", body)
end

function load_checkpoint_source_holds()
    src = read(training_jl_source_path(), String)
    start = findfirst("function load_checkpoint", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("deserialize", body) &&
           occursin("CHECKPOINT_SCHEMA_VERSION.major", body) &&
           occursin("checkpoint isa Checkpoint", body)
end

function checkpoint_metadata_source_row()
    src = read(training_jl_source_path(), String)
    start = findfirst("function resume_training", src)
    start === nothing && return (; holds = false)
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    needles = (
        "optimizer_state = checkpoint.optimizer_state",
        "initial_iteration = checkpoint.iteration",
        "dual_init = dual",
        "rho_init = rho",
        "initial_outer = outer",
        "initial_stage = stage",
        "initial_stage_iteration = stage_iteration",
        "previous_residual_init = previous_residual")
    missing = [n for n in needles if !occursin(n, body)]
    return (;
        missing,
        n_needles = length(needles),
        holds = isempty(missing) && !occursin("compile_network", body))
end

# -- Fingerprint extras -------------------------------------------------------

"""
    hex_fingerprint_row(digest)

A data/experiment fingerprint is a 64-character lowercase hex SHA-256.
"""
function hex_fingerprint_row(digest::AbstractString)
    hex = all(c -> c in "0123456789abcdef", digest)
    return (;
        digest,
        width = length(digest),
        hex,
        holds = length(digest) == 64 && hex)
end

"""
    units_change_fingerprint_row(set)

`ExperimentSet.units` is part of the identity. A unit relabel must change
the hash. Metadata still must not.
"""
function units_change_fingerprint_row(set::ExperimentSet)
    base = experiment_fingerprint(set)
    relabeled = ExperimentSet(
        set.experiments, set.state_names;
        units = fill(:micromolar, length(set.state_names)),
        metadata = set.metadata)
    unit_hash = experiment_fingerprint(relabeled)
    names_changed = ExperimentSet(
        set.experiments,
        [Symbol(string(name, "_x")) for name in set.state_names];
        units = set.units,
        metadata = set.metadata)
    name_hash = experiment_fingerprint(names_changed)
    hex = hex_fingerprint_row(base)
    return (;
        base,
        unit_hash,
        name_hash,
        units_change = base != unit_hash,
        names_change = base != name_hash,
        hex,
        holds = base != unit_hash && base != name_hash && hex.holds)
end

"""
    irregular_times_fingerprint_row(set)

A strictly increasing but irregular time grid is a different experiment
from the uniform grid with the same observations.
"""
function irregular_times_fingerprint_row(set::ExperimentSet)
    exp = first(set.experiments)
    n = length(exp.times)
    n ≥ 3 || return (; holds = false, reason = :too_short)
    shifted = collect(exp.times)
    shifted[2] = (shifted[1] + shifted[2]) / 2 + (shifted[2] - shifted[1]) / 4
    issorted(shifted) && all(diff(shifted) .> 0) ||
        return (; holds = false, reason = :not_increasing)
    replaced = Experiment(
        exp.name, shifted, exp.observations, exp.u0;
        mask = exp.mask, metadata = exp.metadata)
    rest = set.experiments[2:end]
    warped = ExperimentSet(vcat([replaced], rest), set.state_names;
        units = set.units)
    base = experiment_fingerprint(set)
    warped_hash = experiment_fingerprint(warped)
    return (;
        base,
        warped_hash,
        n_points = n,
        changes = base != warped_hash,
        holds = base != warped_hash)
end

"""
    noise_fingerprint_row()

Two compiled-once sets that differ only in `noise_σ` must not share a
fingerprint. Zero-noise repeats with the same RNG share a fingerprint.
"""
function noise_fingerprint_row()
    net = build_linear_test_network()
    ics = [[0.22, 0.14], [0.30, 0.18]]
    quiet = generate_experiment_set(
        MersenneTwister(71); network = net,
        initial_conditions = ics, tspan = (0.0, 0.5), n_points = 6,
        noise_σ = 0.0, truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    quiet2 = generate_experiment_set(
        MersenneTwister(71); network = net,
        initial_conditions = ics, tspan = (0.0, 0.5), n_points = 6,
        noise_σ = 0.0, truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    noisy = generate_experiment_set(
        MersenneTwister(71); network = net,
        initial_conditions = ics, tspan = (0.0, 0.5), n_points = 6,
        noise_σ = 0.05, truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    a = experiment_fingerprint(quiet)
    b = experiment_fingerprint(quiet2)
    c = experiment_fingerprint(noisy)
    return (;
        quiet = a,
        quiet_repeat = b,
        noisy = c,
        compiled_once = experiment_set_is_compiled_once(quiet) &&
                        experiment_set_is_compiled_once(noisy),
        same_zero = a == b,
        noise_changes = a != c,
        holds = a == b && a != c &&
                experiment_set_is_compiled_once(quiet) &&
                experiment_set_is_compiled_once(noisy))
end

"""
    generated_data_fingerprint_row()

`generate_from_compiled_model` at noise 0 is deterministic. The same
`(u0, tspan, n_points)` hashes identically; a different IC does not.
The second generate does not call `compile_network`.
"""
function generated_data_fingerprint_row()
    net = build_linear_test_network()
    rng = MersenneTwister(73)
    model, p0 = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), p0.nn)
    u0 = [0.22, 0.14]
    other = [0.30, 0.18]
    tspan = (0.0, 0.5)
    n_points = 6
    t1, clean1, _, _ = generate_from_compiled_model(
        model, packed, MersenneTwister(1);
        u0 = u0, tspan = tspan, n_points = n_points, noise_σ = 0.0)
    n = with_compile_network_counter() do counter
        t2, clean2, _, _ = generate_from_compiled_model(
            model, packed, MersenneTwister(2);
            u0 = u0, tspan = tspan, n_points = n_points, noise_σ = 0.0)
        _, clean3, _, _ = generate_from_compiled_model(
            model, packed, MersenneTwister(3);
            u0 = other, tspan = tspan, n_points = n_points, noise_σ = 0.0)
        a = data_fingerprint(clean1, t1, u0)
        b = data_fingerprint(clean2, t2, u0)
        c = data_fingerprint(clean3, t2, other)
        return (;
            a, b, c,
            compiles = counter[],
            same_ic = a == b,
            other_ic = a != c,
            matches = clean1 ≈ clean2,
            holds = counter[] == 0 && a == b && a != c && clean1 ≈ clean2)
    end
    return n
end

"""
    unique_claim_from_compiled_fingerprint_row()

Compile the unique-claim truth once, then generate the smoke set from
that stored model. Fingerprints and `compile_network` stay at zero
while the set is built.
"""
function unique_claim_from_compiled_fingerprint_row()
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    truth = compile_ground_truth_model(
        MersenneTwister(103), net; truth_params = truth_params)
    fp = unique_claim_fingerprint(; smoke = true)
    n = with_compile_network_counter() do counter
        set = generate_experiment_set_from_compiled_model(
            truth, MersenneTwister(103);
            initial_conditions = unique_claim_protocol_ics(; smoke = true),
            tspan = fp.tspan, n_points = fp.n_points, noise_σ = 0.0)
        digest = experiment_fingerprint(set)
        ics = [experiment_fingerprint(ExperimentSet(
            [exp], set.state_names; units = set.units))
               for exp in set.experiments]
        return (;
            compiles = counter[],
            digest,
            n_ics = length(set),
            n_points = size(first(set.experiments).observations, 2),
            compiled_once = experiment_set_is_compiled_once(set),
            distinct = length(unique(ics)) == length(ics),
            hex = hex_fingerprint_row(digest),
            holds = counter[] == 0 &&
                    experiment_set_is_compiled_once(set) &&
                    length(set) == fp.n_ics &&
                    size(first(set.experiments).observations, 2) == fp.n_points &&
                    length(unique(ics)) == length(ics))
    end
    return n
end

function unique_claim_ic_fingerprint_uniqueness_row()
    net = build_linear_test_network()
    set = generate_experiment_set(
        MersenneTwister(79); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18], [0.18, 0.12]],
        tspan = (0.0, 0.6), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    hashes = [experiment_fingerprint(ExperimentSet(
        [exp], set.state_names; units = set.units))
              for exp in set.experiments]
    return (;
        n_ics = length(set),
        hashes,
        distinct = length(unique(hashes)) == length(hashes),
        compiled_once = experiment_set_is_compiled_once(set),
        holds = length(unique(hashes)) == length(hashes) &&
                length(set) == 3 &&
                experiment_set_is_compiled_once(set))
end

function csv_experiment_fingerprint_row()
    path = joinpath(pkgdir(BioDynaX), "examples", "data", "unknown_inhibition.csv")
    isfile(path) || return (; holds = false, reason = :missing_csv)
    exp, names = experiment_from_csv(path)
    set = ExperimentSet([exp], collect(Symbol, names))
    again_exp, again_names = experiment_from_csv(path)
    again = ExperimentSet([again_exp], collect(Symbol, again_names))
    a = experiment_fingerprint(set)
    b = experiment_fingerprint(again)
    return (;
        path,
        n_points = length(exp.times),
        nstates = size(exp.observations, 1),
        same = a == b,
        hex = hex_fingerprint_row(a),
        holds = a == b && length(a) == 64)
end

# -- Batch extras -------------------------------------------------------------

"""
    batch_remainder_row(set; batch_size)

The last batch may be shorter than `batch_size`. The partition must
still cover every IC exactly once. No dummy experiment is appended.
"""
function batch_remainder_row(set::ExperimentSet; batch_size::Int = 2)
    names = [exp.name for exp in set.experiments]
    batches = experiment_batches(set, batch_size; shuffle = false)
    widths = [length(batch) for batch in batches]
    flat = reduce(vcat, [[exp.name for exp in batch] for batch in batches])
    last_short = length(set) % batch_size != 0 ?
        last(widths) == length(set) % batch_size : last(widths) == batch_size
    return (;
        n_ics = length(set),
        batch_size,
        n_batches = length(batches),
        widths,
        last_short,
        covers = issetequal(flat, names),
        no_pad = sum(widths) == length(set),
        no_dup = length(flat) == length(unique(flat)),
        holds = issetequal(flat, names) &&
                sum(widths) == length(set) &&
                length(flat) == length(unique(flat)) &&
                last_short)
end

function unique_claim_batch_row()
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    set = unique_claim_experiment_set(
        MersenneTwister(103), net; smoke = true,
        truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6))
    row = experiment_batch_row(set; batch_size = 1)
    remainder = batch_remainder_row(set; batch_size = 1)
    return (;
        row,
        remainder,
        n_ics = length(set),
        smoke = true,
        holds = row.holds && remainder.holds && length(set) == 1)
end

function shuffle_seed_independence_row(set::ExperimentSet; batch_size::Int = 2)
    a = experiment_batches(set, batch_size; shuffle = true, rng = MersenneTwister(3))
    b = experiment_batches(set, batch_size; shuffle = true, rng = MersenneTwister(3))
    c = experiment_batches(set, batch_size; shuffle = true, rng = MersenneTwister(11))
    an = reduce(vcat, [[exp.name for exp in batch] for batch in a])
    bn = reduce(vcat, [[exp.name for exp in batch] for batch in b])
    cn = reduce(vcat, [[exp.name for exp in batch] for batch in c])
    names = [exp.name for exp in set.experiments]
    return (;
        same_seed = an == bn,
        other_seed_covers = issetequal(cn, names),
        same_seed_covers = issetequal(an, names),
        holds = an == bn && issetequal(an, names) && issetequal(cn, names))
end

# -- Masked / frozen / resume extras ------------------------------------------

"""
    masked_fingerprint_train_row()

Hiding one state changes the experiment fingerprint and still trains
without `compile_network`.
"""
function masked_fingerprint_train_row()
    net = build_linear_test_network()
    rng = MersenneTwister(83)
    model, p0 = build_ude_model(rng, net)
    truth = (k_ba = 0.8, k_a = 1.2, k_b = 0.5)
    set = generate_experiment_set(
        MersenneTwister(83); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    experiments = map(set.experiments) do experiment
        mask = copy(experiment.mask)
        mask[1, :] .= false
        mask[1, 1] = true
        Experiment(
            experiment.name, experiment.times, experiment.observations,
            experiment.u0; mask = mask, metadata = experiment.metadata)
    end
    masked = ExperimentSet(experiments, set.state_names; units = set.units)
    base = experiment_fingerprint(set)
    masked_hash = experiment_fingerprint(masked)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(
            init, masked, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        fingerprint_changes = base != masked_hash,
        n_ics = length(masked),
        holds = n == 0 && base != masked_hash)
end

"""
    frozen_phys_checkpoint_row(; dir)

A frozen physical parameter survives checkpoint resume. Resume does not
compile. The frozen name is still the same numeric value.
"""
function frozen_phys_checkpoint_row(; dir)
    net = build_linear_test_network()
    rng = MersenneTwister(89)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(89); network = net,
        initial_conditions = [[0.22, 0.14]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    exp = first(set.experiments)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    frozen = [:k_ba]
    names = Tuple(parameter_schema(model).phys_names)
    before = NamedTuple{names}(ntuple(
        i -> Float64(init.phys[i]), length(names)))
    path = joinpath(dir, "frozen.ckpt")
    config = TrainingConfig(
        adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6,
        frozen_phys = frozen)
    tspan = (first(exp.times), last(exp.times))
    first_fit = train_ude(
        init, exp.observations, exp.times, exp.u0, tspan, model;
        config = config, verbose = false, checkpoint_path = path,
        checkpoint_every = 1, seed = 5)
    isfile(path) || return (; holds = false, reason = :missing_ckpt)
    ckpt = load_checkpoint(path)
    n = with_compile_network_counter() do counter
        resume_training(
            ckpt, exp.observations, exp.times, exp.u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6,
                frozen_phys = frozen),
            verbose = false)
        counter[]
    end
    after = NamedTuple{names}(ntuple(
        i -> Float64(first_fit.params.phys[i]), length(names)))
    frozen_held = all(name -> getfield(before, name) ≈ getfield(after, name),
                      frozen)
    return (;
        compiles = n,
        frozen,
        frozen_held,
        has_state = ckpt.optimizer_state !== nothing,
        holds = n == 0 && frozen_held && ckpt.optimizer_state !== nothing)
end

"""
    resume_equivalence_row(; dir)

Two Adam steps plus a one-step resume stay compile-free. A three-step
fresh run also stays compile-free. This does not claim bit-identical
losses; it claims the resume path does not rebuild the UDE.
"""
function resume_equivalence_row(; dir)
    net = build_linear_test_network()
    rng = MersenneTwister(97)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(97); network = net,
        initial_conditions = [[0.22, 0.14]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    exp = first(set.experiments)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    tspan = (first(exp.times), last(exp.times))
    path = joinpath(dir, "equiv.ckpt")
    short = TrainingConfig(
        adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6)
    first_fit = train_ude(
        init, exp.observations, exp.times, exp.u0, tspan, model;
        config = short, verbose = false, checkpoint_path = path,
        checkpoint_every = 1, seed = 7)
    isfile(path) || return (; holds = false, reason = :missing_ckpt)
    ckpt = load_checkpoint(path)
    n_resume = with_compile_network_counter() do counter
        resume_training(
            ckpt, exp.observations, exp.times, exp.u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6),
            verbose = false)
        counter[]
    end
    n_fresh = with_compile_network_counter() do counter
        train_ude(
            init, exp.observations, exp.times, exp.u0, tspan, model;
            config = TrainingConfig(
                adam_iterations = 3, bfgs_iterations = 0, log_every = 10^6),
            verbose = false, seed = 7)
        counter[]
    end
    return (;
        resume_compiles = n_resume,
        fresh_compiles = n_fresh,
        first_loss = first_fit.final_loss,
        iteration = ckpt.iteration,
        has_state = ckpt.optimizer_state !== nothing,
        holds = n_resume == 0 && n_fresh == 0 &&
                isfinite(first_fit.final_loss) &&
                ckpt.optimizer_state !== nothing)
end

function optimizer_state_checkpoint_row(; dir)
    net = build_linear_test_network()
    rng = MersenneTwister(101)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(101); network = net,
        initial_conditions = [[0.22, 0.14]],
        tspan = (0.0, 0.4), n_points = 5, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    exp = first(set.experiments)
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    tspan = (first(exp.times), last(exp.times))
    path = joinpath(dir, "optstate.ckpt")
    result = train_ude(
        init, exp.observations, exp.times, exp.u0, tspan, model;
        config = TrainingConfig(
            adam_iterations = 2, bfgs_iterations = 0, log_every = 10^6),
        verbose = false, checkpoint_path = path, checkpoint_every = 1,
        seed = 3)
    isfile(path) || return (; holds = false, reason = :missing_ckpt)
    ckpt = load_checkpoint(path)
    from_diag = optimizer_state_from_result(result)
    return (;
        ckpt_state = ckpt.optimizer_state !== nothing,
        diag_state = from_diag !== nothing,
        schema = ckpt.schema_version,
        holds = ckpt.optimizer_state !== nothing &&
                from_diag !== nothing &&
                ckpt.schema_version.major == CHECKPOINT_SCHEMA_VERSION.major)
end

# -- Additional generate + train fixtures -------------------------------------

function _short_train_config()
    TrainingConfig(adam_iterations = 1, bfgs_iterations = 0, log_every = 10^6)
end

function mm_unknown_generate_train_row()
    truth_net = build_mm_recovery_network(; known = true)
    train_net = build_mm_recovery_network(; known = false)
    truth_params = (k_prod = 0.9, vmax = 1.6, km = 0.45, k_rs = 1.0, k_r = 0.6)
    set = generate_experiment_set(
        MersenneTwister(107); network = truth_net,
        initial_conditions = [[0.30, 0.25], [0.22, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = truth_params)
    rng = MersenneTwister(107)
    model, p0 = build_ude_model(rng, train_net)
    init = pack_parameters((k_prod = 0.8, k_rs = 0.9, k_r = 0.7), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(init, set, model; config = _short_train_config(),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = neural_head_count(model),
        recovery_admits = unique_claim_recovery_admits(train_net),
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                neural_head_count(model) == 1 &&
                unique_claim_recovery_admits(train_net))
end

function competitive_known_generate_train_row()
    net = build_competitive_test_network(; known = true)
    truth = (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6, k_s = 0.8, k_i = 0.5)
    set = generate_experiment_set(
        MersenneTwister(109); network = net,
        initial_conditions = [[0.25, 0.45, 0.20], [0.30, 0.40, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    rng = MersenneTwister(109)
    model, p0 = build_ude_model(rng, net)
    init = pack_parameters(truth, p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(init, set, model; config = _short_train_config(),
            verbose = false)
        counter[]
    end
    fp = experiment_fingerprint_row(set)
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = neural_head_count(model),
        fingerprint = fp,
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                neural_head_count(model) == 0 && fp.holds)
end

function competitive_unknown_generate_train_row()
    truth_net = build_competitive_test_network(; known = true)
    train_net = build_competitive_test_network(; known = false)
    truth = (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6, k_s = 0.8, k_i = 0.5)
    set = generate_experiment_set(
        MersenneTwister(113); network = truth_net,
        initial_conditions = [[0.25, 0.45, 0.20], [0.30, 0.40, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = truth)
    rng = MersenneTwister(113)
    model, p0 = build_ude_model(rng, train_net)
    init = pack_parameters((k_in = 1.0, k_s = 0.7, k_i = 0.6), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(init, set, model; config = _short_train_config(),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = neural_head_count(model),
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                neural_head_count(model) ≥ 1)
end

function skipped_middle_generate_train_row()
    net = build_skipped_middle_unknown_network()
    rng = MersenneTwister(127)
    model, p0 = build_ude_model(rng, net)
    ics = [[0.22, 0.18, 0.16, 0.14], [0.28, 0.20, 0.18, 0.16]]
    set = generate_experiment_set(
        MersenneTwister(127); network = net,
        initial_conditions = ics, tspan = (0.0, 0.4), n_points = 5,
        noise_σ = 0.0)
    schema = parameter_schema(model)
    phys = NamedTuple{Tuple(schema.phys_names)}(
        ntuple(_ -> 0.8, length(schema.phys_names)))
    init = pack_parameters(phys, p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(init, set, model; config = _short_train_config(),
            verbose = false)
        counter[]
    end
    arch = compiled_nn_architecture(model, init)
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        nstates = model.compiled.nstates,
        n_heads = arch.n_heads,
        dense = arch.dense,
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                model.compiled.nstates == 4 && arch.dense)
end

function zero_hole_generate_fingerprint_row()
    net = build_zero_unknown_linear_network()
    set = generate_experiment_set(
        MersenneTwister(131); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    fp = experiment_fingerprint_row(set)
    batches = experiment_batch_row(set; batch_size = 1)
    return (;
        compiled_once = experiment_set_is_compiled_once(set),
        holes = count_unknown_destructions(net),
        fingerprint = fp,
        batches,
        validate_open = validate_network(net) === net,
        recovery_admits = unique_claim_recovery_admits(net),
        holds = experiment_set_is_compiled_once(set) &&
                count_unknown_destructions(net) == 0 &&
                fp.holds && batches.holds &&
                validate_network(net) === net &&
                unique_claim_recovery_admits(net) == false)
end

function default_example_generate_train_row()
    net = DEFAULT_EXAMPLE_NETWORK
    set = generate_experiment_set(
        MersenneTwister(137); network = net,
        initial_conditions = [[0.20, 0.10], [0.30, 0.15]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0)
    rng = MersenneTwister(137)
    model, p0 = build_ude_model(rng, net)
    init = p0
    n = with_compile_network_counter() do counter
        train_experiments(init, set, model; config = _short_train_config(),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = neural_head_count(model),
        dense = neural_index_is_dense(model),
        duplicate = default_example_has_duplicate_unknown_declaration(),
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                neural_head_count(model) == 1 &&
                neural_index_is_dense(model) &&
                default_example_has_duplicate_unknown_declaration())
end

function mm_known_generate_row()
    net = build_mm_recovery_network(; known = true)
    set = generate_experiment_set(
        MersenneTwister(139); network = net,
        initial_conditions = [[0.30, 0.25], [0.22, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_prod = 0.9, vmax = 1.6, km = 0.45, k_rs = 1.0, k_r = 0.6))
    fp = experiment_fingerprint_row(set)
    return (;
        compiled_once = experiment_set_is_compiled_once(set),
        holes = count_unknown_destructions(net),
        fingerprint = fp,
        n_ics = length(set),
        holds = experiment_set_is_compiled_once(set) &&
                count_unknown_destructions(net) == 0 && fp.holds)
end

function mm_test_generate_train_row()
    net = build_mm_test_network()
    rng = MersenneTwister(149)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(149); network = net,
        initial_conditions = [[0.30, 0.25], [0.22, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0)
    init = pack_parameters(
        (vmax = 1.6, km = 0.45, k_se = 0.8, k_s = 0.5, k_e = 0.4), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments(init, set, model; config = _short_train_config(),
            verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = neural_head_count(model),
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                neural_head_count(model) == 0)
end

function repressilator_generate_row()
    net = build_repressilator_network()
    set = generate_experiment_set(
        MersenneTwister(151); network = net,
        initial_conditions = [[0.30, 0.20, 0.25], [0.22, 0.28, 0.18]],
        tspan = (0.0, 0.4), n_points = 5, noise_σ = 0.0)
    fp = experiment_fingerprint_row(set)
    batches = experiment_batch_row(set; batch_size = 1)
    return (;
        compiled_once = experiment_set_is_compiled_once(set),
        nstates = length(state_nodes(net)),
        holes = count_unknown_destructions(net),
        fingerprint = fp,
        batches,
        holds = experiment_set_is_compiled_once(set) &&
                length(state_nodes(net)) == 3 &&
                count_unknown_destructions(net) == 0 &&
                fp.holds && batches.holds)
end

# -- Warmup joint generate + train --------------------------------------------

"""
    remapped_warmup_generate_train_row()

The remapped multi-head tree is generated once and trained with
`train_experiments_with_warmup`. `compile_network` stays at zero.
"""
function remapped_warmup_generate_train_row()
    net = build_remapped_two_regulator_network()
    rng = MersenneTwister(157)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(157); network = net,
        initial_conditions = [
            remapped_two_regulator_state(),
            [0.18, 0.16, 0.22, 0.12]
        ],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = remapped_two_regulator_phys_truth())
    init = pack_parameters(remapped_two_regulator_phys_truth(), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments_with_warmup(
            init, set, model;
            config = _short_train_config(), verbose = false)
        counter[]
    end
    arch = compiled_nn_architecture(model, init)
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = arch.n_heads,
        arities = arch.arities,
        dense = arch.dense,
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                arch.n_heads == 2 && arch.arities == [1, 2] && arch.dense)
end

function linear_warmup_generate_train_row()
    net = build_linear_test_network()
    rng = MersenneTwister(163)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(163); network = net,
        initial_conditions = [[0.22, 0.14], [0.30, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    init = pack_parameters((k_ba = 1.0, k_a = 1.0, k_b = 0.6), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments_with_warmup(
            init, set, model;
            config = _short_train_config(), verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        holds = n == 0 && experiment_set_is_compiled_once(set))
end

function hill_warmup_from_compiled_row()
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    train_net = build_hill_recovery_network(; known = false, hill_order = 2)
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    truth = compile_ground_truth_model(
        MersenneTwister(167), truth_net; truth_params = truth_params)
    fp = unique_claim_fingerprint(; smoke = true)
    set = generate_experiment_set_from_compiled_model(
        truth, MersenneTwister(167);
        initial_conditions = unique_claim_protocol_ics(; smoke = true),
        tspan = fp.tspan, n_points = fp.n_points, noise_σ = 0.0)
    rng = MersenneTwister(167)
    model, p0 = build_ude_model(rng, train_net)
    init = pack_parameters((k_prod = 0.8, k_rs = 0.9, k_r = 0.7), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments_with_warmup(
            init, set, model;
            config = _short_train_config(), verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_ics = length(set),
        n_points = size(first(set.experiments).observations, 2),
        n_heads = neural_head_count(model),
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                length(set) == fp.n_ics &&
                neural_head_count(model) == 1)
end

function two_regulator_warmup_generate_train_row()
    net = build_two_regulator_unknown_network()
    rng = MersenneTwister(173)
    model, p0 = build_ude_model(rng, net)
    set = generate_experiment_set(
        MersenneTwister(173); network = net,
        initial_conditions = [[0.25, 0.20, 0.15], [0.30, 0.22, 0.18]],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_es = 0.8, k_i = 0.5, k_e = 0.4))
    init = pack_parameters((k_es = 1.0, k_i = 0.6, k_e = 0.5), p0.nn)
    n = with_compile_network_counter() do counter
        train_experiments_with_warmup(
            init, set, model;
            config = _short_train_config(), verbose = false)
        counter[]
    end
    return (;
        compiles = n,
        compiled_once = experiment_set_is_compiled_once(set),
        n_heads = neural_head_count(model),
        holds = n == 0 && experiment_set_is_compiled_once(set) &&
                neural_head_count(model) == 1)
end

# -- Catalog / matrix / formatter ---------------------------------------------

function experiment_checkpoint_extended_matrix(; dir)
    units = units_change_fingerprint_row(linear_probe_set())
    irregular = irregular_times_fingerprint_row(linear_probe_set())
    noise = noise_fingerprint_row()
    generated = generated_data_fingerprint_row()
    from_compiled = unique_claim_from_compiled_fingerprint_row()
    uniqueness = unique_claim_ic_fingerprint_uniqueness_row()
    csv = csv_experiment_fingerprint_row()
    remainder = batch_remainder_row(linear_probe_set(); batch_size = 2)
    claim_batch = unique_claim_batch_row()
    shuffle = shuffle_seed_independence_row(linear_probe_set(); batch_size = 2)
    masked = masked_fingerprint_train_row()
    frozen = frozen_phys_checkpoint_row(; dir = dir)
    equiv = resume_equivalence_row(; dir = dir)
    optstate = optimizer_state_checkpoint_row(; dir = dir)
    mm_u = mm_unknown_generate_train_row()
    comp_k = competitive_known_generate_train_row()
    comp_u = competitive_unknown_generate_train_row()
    middle = skipped_middle_generate_train_row()
    zero = zero_hole_generate_fingerprint_row()
    default = default_example_generate_train_row()
    mm_k = mm_known_generate_row()
    mm_t = mm_test_generate_train_row()
    repress = repressilator_generate_row()
    remap_w = remapped_warmup_generate_train_row()
    linear_w = linear_warmup_generate_train_row()
    hill_w = hill_warmup_from_compiled_row()
    two_w = two_regulator_warmup_generate_train_row()
    meta = checkpoint_metadata_source_row()
    return (;
        units, irregular, noise, generated, from_compiled, uniqueness, csv,
        remainder, claim_batch, shuffle, masked, frozen, equiv, optstate,
        mm_u, comp_k, comp_u, middle, zero, default, mm_k, mm_t, repress,
        remap_w, linear_w, hill_w, two_w, meta,
        holds = units.holds && irregular.holds && noise.holds &&
                generated.holds && from_compiled.holds && uniqueness.holds &&
                csv.holds && remainder.holds && claim_batch.holds &&
                shuffle.holds && masked.holds && frozen.holds &&
                equiv.holds && optstate.holds && mm_u.holds &&
                comp_k.holds && comp_u.holds && middle.holds &&
                zero.holds && default.holds && mm_k.holds && mm_t.holds &&
                repress.holds && remap_w.holds && linear_w.holds &&
                hill_w.holds && two_w.holds && meta.holds)
end

function linear_probe_set()
    return generate_experiment_set(
        MersenneTwister(181); network = build_linear_test_network(),
        initial_conditions = [
            [0.22, 0.14], [0.30, 0.18], [0.18, 0.12], [0.26, 0.16],
            [0.24, 0.20]
        ],
        tspan = (0.0, 0.5), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
end

function experiment_checkpoint_fixture_names()
    return (
        :linear, :remap, :two, :hill, :dual, :six, :skipped, :claim, :ckpt,
        :units, :irregular, :noise, :generated, :from_compiled, :uniqueness,
        :csv, :remainder, :claim_batch, :shuffle, :masked, :frozen, :equiv,
        :optstate, :mm_unknown, :competitive_known, :competitive_unknown,
        :skipped_middle, :zero_hole, :default_example, :mm_known, :mm_test,
        :repressilator, :remap_warmup, :linear_warmup, :hill_warmup,
        :two_warmup, :metadata_source)
end

function experiment_checkpoint_row_index()
    names = experiment_checkpoint_fixture_names()
    return (;
        n = length(names),
        names,
        unique = length(unique(names)) == length(names),
        holds = length(unique(names)) == length(names) && length(names) ≥ 30)
end

function format_experiment_checkpoint_index()
    io = IOBuffer()
    println(io, "| fixture | role |")
    println(io, "|---|---|")
    println(io, "| linear | generate + train_experiments |")
    println(io, "| remap | remapped multi-head generate + train |")
    println(io, "| two | two-regulator D(S,I) generate + train |")
    println(io, "| hill | unique-claim smoke generate + train |")
    println(io, "| dual | dual-head generate + train |")
    println(io, "| six | six-state generate + train |")
    println(io, "| skipped | skipped-duplicate generate + train |")
    println(io, "| claim | unique-claim smoke fingerprint set |")
    println(io, "| ckpt | linear checkpoint resume |")
    println(io, "| units | units and state-name identity |")
    println(io, "| irregular | irregular time-grid identity |")
    println(io, "| noise | noise_σ changes the hash |")
    println(io, "| generated | generate_from_compiled_model hash |")
    println(io, "| from_compiled | smoke set from stored truth |")
    println(io, "| uniqueness | per-IC fingerprints are distinct |")
    println(io, "| csv | committed demo CSV is stable |")
    println(io, "| remainder | last batch may be short |")
    println(io, "| claim_batch | smoke set batches once |")
    println(io, "| shuffle | seed-3 shuffle is reproducible |")
    println(io, "| masked | hidden state changes hash, train stays compile-free |")
    println(io, "| frozen | frozen_phys survives resume |")
    println(io, "| equiv | resume does not compile |")
    println(io, "| optstate | checkpoint and diagnostics keep Adam state |")
    println(io, "| mm_unknown | known MM generate, unknown train |")
    println(io, "| competitive_known | mechanistic competitive train |")
    println(io, "| competitive_unknown | known generate, unknown train |")
    println(io, "| skipped_middle | remapped 1:2 heads generate + train |")
    println(io, "| zero_hole | known linear fingerprints; recovery rejects |")
    println(io, "| default_example | p53/Mdm2 duplicate-unknown train |")
    println(io, "| mm_known | known MM fingerprints |")
    println(io, "| mm_test | known saturation generate + train |")
    println(io, "| repressilator | three-state known generate |")
    println(io, "| remap_warmup | remapped train_experiments_with_warmup |")
    println(io, "| linear_warmup | linear warmup train |")
    println(io, "| hill_warmup | smoke generate + warmup train |")
    println(io, "| two_warmup | two-regulator warmup train |")
    println(io, "| metadata_source | resume_training restores AL fields |")
    return String(take!(io))
end

function experiment_checkpoint_index_holds()
    text = format_experiment_checkpoint_index()
    index = experiment_checkpoint_row_index()
    return index.holds &&
           occursin("remap_warmup", text) &&
           occursin("from_compiled", text) &&
           occursin("frozen_phys", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function experiment_checkpoint_fixture_matrix_namedtuple(matrix)
    extra = matrix.extra
    return (;
        linear = matrix.linear.holds,
        remap = matrix.remap.holds,
        two = matrix.two.holds,
        hill = matrix.hill.holds,
        dual = matrix.dual.holds,
        six = matrix.six.holds,
        skipped = matrix.skipped.holds,
        claim = matrix.claim.holds,
        ckpt = matrix.ckpt.holds,
        extra = extra.holds,
        remap_compiles = matrix.remap.compiles,
        warmup_compiles = extra.remap_w.compiles,
        generated_compiles = extra.generated.compiles,
        from_compiled_compiles = extra.from_compiled.compiles,
        holds = matrix.holds)
end

function experiment_checkpoint_docs_mention_helpers()
    path = experiment_checkpoint_docs_path()
    isfile(path) || return false
    text = read(path, String)
    return occursin("experiment_fingerprint_row", text) &&
           occursin("experiment_batch_row", text) &&
           occursin("checkpoint_resume_row", text) &&
           occursin("remapped_generate_train_row", text) &&
           occursin("train_experiments_with_warmup", text) &&
           occursin("generate_experiment_set_from_compiled_model", text)
end
