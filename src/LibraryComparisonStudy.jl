###############################################################################
# Library comparison study (not exported).
#
# Runs a trained-model library comparison over several seeds and
# observation-noise levels and scores each discovery library (graph-local,
# global, wrong graph) against the true support of the unknown Hill term. One
# training per (fixture, seed, noise level); every discovery variant and
# library of that pair shares the training. A row records support recall,
# precision, and F1, the number of extra terms, the hybrid residual on the
# first training experiment, the mean hybrid residual on held-out
# experiments, the neural-rate error, and wall times.
#
# Fixtures:
#   :four_state  the trained-model library check (`evaluate_trained_graph_local`):
#                states S, R, Q, Z; three initial conditions; designed sample
#                coordinates. With `design = :constant` (the check's own
#                design, S fixed at 0.4) the discovery variant :study is that
#                check's configuration, unchanged; `design = :varying` spreads
#                S over its observed range on the same samples of R, Q, Z.
#   :two_state   the reference-protocol network (`build_hill_recovery_network`):
#                nine initial conditions split 7/2; samples on the regulator
#                grid of the training experiments. That network declares its
#                unknown term as a reaction only, and the interaction graph is
#                built from edges, so the discovery networks of this fixture
#                add the unknown edge explicitly (R -> S, or S -> S for the
#                wrong graph); training uses the reference network as is.
#
# Discovery variants, all on the same learned-rate samples of a training:
#   :study        the library check's configuration: libraries built by
#                 `local_basis` (the target state is part of every library),
#                 no bootstrap, samples in their generated order.
#   :bootstrap    the same libraries with the reference protocol's bootstrap
#                 (consensus refit and nested pruning), samples unpermuted.
#   :parents      libraries over the parent states only (the target state is
#                 excluded), no bootstrap, samples unpermuted.
#   :reference    parent-only libraries, the reference protocol's bootstrap,
#                 discovery seed, and sample permutation; for the graph-local
#                 library this is exactly `discover_unknown_rate` with
#                 `rate_discovery_config()`.
###############################################################################

"""Seeds of the default library comparison study."""
const LIBRARY_STUDY_SEEDS = (103, 107, 111, 113, 127)

"""Observation-noise standard deviations of the default library comparison study."""
const LIBRARY_STUDY_NOISE_LEVELS = (0.0, 0.02, 0.05)

"""Libraries compared by the study, in the order of the report."""
const LIBRARY_STUDY_LIBRARIES = (:graph_local, :global, :wrong_graph)

"""Fixtures of the study."""
const LIBRARY_STUDY_FIXTURES = (:four_state, :two_state)

"""Discovery variants of the study, in the order of the report."""
const LIBRARY_STUDY_VARIANTS = (:study, :bootstrap, :parents, :reference)

"""
Held-out initial conditions (S, R, Q, Z) of the four-state fixture. They lie
inside the range of the training initial conditions and are never used for
training.
"""
const LIBRARY_STUDY_HOLDOUT_ICS = [
    [0.60, 0.70, 0.30, 0.25],
    [0.25, 0.90, 0.60, 0.10]]

"""Offset added to the study seed for the four-state held-out data generator."""
const LIBRARY_STUDY_HOLDOUT_SEED_OFFSET = 7919

"""
True unknown term of the four-state fixture: `D(R) = vmax R^n / (K^n + R^n)`
on state 2 (R), with the values of `m4b_truth_params`.
"""
const LIBRARY_STUDY_TRUTH = (vmax = 1.7, K = 0.6, n = 2, variable = 2)

"""
True unknown term and truth parameters of the two-state fixture (the
reference protocol's values).
"""
const LIBRARY_STUDY_TWO_STATE_TRUTH = (vmax = 1.8, K = 0.55, n = 2, variable = 2)
const LIBRARY_STUDY_TWO_STATE_PARAMS = (
    k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)

"""Training budgets of the two-state fixture (nine initial conditions, 7/2 split)."""
const LIBRARY_STUDY_TWO_STATE_BUDGET = (
    protocol = (
        n_points = UNIQUE_CLAIM_PROTOCOL.n_points,
        tspan = UNIQUE_CLAIM_PROTOCOL.tspan,
        adam_iterations = UNIQUE_CLAIM_PROTOCOL.adam_iterations,
        bfgs_iterations = UNIQUE_CLAIM_PROTOCOL.bfgs_iterations,
        n_sample_points = 80, x_seed = 619),
    smoke = (
        n_points = UNIQUE_CLAIM_PROTOCOL.smoke_n_points,
        tspan = UNIQUE_CLAIM_PROTOCOL.tspan,
        adam_iterations = 2, bfgs_iterations = 0,
        n_sample_points = 24, x_seed = 619))

"""
Sample coordinate designs of the four-state fixture: `:constant` keeps the
target state S at 0.4 on every sample (the library check's design);
`:varying` spreads S over the range observed in the training experiments.
The two-state fixture always varies S.
"""
const LIBRARY_STUDY_DESIGNS = (:constant, :varying)

"""Default sample coordinate design of the four-state fixture."""
const LIBRARY_STUDY_DEFAULT_DESIGN = :constant

"""Default design of a fixture: `LIBRARY_STUDY_DEFAULT_DESIGN` for the four-state fixture, `:varying` for the two-state fixture."""
function library_study_default_design(fixture::Symbol)
    fixture === :four_state && return LIBRARY_STUDY_DEFAULT_DESIGN
    fixture === :two_state && return :varying
    throw(ArgumentError(
        "fixture must be one of $(LIBRARY_STUDY_FIXTURES); got $(fixture)"))
end

"""Column order of a study row and of the CSV file."""
const LIBRARY_STUDY_COLUMNS = (
    :seed, :noise, :library, :success,
    :support_recall, :support_precision, :support_f1, :extra_terms, :extras,
    :data_residual, :holdout_residual, :nn_rate_rmse,
    :train_time_s, :run_time_s, :fixture, :variant, :design)

"""Columns of the CSV files written before fixtures and variants existed."""
const LIBRARY_STUDY_COLUMNS_V1 = LIBRARY_STUDY_COLUMNS[1:14]

"""Columns of the CSV files written before the design column existed (0.11)."""
const LIBRARY_STUDY_COLUMNS_V2 = LIBRARY_STUDY_COLUMNS[1:16]

# -- Scores --------------------------------------------------------------------

"""Label of a monomial key with state names (`1`, `R^2`, `Q*Z`)."""
function _library_study_term_label(key, names)
    vars, powers = key
    isempty(vars) && return "1"
    parts = String[]
    for (variable, power) in zip(vars, powers)
        name = string(names[variable])
        push!(parts, power == 1 ? name : string(name, "^", power))
    end
    return join(parts, "*")
end

"""
    library_study_scores(discovery, truth; names)

Support scores of the first candidate of `discovery` against `truth`
(`(numerator, denominator)` sets of monomial keys): recall, precision, and
F1 of the combined numerator and denominator support, and the extra
monomials (those in the candidate but not in the truth, counted once even
when they appear in both numerator and denominator) with their labels. A
failed discovery scores zero with no extras.
"""
function library_study_scores(discovery::DiscoveryResult, truth; names)
    candidate = discovery.success && !isempty(discovery.candidates) ?
                discovery.candidates[1] : nothing
    candidate === nothing && return (;
        success = false, candidate = nothing,
        support_recall = 0.0, support_precision = 0.0, support_f1 = 0.0,
        extra_terms = 0, extras = String[])
    scores = support_f1(candidate, truth.numerator, truth.denominator)
    recovered = active_support(candidate)
    truth_keys = union(truth.numerator, truth.denominator)
    extras = String[]
    for key in sort!(collect(union(recovered.numerator, recovered.denominator));
        by = string)
        key in truth_keys && continue
        push!(extras, _library_study_term_label(key, names))
    end
    return (;
        success = true, candidate,
        support_recall = scores.combined.recall,
        support_precision = scores.combined.precision,
        support_f1 = scores.combined.f1,
        extra_terms = length(extras),
        extras)
end

# -- Residuals -----------------------------------------------------------------

"""
Right-hand side that replaces the neural destruction term by a discovered
rate evaluated on the state rows `rows` (all states for candidates regressed
on the full state, the parent rows for parent-only libraries). Same
composition as `compose_hybrid_rhs`, which passes the term's regulators.
"""
function _library_study_hybrid_rhs(model::UDEModel, p, term::NeuralDestructionTerm,
        rate_fn, rows)
    return function (u, _, t)
        du = ude_system(u, p, t, model)::typeof(u)
        nn_D = _destruction_contribution(
            term, term.target, u, p, model.nn, model.st)
        hat_D = rate_fn(u[rows])
        du[term.target] += (nn_D - hat_D) * u[term.target]
        return du
    end
end

"""RMSE of the hybrid model against one experiment; `Inf` when the solve fails."""
function _library_study_residual(model, p, term, rate_fn, rows, experiment::Experiment)
    rhs = _library_study_hybrid_rhs(model, p, term, rate_fn, rows)
    times = experiment.times
    prob = SciMLBase.ODEProblem(rhs, experiment.u0, (first(times), last(times)))
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return Inf
    pred = Array(sol)
    size(pred) == size(experiment.observations) || return Inf
    return sqrt(mean(abs2, pred .- experiment.observations))
end

# -- Discovery variants --------------------------------------------------------

"""
Discovery for one library and variant on the samples `X` (all states) with
the learned rate `D`. Returns the `DiscoveryResult` and the state rows its
candidates index (`rows`), so that scores use the right truth variable and
residuals the right inputs.
"""
function _library_study_discover(variant::Symbol, library::Symbol, X, D, times,
        networks, parent_rows, all_rows, budget;
        stability_selection = nothing)
    if variant === :study || variant === :bootstrap
        network = library === :wrong_graph ? networks.wrong : networks.ude
        scope = library === :global ? :global : :graph
        config = variant === :study ?
                 rate_discovery_config(scope = scope, bootstrap = budget.bootstrap,
            seed = budget.discovery_seed) :
                 rate_discovery_config(scope = scope)
        dX = zeros(eltype(X), size(X))
        dX[1, :] .= vec(D)
        result = discover_equations(X, times, network;
            derivatives = dX, targets = 1, config = config, verbose = false,
            stability_selection = stability_selection)
        return result, all_rows
    end
    rows = library === :graph_local ? parent_rows.true_parent :
           library === :wrong_graph ? parent_rows.wrong_parent :
           parent_rows.all_parents
    R = Matrix(X[rows, :])
    if variant === :parents
        network = _rate_network_from_samples(R)
        dX = zeros(eltype(R), size(R))
        dX[1, :] .= vec(D)
        config = rate_discovery_config(bootstrap = budget.bootstrap,
            seed = budget.discovery_seed)
        result = discover_equations(R, times, network;
            derivatives = dX, targets = 1, config = config, verbose = false,
            stability_selection = stability_selection)
        return result, rows
    elseif variant === :reference
        result = discover_unknown_rate(R, times, Matrix(D);
            config = rate_discovery_config(), verbose = false,
            stability_selection = stability_selection)
        return result, rows
    end
    throw(ArgumentError(
        "variant must be one of $(LIBRARY_STUDY_VARIANTS); got $(variant)"))
end

"""Truth support keyed on the position of the regulator among `rows`; an absent regulator gives an unreachable key."""
function _library_study_truth(truth, rows)
    position = findfirst(==(truth.variable), rows)
    variable = position === nothing ? -1 : position
    return hill_rate_support(truth.n; variable = variable)
end

# -- Fixtures ------------------------------------------------------------------

"""
One trained model of a fixture with everything the discovery variants need:
the model, its parameters, the neural term, the sample coordinates `X` (all
states), the learned rate `D`, the training and held-out experiment sets,
the networks, the parent rows of each library, the truth, the state names,
and the wall times. The `:study` variant of the four-state fixture is the
`TrainedGraphLocalEvidence` itself, so its rows reproduce the library check.
"""
function _library_study_train(fixture::Symbol, seed, noise_σ, kind, training_call,
        holdout_ics, stability_selection, design)
    if fixture === :four_state
        return _library_study_train_four_state(
            seed, noise_σ, kind, training_call, holdout_ics, stability_selection, design)
    elseif fixture === :two_state
        design === :varying || throw(ArgumentError(
            "the two-state fixture always varies S; design must be :varying"))
        return _library_study_train_two_state(seed, noise_σ, kind, training_call)
    end
    throw(ArgumentError(
        "fixture must be one of $(LIBRARY_STUDY_FIXTURES); got $(fixture)"))
end

function _library_study_train_four_state(seed, noise_σ, kind, training_call,
        holdout_ics, stability_selection, design)
    budget = m4b_budget(kind)
    train_time = Ref(NaN)
    timed_call = function (args...; kwargs...)
        train_started = time()
        result = training_call(args...; kwargs...)
        train_time[] = time() - train_started
        return result
    end
    started = time()
    evidence = evaluate_trained_graph_local(;
        kind, training_call = timed_call, seed, noise_σ, stability_selection, design)
    run_time = time() - started
    train_set = library_study_training_set(kind; seed, noise_σ)
    holdout_set = library_study_training_set(kind;
        seed = seed + LIBRARY_STUDY_HOLDOUT_SEED_OFFSET, noise_σ,
        initial_conditions = holdout_ics)
    return (;
        fixture = :four_state,
        model = evidence.model, params = evidence.training.params,
        term = evidence.term, X = evidence.X, D = evidence.D, times = evidence.times,
        train_set, holdout_set,
        networks = (;
            ude = build_three_state_unknown_network(;
                known = false, with_distractor = true, parent = 2),
            wrong = build_wrong_graph_unknown_network(;
                known = false, with_distractor = true)),
        parent_rows = (; true_parent = [2], wrong_parent = [3], all_parents = [2, 3, 4]),
        all_rows = collect(1:4),
        truth = LIBRARY_STUDY_TRUTH,
        names = [node.name for node in evidence.model.network.nodes],
        budget = (; bootstrap = budget.bootstrap,
            discovery_seed = budget.discovery_seed),
        study_discoveries = Dict{Symbol, DiscoveryResult}(
            :graph_local => evidence.graph_discovery,
            :global => evidence.global_discovery,
            :wrong_graph => evidence.wrong_graph_discovery),
        train_time = train_time[], run_time)
end

"""
Designed two-state sample coordinates: R on the regulator grid of the
training experiments (as in the reference protocol) and S spread over its
observed range in a fixed shuffled order, so that S and R are not collinear.
The neural term reads R only, so the learned rate does not depend on S.
"""
function _library_study_two_state_coordinates(train_set, term, n_points, x_seed)
    r = collect(_regulator_grid(train_set, term; npoints = n_points))
    s_values = reduce(vcat,
        (experiment.observations[1, :] for experiment in train_set.experiments))
    lo, hi = extrema(s_values)
    s = collect(range(lo, hi; length = n_points))
    s = s[randperm(MersenneTwister(x_seed), n_points)]
    return permutedims(hcat(s, r))
end

"""
Two-state network with the unknown Hill edge `parent -> S` declared as an
edge, for library construction. `build_hill_recovery_network` declares the
unknown term as a reaction only, and `local_basis` reads parents from the
interaction graph, which is built from edges.
"""
function _library_study_two_state_graph_network(; parent::Int = 2)
    base = build_hill_recovery_network(; known = false, hill_order = 2, parent = parent)
    edges = [EdgeSpec(source = parent, target = 1, kind = INHIBITION,
        family = HILL, known = false)]
    return BiologicalNetwork(base.nodes, edges; reactions = base.reactions)
end

function _library_study_train_two_state(seed, noise_σ, kind, training_call)
    budget = kind === :smoke ? LIBRARY_STUDY_TWO_STATE_BUDGET.smoke :
             kind === :protocol ? LIBRARY_STUDY_TWO_STATE_BUDGET.protocol :
             throw(ArgumentError("kind must be :smoke or :protocol; got $(kind)"))
    truth_net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = build_hill_recovery_network(; known = false, hill_order = 2)
    graph_net = _library_study_two_state_graph_network(; parent = 2)
    wrong_net = _library_study_two_state_graph_network(; parent = 1)
    started = time()
    set = generate_recovery_experiments(
        MersenneTwister(seed), truth_net, LIBRARY_STUDY_TWO_STATE_PARAMS;
        tspan = budget.tspan, n_points = budget.n_points,
        noise_σ = Float64(noise_σ))
    split = unique_claim_experiment_split(set)
    model, p0 = build_ude_model(MersenneTwister(seed), ude_net)
    train_started = time()
    training = training_call(model, p0, split.train;
        adam = budget.adam_iterations, bfgs = budget.bfgs_iterations)
    training isa TrainingResult || throw(ArgumentError(
        "training_call must return a TrainingResult"))
    train_time = time() - train_started
    term = only_unknown_destruction(model)
    X = _library_study_two_state_coordinates(
        split.train, term, budget.n_sample_points, budget.x_seed)
    (_, D, _) = sample_unknown_destruction(model, training.params, X)
    D = Matrix{Float64}(D)
    times = dummy_trained_graph_local_times(size(X, 2))
    return (;
        fixture = :two_state,
        model, params = training.params, term, X = Matrix{Float64}(X), D, times,
        train_set = split.train, holdout_set = split.holdout,
        networks = (; ude = graph_net, wrong = wrong_net),
        parent_rows = (; true_parent = [2], wrong_parent = [1], all_parents = [2]),
        all_rows = collect(1:2),
        truth = LIBRARY_STUDY_TWO_STATE_TRUTH,
        names = [node.name for node in model.network.nodes],
        budget = (; bootstrap = 0, discovery_seed = M4B_PROTOCOL.discovery_seed),
        study_discoveries = nothing,
        train_time, run_time = time() - started)
end

# -- Runs ----------------------------------------------------------------------

"""
    library_comparison_run(; seed, noise_σ, kind=:protocol, fixture=:four_state,
                           variants=(:study,), libraries=LIBRARY_STUDY_LIBRARIES,
                           holdout_ics=LIBRARY_STUDY_HOLDOUT_ICS,
                           training_call=fit_unknown_destruction,
                           stability_selection=nothing, design=nothing,
                           on_row=nothing)

One study run: one training of `fixture` at `seed` and `noise_σ`, then one
row per requested variant and library with the columns in
`LIBRARY_STUDY_COLUMNS`. For the four-state fixture with `design =
:constant` and the `:study` variant the discoveries are those of
`evaluate_trained_graph_local(; kind, seed, noise_σ)`, unchanged.

`design` is the sample coordinate design (`LIBRARY_STUDY_DESIGNS`): for the
four-state fixture `:constant` keeps S at 0.4 on every sample and
`:varying` spreads S over its observed range; `nothing` selects the
fixture's default (`LIBRARY_STUDY_DEFAULT_DESIGN` for the four-state
fixture). The two-state fixture always varies S and accepts only
`:varying`. The design is recorded in the `design` column.

`data_residual` is the hybrid residual on the first training experiment,
`holdout_residual` the mean hybrid residual on the held-out experiments (two
fixed initial conditions with the seed `seed +
LIBRARY_STUDY_HOLDOUT_SEED_OFFSET` for the four-state fixture; experiments 8
and 9 of the 7/2 split for the two-state fixture), and `nn_rate_rmse` the
relative error of the learned rate against the true Hill rate on the sample
coordinates. `train_time_s` is the wall time of the training call and
`run_time_s` the wall time of the training run including sampling; both are
the same for the rows of one run. `stability_selection` is passed to every
discovery (off by default). `on_row` is called with each row as soon as it
exists. Not exported.
"""
function library_comparison_run(;
        seed::Integer,
        noise_σ::Real,
        kind::Symbol = :protocol,
        fixture::Symbol = :four_state,
        variants = (:study,),
        libraries = LIBRARY_STUDY_LIBRARIES,
        holdout_ics = LIBRARY_STUDY_HOLDOUT_ICS,
        training_call = fit_unknown_destruction,
        stability_selection::Union{Nothing, StabilitySelection} = nothing,
        design::Union{Nothing, Symbol} = nothing,
        on_row = nothing)
    design = design === nothing ? library_study_default_design(fixture) : design
    design in LIBRARY_STUDY_DESIGNS || throw(ArgumentError(
        "design must be one of $(LIBRARY_STUDY_DESIGNS); got $(design)"))
    for library in libraries
        library in LIBRARY_STUDY_LIBRARIES || throw(ArgumentError(
            "library must be one of $(LIBRARY_STUDY_LIBRARIES); got $(library)"))
    end
    for variant in variants
        variant in LIBRARY_STUDY_VARIANTS || throw(ArgumentError(
            "variant must be one of $(LIBRARY_STUDY_VARIANTS); got $(variant)"))
    end
    trained = _library_study_train(
        fixture, seed, noise_σ, kind, training_call, holdout_ics, stability_selection,
        design)
    r = vec(trained.X[trained.truth.variable, :])
    nn_rate_rmse = rate_rel_rmse(vec(trained.D),
        hill_rate_truth(r; vmax = trained.truth.vmax, K = trained.truth.K,
            n = trained.truth.n))
    rows = NamedTuple[]
    for variant in variants, library in libraries
        if variant === :study && trained.study_discoveries !== nothing
            discovery = trained.study_discoveries[library]
            discovery_rows = trained.all_rows
        else
            discovery, discovery_rows = _library_study_discover(
                variant, library, trained.X, trained.D, trained.times,
                trained.networks, trained.parent_rows, trained.all_rows,
                trained.budget; stability_selection)
        end
        truth = _library_study_truth(trained.truth, discovery_rows)
        scores = library_study_scores(discovery, truth;
            names = trained.names[discovery_rows])
        data_residual = NaN
        holdout_residual = NaN
        if scores.candidate !== nothing
            rate_fn = equation_to_function(scores.candidate)
            data_residual = _library_study_residual(
                trained.model, trained.params, trained.term, rate_fn, discovery_rows,
                trained.train_set.experiments[1])
            holdout_residual = mean(
                _library_study_residual(
                    trained.model, trained.params, trained.term, rate_fn,
                    discovery_rows, experiment)
            for experiment in trained.holdout_set.experiments)
        end
        row = (;
            seed = Int(seed), noise = Float64(noise_σ), library,
            success = scores.success,
            support_recall = scores.support_recall,
            support_precision = scores.support_precision,
            support_f1 = scores.support_f1,
            extra_terms = scores.extra_terms,
            extras = join(scores.extras, ";"),
            data_residual, holdout_residual, nn_rate_rmse,
            train_time_s = trained.train_time, run_time_s = trained.run_time,
            fixture, variant, design)
        push!(rows, row)
        on_row === nothing || on_row(row)
    end
    return rows
end

"""
    library_comparison_study(; seeds=LIBRARY_STUDY_SEEDS,
                             noise_levels=LIBRARY_STUDY_NOISE_LEVELS,
                             libraries=LIBRARY_STUDY_LIBRARIES,
                             variants=(:study,), fixture=:four_state,
                             kind=:protocol,
                             skip=(seed, noise, library, variant) -> false,
                             on_row=nothing, verbose=false, kwargs...)

Tidy table of the library comparison study: one row (a `NamedTuple` with the
columns in `LIBRARY_STUDY_COLUMNS`) per seed, noise level, variant, and
library. Each (seed, noise level) pair is one `library_comparison_run`, so
its variants and libraries share one training. Rows for which `skip`
returns `true` are not computed, and a pair whose rows are all skipped is
not trained, which lets a script resume from rows it has already written.
`on_row` and the remaining keywords are passed to `library_comparison_run`.
The default study is five seeds, three noise levels, and all three
libraries (15 trainings per fixture). The smoke configuration used in the
test suite is `kind = :smoke` with one seed and one noise level. Not
exported.
"""
function library_comparison_study(;
        seeds = LIBRARY_STUDY_SEEDS,
        noise_levels = LIBRARY_STUDY_NOISE_LEVELS,
        libraries = LIBRARY_STUDY_LIBRARIES,
        variants = (:study,),
        fixture::Symbol = :four_state,
        kind::Symbol = :protocol,
        skip = (seed, noise, library, variant) -> false,
        on_row = nothing,
        verbose::Bool = false,
        kwargs...)
    rows = NamedTuple[]
    for seed in seeds, noise_σ in noise_levels
        wanted = [(variant, library) for variant in variants
                  for library in libraries
                  if !skip(seed, noise_σ, library, variant)]
        isempty(wanted) && continue
        wanted_variants = Tuple(unique(first.(wanted)))
        wanted_libraries = Tuple(unique(last.(wanted)))
        verbose && println(fixture, " seed ", seed, ", noise ", noise_σ, ": training")
        run_rows = library_comparison_run(;
            seed, noise_σ, kind, fixture, variants = wanted_variants,
            libraries = wanted_libraries,
            on_row = row -> ((row.variant, row.library) in wanted &&
                             on_row !== nothing && on_row(row)),
            kwargs...)
        # A pair may need only some (variant, library) combinations.
        filter!(row -> (row.variant, row.library) in wanted, run_rows)
        append!(rows, run_rows)
        if verbose
            for row in run_rows
                println("  ", rpad(string(row.variant), 10), rpad(string(row.library), 12),
                    " F1 ", _format_protocol_value(row.support_f1),
                    "  recall ", _format_protocol_value(row.support_recall),
                    "  extras ", row.extra_terms,
                    "  holdout ", _format_protocol_value(row.holdout_residual))
            end
        end
    end
    return rows
end

"""Smoke configuration of the study: one seed, one noise level, the smoke budget."""
function library_comparison_smoke(; fixture::Symbol = :four_state, kwargs...)
    return library_comparison_study(;
        seeds = (M4B_SMOKE.seed,), noise_levels = (M4B_SMOKE.noise_σ,),
        kind = :smoke, fixture, kwargs...)
end

# -- CSV persistence -----------------------------------------------------------

_library_study_csv_field(value::AbstractString) = string('"', value, '"')
_library_study_csv_field(value::Symbol) = string(value)
_library_study_csv_field(value::Bool) = value ? "true" : "false"
_library_study_csv_field(value::Integer) = string(value)
_library_study_csv_field(value::Real) = repr(Float64(value))

"""Header line of the study CSV."""
library_study_csv_header() = join(string.(LIBRARY_STUDY_COLUMNS), ",")

"""One CSV line for a study row (no newline)."""
function library_study_csv_line(row)
    return join(
        (_library_study_csv_field(getproperty(row, column))
        for column in LIBRARY_STUDY_COLUMNS),
        ",")
end

"""
    append_library_study_row(path, row)

Append `row` to the CSV at `path`, writing the header first when the file does
not exist yet. The file is flushed after every row.
"""
function append_library_study_row(path::AbstractString, row)
    fresh = !isfile(path) || filesize(path) == 0
    mkpath(dirname(abspath(path)))
    open(path, "a") do io
        fresh && println(io, library_study_csv_header())
        println(io, library_study_csv_line(row))
        flush(io)
    end
    return path
end

function _library_study_parse(column::Symbol, text::AbstractString)
    (column === :library || column === :fixture || column === :variant ||
     column === :design) && return Symbol(text)
    column === :success && return text == "true"
    column === :extras && return String(strip(text, '"'))
    (column === :seed || column === :extra_terms) && return parse(Int, text)
    return parse(Float64, text)
end

"""
    read_library_study_csv(path)

Rows written by `append_library_study_row`, as `NamedTuple`s with the columns
in `LIBRARY_STUDY_COLUMNS`. Files written before the `fixture` and `variant`
columns existed are read with `fixture = :four_state` and `variant = :study`;
files written before the `design` column existed (0.11) are read with the
design those runs used (`:constant` for the four-state fixture, `:varying`
for the two-state fixture). An absent or empty file gives an empty vector.
"""
function read_library_study_csv(path::AbstractString)
    rows = NamedTuple[]
    (isfile(path) && filesize(path) > 0) || return rows
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && return rows
    header = Symbol.(split(lines[1], ","))
    columns = header == collect(LIBRARY_STUDY_COLUMNS) ? LIBRARY_STUDY_COLUMNS :
              header == collect(LIBRARY_STUDY_COLUMNS_V2) ? LIBRARY_STUDY_COLUMNS_V2 :
              header == collect(LIBRARY_STUDY_COLUMNS_V1) ? LIBRARY_STUDY_COLUMNS_V1 :
              throw(ArgumentError("unexpected header in $(path): $(lines[1])"))
    for line in lines[2:end]
        fields = split(line, ","; limit = length(columns))
        length(fields) == length(columns) || throw(ArgumentError(
            "malformed line in $(path): $(line)"))
        values = Any[_library_study_parse(column, field)
                     for (column, field) in zip(columns, fields)]
        if columns === LIBRARY_STUDY_COLUMNS_V1
            push!(values, :four_state, :study, :constant)
        elseif columns === LIBRARY_STUDY_COLUMNS_V2
            push!(values, values[15] === :two_state ? :varying : :constant)
        end
        push!(rows, NamedTuple{LIBRARY_STUDY_COLUMNS}(Tuple(values)))
    end
    return rows
end

"""Set of `(seed, noise, library, variant)` keys present in `rows`."""
function library_study_keys(rows)
    return Set((row.seed, row.noise, row.library, row.variant) for row in rows)
end

# -- Summary -------------------------------------------------------------------

"""
    library_study_summary(rows; metrics=(:support_f1, :extra_terms, :holdout_residual))

Median and interquartile range of each metric per fixture, design, variant,
library, and noise level. Returns one `NamedTuple` per group with `fixture`,
`design`, `variant`, `library`, `noise`, `n`, and for each metric
`<metric>_median`, `<metric>_q25`, `<metric>_q75`, computed over the finite
values only. Groups follow `LIBRARY_STUDY_FIXTURES`, `LIBRARY_STUDY_DESIGNS`,
`LIBRARY_STUDY_VARIANTS`, and `LIBRARY_STUDY_LIBRARIES`; noise levels are
sorted.
"""
function library_study_summary(rows;
        metrics = (:support_f1, :extra_terms, :holdout_residual))
    fixtures = [f for f in LIBRARY_STUDY_FIXTURES if any(r -> r.fixture === f, rows)]
    designs = [d for d in LIBRARY_STUDY_DESIGNS if any(r -> r.design === d, rows)]
    variants = [v for v in LIBRARY_STUDY_VARIANTS if any(r -> r.variant === v, rows)]
    libraries = [l for l in LIBRARY_STUDY_LIBRARIES if any(r -> r.library === l, rows)]
    noises = sort!(unique(Float64[r.noise for r in rows]))
    summary = NamedTuple[]
    for fixture in fixtures, design in designs, variant in variants,
        library in libraries, noise in noises

        group = [r
                 for r in rows
                 if r.fixture === fixture && r.design === design &&
                    r.variant === variant && r.library === library && r.noise == noise]
        isempty(group) && continue
        entry = Pair{Symbol, Any}[:fixture => fixture, :design => design,
            :variant => variant, :library => library, :noise => noise,
            :n => length(group)]
        for metric in metrics
            values = Float64[getproperty(r, metric) for r in group]
            finite = filter(isfinite, values)
            if isempty(finite)
                push!(entry, Symbol(metric, :_median) => NaN,
                    Symbol(metric, :_q25) => NaN, Symbol(metric, :_q75) => NaN)
            else
                push!(entry, Symbol(metric, :_median) => median(finite),
                    Symbol(metric, :_q25) => quantile(finite, 0.25),
                    Symbol(metric, :_q75) => quantile(finite, 0.75))
            end
        end
        push!(summary, (; entry...))
    end
    return summary
end

function _library_study_iqr_cell(entry, metric; digits = 2)
    m = getproperty(entry, Symbol(metric, :_median))
    lo = getproperty(entry, Symbol(metric, :_q25))
    hi = getproperty(entry, Symbol(metric, :_q75))
    fmt(x) = isfinite(x) ? string(round(x; digits)) : "NA"
    return string(fmt(m), " [", fmt(lo), ", ", fmt(hi), "]")
end

"""
    format_library_study_summary(summary; metrics=(:support_f1, :extra_terms, :holdout_residual))

Markdown table of a `library_study_summary`: one row per group, cells as
`median [q25, q75]`. The fixture, design, and variant columns are shown
when the summary has more than one fixture, design, or variant.
"""
function format_library_study_summary(summary;
        metrics = (:support_f1, :extra_terms, :holdout_residual))
    show_fixture = length(unique(entry.fixture for entry in summary)) > 1
    show_design = length(unique(entry.design for entry in summary)) > 1
    show_variant = length(unique(entry.variant for entry in summary)) > 1
    io = IOBuffer()
    print(io, "|")
    show_fixture && print(io, " fixture |")
    show_design && print(io, " design |")
    show_variant && print(io, " variant |")
    println(io, " library | noise | n | ", join(string.(metrics), " | "), " |")
    println(io, "|",
        repeat("---|", 3 + show_fixture + show_design + show_variant + length(metrics)))
    for entry in summary
        cells = [_library_study_iqr_cell(entry, metric;
                     digits = metric === :holdout_residual ? 4 : 2)
                 for metric in metrics]
        print(io, "|")
        show_fixture && print(io, " ", entry.fixture, " |")
        show_design && print(io, " ", entry.design, " |")
        show_variant && print(io, " ", entry.variant, " |")
        println(io, " ", entry.library, " | ", entry.noise, " | ", entry.n, " | ",
            join(cells, " | "), " |")
    end
    return String(take!(io))
end
