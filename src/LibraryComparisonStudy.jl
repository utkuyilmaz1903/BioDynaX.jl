###############################################################################
# Library comparison study (not exported).
#
# Runs the trained-model library comparison (`evaluate_trained_graph_local`)
# over several seeds and observation-noise levels and scores each library
# (graph-local, global, wrong graph) against the true support of the unknown
# Hill term. One training per (seed, noise level); the three discoveries
# share that training. Every row records support recall, precision, and F1,
# the number of extra terms, the hybrid residual on the first training
# experiment, the mean hybrid residual on held-out experiments, the
# neural-rate error, and wall times.
###############################################################################

"""Seeds of the default library comparison study."""
const LIBRARY_STUDY_SEEDS = (103, 107, 111, 113, 127)

"""Observation-noise standard deviations of the default library comparison study."""
const LIBRARY_STUDY_NOISE_LEVELS = (0.0, 0.02, 0.05)

"""Libraries compared by the study, in the order of the report."""
const LIBRARY_STUDY_LIBRARIES = (:graph_local, :global, :wrong_graph)

"""
Held-out initial conditions (S, R, Q, Z) of the library comparison study.
They lie inside the range of the training initial conditions and are never
used for training.
"""
const LIBRARY_STUDY_HOLDOUT_ICS = [
    [0.60, 0.70, 0.30, 0.25],
    [0.25, 0.90, 0.60, 0.10]]

"""Offset added to the study seed for the held-out data generator."""
const LIBRARY_STUDY_HOLDOUT_SEED_OFFSET = 7919

"""
True unknown term of the study network: `D(R) = vmax R^n / (K^n + R^n)` on
state 2 (R), with the values of `m4b_truth_params`.
"""
const LIBRARY_STUDY_TRUTH = (vmax = 1.7, K = 0.6, n = 2, variable = 2)

"""Column order of a study row and of the CSV file."""
const LIBRARY_STUDY_COLUMNS = (
    :seed, :noise, :library, :success,
    :support_recall, :support_precision, :support_f1, :extra_terms, :extras,
    :data_residual, :holdout_residual, :nn_rate_rmse,
    :train_time_s, :run_time_s)

function _library_study_discovery(evidence::TrainedGraphLocalEvidence, library::Symbol)
    library === :graph_local && return evidence.graph_discovery
    library === :global && return evidence.global_discovery
    library === :wrong_graph && return evidence.wrong_graph_discovery
    throw(ArgumentError(
        "library must be one of $(LIBRARY_STUDY_LIBRARIES); got $(library)"))
end

"""Label of a monomial key with the study's state names (`1`, `R^2`, `Q*Z`)."""
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
F1 of the combined numerator and denominator support, the number of extra
terms, and their labels. A failed discovery scores zero with no extras.
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
        extra_terms = scores.combined.fp,
        extras)
end

"""
Right-hand side that replaces the neural destruction term by a discovered
rate evaluated on the full state vector. Same composition as
`compose_hybrid_rhs`, but the discovered candidates of the library study
are regressed on all states, not only on the term's regulators.
"""
function _library_study_hybrid_rhs(model::UDEModel, p, term::NeuralDestructionTerm,
        rate_fn)
    return function (u, _, t)
        du = ude_system(u, p, t, model)::typeof(u)
        nn_D = _destruction_contribution(
            term, term.target, u, p, model.nn, model.st)
        hat_D = rate_fn(u)
        du[term.target] += (nn_D - hat_D) * u[term.target]
        return du
    end
end

"""RMSE of the hybrid model against one experiment; `Inf` when the solve fails."""
function _library_study_residual(model, p, term, rate_fn, experiment::Experiment)
    rhs = _library_study_hybrid_rhs(model, p, term, rate_fn)
    times = experiment.times
    prob = SciMLBase.ODEProblem(rhs, experiment.u0, (first(times), last(times)))
    sol = solve(prob, Tsit5(); saveat = times, sensealg = nothing)
    SciMLBase.successful_retcode(sol) || return Inf
    pred = Array(sol)
    size(pred) == size(experiment.observations) || return Inf
    return sqrt(mean(abs2, pred .- experiment.observations))
end

"""
    library_comparison_run(; seed, noise_σ, kind=:protocol,
                           libraries=LIBRARY_STUDY_LIBRARIES,
                           holdout_ics=LIBRARY_STUDY_HOLDOUT_ICS,
                           training_call=fit_unknown_destruction,
                           on_row=nothing)

One study run: `evaluate_trained_graph_local(; kind, seed, noise_σ)` (one
training, one learned-rate sample, three discoveries), then one row per
requested library with the columns in `LIBRARY_STUDY_COLUMNS`.

`data_residual` is the hybrid residual on the first training experiment,
`holdout_residual` the mean hybrid residual on the experiments generated from
`holdout_ics` with the same noise level and the seed
`seed + LIBRARY_STUDY_HOLDOUT_SEED_OFFSET`, and `nn_rate_rmse` the relative
error of the learned rate against the true Hill rate on the designed sample
coordinates. `train_time_s` is the wall time of the training call and
`run_time_s` the wall time of the whole single run; both are the same for
the rows of one run. `on_row` is called with each row as soon as it exists.
Not exported.
"""
function library_comparison_run(;
        seed::Integer,
        noise_σ::Real,
        kind::Symbol = :protocol,
        libraries = LIBRARY_STUDY_LIBRARIES,
        holdout_ics = LIBRARY_STUDY_HOLDOUT_ICS,
        training_call = fit_unknown_destruction,
        on_row = nothing)
    for library in libraries
        library in LIBRARY_STUDY_LIBRARIES || throw(ArgumentError(
            "library must be one of $(LIBRARY_STUDY_LIBRARIES); got $(library)"))
    end
    train_time = Ref(NaN)
    timed_call = function (args...; kwargs...)
        started = time()
        result = training_call(args...; kwargs...)
        train_time[] = time() - started
        return result
    end
    started = time()
    evidence = evaluate_trained_graph_local(;
        kind, training_call = timed_call, seed, noise_σ)
    run_time = time() - started
    train_set = library_study_training_set(kind; seed, noise_σ)
    holdout_set = library_study_training_set(kind;
        seed = seed + LIBRARY_STUDY_HOLDOUT_SEED_OFFSET, noise_σ,
        initial_conditions = holdout_ics)
    names = [node.name for node in evidence.model.network.nodes]
    truth = hill_rate_support(LIBRARY_STUDY_TRUTH.n;
        variable = LIBRARY_STUDY_TRUTH.variable)
    r = vec(evidence.X[LIBRARY_STUDY_TRUTH.variable, :])
    nn_rate_rmse = rate_rel_rmse(vec(evidence.D),
        hill_rate_truth(r; vmax = LIBRARY_STUDY_TRUTH.vmax,
            K = LIBRARY_STUDY_TRUTH.K, n = LIBRARY_STUDY_TRUTH.n))
    rows = NamedTuple[]
    for library in libraries
        discovery = _library_study_discovery(evidence, library)
        scores = library_study_scores(discovery, truth; names)
        data_residual = NaN
        holdout_residual = NaN
        if scores.candidate !== nothing
            rate_fn = equation_to_function(scores.candidate)
            data_residual = _library_study_residual(
                evidence.model, evidence.training.params, evidence.term, rate_fn,
                train_set.experiments[1])
            holdout_residual = mean(
                _library_study_residual(
                    evidence.model, evidence.training.params, evidence.term,
                    rate_fn, experiment)
            for experiment in holdout_set.experiments)
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
            train_time_s = train_time[], run_time_s = run_time)
        push!(rows, row)
        on_row === nothing || on_row(row)
    end
    return rows
end

"""
    library_comparison_study(; seeds=LIBRARY_STUDY_SEEDS,
                             noise_levels=LIBRARY_STUDY_NOISE_LEVELS,
                             libraries=LIBRARY_STUDY_LIBRARIES,
                             kind=:protocol, skip=(seed, noise, library) -> false,
                             on_row=nothing, verbose=false, kwargs...)

Tidy table of the library comparison study: one row (a `NamedTuple` with the
columns in `LIBRARY_STUDY_COLUMNS`) per seed, noise level, and library. Each
(seed, noise level) pair is one `library_comparison_run`; the libraries of a
pair share one training. Rows for which `skip` returns `true` are not
computed, and a pair whose libraries are all skipped is not trained, which
lets a script resume from rows it has already written. `on_row` is passed
through to `library_comparison_run`. The default study is five seeds, three
noise levels, and all three libraries (15 trainings). The smoke
configuration used in the test suite is `kind = :smoke` with one seed and
one noise level. Not exported.
"""
function library_comparison_study(;
        seeds = LIBRARY_STUDY_SEEDS,
        noise_levels = LIBRARY_STUDY_NOISE_LEVELS,
        libraries = LIBRARY_STUDY_LIBRARIES,
        kind::Symbol = :protocol,
        skip = (seed, noise, library) -> false,
        on_row = nothing,
        verbose::Bool = false,
        kwargs...)
    rows = NamedTuple[]
    for seed in seeds, noise_σ in noise_levels
        wanted = Tuple(l for l in libraries if !skip(seed, noise_σ, l))
        isempty(wanted) && continue
        verbose && println("seed ", seed, ", noise ", noise_σ, ": training")
        run_rows = library_comparison_run(;
            seed, noise_σ, kind, libraries = wanted, on_row, kwargs...)
        append!(rows, run_rows)
        if verbose
            for row in run_rows
                println("  ", rpad(string(row.library), 12),
                    " F1 ", _format_protocol_value(row.support_f1),
                    "  extras ", row.extra_terms,
                    "  holdout ", _format_protocol_value(row.holdout_residual))
            end
        end
    end
    return rows
end

"""Smoke configuration of the study: one seed, one noise level, the smoke budget."""
function library_comparison_smoke(; kwargs...)
    return library_comparison_study(;
        seeds = (M4B_SMOKE.seed,), noise_levels = (M4B_SMOKE.noise_σ,),
        kind = :smoke, kwargs...)
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
    column === :library && return Symbol(text)
    column === :success && return text == "true"
    column === :extras && return String(strip(text, '"'))
    (column === :seed || column === :extra_terms) && return parse(Int, text)
    return parse(Float64, text)
end

"""
    read_library_study_csv(path)

Rows written by `append_library_study_row`, as `NamedTuple`s with the columns
in `LIBRARY_STUDY_COLUMNS`. An absent or empty file gives an empty vector.
"""
function read_library_study_csv(path::AbstractString)
    rows = NamedTuple[]
    (isfile(path) && filesize(path) > 0) || return rows
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && return rows
    header = Symbol.(split(lines[1], ","))
    header == collect(LIBRARY_STUDY_COLUMNS) || throw(ArgumentError(
        "unexpected header in $(path): $(lines[1])"))
    for line in lines[2:end]
        fields = split(line, ","; limit = length(LIBRARY_STUDY_COLUMNS))
        length(fields) == length(LIBRARY_STUDY_COLUMNS) || throw(ArgumentError(
            "malformed line in $(path): $(line)"))
        values = Tuple(_library_study_parse(column, field)
        for (column, field) in zip(LIBRARY_STUDY_COLUMNS, fields))
        push!(rows, NamedTuple{LIBRARY_STUDY_COLUMNS}(values))
    end
    return rows
end

"""Set of `(seed, noise, library)` keys present in `rows`."""
function library_study_keys(rows)
    return Set((row.seed, row.noise, row.library) for row in rows)
end

# -- Summary -------------------------------------------------------------------

"""
    library_study_summary(rows; metrics=(:support_f1, :extra_terms, :holdout_residual))

Median and interquartile range of each metric per library and noise level.
Returns one `NamedTuple` per (library, noise) with `n`, and for each metric
`<metric>_median`, `<metric>_q25`, `<metric>_q75`, computed over the finite
values only. Libraries follow `LIBRARY_STUDY_LIBRARIES` and noise levels are
sorted.
"""
function library_study_summary(rows;
        metrics = (:support_f1, :extra_terms, :holdout_residual))
    libraries = [l for l in LIBRARY_STUDY_LIBRARIES if any(r -> r.library === l, rows)]
    noises = sort!(unique(Float64[r.noise for r in rows]))
    summary = NamedTuple[]
    for library in libraries, noise in noises
        group = [r for r in rows if r.library === library && r.noise == noise]
        isempty(group) && continue
        entry = Pair{Symbol, Any}[:library => library, :noise => noise,
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

Markdown table of a `library_study_summary`: one row per library and noise
level, cells as `median [q25, q75]`.
"""
function format_library_study_summary(summary;
        metrics = (:support_f1, :extra_terms, :holdout_residual))
    io = IOBuffer()
    println(io, "| library | noise | n | ", join(string.(metrics), " | "), " |")
    println(io, "|---|---|---|", repeat("---|", length(metrics)))
    for entry in summary
        cells = [_library_study_iqr_cell(entry, metric;
                     digits = metric === :holdout_residual ? 4 : 2)
                 for metric in metrics]
        println(io, "| ", entry.library, " | ", entry.noise, " | ", entry.n, " | ",
            join(cells, " | "), " |")
    end
    return String(take!(io))
end
