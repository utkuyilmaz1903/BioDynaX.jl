###############################################################################
# Multi-term study (0.16): two or three unknown destruction terms, with the
# single-unknown control of the same network, on a seed × noise grid, with
# and without stability selection. Rows are appended to a CSV as each run
# finishes and a rerun skips the runs already recorded. Not exported; the
# command line lives in benchmark/multi_term_study.jl.
###############################################################################

# -- Fixtures ------------------------------------------------------------------

const MULTI_TERM_STUDY_SEEDS = (103, 107, 111, 113, 127)
const MULTI_TERM_STUDY_NOISE_LEVELS = (0.0, 0.02, 0.05)
const MULTI_TERM_STUDY_FIXTURES = (:separate, :coupled, :three)

function _hill(name, target, regulator, vmax, K; known)
    ReactionSpec(name = name,
        stoichiometry = Dict(target => -1.0), regulators = [regulator], known = known,
        family = HILL, metadata = HillMetadata(
            vmax_param = vmax, k_param = K, hill_order = 2))
end
function _mass(name, target, regulator, rate)
    ReactionSpec(name = name,
        stoichiometry = Dict(target => 1.0), regulators = [regulator],
        metadata = MassActionMetadata(rate_param = rate))
end
function _decay(name, target, rate)
    ReactionSpec(name = name, stoichiometry = Dict(target => -1.0),
        regulators = Int[], metadata = LinearDecayMetadata(rate_param = rate))
end

"""
    build_two_term_separate_network(; unknown = (:S1, :S2))

Four states `S1, R1, S2, R2`: two copies of the tutorial's two-state motif
(`S` produced in proportion to `R` and degraded by a Hill term in `R`; `R`
produced from `S` and decaying linearly), joined by the production of `S2`
from `R1`. The unknown terms sit on `S1` (regulator `R1`) and `S2`
(regulator `R2`): two nodes that do not regulate each other's term, with
distinct regulators. `unknown` lists the nodes whose Hill degradation is
unknown (`()` gives the fully known truth).
"""
function build_two_term_separate_network(; unknown = (:S1, :S2))
    nodes = [NodeSpec(name = :S1), NodeSpec(name = :R1), NodeSpec(name = :S2),
        NodeSpec(name = :R2)]
    reactions = [
        _mass(:produce_s1, 1, 2, :k_prod1),
        _hill(:hill_s1, 1, 2, :vmax1, :K1; known = !(:S1 in unknown)),
        _mass(:produce_r1, 2, 1, :k_rs1),
        _decay(:decay_r1, 2, :k_r1),
        _mass(:produce_s2, 3, 2, :k_prod2),
        _hill(:hill_s2, 3, 4, :vmax2, :K2; known = !(:S2 in unknown)),
        _mass(:produce_r2, 4, 3, :k_rs2),
        _decay(:decay_r2, 4, :k_r2)]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

const TWO_TERM_SEPARATE_TRUTH = (k_prod1 = 0.9, vmax1 = 1.8, K1 = 0.55, k_rs1 = 1.0,
    k_r1 = 0.6, k_prod2 = 0.7, vmax2 = 1.5, K2 = 0.5, k_rs2 = 0.9, k_r2 = 0.5)

"""
    build_two_term_coupled_network(; unknown = (:A, :B))

Three states `A, B, C`: `A` and `B` are produced from `C`, `C` decays
linearly, and each of `A` and `B` is degraded by a Hill term in the other.
The unknown terms sit on adjacent nodes: the regulator of each unknown term
is the other unknown node. Same graph as `build_dual_unknown_network`, with
named Hill parameters so that the fully known truth and the single-unknown
controls can be built (`unknown = ()`, `(:A,)`, `(:B,)`).
"""
function build_two_term_coupled_network(; unknown = (:A, :B))
    nodes = [NodeSpec(name = :A), NodeSpec(name = :B), NodeSpec(name = :C)]
    reactions = [
        _mass(:drive_a, 1, 3, :k_ca),
        _mass(:drive_b, 2, 3, :k_cb),
        _hill(:a_decay, 1, 2, :vmax_a, :K_a; known = !(:A in unknown)),
        _hill(:b_decay, 2, 1, :vmax_b, :K_b; known = !(:B in unknown)),
        _decay(:c_decay, 3, :k_c)]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

const TWO_TERM_COUPLED_TRUTH = (k_ca = 0.8, k_cb = 0.9, vmax_a = 1.5, K_a = 0.5,
    vmax_b = 1.2, K_b = 0.6, k_c = 0.5)

"""
    build_three_term_network(; unknown = (:S1, :R1, :S2))

`build_two_term_separate_network` with the linear decay of `R1` replaced by a
Hill term in `S1`, so that three terms can be unknown: `S1` and `R1` regulate
each other's term (adjacent), `S2` is regulated by `R2` (separate).
"""
function build_three_term_network(; unknown = (:S1, :R1, :S2))
    nodes = [NodeSpec(name = :S1), NodeSpec(name = :R1), NodeSpec(name = :S2),
        NodeSpec(name = :R2)]
    reactions = [
        _mass(:produce_s1, 1, 2, :k_prod1),
        _hill(:hill_s1, 1, 2, :vmax1, :K1; known = !(:S1 in unknown)),
        _mass(:produce_r1, 2, 1, :k_rs1),
        _hill(:hill_r1, 2, 1, :vmax_r1, :K_r1; known = !(:R1 in unknown)),
        _mass(:produce_s2, 3, 2, :k_prod2),
        _hill(:hill_s2, 3, 4, :vmax2, :K2; known = !(:S2 in unknown)),
        _mass(:produce_r2, 4, 3, :k_rs2),
        _decay(:decay_r2, 4, :k_r2)]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

const THREE_TERM_TRUTH = (k_prod1 = 0.9, vmax1 = 1.8, K1 = 0.55, k_rs1 = 1.0,
    vmax_r1 = 1.0, K_r1 = 0.6, k_prod2 = 0.7, vmax2 = 1.5, K2 = 0.5, k_rs2 = 0.9,
    k_r2 = 0.5)

"""Per fixture: the builder, the truth parameters, the unknown nodes, and each node's true Hill (vmax, K) and production parameter."""
function multi_term_fixture(fixture::Symbol)
    fixture == :separate && return (;
        build = build_two_term_separate_network, truth = TWO_TERM_SEPARATE_TRUTH,
        nodes = (:S1, :S2), nstates = 4,
        hill = Dict(:S1 => (:vmax1, :K1), :S2 => (:vmax2, :K2)),
        production = Dict(:S1 => :k_prod1, :S2 => :k_prod2))
    fixture == :coupled && return (;
        build = build_two_term_coupled_network, truth = TWO_TERM_COUPLED_TRUTH,
        nodes = (:A, :B), nstates = 3,
        hill = Dict(:A => (:vmax_a, :K_a), :B => (:vmax_b, :K_b)),
        production = Dict(:A => :k_ca, :B => :k_cb))
    fixture == :three && return (;
        build = build_three_term_network, truth = THREE_TERM_TRUTH,
        nodes = (:S1, :R1, :S2), nstates = 4,
        hill = Dict(:S1 => (:vmax1, :K1), :R1 => (:vmax_r1, :K_r1), :S2 => (:vmax2, :K2)),
        production = Dict(:S1 => :k_prod1, :R1 => :k_rs1, :S2 => :k_prod2))
    throw(ArgumentError("unknown multi-term fixture $(fixture); use :separate, :coupled or :three"))
end

"""Nine initial conditions per state count, drawn once from a fixed seed in [0.2, 1.2]."""
function multi_term_study_ics(nstates::Int; n_ics::Int = REFERENCE_PROTOCOL.n_ics)
    rng = MersenneTwister(1603 + nstates)
    return [0.2 .+ rand(rng, nstates) for _ in 1:n_ics]
end

# -- One run -------------------------------------------------------------------

const MULTI_TERM_STUDY_COLUMNS = (
    :fixture, :unknown, :seed, :noise, :variant, :node, :success,
    :support_recall, :support_precision, :support_f1, :extra_terms, :extras,
    :nn_rate_rmse, :nn_rate_bias, :collinearity, :unidentifiable_edge,
    :data_residual, :holdout_residual, :cross_term_max, :cross_term_pairs,
    :train_time_s, :run_time_s)

_mts_field(x::AbstractString) = '"' * replace(x, '"' => "'") * '"'
_mts_field(x::Symbol) = string(x)
_mts_field(x::Bool) = string(x)
function _mts_field(x::Real)
    isnan(x) ? "NaN" : isinf(x) ? (x > 0 ? "Inf" : "-Inf") :
    string(Float64(x))
end
_mts_field(x::Integer) = string(x)

function multi_term_csv_line(row)
    return join((_mts_field(getproperty(row, c)) for c in MULTI_TERM_STUDY_COLUMNS), ",")
end

function append_multi_term_rows(path::AbstractString, rows)
    fresh = !isfile(path) || filesize(path) == 0
    mkpath(dirname(abspath(path)))
    open(path, "a") do io
        fresh && println(io, join(string.(MULTI_TERM_STUDY_COLUMNS), ","))
        for row in rows
            println(io, multi_term_csv_line(row))
        end
        flush(io)
    end
    return path
end

function _mts_parse(column, value)
    column in (:fixture, :unknown, :variant, :node) && return Symbol(value)
    column in (:extras, :cross_term_pairs) && return strip(value, '"')
    column in (:success, :unidentifiable_edge) && return value == "true"
    column == :seed && return parse(Int, value)
    column == :extra_terms && return parse(Int, value)
    return parse(Float64, value)
end

function read_multi_term_csv(path::AbstractString)
    rows = NamedTuple[]
    (isfile(path) && filesize(path) > 0) || return rows
    lines = filter(!isempty, strip.(readlines(path)))
    header = Symbol.(split(lines[1], ","))
    header == collect(MULTI_TERM_STUDY_COLUMNS) || throw(ArgumentError(
        "$(path) has columns $(header); expected $(MULTI_TERM_STUDY_COLUMNS)"))
    for line in lines[2:end]
        fields = _split_csv_line(line)
        length(fields) == length(header) || continue
        push!(rows,
            NamedTuple{MULTI_TERM_STUDY_COLUMNS}(Tuple(
                _mts_parse(c, f) for (c, f) in zip(header, fields))))
    end
    return rows
end

function _split_csv_line(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    quoted = false
    for ch in line
        if ch == '"'
            quoted = !quoted
            print(buf, ch)
        elseif ch == ',' && !quoted
            push!(fields, String(take!(buf)))
        else
            print(buf, ch)
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end

"""Signed mean relative error of the learned rate against the true rate on the grid."""
function _relative_bias(estimate, truth)
    e = vec(Float64.(estimate))
    t = vec(Float64.(truth))
    keep = t .> 0
    any(keep) || return NaN
    return mean((e[keep] .- t[keep]) ./ t[keep])
end

function _score_term(candidate, R, D, truth_vmax, truth_K)
    truth = hill_rate_truth(vec(R); vmax = truth_vmax, K = truth_K, n = 2)
    support = hill_rate_support(2)
    scores = candidate === nothing ? nothing :
             support_f1(candidate, support.numerator, support.denominator).combined
    extras = candidate === nothing ? String[] :
             discovered_support_extras(candidate, support.numerator, support.denominator)
    return (;
        support_recall = scores === nothing ? 0.0 : scores.recall,
        support_precision = scores === nothing ? 0.0 : scores.precision,
        support_f1 = scores === nothing ? 0.0 : scores.f1,
        extra_terms = length(extras),
        extras = join(extras, ";"),
        nn_rate_rmse = rate_rel_rmse(vec(D), truth),
        nn_rate_bias = _relative_bias(vec(D), truth))
end

"""
    multi_term_study_run(; fixture, seed, noise_σ, unknown = :all, training, verbose)

One training of the study: the fixture's network with the nodes in
`unknown` (`:all`, or one node for the single-unknown control) marked
unknown, the reference-protocol data settings (9 initial conditions, 50
points on `(0, 8)`, `holdout = 2`), `discover_unknown_terms` with the
reference training defaults, then per term the discovery without and with
stability selection. Returns the CSV rows, one per variant and term.
"""
function multi_term_study_run(; fixture::Symbol, seed::Integer, noise_σ::Real,
        unknown = :all,
        training::TrainingConfig = TrainingConfig(
            adam_iterations = REFERENCE_PROTOCOL.adam_iterations,
            bfgs_iterations = REFERENCE_PROTOCOL.bfgs_iterations, log_every = 10^6),
        n_points::Int = REFERENCE_PROTOCOL.n_points,
        n_ics::Int = REFERENCE_PROTOCOL.n_ics,
        verbose::Bool = false)
    fx = multi_term_fixture(fixture)
    unknown_nodes = unknown === :all ? collect(fx.nodes) : [Symbol(unknown)]
    all(n -> n in fx.nodes, unknown_nodes) || throw(ArgumentError(
        "unknown must be :all or one of $(fx.nodes)"))
    truth_net = fx.build(; unknown = ())
    ude_net = fx.build(; unknown = Tuple(unknown_nodes))
    ics = multi_term_study_ics(fx.nstates; n_ics = n_ics)
    set = generate_experiment_set(MersenneTwister(seed); network = truth_net,
        initial_conditions = ics, tspan = REFERENCE_PROTOCOL.tspan, n_points = n_points,
        noise_σ = Float64(noise_σ), truth_params = fx.truth)
    label = Symbol(join(string.(unknown_nodes), "+"))
    started = time()
    t_train = @elapsed result = discover_unknown_terms(ude_net, set; training = training,
        holdout = 2, rng = MersenneTwister(seed), verbose = verbose,
        known_support = Dict(n => hill_rate_support(2) for n in unknown_nodes),
        production_param = Dict(n => fx.production[n] for n in unknown_nodes))
    cross_max = isempty(result.cross_term) ? NaN :
                maximum(p.collinearity for p in result.cross_term)
    cross_pairs = join(
        ("$(p.nodes[1])-$(p.nodes[2]):$(round(p.collinearity; digits = 4))"
        for p in result.cross_term),
        ";")
    rows = NamedTuple[]
    for (variant, stability) in ((:plain, nothing), (:stability, StabilitySelection()))
        pairs = Pair{Any, Any}[]
        per_term = []
        for term in result
            found = variant == :plain ? term.discovery :
                    discover_unknown_rate(term.samples.R,
                collect(range(0.0, 1.0; length = size(term.samples.R, 2))), term.samples.D;
                config = rate_discovery_config(), verbose = false,
                stability_selection = stability)
            candidate = found.success && !isempty(found.candidates) ? found.candidates[1] :
                        nothing
            candidate === nothing ||
                push!(pairs, term.term => equation_to_function(candidate))
            push!(per_term, (term, found, candidate))
        end
        complete = length(pairs) == length(result)
        first_exp = first(set.experiments)
        holdout_set = set.experiments[(end - 1):end]
        function residual_of(e)
            hybrid_data_residual(result.model, result.params, pairs, e.u0,
                (first(e.times), last(e.times)), e.times, e.observations; mask = e.mask)
        end
        data_residual = complete ? residual_of(first_exp) : Inf
        holdout_residual = complete ? mean(residual_of(e) for e in holdout_set) : Inf
        for (term, found, candidate) in per_term
            vmax_name, K_name = fx.hill[term.node]
            score = _score_term(candidate, term.samples.R, term.samples.D,
                getproperty(fx.truth, vmax_name), getproperty(fx.truth, K_name))
            push!(rows,
                (;
                    fixture = fixture, unknown = label, seed = Int(seed), noise = Float64(noise_σ),
                    variant = variant, node = term.node,
                    success = candidate !== nothing,
                    score...,
                    collinearity = Float64(term.identifiability.collinearity),
                    unidentifiable_edge = Bool(term.identifiability.unidentifiable_edge),
                    data_residual = Float64(data_residual),
                    holdout_residual = Float64(holdout_residual),
                    cross_term_max = Float64(cross_max), cross_term_pairs = cross_pairs,
                    train_time_s = Float64(t_train), run_time_s = time() - started))
        end
    end
    return rows
end

# -- The grid -------------------------------------------------------------------

_mts_key(row) = (row.fixture, row.unknown, row.seed, row.noise)

"""
    multi_term_study(; fixtures = (:separate, :coupled), seeds, noise_levels, controls = true,
                     out = "benchmark/results/multi_term_study.csv", verbose = true)

Run every combination of fixture, unknown-term configuration (all unknown,
plus each single-unknown control when `controls` is true), seed and noise
level that is not yet in `out`, appending its rows as it finishes.
"""
function multi_term_study(; fixtures = (:separate, :coupled),
        seeds = MULTI_TERM_STUDY_SEEDS, noise_levels = MULTI_TERM_STUDY_NOISE_LEVELS,
        controls::Bool = true, out::AbstractString = "benchmark/results/multi_term_study.csv",
        verbose::Bool = true, kwargs...)
    done = Set(_mts_key(r) for r in read_multi_term_csv(out))
    total = 0
    skipped = 0
    for fixture in fixtures
        fx = multi_term_fixture(fixture)
        configurations = Any[:all]
        controls && append!(configurations, collect(fx.nodes))
        for noise in noise_levels, seed in seeds, unknown in configurations
            label = unknown === :all ? Symbol(join(string.(fx.nodes), "+")) :
                    Symbol(unknown)
            key = (fixture, label, Int(seed), Float64(noise))
            if key in done
                skipped += 1
                continue
            end
            verbose &&
                println("== multi-term study: fixture ", fixture, ", unknown ", label,
                    ", seed ", seed, ", noise ", noise)
            t = @elapsed rows = multi_term_study_run(; fixture = fixture, seed = seed,
                noise_σ = noise, unknown = unknown, kwargs...)
            append_multi_term_rows(out, rows)
            push!(done, key)
            total += 1
            verbose && println("   done in ", round(t; digits = 1), " s; rows written ",
                length(rows))
        end
    end
    verbose && println("multi-term study: ", total, " runs added, ",
        skipped, " already recorded, in ", out)
    return read_multi_term_csv(out)
end

# -- Summary --------------------------------------------------------------------

_median(v) = isempty(v) ? NaN : median(v)
_iqr(v) = isempty(v) ? (NaN, NaN) : (quantile(v, 0.25), quantile(v, 0.75))

"""
    multi_term_compensation(rows) -> Vector{NamedTuple}

For every run with all terms unknown (one row per term and variant), whether
the terms compensated: the signed biases of the learned rates have opposite
signs, and each is larger in magnitude than the bias of the same term in the
single-unknown control of the same fixture, seed, noise level and variant.
Returns one entry per run and variant with `compensated`, the biases, the
control biases and the cross-term collinearity.
"""
function multi_term_compensation(rows)
    controls = Dict{Tuple{Symbol, Symbol, Int, Float64, Symbol}, Float64}()
    for r in rows
        r.unknown == r.node || continue   # a single-unknown control row
        controls[(r.fixture, r.node, r.seed, r.noise, r.variant)] = r.nn_rate_bias
    end
    groups = Dict{Tuple{Symbol, Symbol, Int, Float64, Symbol}, Vector{Any}}()
    for r in rows
        r.unknown == r.node && continue
        push!(get!(groups, (r.fixture, r.unknown, r.seed, r.noise, r.variant), Any[]), r)
    end
    out = NamedTuple[]
    for (key, group) in sort!(collect(groups); by = first)
        fixture, unknown, seed, noise, variant = key
        biases = [r.nn_rate_bias for r in group]
        nodes = [r.node for r in group]
        control = [get(controls, (fixture, n, seed, noise, variant), NaN) for n in nodes]
        opposite = length(biases) ≥ 2 && any(b > 0 for b in biases) &&
                   any(b < 0 for b in biases)
        larger = all(isfinite(c) ? abs(b) > abs(c) : false
        for (b, c) in zip(biases, control))
        push!(out,
            (; fixture, unknown, seed, noise, variant, nodes, biases,
                control_biases = control, compensated = opposite && larger,
                cross_term = first(group).cross_term_max,
                f1 = [r.support_f1 for r in group],
                control_f1 = [r.support_f1
                              for r in rows
                              if r.unknown == r.node && r.fixture == fixture &&
                                     r.seed == seed && r.noise == noise &&
                                     r.variant == variant]))
    end
    return out
end

"""
    multi_term_study_summary(rows) -> Vector{NamedTuple}

Per fixture, unknown-term configuration, noise level, variant and node:
number of runs, median and interquartile range of support F1, of the
relative rate RMSE, of the signed bias, of the held-out residual, of the
cross-term collinearity, and of the training time.
"""
function multi_term_study_summary(rows)
    groups = Dict{Tuple{Symbol, Symbol, Float64, Symbol, Symbol}, Vector{Any}}()
    for r in rows
        push!(get!(groups, (r.fixture, r.unknown, r.noise, r.variant, r.node), Any[]), r)
    end
    out = NamedTuple[]
    for (key, group) in sort!(collect(groups); by = first)
        fixture, unknown, noise, variant, node = key
        f1 = [r.support_f1 for r in group]
        rmse = [r.nn_rate_rmse for r in group]
        bias = [r.nn_rate_bias for r in group]
        hold = [r.holdout_residual for r in group if isfinite(r.holdout_residual)]
        cross = [r.cross_term_max for r in group if isfinite(r.cross_term_max)]
        train = [r.train_time_s for r in group]
        push!(out,
            (; fixture, unknown, noise, variant, node, n = length(group),
                successes = count(r -> r.success, group),
                f1_median = _median(f1), f1_iqr = _iqr(f1),
                rmse_median = _median(rmse), rmse_iqr = _iqr(rmse),
                bias_median = _median(bias), bias_iqr = _iqr(bias),
                holdout_median = _median(hold),
                cross_median = _median(cross), cross_iqr = _iqr(cross),
                train_median = _median(train)))
    end
    return out
end

_fmt3(x) = isnan(x) ? "NA" : string(round(x; digits = 3))
_fmt_iqr(t) = string("[", _fmt3(t[1]), ", ", _fmt3(t[2]), "]")

"""Markdown table of `multi_term_study_summary` for one variant."""
function format_multi_term_summary(summary; variant::Symbol = :plain)
    io = IOBuffer()
    println(io,
        "| fixture | unknown | noise | term | runs | F1 median [IQR] | rate RMSE median | bias median | held-out residual | cross-term median | training s |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary
        s.variant == variant || continue
        println(
            io, "| ", s.fixture, " | ", s.unknown, " | ", s.noise, " | ", s.node, " | ",
            s.successes, "/", s.n, " | ", _fmt3(s.f1_median), " ", _fmt_iqr(s.f1_iqr),
            " | ", _fmt3(s.rmse_median), " | ", _fmt3(s.bias_median), " | ",
            _fmt3(s.holdout_median), " | ", _fmt3(s.cross_median), " | ",
            isnan(s.train_median) ? "NA" : string(round(Int, s.train_median)), " |")
    end
    return String(take!(io))
end

"""
    cross_term_threshold_from_study(rows) -> NamedTuple

The cross-term values of the two-unknown runs split by whether the terms
compensated (`multi_term_compensation`). Returns the two groups' medians and
ranges and the midpoint between the highest non-compensated value and the
lowest compensated value when the groups separate, `nothing` otherwise.
"""
function cross_term_threshold_from_study(rows)
    comp = multi_term_compensation(rows)
    yes = [c.cross_term for c in comp if c.compensated && isfinite(c.cross_term)]
    no = [c.cross_term for c in comp if !c.compensated && isfinite(c.cross_term)]
    separates = !isempty(yes) && !isempty(no) && minimum(yes) > maximum(no)
    return (; n_compensated = length(yes), n_not = length(no),
        compensated_range = isempty(yes) ? (NaN, NaN) : extrema(yes),
        not_compensated_range = isempty(no) ? (NaN, NaN) : extrema(no),
        compensated_median = _median(yes), not_compensated_median = _median(no),
        separates, midpoint = separates ? (maximum(no) + minimum(yes)) / 2 : nothing)
end
