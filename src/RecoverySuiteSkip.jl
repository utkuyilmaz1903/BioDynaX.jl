###############################################################################
# Recovery-suite skip: unused sections must not train a unique-claim UDE.
#
# run_recovery_suite already gates each section with `if :name in wanted`.
# This file names the work each section does, counts `_train_unknown_edge`,
# and fails the suite if a skipped unique-claim section still trains.
# Does not drop protocol ICs, points, or seeds. Does not grow exports.
###############################################################################

"""Source strings that prove the skip instrument stays wired."""
const RECOVERY_SUITE_SKIP_MUST_CONTAIN = (
    "struct RecoverySuiteSectionSpec",
    "function recovery_suite_plan",
    "function with_train_unknown_edge_counter",
    "function recovery_suite_section_body",
    "function skipped_unique_claim_does_not_train")

const RECOVERY_SUITE_SKIP_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network")

function recovery_suite_skip_locked_sentences()
    return (;
        skip = "A skipped recovery-suite section does not call _train_unknown_edge.",
        plan = "recovery_suite_plan lists which requested sections train a unique-claim UDE and which requested sections are skipped.",
        counter = "with_train_unknown_edge_counter fails the suite if a skipped unique-claim section still trains.",
        default = "The default suite still runs :ude_discovery and :mm_unknown; skip is opt-in through the sections keyword.")
end

function recovery_jl_source_path()
    joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
end

function recovery_suite_skip_source_path()
    joinpath(pkgdir(BioDynaX), "src", "RecoverySuiteSkip.jl")
end

# -- Train counter ------------------------------------------------------------

const TRAIN_UNKNOWN_EDGE_COUNTER = Ref{Union{Nothing,Base.RefValue{Int}}}(nothing)

function _note_train_unknown_edge()
    counter = TRAIN_UNKNOWN_EDGE_COUNTER[]
    counter === nothing && return nothing
    counter[] += 1
    return nothing
end

"""
    with_train_unknown_edge_counter(f)

Run `f(counter)` while `_train_unknown_edge` increments `counter`.
Nested calls restore the previous counter.
"""
function with_train_unknown_edge_counter(f)
    counter = Ref(0)
    previous = TRAIN_UNKNOWN_EDGE_COUNTER[]
    TRAIN_UNKNOWN_EDGE_COUNTER[] = counter
    try
        return f(counter)
    finally
        TRAIN_UNKNOWN_EDGE_COUNTER[] = previous
    end
end

function train_unknown_edge_call_count()
    counter = TRAIN_UNKNOWN_EDGE_COUNTER[]
    return counter === nothing ? 0 : counter[]
end

# -- Typed section spec -------------------------------------------------------

"""
    RecoverySuiteSectionSpec

Work a `run_recovery_suite` section is allowed to do. Unique-claim
trainers set `trains_unknown_edge = true`. `validate_network` stays
open; admission is a separate instrument.
"""
struct RecoverySuiteSectionSpec
    name::Symbol
    kind::Symbol
    hole_policy::Symbol
    expected_holes::Union{Nothing,Int}
    trains_unknown_edge::Bool
    trains_ude::Bool
    trains_experiments::Bool
    discovers::Bool
    compiles::Bool
    uses_admit::Bool
    uses_generate_experiment_set::Bool
    uses_shared_rng::Bool
    default_suite::Bool
end

function RecoverySuiteSectionSpec(name::Symbol; kind::Symbol,
        hole_policy::Symbol, expected_holes,
        trains_unknown_edge::Bool = false,
        trains_ude::Bool = false,
        trains_experiments::Bool = false,
        discovers::Bool = false,
        compiles::Bool = true,
        uses_admit::Bool = false,
        uses_generate_experiment_set::Bool = false,
        uses_shared_rng::Bool = false,
        default_suite::Bool = false)
    return RecoverySuiteSectionSpec(
        name, kind, hole_policy, expected_holes,
        trains_unknown_edge, trains_ude, trains_experiments, discovers,
        compiles, uses_admit, uses_generate_experiment_set,
        uses_shared_rng, default_suite)
end

function recovery_suite_default_sections()
    return (:linear, :mm, :hill, :competitive,
            :ude_discovery, :mm_unknown, :ablation)
end

function recovery_suite_unique_claim_trainers()
    return (:ude_discovery, :mm_unknown)
end

function recovery_suite_section_spec(section::Symbol)
    kind = recovery_suite_section_kind(section)
    policy = recovery_suite_hole_policy(section)
    holes = recovery_suite_expected_holes(section)
    default = section in recovery_suite_default_sections()
    section === :linear && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_ude = true, compiles = true, uses_shared_rng = true,
        default_suite = default)
    section === :mm && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_ude = true, compiles = true, uses_shared_rng = true,
        default_suite = default)
    section === :hill && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_ude = true, compiles = true, uses_shared_rng = true,
        default_suite = default)
    section === :competitive && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_ude = true, compiles = true, uses_shared_rng = true,
        default_suite = default)
    section === :ude_discovery && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_unknown_edge = true, compiles = true, uses_admit = true,
        uses_generate_experiment_set = true, uses_shared_rng = true,
        default_suite = default)
    section === :mm_unknown && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_unknown_edge = true, compiles = true, uses_admit = true,
        uses_generate_experiment_set = true, uses_shared_rng = true,
        default_suite = default)
    section === :ablation && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        discovers = true, compiles = false, default_suite = default)
    section === :three_state && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        discovers = true, compiles = true)
    section === :wrong_graph && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        discovers = true, compiles = true)
    section === :six_state && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        discovers = true, compiles = true)
    section === :six_state_wrong_graph && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        discovers = true, compiles = true)
    section === :identifiability && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        compiles = true, uses_shared_rng = true)
    section === :ident_interventions && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_ude = true, discovers = true, compiles = true,
        uses_admit = true, uses_shared_rng = true)
    section === :partial_obs && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        trains_experiments = true, discovers = true, compiles = true,
        uses_admit = true, uses_shared_rng = true)
    section === :competitive_unknown && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        discovers = true, compiles = true)
    section === :literature && return RecoverySuiteSectionSpec(
        section; kind, hole_policy = policy, expected_holes = holes,
        compiles = true, uses_shared_rng = true)
    throw(ArgumentError("unknown recovery suite section $section"))
end

function recovery_suite_section_specs()
    return [recovery_suite_section_spec(section)
            for section in recovery_suite_sections()]
end

function recovery_suite_spec_matrix()
    specs = recovery_suite_section_specs()
    names = [spec.name for spec in specs]
    trainers = [spec.name for spec in specs if spec.trains_unknown_edge]
    default_trainers = [spec.name for spec in specs
                        if spec.trains_unknown_edge && spec.default_suite]
    unique_claim = [spec.name for spec in specs if spec.kind === :unique_claim]
    open_known = [spec.name for spec in specs if spec.kind === :known_kinetics]
    return (;
        n = length(specs),
        names = Tuple(names),
        trainers = Tuple(trainers),
        default_trainers = Tuple(default_trainers),
        unique_claim = Tuple(unique_claim),
        open_known = Tuple(open_known),
        holds = length(specs) == length(recovery_suite_sections()) &&
                issetequal(trainers, recovery_suite_unique_claim_trainers()) &&
                issetequal(default_trainers, recovery_suite_unique_claim_trainers()) &&
                issetequal(unique_claim, recovery_suite_unique_claim_sections()) &&
                issetequal(open_known, RECOVERY_SUITE_KNOWN_KINETICS_SECTIONS))
end

# -- Plan ---------------------------------------------------------------------

"""
    RecoverySuitePlan

Requested sections, skipped catalog sections, and which of the requested
ones would call `_train_unknown_edge`.
"""
struct RecoverySuitePlan
    requested::Vector{Symbol}
    skipped::Vector{Symbol}
    train_unknown_edge::Vector{Symbol}
    train_ude::Vector{Symbol}
    train_experiments::Vector{Symbol}
    discover::Vector{Symbol}
    compile::Vector{Symbol}
    admit::Vector{Symbol}
end

function recovery_suite_plan(sections = recovery_suite_default_sections())
    requested = collect(Symbol, sections)
    known = collect(recovery_suite_sections())
    skipped = [section for section in known if !(section in requested)]
    specs = [recovery_suite_section_spec(section) for section in requested]
    return RecoverySuitePlan(
        requested,
        skipped,
        [spec.name for spec in specs if spec.trains_unknown_edge],
        [spec.name for spec in specs if spec.trains_ude],
        [spec.name for spec in specs if spec.trains_experiments],
        [spec.name for spec in specs if spec.discovers],
        [spec.name for spec in specs if spec.compiles],
        [spec.name for spec in specs if spec.uses_admit])
end

function recovery_suite_plan_namedtuple(plan::RecoverySuitePlan)
    return (;
        requested = Tuple(plan.requested),
        skipped = Tuple(plan.skipped),
        train_unknown_edge = Tuple(plan.train_unknown_edge),
        train_ude = Tuple(plan.train_ude),
        train_experiments = Tuple(plan.train_experiments),
        discover = Tuple(plan.discover),
        compile = Tuple(plan.compile),
        admit = Tuple(plan.admit),
        n_requested = length(plan.requested),
        n_skipped = length(plan.skipped))
end

function recovery_suite_skipped_unique_claim_trainers(sections)
    plan = recovery_suite_plan(sections)
    return [section for section in recovery_suite_unique_claim_trainers()
            if section in plan.skipped]
end

function recovery_suite_would_train_unknown_edge(
        sections = recovery_suite_default_sections())
    return !isempty(recovery_suite_plan(sections).train_unknown_edge)
end

# -- Source body extraction ---------------------------------------------------

function recovery_suite_runner_source()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function run_recovery_suite", src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    return nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
end

function recovery_suite_section_body(section::Symbol)
    src = recovery_suite_runner_source()
    needle = "if :$section in wanted"
    start = findfirst(needle, src)
    start === nothing && return ""
    rest = src[first(start):end]
    nxt = findnext(r"\n    if :", rest, length(needle) + 1)
    closing = findnext("return report", rest, 1)
    stop = if nxt === nothing
        closing === nothing ? length(rest) : first(closing) - 1
    else
        close_at = closing === nothing ? typemax(Int) : first(closing)
        min(first(nxt) - 1, close_at)
    end
    return rest[1:stop]
end

function recovery_suite_section_is_gated(section::Symbol)
    body = recovery_suite_section_body(section)
    return startswith(body, "if :$section in wanted")
end

function recovery_suite_all_sections_gated()
    return all(recovery_suite_section_is_gated, recovery_suite_sections())
end

function recovery_suite_section_source_row(section::Symbol)
    spec = recovery_suite_section_spec(section)
    body = recovery_suite_section_body(section)
    isempty(body) && return (;
        section, gated = false, trains_unknown_edge = false,
        trains_ude = false, trains_experiments = false, discovers = false,
        uses_admit = false, uses_generate = false, holds = false)
    trains_unknown = occursin("_train_unknown_edge", body)
    trains_ude = occursin("train_ude(", body)
    trains_experiments = occursin("train_experiments(", body) &&
        !occursin("train_experiments_with_warmup", body)
    discovers = occursin("discover_equations(", body) ||
                occursin("discover_unknown_rate(", body)
    uses_admit = occursin("admit_recovery_suite_network", body)
    uses_generate = occursin("generate_experiment_set(", body) ||
        (spec.uses_generate_experiment_set && occursin("_train_unknown_edge", body))
    gated = recovery_suite_section_is_gated(section)
    generate_ok = spec.uses_generate_experiment_set ? uses_generate :
        !occursin("generate_experiment_set(", body)
    holds = gated &&
            trains_unknown == spec.trains_unknown_edge &&
            trains_ude == spec.trains_ude &&
            uses_admit == spec.uses_admit &&
            generate_ok
    return (;
        section,
        gated,
        trains_unknown_edge = trains_unknown,
        trains_ude,
        trains_experiments,
        discovers,
        uses_admit,
        uses_generate,
        holds)
end

function recovery_suite_section_source_matrix()
    rows = [recovery_suite_section_source_row(section)
            for section in recovery_suite_sections()]
    trainer_bodies = [row for row in rows if row.trains_unknown_edge]
    return (;
        rows,
        n = length(rows),
        gated = all(row -> row.gated, rows),
        trainer_sections = Tuple(row.section for row in trainer_bodies),
        holds = all(row -> row.holds, rows) &&
                issetequal(
                    [row.section for row in trainer_bodies],
                    recovery_suite_unique_claim_trainers()))
end

function train_unknown_edge_only_in_unique_claim_source()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function _train_unknown_edge", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    definition = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    occursin("_note_train_unknown_edge", definition) || return false
    runner = recovery_suite_runner_source()
    ude = recovery_suite_section_body(:ude_discovery)
    mm = recovery_suite_section_body(:mm_unknown)
    other = replace(replace(runner, ude => ""), mm => "")
    return occursin("_train_unknown_edge", ude) &&
           occursin("_train_unknown_edge", mm) &&
           !occursin("_train_unknown_edge(", other)
end

function recovery_suite_default_sections_source()
    src = recovery_suite_runner_source()
    return occursin("sections = (:linear, :mm, :hill, :competitive,", src) &&
           occursin(":ude_discovery, :mm_unknown, :ablation)", src)
end

# -- Skip reports -------------------------------------------------------------

"""
    skipped_unique_claim_does_not_train(sections; kwargs...)

Run `run_recovery_suite` on `sections` and return the `_train_unknown_edge`
count. Callers must not pass `:ude_discovery` or `:mm_unknown` when they
only want the skip oracle.
"""
function skipped_unique_claim_does_not_train(sections;
        rng::AbstractRNG = MersenneTwister(1), kwargs...)
    plan = recovery_suite_plan(sections)
    n = with_train_unknown_edge_counter() do counter
        report = run_recovery_suite(rng; sections = sections, kwargs...)
        return (;
            counter = counter[],
            keys = Tuple(sort(collect(keys(report)))),
            report)
    end
    skipped_trainers = recovery_suite_skipped_unique_claim_trainers(sections)
    return (;
        plan = recovery_suite_plan_namedtuple(plan),
        skipped_trainers = Tuple(skipped_trainers),
        counter = n.counter,
        keys = n.keys,
        holds = n.counter == length(plan.train_unknown_edge) &&
                issetequal(n.keys, plan.requested) &&
                all(section -> !(section in n.keys), skipped_trainers))
end

function skip_linear_only_report()
    return skipped_unique_claim_does_not_train(
        (:linear,);
        linear_adam = 1, linear_bfgs = 0)
end

function skip_mm_only_report()
    return skipped_unique_claim_does_not_train(
        (:mm,);
        mm_adam = 1, mm_bfgs = 0)
end

function skip_known_kinetics_report()
    return skipped_unique_claim_does_not_train(
        (:linear, :mm, :hill, :competitive);
        linear_adam = 1, linear_bfgs = 0,
        mm_adam = 1, mm_bfgs = 0,
        hill_adam = 1, hill_bfgs = 0,
        competitive_adam = 1, competitive_bfgs = 0)
end

function skip_ablation_only_report()
    return skipped_unique_claim_does_not_train((:ablation,))
end

function skip_identifiability_only_report()
    return skipped_unique_claim_does_not_train((:identifiability,))
end

function skip_literature_only_report()
    return skipped_unique_claim_does_not_train((:literature,))
end

function skip_three_state_only_report()
    return skipped_unique_claim_does_not_train((:three_state,))
end

function skip_wrong_graph_only_report()
    return skipped_unique_claim_does_not_train((:wrong_graph,))
end

function skip_competitive_unknown_only_report()
    return skipped_unique_claim_does_not_train((:competitive_unknown,))
end

function skip_empty_unique_claim_plan()
    plan = recovery_suite_plan((:linear, :ablation, :literature))
    return (;
        plan = recovery_suite_plan_namedtuple(plan),
        skipped_trainers = Tuple(recovery_suite_skipped_unique_claim_trainers(
            (:linear, :ablation, :literature))),
        would_train = recovery_suite_would_train_unknown_edge(
            (:linear, :ablation, :literature)),
        holds = isempty(plan.train_unknown_edge) &&
                issetequal(plan.skipped,
                    setdiff(collect(recovery_suite_sections()),
                            [:linear, :ablation, :literature])) &&
                recovery_suite_would_train_unknown_edge(
                    recovery_suite_default_sections()))
end

function default_suite_plan_includes_trainers()
    plan = recovery_suite_plan()
    return (;
        plan = recovery_suite_plan_namedtuple(plan),
        holds = issetequal(plan.train_unknown_edge,
                    collect(recovery_suite_unique_claim_trainers())) &&
                :ude_discovery in plan.requested &&
                :mm_unknown in plan.requested &&
                :ablation in plan.requested)
end

function skip_report_matrix()
    linear = skip_linear_only_report()
    ablation = skip_ablation_only_report()
    ident = skip_identifiability_only_report()
    literature = skip_literature_only_report()
    empty = skip_empty_unique_claim_plan()
    default = default_suite_plan_includes_trainers()
    source = recovery_suite_section_source_matrix()
    specs = recovery_suite_spec_matrix()
    return (;
        linear, ablation, ident, literature, empty, default, source, specs,
        holds = linear.holds && ablation.holds && ident.holds &&
                literature.holds && empty.holds && default.holds &&
                source.holds && specs.holds &&
                linear.counter == 0 && ablation.counter == 0 &&
                ident.counter == 0 && literature.counter == 0)
end

# -- Admit / compile skip honesty ---------------------------------------------

function skipped_section_does_not_admit_source(section::Symbol)
    spec = recovery_suite_section_spec(section)
    spec.uses_admit || return true
    body = recovery_suite_section_body(section)
    return occursin("admit_recovery_suite_network(:$section)", body) &&
           startswith(body, "if :$section in wanted")
end

function skipped_section_does_not_generate_set_source(section::Symbol)
    spec = recovery_suite_section_spec(section)
    spec.uses_generate_experiment_set || return true
    body = recovery_suite_section_body(section)
    return occursin("_train_unknown_edge", body) &&
           startswith(body, "if :$section in wanted")
end

function unique_claim_skip_source_holds()
    return all(skipped_section_does_not_admit_source,
               recovery_suite_unique_claim_sections()) &&
           all(skipped_section_does_not_generate_set_source,
               recovery_suite_unique_claim_trainers()) &&
           train_unknown_edge_only_in_unique_claim_source() &&
           recovery_suite_all_sections_gated() &&
           recovery_suite_default_sections_source()
end

function recovery_suite_skip_fixture_paths()
    specs = recovery_suite_spec_matrix()
    source = recovery_suite_section_source_matrix()
    empty = skip_empty_unique_claim_plan()
    default = default_suite_plan_includes_trainers()
    linear = skip_linear_only_report()
    return (;
        specs, source, empty, default, linear,
        holds = specs.holds && source.holds && empty.holds &&
                default.holds && linear.holds)
end

# -- Docs / contract ----------------------------------------------------------

function recovery_suite_section_cost_row(section::Symbol)
    spec = recovery_suite_section_spec(section)
    return (;
        section,
        kind = spec.kind,
        hole_policy = spec.hole_policy,
        expected_holes = spec.expected_holes,
        trains_unknown_edge = spec.trains_unknown_edge,
        trains_ude = spec.trains_ude,
        trains_experiments = spec.trains_experiments,
        discovers = spec.discovers,
        compiles = spec.compiles,
        uses_admit = spec.uses_admit,
        default_suite = spec.default_suite,
        skip_avoids_unknown_edge = !spec.trains_unknown_edge || spec.default_suite)
end

function recovery_suite_cost_matrix()
    rows = [recovery_suite_section_cost_row(section)
            for section in recovery_suite_sections()]
    trainer_cost = [row for row in rows if row.trains_unknown_edge]
    return (;
        rows,
        n = length(rows),
        trainer_sections = Tuple(row.section for row in trainer_cost),
        holds = length(rows) == length(recovery_suite_sections()) &&
                issetequal(
                    [row.section for row in trainer_cost],
                    recovery_suite_unique_claim_trainers()))
end

function skip_linear_compile_report()
    plan = recovery_suite_plan((:linear,))
    n_train = with_train_unknown_edge_counter() do train
        n_compile = with_compile_network_counter() do compile
            run_recovery_suite(
                MersenneTwister(1);
                sections = (:linear,),
                linear_adam = 1, linear_bfgs = 0)
            return (; train = train[], compile = compile[])
        end
        return n_compile
    end
    return (;
        plan = recovery_suite_plan_namedtuple(plan),
        train = n_train.train,
        compile = n_train.compile,
        holds = n_train.train == 0 && n_train.compile ≥ 1 &&
                isempty(plan.train_unknown_edge))
end

function ident_interventions_does_not_train_unknown_edge_source()
    body = recovery_suite_section_body(:ident_interventions)
    return occursin("train_ude(", body) &&
           occursin("admit_recovery_suite_network(:ident_interventions)", body) &&
           !occursin("_train_unknown_edge", body) &&
           startswith(body, "if :ident_interventions in wanted")
end

function partial_obs_does_not_train_unknown_edge_source()
    body = recovery_suite_section_body(:partial_obs)
    return occursin("train_experiments(", body) &&
           occursin("admit_recovery_suite_network(:partial_obs)", body) &&
           !occursin("_train_unknown_edge", body) &&
           startswith(body, "if :partial_obs in wanted")
end

function unique_claim_non_trainers_source_hold()
    return ident_interventions_does_not_train_unknown_edge_source() &&
           partial_obs_does_not_train_unknown_edge_source()
end

function recovery_suite_expected_report_keys(section::Symbol)
    section === :linear && return (:rmse, :rel, :final_loss)
    section === :mm && return (:rmse, :rel, :final_loss)
    section === :hill && return (:rmse, :rel, :final_loss)
    section === :competitive && return (:rmse, :rel, :final_loss)
    section === :ude_discovery && return (
        :locked_kpis, :protocol_result)
    section === :mm_unknown && return (
        :locked_kpis, :protocol_result)
    section === :ablation && return (
        :local_terms, :global_terms, :local_success, :global_success,
        :local_f1, :global_f1, :local_false_parent, :global_false_parent)
    section === :three_state && return (
        :graph_parents, :local_success, :global_success,
        :local_has_true_parent, :local_false_parent)
    section === :wrong_graph && return (
        :graph_parents, :local_success, :local_has_true_parent,
        :local_false_parent)
    section === :six_state && return (
        :nstates, :graph_parents, :local_success, :Z_in_local_library)
    section === :six_state_wrong_graph && return (
        :nstates, :graph_parents, :local_success, :local_has_true_parent)
    section === :identifiability && return (
        :unidentifiable_edge, :collinearity)
    section === :ident_interventions && return (
        :nominal_unidentifiable, :frozen_k_prod_unchanged,
        :tradeoff_broken)
    section === :partial_obs && return (
        :subsample_success, :mask_used, :ude_mask_train_claimed)
    section === :competitive_unknown && return (
        :compiled_regulators, :two_parent_success, :canonical_f1_claimed)
    section === :literature && return (
        :source, :experimental_csv, :unique_claim_protocol,
        :licensed_experimental_series, :finite_trajectory)
    throw(ArgumentError("unknown recovery suite section $section"))
end

function recovery_suite_report_keys_hold(section::Symbol, payload)
    expected = recovery_suite_expected_report_keys(section)
    return all(key -> hasproperty(payload, key) || haskey(payload, key), expected)
end

function recovery_suite_report_key_matrix()
    keys = Dict(section => recovery_suite_expected_report_keys(section)
                for section in recovery_suite_sections())
    return (;
        n = length(keys),
        ude = keys[:ude_discovery],
        literature = keys[:literature],
        holds = length(keys) == length(recovery_suite_sections()) &&
                :protocol_result in keys[:ude_discovery] &&
                :experimental_csv in keys[:literature] &&
                :Z_in_local_library in keys[:six_state])
end

function skip_report_has_expected_keys(report, section::Symbol)
    haskey(report.report, section) || return false
    return recovery_suite_report_keys_hold(section, report.report[section])
end

function skip_known_kinetics_key_report()
    report = skip_known_kinetics_report()
    keys_ok = all(section -> skip_report_has_expected_keys(report, section),
                  (:linear, :mm, :hill, :competitive))
    return (;
        report,
        keys_ok,
        holds = report.holds && keys_ok && report.counter == 0)
end

function skip_graph_prior_report()
    three = skip_three_state_only_report()
    wrong = skip_wrong_graph_only_report()
    return (;
        three, wrong,
        holds = three.holds && wrong.holds &&
                three.counter == 0 && wrong.counter == 0 &&
                skip_report_has_expected_keys(three, :three_state) &&
                skip_report_has_expected_keys(wrong, :wrong_graph))
end

function skip_competitive_unknown_key_report()
    report = skip_competitive_unknown_only_report()
    return (;
        report,
        holds = report.holds && report.counter == 0 &&
                skip_report_has_expected_keys(report, :competitive_unknown))
end

"""Needles that must appear in each gated section body."""
const RECOVERY_SUITE_SECTION_NEEDLES = (
    linear = ("train_ude(", "build_linear_test_network"),
    mm = ("train_ude(", "build_mm_test_network"),
    hill = ("train_ude(", "build_hill_recovery_network"),
    competitive = ("train_ude(", "build_competitive_test_network"),
    ude_discovery = ("_train_unknown_edge",
                     "admit_recovery_suite_network(:ude_discovery)",
                     "UNIQUE_CLAIM_PROTOCOL.tspan",
                     "UNIQUE_CLAIM_PROTOCOL.n_points"),
    mm_unknown = ("_train_unknown_edge",
                  "admit_recovery_suite_network(:mm_unknown)",
                  "UNIQUE_CLAIM_PROTOCOL.tspan",
                  "UNIQUE_CLAIM_PROTOCOL.n_points"),
    ablation = ("discover_equations(", "build_rate_ablation_network",
                "scope = :graph", "scope = :global"),
    three_state = ("discover_equations(", "build_three_state_unknown_network"),
    wrong_graph = ("discover_equations(", "build_wrong_graph_unknown_network"),
    six_state = ("discover_equations(", "build_six_state_unknown_network",
                 "Z_in_local_library"),
    six_state_wrong_graph = ("discover_equations(",
                             "build_six_state_wrong_graph_network"),
    identifiability = ("production_destruction_tradeoff(",
                       "build_hill_recovery_network"),
    ident_interventions = ("train_ude(", "frozen_phys = [:k_prod]",
                           "admit_recovery_suite_network(:ident_interventions)"),
    partial_obs = ("train_experiments(",
                   "admit_recovery_suite_network(:partial_obs)",
                   "ude_mask_train_claimed = false"),
    competitive_unknown = ("discover_unknown_rate(",
                           "canonical_f1_claimed = false"),
    literature = ("build_repressilator_network",
                  "experimental_csv = false",
                  "unique_claim_protocol = false"))

function recovery_suite_section_needles(section::Symbol)
    needles = RECOVERY_SUITE_SECTION_NEEDLES
    hasproperty(needles, section) || throw(ArgumentError(
        "unknown recovery suite section $section"))
    return getfield(needles, section)
end

function recovery_suite_section_needles_hold(section::Symbol)
    body = recovery_suite_section_body(section)
    isempty(body) && return false
    return all(needle -> occursin(needle, body),
               recovery_suite_section_needles(section))
end

function recovery_suite_needles_matrix()
    rows = [(;
        section,
        holds = recovery_suite_section_needles_hold(section),
        gated = recovery_suite_section_is_gated(section))
            for section in recovery_suite_sections()]
    return (;
        rows,
        n = length(rows),
        holds = all(row -> row.holds && row.gated, rows))
end

function recovery_suite_skip_index_row(section::Symbol)
    spec = recovery_suite_section_spec(section)
    source = recovery_suite_section_source_row(section)
    cost = recovery_suite_section_cost_row(section)
    return (;
        section,
        kind = spec.kind,
        hole_policy = spec.hole_policy,
        expected_holes = spec.expected_holes,
        trains_unknown_edge = spec.trains_unknown_edge,
        trains_ude = spec.trains_ude,
        trains_experiments = spec.trains_experiments,
        discovers = spec.discovers,
        compiles = spec.compiles,
        uses_admit = spec.uses_admit,
        default_suite = spec.default_suite,
        uses_shared_rng = spec.uses_shared_rng,
        gated = source.gated,
        needles = recovery_suite_section_needles(section),
        report_keys = recovery_suite_expected_report_keys(section),
        skip_avoids_unknown_edge = !spec.trains_unknown_edge,
        holds = source.holds &&
                recovery_suite_section_needles_hold(section) &&
                cost.section === section)
end

function recovery_suite_skip_index()
    rows = [recovery_suite_skip_index_row(section)
            for section in recovery_suite_sections()]
    shared = [row.section for row in rows if row.uses_shared_rng]
    trainers = [row.section for row in rows if row.trains_unknown_edge]
    return (;
        rows,
        n = length(rows),
        shared_rng = Tuple(shared),
        trainers = Tuple(trainers),
        holds = all(row -> row.holds, rows) &&
                length(rows) == length(recovery_suite_sections()) &&
                issetequal(trainers, recovery_suite_unique_claim_trainers()))
end

function recovery_suite_shared_rng_honesty()
    shared = [spec.name for spec in recovery_suite_section_specs()
              if spec.uses_shared_rng]
    trainers = collect(recovery_suite_unique_claim_trainers())
    return (;
        shared = Tuple(shared),
        trainers = Tuple(trainers),
        skip_linear_changes_ude_rng = :linear in shared &&
            :ude_discovery in shared,
        default_keeps_order = recovery_suite_default_sections()[1] === :linear &&
            recovery_suite_default_sections()[5] === :ude_discovery,
        holds = :linear in shared && :ude_discovery in shared &&
                :mm_unknown in shared &&
                recovery_suite_default_sections()[5] === :ude_discovery)
end

function format_recovery_suite_skip_index()
    index = recovery_suite_skip_index()
    lines = String[]
    push!(lines, "section kind policy holes train_unknown default")
    for row in index.rows
        holes = row.expected_holes === nothing ? "NA" : string(row.expected_holes)
        push!(lines, join((
            row.section,
            row.kind,
            row.hole_policy,
            holes,
            row.trains_unknown_edge,
            row.default_suite), " "))
    end
    return join(lines, "\n")
end

function format_recovery_suite_skip_markdown()
    index = recovery_suite_skip_index()
    lines = String[
        "| section | kind | policy | holes | `_train_unknown_edge` | default |",
        "|---|---|---|---|---|---|"]
    for row in index.rows
        holes = row.expected_holes === nothing ? "NA" : string(row.expected_holes)
        push!(lines, "| `$(row.section)` | $(row.kind) | $(row.hole_policy) | $holes | $(row.trains_unknown_edge) | $(row.default_suite) |")
    end
    return join(lines, "\n")
end

function skip_default_minus_trainers_report()
    sections = filter(
        section -> !(section in recovery_suite_unique_claim_trainers()),
        collect(recovery_suite_default_sections()))
    return skipped_unique_claim_does_not_train(
        Tuple(sections);
        linear_adam = 1, linear_bfgs = 0,
        mm_adam = 1, mm_bfgs = 0,
        hill_adam = 1, hill_bfgs = 0,
        competitive_adam = 1, competitive_bfgs = 0)
end

function recovery_suite_benchmark_path()
    joinpath(pkgdir(BioDynaX), "benchmark", "recovery_suite.jl")
end

function recovery_suite_seeds_path()
    joinpath(pkgdir(BioDynaX), "benchmark", "recovery_seeds.jl")
end

function recovery_suite_sindy_baseline_path()
    joinpath(pkgdir(BioDynaX), "benchmark", "sindy_baseline.jl")
end

function recovery_suite_benchmark_fast_skips_trainers()
    src = read(recovery_suite_benchmark_path(), String)
    return occursin("sections = (:linear, :mm, :hill, :competitive, :ablation)", src) &&
           occursin("sections = (:ude_discovery, :mm_unknown)", src)
end

function recovery_suite_seeds_uses_ude_only()
    src = read(recovery_suite_seeds_path(), String)
    return occursin("sections = (:ude_discovery,)", src)
end

function recovery_suite_sindy_baseline_uses_ablation_only()
    src = read(recovery_suite_sindy_baseline_path(), String)
    return occursin("sections = (:ablation,)", src)
end

function recovery_suite_benchmark_skip_source_holds()
    return recovery_suite_benchmark_fast_skips_trainers() &&
           recovery_suite_seeds_uses_ude_only() &&
           recovery_suite_sindy_baseline_uses_ablation_only()
end

function recovery_suite_skip_markdown_holds()
    text = format_recovery_suite_skip_markdown()
    return occursin("| `ude_discovery` |", text) &&
           occursin("| `mm_unknown` |", text) &&
           occursin("| `ablation` |", text) &&
           occursin("exactly_one", text) &&
           occursin("library_fixture", text)
end

function unique_claim_trainer_keeps_protocol_source()
    ude = recovery_suite_section_body(:ude_discovery)
    mm = recovery_suite_section_body(:mm_unknown)
    return occursin("UNIQUE_CLAIM_PROTOCOL.tspan", ude) &&
           occursin("UNIQUE_CLAIM_PROTOCOL.n_points", ude) &&
           occursin("UNIQUE_CLAIM_PROTOCOL.tspan", mm) &&
           occursin("UNIQUE_CLAIM_PROTOCOL.n_points", mm) &&
           !occursin("n_ics = 1", ude) &&
           !occursin("n_points = 8", ude)
end

function recovery_suite_skip_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "recovery-suite-skip.md")
end

function recovery_suite_skip_source_holds()
    src = read(recovery_suite_skip_source_path(), String)
    impl = read(recovery_jl_source_path(), String)
    docs = isfile(recovery_suite_skip_docs_path()) ?
        read(recovery_suite_skip_docs_path(), String) : ""
    return all(occursin(needle, src) for needle in RECOVERY_SUITE_SKIP_MUST_CONTAIN) &&
           !occursin("support_f1_ude = 0.99", impl) &&
           !occursin("support_f1_ude = 0.99", docs) &&
           !occursin("function validate_network", docs)
end

function recovery_suite_skip_docs_hold()
    path = recovery_suite_skip_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(recovery_suite_skip_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("recovery-suite-skip.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function recovery_suite_skip_landing_docs_hold()
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    arch = read(joinpath(pkgdir(BioDynaX), "docs", "src", "architecture.md"), String)
    sentences = recovery_suite_skip_locked_sentences()
    return occursin("recovery-suite-skip", howto) &&
           occursin("_train_unknown_edge", howto) &&
           occursin(sentences.skip, arch)
end

function recovery_suite_skip_source_violations()
    src = read(recovery_suite_skip_source_path(), String)
    impl = read(recovery_jl_source_path(), String)
    docs = isfile(recovery_suite_skip_docs_path()) ?
        read(recovery_suite_skip_docs_path(), String) : ""
    missing = [s for s in RECOVERY_SUITE_SKIP_MUST_CONTAIN if !occursin(s, src)]
    forbidden = String[]
    occursin("support_f1_ude = 0.99", impl) &&
        push!(forbidden, "Recovery.jl: support_f1_ude = 0.99")
    occursin("support_f1_ude = 0.99", docs) &&
        push!(forbidden, "docs: support_f1_ude = 0.99")
    occursin("function validate_network", docs) &&
        push!(forbidden, "docs: function validate_network")
    return (; missing, forbidden)
end

function recovery_suite_skip_contract_holds()
    return recovery_suite_skip_source_holds() &&
           unique_claim_skip_source_holds() &&
           unique_claim_non_trainers_source_hold() &&
           unique_claim_trainer_keeps_protocol_source() &&
           recovery_suite_needles_matrix().holds &&
           recovery_suite_benchmark_skip_source_holds() &&
           recovery_suite_skip_docs_hold() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source()
end
