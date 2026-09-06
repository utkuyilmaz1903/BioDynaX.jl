# Source-reading and consistency checks that lived in src/ up to 0.11. They
# read the package source files at test time and compare fixture outputs, so
# they are test helpers, not package code. Loaded by test/runtests.jl through
# test/support/support.jl; every function keeps the name it had in src/.

function compile_mechanism_source_violations()
    src = read(compile_mechanism_source_path(), String)
    missing = [s for s in COMPILER_REINDEX_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in COMPILER_REINDEX_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

function denominator_domain_index_holds()
    text = format_denominator_domain_index()
    names = denominator_domain_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("domain grid", text) &&
           occursin("typemax", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function domain_grid_clips_source_holds()
    src = read(discovery_jl_source_path_for_denominator(), String)
    start = findfirst("function _denominator_domain_grid", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("max.(zero.(lo), lo .- pad)", body) &&
           occursin("n ≤ 0 && return", body)
end

function explicit_path_skips_domain_grid_source_holds()
    src = read(discovery_jl_source_path_for_denominator(), String)
    start = findfirst("function _discover_explicit", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return !occursin("_denominator_domain_grid", body) &&
           !occursin("_check_denominator_safety", body)
end

function implicit_discovery_uses_domain_grid_source_holds()
    src = read(discovery_jl_source_path_for_denominator(), String)
    start = findfirst("function _discover_implicit", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("_denominator_domain_grid", body) &&
           occursin("_check_denominator_safety", body) &&
           occursin("train_X", body) &&
           occursin("val_X", body) &&
           occursin("domain_X", body)
end

function extras_path_calls_split_source_holds()
    path = joinpath(pkgdir(BioDynaX), "src", "RecoveryPipeline.jl")
    src = read(path, String)
    start = findfirst("function evaluate_recovery", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin(
        "extras_denominator = ude_extras_denominator_row(", body) &&
           occursin(
        "den_violations = denominator_violation_count(candidate, R_grid)", body)
end

function ude_extras_denominator_source_holds()
    src = read(recovery_jl_source_path_for_denominator(), String)
    start = findfirst(
        "function ude_extras_denominator_row(candidate, R_grid;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("denominator_split_counts", body) &&
           occursin("_denominator_domain_grid", body) &&
           occursin("extras_print_label", body) &&
           occursin("extras_print_is_hardcoded_attempt", body)
end

function denominator_split_counts_source_holds()
    src = read(recovery_jl_source_path_for_denominator(), String)
    start = findfirst(
        "function denominator_split_counts(candidate, train_X, val_X, domain_X;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("denominator_violation_count(candidate, train_X", body) &&
           occursin("denominator_violation_count(candidate, val_X", body) &&
           occursin("denominator_violation_count(candidate, domain_X", body)
end

function denominator_violation_count_source_holds()
    src = read(recovery_jl_source_path_for_denominator(), String)
    return occursin(
               "function denominator_violation_count(candidate::ImplicitCandidate, X;", src) &&
           occursin(
               "function denominator_violation_count(::ExplicitCandidate, X;", src) &&
           occursin(
               "function denominator_violation_count(::Nothing, X; floor::Real = 1e-8)", src)
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

function resume_source_holds()
    body = training_jl_checkpoint_source()
    return occursin("optimizer_state = checkpoint.optimizer_state", body) &&
           occursin("initial_iteration = checkpoint.iteration", body) &&
           occursin("train_ude(", body)
end

function failure_mode_index_holds()
    text = format_failure_mode_index()
    index = failure_mode_row_index()
    return index.holds &&
           occursin("InsufficientSamples", text) &&
           occursin("validate open", text) &&
           occursin("0.99 F1", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function extras_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Recovery.jl"), String)
    start = findfirst("function _format_protocol_extras", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("NA", body) &&
           occursin("(none)", body) &&
           occursin("isempty(extras)", body)
end

function discovery_n_samples_entry_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function discover_equations(p_trained, nn, st;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("n_samples ≥ 20", body) &&
           occursin("n_samples must be at least 20", body)
end

function discovery_sample_floor_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function _run_discovery", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("size(X, 2) ≥ 20", body) &&
           occursin("insufficient finite trajectory samples", body) &&
           occursin("empty support: no terms survived thresholding", body)
end

function discovery_retcode_mapper_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function _discovery_retcode", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("DenominatorUnsafe", body) &&
           occursin("SingularLibrary", body) &&
           occursin("InsufficientSamples", body) &&
           occursin("EmptySupport", body) &&
           occursin("DiscoveryFailed", body) &&
           occursin("insufficient", body) &&
           occursin("empty support", body)
end

function suite_library_index_holds()
    text = format_suite_library_index()
    return occursin("three_state", text) &&
           occursin("wrong_graph", text) &&
           occursin("graph_prior", text) &&
           occursin("reference_protocol", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function graph_local_library_index_holds()
    text = format_graph_local_library_index()
    names = graph_local_library_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("wrong graph", text) &&
           occursin("local_has_true_parent_check", text) &&
           !occursin("support_f1_ude = 0.99", text)
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

function recovery_suite_uses_parent_checks_source_holds()
    src = read(recovery_jl_source_path(), String)
    return occursin("local_has_true_parent = local_has_true_parent_check(", src) &&
           occursin("local_false_parent = local_has_false_parent_check(", src) &&
           occursin("if :three_state in wanted", src) &&
           occursin("if :wrong_graph in wanted", src) &&
           occursin("if :six_state in wanted", src)
end

function local_has_true_parent_check_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst(
        "function local_has_true_parent_check(candidate; variable::Int", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("support_uses_variable", body) &&
           occursin("candidate === nothing && return false", body)
end

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

function hybrid_compose_index_holds()
    text = format_hybrid_compose_index()
    names = hybrid_compose_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("ude_system", text) &&
           occursin("export_rhs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function hybrid_compose_honesty_matrix()
    zero = linear_zero_hole_compose_row()
    dual = dual_only_throws_row()
    remap = remapped_compose_row()
    skipped = skipped_duplicate_compose_row()
    failed = failed_export_rhs_row()
    empty = empty_export_rhs_row()
    discover = discover_then_compose_row()
    known = hill_known_generate_unknown_identity_row()
    multi = multi_ic_identity_residual_row()
    constant = constant_rate_changes_residual_row()
    middle = skipped_middle_compose_row()
    mm_known = mm_known_no_compose_row()
    repress = repressilator_no_compose_row()
    sample = sample_destruction_matches_identity_row()
    irregular = irregular_times_residual_row()
    exploding = failed_solve_residual_is_inf_row()
    dual_terms = dual_per_term_compose_row()
    session = session_predict_hybrid_row()
    normalized = normalize_destruction_honesty_row()
    explicit_fn = equation_to_function_explicit_row()
    no_compile = compose_does_not_compile_row()
    shape = residual_shape_guard_row()
    kinetic = kinetic_known_no_compose_row()
    typed = hybrid_compose_typed_matrix()
    smoke = reference_protocol_smoke_identity_row()
    return (;
        zero, dual, remap, skipped, failed, empty, discover, known,
        multi, constant, middle, mm_known, repress, sample, irregular,
        exploding, dual_terms, session, normalized, explicit_fn, no_compile,
        shape, kinetic, typed, smoke,
        holds = zero.holds && dual.holds && remap.holds && skipped.holds &&
                failed.holds && empty.holds && discover.holds &&
                known.holds && multi.holds && constant.holds &&
                middle.holds && mm_known.holds && repress.holds &&
                sample.holds && irregular.holds && exploding.holds &&
                dual_terms.holds && session.holds && normalized.holds &&
                explicit_fn.holds && no_compile.holds &&
                shape.holds && kinetic.holds && typed.holds && smoke.holds)
end

function sample_unknown_destruction_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function sample_unknown_destruction(model::UDEModel, p, X", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("_destruction_contribution", body) &&
           occursin("chosen.regulators", body)
end

function export_rhs_rejects_failure_source_holds()
    src = read(discovery_jl_source_path(), String)
    start = findfirst("function export_rhs", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("result.success", body) &&
           occursin("cannot export RHS from a failed discovery", body)
end

function hybrid_data_residual_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function hybrid_data_residual", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("compose_hybrid_rhs", body) &&
           occursin("SciMLBase.ODEProblem", body) &&
           occursin("sqrt(mean(abs2", body) &&
           occursin("mask", body)
end

function compose_hybrid_rhs_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function compose_hybrid_rhs", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("ude_system(u, p, t, model)", body) &&
           occursin("_destruction_contribution", body) &&
           occursin("rate_fn", body) &&
           occursin("term.regulators", body) &&
           !occursin("compile_network", body)
end

function normalize_destruction_honesty_row()
    values = [0.0, 0.5, -1.0, 2.0]
    scaled, scale = normalize_destruction_samples(values)
    return (;
        scale,
        maxabs = maximum(abs, scaled),
        holds = scale == 2.0 && maximum(abs, scaled) ≈ 1.0 &&
                scaled ≈ values ./ 2.0)
end

function hybrid_residual_index_holds()
    text = format_hybrid_residual_index()
    names = hybrid_residual_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("SciMLBase.solve", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function hybrid_residual_honesty_matrix()
    linear = failed_compose_linear_term_row()
    empty = failed_compose_empty_terms_row()
    dual = failed_compose_dual_only_row()
    export_failed = failed_compose_export_row()
    empty_export = failed_compose_empty_export_row()
    exploding = hybrid_residual_failed_solve_row()
    shape = hybrid_residual_shape_guard_row()
    wrong = failed_compose_wrong_rate_row()
    remap = remapped_residual_solver_row()
    skipped = skipped_duplicate_residual_solver_row()
    middle = skipped_middle_residual_solver_row()
    multi = multi_ic_residual_solver_row()
    known = hill_known_generate_unknown_solver_row()
    session = session_residual_solver_path()
    mm_known = mm_known_no_residual_row()
    repress = repressilator_no_residual_row()
    kinetic = kinetic_known_no_residual_row()
    zero = linear_zero_hole_residual_row()
    noise = noise_does_not_paint_f1_row()
    grid = begin
        built = hybrid_linear_unknown_model(389)
        noise_grid_residual_row(built.model, built.packed, [0.30, 0.25])
    end
    smoke = smoke_vs_protocol_residual_row()
    smoke_self = smoke_identity_on_self_row()
    protocol = protocol_fingerprint_not_dropped_row()
    typed = hybrid_residual_typed_matrix()
    return (;
        linear, empty, dual, export_failed, empty_export, exploding, shape,
        wrong, remap, skipped, middle, multi, known, session, mm_known,
        repress, kinetic, zero, noise, grid, smoke, smoke_self, protocol, typed,
        holds = linear.holds && empty.holds && dual.holds &&
                export_failed.holds && empty_export.holds && exploding.holds &&
                shape.holds && wrong.holds && remap.holds && skipped.holds &&
                middle.holds && multi.holds && known.holds && session.holds &&
                mm_known.holds && repress.holds && kinetic.holds && zero.holds &&
                noise.holds && grid.holds && smoke.holds && smoke_self.holds &&
                protocol.holds && typed.holds)
end

function predict_ude_uses_odeproblem_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Training.jl"), String)
    start = findfirst("function predict_ude(p, u0, tspan, saveat, nn, st;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("SciMLBase.ODEProblem", body) &&
           occursin("solver_config.algorithm", body)
end

function hybrid_residual_model_solve_source_holds()
    src = read(hybrid_residual_source_path(), String)
    start = findfirst(
        "function hybrid_residual_model_solve(model::UDEModel, p, u0, tspan, times, data)",
        src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("SciMLBase.ODEProblem(model", body) &&
           occursin("Tsit5()", body)
end

function hybrid_residual_sciml_solve_source_holds()
    src = read(hybrid_residual_source_path(), String)
    start = findfirst(
        "function hybrid_residual_sciml_solve(model, p, term, rate_fn, u0, tspan, times, data)",
        src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("compose_hybrid_rhs", body) &&
           occursin("SciMLBase.ODEProblem", body) &&
           occursin("Tsit5()", body) &&
           !occursin("Rodas5", body)
end

function hybrid_data_residual_uses_sciml_solve_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function hybrid_data_residual", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("compose_hybrid_rhs", body) &&
           occursin("SciMLBase.ODEProblem", body) &&
           occursin("Tsit5()", body) &&
           occursin("sensealg = nothing", body) &&
           occursin("sqrt(mean(abs2", body)
end

function identifiability_product_index_holds()
    text = format_identifiability_product_index()
    names = identifiability_product_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin("unidentifiable_edge", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function coefficients_are_biological_constants_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "ReferenceProtocol.jl"), String)
    start = findfirst(
        "function coefficients_are_biological_constants(ident)", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("unidentifiable_edge", body) &&
           occursin("!ident.unidentifiable_edge", body)
end

function format_protocol_result_collinearity_source_holds()
    src = read(recovery_jl_source_path(), String)
    start = findfirst("function format_protocol_result(ident;", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\n\"\"\"", rest, 40)
    body = nxt === nothing ? rest[1:min(lastindex(rest), 2500)] :
           rest[1:(first(nxt) - 1)]
    return occursin("coefficients_are_biological_constants", body) &&
           occursin("collinearity", body) &&
           occursin("isfinite(ident.collinearity)", body) &&
           occursin("canonical_hill_from_nn: false", body)
end

function format_production_destruction_warning_source_holds()
    src = read(identifiability_jl_source_path(), String)
    start = findfirst("function format_production_destruction_warning", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\n\"\"\"", rest, 20)
    body = nxt === nothing ? rest[1:min(lastindex(rest), 1200)] :
           rest[1:(first(nxt) - 1)]
    return occursin("not structural", body) &&
           occursin("collinear", body)
end

function production_destruction_tradeoff_source_holds()
    src = read(identifiability_jl_source_path(), String)
    start = findfirst(
        "function production_destruction_tradeoff(", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\n\"\"\"", rest, 40)
    body = nxt === nothing ? rest[1:min(lastindex(rest), 2500)] :
           rest[1:(first(nxt) - 1)]
    return occursin("assess_identifiability", body) &&
           occursin("collinearity", body) &&
           occursin("unidentifiable_edge", body) &&
           occursin("_destruction_contribution", body) &&
           !occursin("StructuralIdentifiability", body)
end

function parameter_schema_pack_index_holds()
    text = format_parameter_schema_pack_index()
    names = parameter_schema_pack_fixture_names()
    return length(unique(names)) == length(names) &&
           occursin(":k_custom", text) &&
           occursin("9 ICs", text) &&
           !occursin("support_f1_ude = 0.99", text)
end

function default_phys_includes_custom_source_holds()
    src = read(parameter_schema_jl_source_path(), String)
    return occursin(":k_custom => 0.8", src)
end

function frozen_phys_source_holds()
    src = read(joinpath(pkgdir(BioDynaX), "src", "Training.jl"), String)
    return occursin("function _zero_frozen_phys_gradient", src) &&
           occursin("function _restore_frozen_phys", src) &&
           occursin("name in frozen", src)
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

function recovery_suite_admission_source_violations()
    path = joinpath(pkgdir(BioDynaX), "src", "Recovery.jl")
    src = read(path, String)
    required = (
        "admit_recovery_suite_network(:ude_discovery)",
        "admit_recovery_suite_network(:mm_unknown)",
        "admit_recovery_suite_network(:ident_interventions)",
        "admit_recovery_suite_network(:partial_obs)",
        "only_unknown_destruction")
    forbidden = (
        "validate_network(ude_net); count_unknown_destructions",
        "if unknown_holes != 1; return validate_network")
    missing = [s for s in required if !occursin(s, src)]
    hits = [s for s in forbidden if occursin(s, src)]
    return (; missing, forbidden = hits)
end

function reference_protocol_non_trainers_source_hold()
    return ident_interventions_does_not_train_unknown_edge_source() &&
           partial_obs_does_not_train_unknown_edge_source()
end

function reference_protocol_skip_source_holds()
    return all(skipped_section_does_not_admit_source,
               recovery_suite_reference_protocol_sections()) &&
           all(skipped_section_does_not_generate_set_source,
               recovery_suite_reference_protocol_trainers()) &&
           train_unknown_edge_only_in_reference_protocol_source() &&
           recovery_suite_all_sections_checked() &&
           recovery_suite_default_sections_source()
end

function reference_protocol_f1_attempt_source_violations()
    src = read(reference_protocol_f1_attempt_path(), String)
    missing = [s for s in REFERENCE_PROTOCOL_F1_ATTEMPT_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s
                 for s in REFERENCE_PROTOCOL_F1_ATTEMPT_MUST_NOT_CONTAIN
                 if occursin(s, src)]
    return (; missing, forbidden)
end

function reference_protocol_example_source_violations()
    src = read(reference_protocol_example_path(), String)
    missing = [s for s in REFERENCE_PROTOCOL_EXAMPLE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in REFERENCE_PROTOCOL_EXAMPLE_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

function sensealg_nobs_honesty_matrix()
    linear = sensealg_nobs_honesty_row(
        build_ude_model(MersenneTwister(7), build_linear_test_network())[1])
    hill = sensealg_nobs_honesty_row(
        build_ude_model(MersenneTwister(11),
        build_hill_recovery_network(; known = false, hill_order = 2))[1])
    remap = sensealg_nobs_honesty_row(
        build_ude_model(MersenneTwister(13),
        build_remapped_two_regulator_network())[1])
    return (;
        linear,
        hill,
        remap,
        holds = linear.holds && hill.holds && remap.holds &&
                linear.neural == false && hill.neural && remap.neural)
end

"""
    sensealg_nobs_honesty_row(model)

`recommend_sensealg` at 20 observations versus the training lock, which
asks for 100. Mechanistic models may prefer `BacksolveAdjoint` on a short
horizon; the training lock still writes the 100-observation adjoint.
Neural holes stay interpolating at both widths.
"""
function sensealg_nobs_honesty_row(model::UDEModel)
    small = recommend_sensealg(model; n_observations = 20)
    large = recommend_sensealg(model; n_observations = 100)
    locked = lock_training_solver(model, SolverConfig())
    neural = neural_training_requires_interpolating(model)
    return (;
        neural,
        small_name = small.name,
        large_name = large.name,
        locked_kind = training_sensealg_kind(locked),
        lock_follows_nobs_100 = training_sensealg_kind(large.sensealg) ===
                                training_sensealg_kind(locked),
        holds = training_sensealg_kind(large.sensealg) ===
                training_sensealg_kind(locked) &&
                (neural ?
                 small.name === :interpolating_neural &&
                 large.name === :interpolating_neural :
                 small.name === :backsolve_mechanistic &&
                 large.name === :interpolating_default))
end

function training_sensealg_honesty_matrix()
    linear = linear_sensealg_honesty()
    hill = hill_ude_sensealg_honesty()
    remap = remapped_sensealg_honesty()
    two = two_regulator_sensealg_honesty()
    return (;
        linear,
        hill,
        remap,
        two,
        holds = linear.holds && hill.holds && remap.holds && two.holds &&
                linear.neural == false &&
                hill.neural && hill.zygote_kind === :interpolating &&
                remap.neural && two.neural)
end

function two_regulator_sensealg_honesty()
    rng = MersenneTwister(19)
    model, _ = build_ude_model(rng, build_two_regulator_unknown_network())
    return recommend_sensealg_honesty_row(model)
end

function remapped_sensealg_honesty()
    rng = MersenneTwister(13)
    model, _ = build_ude_model(rng, build_remapped_two_regulator_network())
    return recommend_sensealg_honesty_row(model)
end

function hill_ude_sensealg_honesty()
    rng = MersenneTwister(11)
    model, _ = build_ude_model(
        rng, build_hill_recovery_network(; known = false, hill_order = 2))
    return recommend_sensealg_honesty_row(model)
end

function linear_sensealg_honesty()
    rng = MersenneTwister(7)
    model, _ = build_ude_model(rng, build_linear_test_network())
    return recommend_sensealg_honesty_row(model)
end

function recommend_sensealg_honesty_row(model::UDEModel;
        n_observations::Int = 100)
    zy = recommend_sensealg(model; policy = ZygoteAD(), n_observations = n_observations)
    prod = recommend_sensealg(
        model; policy = ProductionAD(), n_observations = n_observations)
    neural = neural_training_requires_interpolating(model)
    locked = lock_training_solver(model, SolverConfig())
    return (;
        neural,
        zygote_kind = training_sensealg_kind(zy.sensealg),
        zygote_name = zy.name,
        production_kind = training_sensealg_kind(prod.sensealg),
        production_name = prod.name,
        locked_kind = training_sensealg_kind(locked),
        backsolve_forbidden_for_neural = neural,
        holds = (neural ? zy.name === :interpolating_neural :
                 zy.name === :backsolve_mechanistic ||
                 zy.name === :interpolating_default) &&
                prod.name === :interpolating_production &&
                training_sensealg_is_locked(model, locked) &&
                !(neural && training_sensealg_kind(locked) === :backsolve))
end

function hybrid_compose_fixture_matrix()
    identity = hybrid_compose_identity_matrix()
    honesty = hybrid_compose_honesty_matrix()
    return (;
        identity, honesty,
        holds = identity.holds && honesty.holds)
end

function hybrid_residual_fixture_matrix()
    identity = hybrid_residual_identity_matrix()
    honesty = hybrid_residual_honesty_matrix()
    return (;
        identity, honesty,
        holds = identity.holds && honesty.holds)
end

function reference_protocol_f1_attempt_holds()
    spec = reference_protocol_f1_attempt_spec()
    violations = reference_protocol_f1_attempt_source_violations()
    return spec.is_protocol == false &&
           spec.trains_ude == false &&
           spec.n_ics == 0 &&
           spec.uses_protocol_ics == false &&
           spec.new_atoms == false &&
           spec.support_f1_ude == RECOVERY_THRESHOLDS.support_f1_ude &&
           spec.support_f1_clean == RECOVERY_THRESHOLDS.support_f1_clean &&
           isempty(violations.missing) &&
           isempty(violations.forbidden)
end

function training_reuse_extended_matrix()
    fixture = linear_training_fixture()
    generate = training_session_matches_generate(
        fixture.model, fixture.init, fixture.u0;
        tspan = fixture.tspan, n_points = length(fixture.times))
    ude = train_ude_compile_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    frozen = frozen_phys_warmup_report(
        fixture.init, fixture.set, fixture.model)
    masked = masked_experiment_compile_report(
        fixture.init, fixture.set, fixture.model)
    nobs = sensealg_nobs_honesty_matrix()
    six = six_state_session_path()
    mm = mm_unknown_session_path()
    competitive = competitive_session_path()
    horizon = horizon_curriculum_session_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    roundtrip = optimizer_state_roundtrip_report(
        fixture.init, fixture.set, fixture.model)
    resume = resume_from_diagnostics_report(
        fixture.init, fixture.data, fixture.times, fixture.u0,
        fixture.tspan, fixture.model)
    return (;
        generate,
        ude,
        frozen,
        masked,
        nobs,
        six,
        mm,
        competitive,
        horizon,
        roundtrip,
        resume,
        holds = generate.holds && ude.holds && frozen.holds && masked.holds &&
                nobs.holds && six.holds && mm.holds && competitive.holds &&
                horizon.holds && roundtrip.holds && resume.holds)
end

function competitive_session_path()
    net = build_competitive_test_network(; known = true)
    rng = MersenneTwister(47)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters(
        (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6, k_s = 0.8, k_i = 0.5),
        params.nn)
    remake = training_session_remake_agreement(
        model, packed, [0.25, 0.45, 0.20]; tspan = (0.0, 0.6), n_points = 6)
    generate = training_session_matches_generate(
        model, packed, [0.25, 0.45, 0.20]; tspan = (0.0, 0.6), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        remake,
        generate,
        sense,
        n_heads = neural_head_count(model),
        validate_open = validate_network(net) === net,
        holds = remake.holds && generate.holds && sense.holds &&
                sense.neural == false && neural_head_count(model) == 0)
end

function mm_unknown_session_path()
    net = build_mm_recovery_network(; known = false)
    rng = MersenneTwister(43)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), params.nn)
    remake = training_session_remake_agreement(
        model, packed, [0.30, 0.25]; tspan = (0.0, 0.6), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        remake,
        sense,
        n_heads = neural_head_count(model),
        recovery_admits = reference_protocol_recovery_admits(net),
        holds = remake.holds && sense.holds && sense.neural &&
                neural_head_count(model) == 1 &&
                reference_protocol_recovery_admits(net))
end

function six_state_session_path()
    net = build_six_state_unknown_network(; known = false)
    rng = MersenneTwister(41)
    model, params = build_ude_model(rng, net)
    u0 = [0.22, 0.18, 0.16, 0.14, 0.12, 0.10]
    remake = training_session_remake_agreement(
        model, params, u0; tspan = (0.0, 0.5), n_points = 6)
    generate = training_session_matches_generate(
        model, params, u0; tspan = (0.0, 0.5), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    nobs = sensealg_nobs_honesty_row(model)
    return (;
        remake,
        generate,
        sense,
        nobs,
        n_heads = neural_head_count(model),
        nstates = model.compiled.nstates,
        holds = remake.holds && generate.holds && sense.holds && nobs.holds &&
                neural_head_count(model) == 1 &&
                model.compiled.nstates == 6)
end

function training_reuse_fixture_matrix()
    linear = training_session_remake_agreement(
        build_ude_model(MersenneTwister(7), build_linear_test_network())...,
        [0.22, 0.14])
    dual = dual_unknown_session_path()
    zero = zero_hole_session_path()
    skipped = skipped_duplicate_session_path()
    claim = reference_protocol_warmup_compile_path()
    sense = training_sensealg_honesty_matrix()
    return (;
        linear,
        dual,
        zero,
        skipped,
        claim,
        sense,
        holds = linear.holds && dual.holds && zero.holds && skipped.holds &&
                claim.holds && sense.holds)
end

function skipped_duplicate_session_path()
    net = build_skipped_duplicate_unknown_network()
    rng = MersenneTwister(13)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_b = 0.5, k_c = 0.4), params.nn)
    report = training_session_remake_agreement(
        model, packed, [0.2, 0.3, 0.4]; tspan = (0.0, 0.6), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        report,
        sense,
        n_heads = neural_head_count(model),
        dense = neural_index_is_dense(model),
        holds = report.holds && sense.holds && sense.neural &&
                neural_index_is_dense(model))
end

function zero_hole_session_path()
    net = build_zero_unknown_linear_network()
    rng = MersenneTwister(7)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_ba = 0.8, k_a = 1.2, k_b = 0.5), params.nn)
    report = training_session_remake_agreement(
        model, packed, [0.22, 0.14]; tspan = (0.0, 0.8), n_points = 6)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        report,
        sense,
        n_heads = neural_head_count(model),
        validate_open = validate_network(net) === net,
        holds = report.holds && sense.holds && sense.neural == false &&
                neural_head_count(model) == 0)
end

function dual_unknown_session_path()
    net = build_dual_unknown_network()
    rng = MersenneTwister(21)
    model, params = build_ude_model(rng, net)
    packed = pack_parameters((k_ca = 0.8, k_cb = 0.9, k_c = 0.5), params.nn)
    set = generate_experiment_set(
        MersenneTwister(21); network = net,
        initial_conditions = [[0.22, 0.18, 0.16], [0.30, 0.24, 0.20]],
        tspan = (0.0, 0.8), n_points = 6, noise_σ = 0.0,
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
    remake = training_session_multi_ic_agreement(model, packed, set)
    sense = recommend_sensealg_honesty_row(model)
    return (;
        remake,
        sense,
        n_heads = neural_head_count(model),
        recovery_admits = reference_protocol_recovery_admits(net),
        holds = remake.holds && sense.holds && sense.neural &&
                neural_head_count(model) == 2 &&
                reference_protocol_recovery_admits(net) == false)
end

function reference_protocol_warmup_compile_path(; smoke::Bool = true)
    net = build_hill_recovery_network(; known = false, hill_order = 2)
    truth = build_hill_recovery_network(; known = true, hill_order = 2)
    rng = MersenneTwister(103)
    model, p0 = build_ude_model(rng, net)
    set = reference_protocol_experiment_set(
        MersenneTwister(103), truth;
        smoke = smoke,
        truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6))
    names = Tuple(parameter_schema(model).phys_names)
    guess = NamedTuple{names}(ntuple(_ -> 0.8, length(names)))
    p_init = pack_parameters(guess, p0.nn)
    config = reference_protocol_training_config(
        model = model,
        adam_iterations = 1,
        bfgs_iterations = 0)
    n = with_compile_network_counter() do counter
        warmup_first_experiment(
            p_init, set, model; config = config, verbose = false)
        counter[]
    end
    rec = recommend_sensealg_honesty_row(model)
    return (;
        n_ics = length(set.experiments),
        compiles = n,
        sensealg = rec,
        compiled_once = experiment_set_is_compiled_once(set),
        holds = n == 0 && rec.holds && rec.neural &&
                experiment_set_is_compiled_once(set))
end

const TRAINING_REUSE_MUST_CONTAIN = (
    "mutable struct TrainingSolveSession",
    "function lock_training_solver",
    "function predict_ude_session",
    "function warmup_first_experiment",
    "function with_compile_network_counter",
    "function training_session_matches_generate",
    "function train_ude_compile_report",
    "function frozen_phys_warmup_report",
    "function sensealg_nobs_honesty_row",
    "optimizer_state")
