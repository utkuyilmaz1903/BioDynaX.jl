@testset "generate_experiment_set compiles the ground-truth model once" begin
    @test generate_experiment_set_uses_compiled_once()
    @test generate_from_compiled_model_uses_sciml_odeproblem()
    @test datagen_experiment_set_source_holds()
    @test compiled_path_datagen_source_holds()
    @test !(:compile_ground_truth_model in names(BioDynaX))
    @test !(:generate_experiment_set_from_compiled_model in names(BioDynaX))
    @test !(:CompiledPathRow in names(BioDynaX))
    @test !(:joint_compiled_path in names(BioDynaX))
    @test !(:recovery_suite_admission_matrix in names(BioDynaX))
end

@testset "multi-IC unknown networks share one compiled parameter tree" begin
    net = build_dual_unknown_network()
    row = compiled_experiment_set_row(
        net;
        rng = MersenneTwister(21),
        initial_conditions = [[0.22, 0.18, 0.16], [0.30, 0.24, 0.20], [0.15, 0.12, 0.18]],
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
    @test row.holds
    @test row.n_ics == 3
    @test row.arch.n_heads == 2
    @test packed_nn_head_count(row.truth.parameters) == 2
    first_p = first(row.set.experiments).metadata[:truth_parameters]
    for exp in row.set.experiments
        @test exp.metadata[:truth_parameters] === first_p
        @test size(exp.observations, 1) == 3
        @test size(exp.observations, 2) == 8
        @test all(isfinite, exp.observations)
    end
    @test row.set.metadata[:compiled_once] === true
    @test experiment_set_is_compiled_once(row.rebuilt)
end

@testset "unique-claim experiment set is compiled-once and fingerprint-sized" begin
    path = unique_claim_compiled_path(; smoke = true)
    @test path.holds
    @test path.compiled_once
    @test path.matches_fingerprint
    @test length(path.set.experiments) == 1
    @test size(first(path.set.experiments).observations, 2) == 8
    @test path.set.metadata[:unique_claim_fingerprint_kind] === :smoke
    @test path.recovery_admits
    @test path.validate_open
    @test path.sciml.holds
    @test unique_claim_fingerprint_is_smoke(path.fingerprint)
    @test assert_unique_claim_protocol_row(path.protocol) === path.protocol
end

@testset "SciML ODEProblem(model) matches generate_from_compiled_model" begin
    linear = sciml_compiled_generate_agreement(
        build_linear_test_network();
        rng = MersenneTwister(7),
        u0 = [0.22, 0.14],
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    @test linear.holds
    @test linear.matches_odeproblem
    @test linear.matches_inplace
    @test linear.matches_remake
    @test linear.matches_sciml_solve
    two = two_regulator_sciml_path()
    @test two.holds
    skipped = skipped_duplicate_sciml_path()
    @test skipped.holds
    remap = remapped_two_regulator_compiled_path()
    @test remap.holds
    @test remap.row.sciml.matches_odeproblem
    @test remap.row.arch.arities == [1, 2]
    @test remap.row.recovery_admits == false
end

@testset "joint compiled path joins generate, remap, admission, protocol row" begin
    remap = remapped_two_regulator_compiled_path()
    @test joint_compiled_path_holds(remap.row)
    named = compiled_path_row_namedtuple(remap.row)
    @test named.n_heads == 2
    @test named.compiled_once
    @test named.recovery_admits == false
    @test named.sciml_holds
    dual = dual_unknown_compiled_path()
    @test dual.holds
    @test dual.open.admission.admitted
    @test dual.closed.admission.admitted == false
    zero = zero_hole_compiled_path()
    @test zero.holds
    @test zero.row.admission.admitted
    @test zero.closed.admitted == false
    default = default_example_compiled_once_path()
    @test default.holds
    @test default.holes == 1
end

@testset "suite admission matrix covers every section including 0/2 probes" begin
    @test compiled_path_admission_source_holds()
    matrix = recovery_suite_admission_matrix()
    @test matrix.holds
    @test matrix.n_sections == length(RECOVERY_SUITE_SECTION_KINDS)
    @test matrix.unique_claim == 4
    kinds = recovery_suite_section_kinds()
    @test Set(recovery_suite_sections()) == Set(keys(kinds))
    for row in matrix.rows
        @test row.unknown_holes == row.expected_holes
        @test row.single_hole_in_validate_network == false
        if row.policy === :exactly_one
            @test row.unknown_holes == 1
            @test row.admitted
            @test recovery_suite_admits_hole_count(row.section, 0) == false
            @test recovery_suite_admits_hole_count(row.section, 2) == false
        elseif row.policy === :library_fixture
            @test row.section === :ablation
            @test row.compiles == false
            @test row.admitted
        else
            @test row.admitted
            @test recovery_suite_admits_hole_count(row.section, 0)
            @test recovery_suite_admits_hole_count(row.section, 2)
        end
    end
    zero_dual = recovery_suite_zero_dual_matrix()
    @test zero_dual.holds
    @test recovery_suite_unique_claim_sections_reject_zero_and_dual().holds
    @test recovery_suite_open_sections_admit_zero_and_dual().holds
    joint = joint_admission_and_compiled_matrix()
    @test joint.holds
end

@testset "compiled-path docs and export/threshold locks" begin
    @test compiled_path_docs_hold()
    @test compiled_path_landing_docs_hold()
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    sentences = unique_claim_locked_sentences()
    @test haskey(sentences, :compiled_once)
    @test haskey(sentences, :sciml_generate)
    @test haskey(sentences, :admission_matrix)
    @test haskey(sentences, :joint)
    page = read(joinpath(pkgdir(BioDynaX), "docs", "src", "unique-claim.md"), String)
    @test occursin(sentences.compiled_once, page)
    @test occursin(sentences.sciml_generate, page)
    @test occursin(sentences.admission_matrix, page)
    @test occursin(sentences.joint, page)
    @test compiled_path_contract_holds()
end
