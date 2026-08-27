@testset "experiment checkpoint helpers are not exported" begin
    @test !(:experiment_fingerprint_row in names(BioDynaX))
    @test !(:experiment_batch_row in names(BioDynaX))
    @test !(:checkpoint_resume_row in names(BioDynaX))
    @test !(:remapped_generate_train_row in names(BioDynaX))
    @test !(:experiment_checkpoint_fixture_matrix in names(BioDynaX))
    @test !(:train_experiments_with_warmup in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.experiment_fingerprint_source_holds()
    @test BioDynaX.experiment_batches_source_holds()
    @test BioDynaX.resume_source_holds()
    @test BioDynaX.save_checkpoint_source_holds()
    @test BioDynaX.load_checkpoint_source_holds()
    @test BioDynaX.experiment_checkpoint_source_holds()
    @test BioDynaX.experiment_checkpoint_docs_hold()
    @test BioDynaX.experiment_checkpoint_landing_docs_hold()
    @test BioDynaX.experiment_checkpoint_docs_mention_helpers()
    @test BioDynaX.experiment_checkpoint_index_holds()
    @test BioDynaX.experiment_checkpoint_contract() ==
          BioDynaX.experiment_checkpoint_locked_sentences().remapped
    violations = BioDynaX.experiment_checkpoint_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    meta = BioDynaX.checkpoint_metadata_source_row()
    @test meta.holds
end

@testset "fingerprint metadata does not change the hash" begin
    set = BioDynaX.linear_probe_set()
    row = BioDynaX.experiment_fingerprint_row(set)
    @test row.holds
    @test row.stable
    @test row.metadata_ignored
    @test row.mask_changes
    @test row.u0_changes
    @test row.hex64
    hex = BioDynaX.hex_fingerprint_row(row.first_hash)
    @test hex.holds
    units = BioDynaX.units_change_fingerprint_row(set)
    @test units.holds
    irregular = BioDynaX.irregular_times_fingerprint_row(set)
    @test irregular.holds
end

@testset "generated trajectories hash the same IC and not another" begin
    row = BioDynaX.generated_data_fingerprint_row()
    @test row.holds
    @test row.compiles == 0
    @test row.same_ic
    @test row.other_ic
    noise = BioDynaX.noise_fingerprint_row()
    @test noise.holds
    uniqueness = BioDynaX.unique_claim_ic_fingerprint_uniqueness_row()
    @test uniqueness.holds
    csv = BioDynaX.csv_experiment_fingerprint_row()
    @test csv.holds
end

@testset "unique-claim smoke set fingerprints from a stored model" begin
    compiled = BioDynaX.unique_claim_from_compiled_fingerprint_row()
    @test compiled.holds
    @test compiled.compiles == 0
    @test compiled.n_ics == 1
    @test compiled.n_points == 8
    claim = BioDynaX.unique_claim_fingerprint_set_row()
    @test claim.holds
    @test claim.compiled_once
    @test claim.n_ics == 1
    @test claim.n_points == 8
end

@testset "batches cover ICs without padding" begin
    set = BioDynaX.linear_probe_set()
    row = BioDynaX.experiment_batch_row(set; batch_size = 2)
    @test row.holds
    @test row.sequential_covers
    @test row.shuffled_covers
    @test row.shuffle_reproducible
    @test row.no_pad
    remainder = BioDynaX.batch_remainder_row(set; batch_size = 2)
    @test remainder.holds
    @test remainder.last_short
    @test remainder.n_ics == 5
    shuffle = BioDynaX.shuffle_seed_independence_row(set; batch_size = 2)
    @test shuffle.holds
    weights = BioDynaX.experiment_weight_row(set)
    @test weights.holds
    claim_batch = BioDynaX.unique_claim_batch_row()
    @test claim_batch.holds
end

@testset "checkpoint resume stays compile-free" begin
    tmp = mktempdir()
    schema = BioDynaX.checkpoint_schema_row()
    @test schema.holds
    @test schema.version == v"1.0.0"
    ckpt = BioDynaX.linear_checkpoint_fixture_row(; dir = tmp)
    @test ckpt.holds
    @test ckpt.resume.compiles == 0
    @test ckpt.resume.diag_compiles == 0
    @test ckpt.artifact.holds
    frozen = BioDynaX.frozen_phys_checkpoint_row(; dir = tmp)
    @test frozen.holds
    @test frozen.compiles == 0
    equiv = BioDynaX.resume_equivalence_row(; dir = tmp)
    @test equiv.holds
    @test equiv.resume_compiles == 0
    @test equiv.fresh_compiles == 0
    optstate = BioDynaX.optimizer_state_checkpoint_row(; dir = tmp)
    @test optstate.holds
end

@testset "joint generate+train stays compile-free" begin
    remap = BioDynaX.remapped_generate_train_row()
    @test remap.holds
    @test remap.compiles == 0
    @test remap.n_heads == 2
    @test remap.arities == [1, 2]
    warmup = BioDynaX.remapped_warmup_generate_train_row()
    @test warmup.holds
    @test warmup.compiles == 0
    two = BioDynaX.two_regulator_generate_train_row()
    @test two.holds
    linear = BioDynaX.linear_generate_train_row()
    @test linear.holds
    hill = BioDynaX.hill_ude_generate_train_row()
    @test hill.holds
    dual = BioDynaX.dual_generate_train_row()
    @test dual.holds
    six = BioDynaX.six_state_generate_train_row()
    @test six.holds
    skipped = BioDynaX.skipped_duplicate_generate_train_row()
    @test skipped.holds
end

@testset "additional generate+train fixtures stay compile-free" begin
    mm_u = BioDynaX.mm_unknown_generate_train_row()
    @test mm_u.holds
    @test mm_u.compiles == 0
    comp_k = BioDynaX.competitive_known_generate_train_row()
    @test comp_k.holds
    comp_u = BioDynaX.competitive_unknown_generate_train_row()
    @test comp_u.holds
    middle = BioDynaX.skipped_middle_generate_train_row()
    @test middle.holds
    zero = BioDynaX.zero_hole_generate_fingerprint_row()
    @test zero.holds
    @test zero.holes == 0
    @test zero.recovery_admits == false
    @test zero.validate_open
    default = BioDynaX.default_example_generate_train_row()
    @test default.holds
    mm_k = BioDynaX.mm_known_generate_row()
    @test mm_k.holds
    mm_t = BioDynaX.mm_test_generate_train_row()
    @test mm_t.holds
    repress = BioDynaX.repressilator_generate_row()
    @test repress.holds
    linear_w = BioDynaX.linear_warmup_generate_train_row()
    @test linear_w.holds
    hill_w = BioDynaX.hill_warmup_from_compiled_row()
    @test hill_w.holds
    two_w = BioDynaX.two_regulator_warmup_generate_train_row()
    @test two_w.holds
    masked = BioDynaX.masked_fingerprint_train_row()
    @test masked.holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"ExperimentCheckpoint.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "docs", "src", "experiment-checkpoint.md"))
    @test isfile(joinpath(@__DIR__, "..", "src", "ExperimentCheckpoint.jl"))
    make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    @test occursin("experiment-checkpoint.md", make)
    howto = read(joinpath(@__DIR__, "..", "docs", "src", "howto.md"), String)
    @test occursin("experiment-checkpoint", howto)
    @test occursin("experiment_fingerprint", howto)
    landing = read(joinpath(@__DIR__, "..", "docs", "src", "sciml.md"), String)
    @test occursin(BioDynaX.experiment_checkpoint_contract(), landing)
    @test occursin("experiment-checkpoint", join(BioDynaX.unique_claim_user_doc_paths(), " "))
    @test BioDynaX.experiment_checkpoint_contract_holds()
end
