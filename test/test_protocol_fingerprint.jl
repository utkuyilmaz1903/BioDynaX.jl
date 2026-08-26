@testset "UNIQUE_CLAIM_PROTOCOL names smoke as a different object" begin
    proto = UNIQUE_CLAIM_PROTOCOL
    @test proto.seed == 103
    @test proto.n_ics == 9
    @test proto.smoke_n_ics == 1
    @test proto.n_points == 50
    @test proto.smoke_n_points == 8
    @test proto.adam_iterations == 100
    @test proto.bfgs_iterations == 50
    @test proto.tspan == (0.0, 8.0)
    @test proto.bootstrap == 8
    @test proto.discovery_seed == 3
    @test proto.observation_noise == 0.0
    @test proto.n_ics != proto.smoke_n_ics
    @test proto.n_points != proto.smoke_n_points
    @test recovery_thresholds_hold()
    @test recovery_thresholds_lock() == RECOVERY_THRESHOLDS
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test RECOVERY_THRESHOLDS.support_f1_ude < RECOVERY_THRESHOLDS.support_f1_clean
    @test !(:UNIQUE_CLAIM_PROTOCOL in names(BioDynaX))
    @test !(:unique_claim_is_protocol in names(BioDynaX))
    @test !(:unique_claim_reproduction in names(BioDynaX))
end

@testset "protocol helpers distinguish 9-IC job from 1-IC smoke" begin
    @test unique_claim_protocol_kind() === :protocol
    @test unique_claim_protocol_kind(; smoke = true) === :smoke
    @test unique_claim_protocol_n_ics() == 9
    @test unique_claim_protocol_n_ics(; smoke = true) == 1
    @test unique_claim_protocol_n_points() == 50
    @test unique_claim_protocol_n_points(; smoke = true) == 8

    protocol_ics = unique_claim_protocol_ics()
    smoke_ics = unique_claim_protocol_ics(; smoke = true)
    @test length(protocol_ics) == 9
    @test length(smoke_ics) == 1
    @test smoke_ics == protocol_ics[1:1]
    @test protocol_ics == [
        [0.25, 0.20], [0.80, 0.35], [0.40, 1.10], [1.20, 0.70], [0.15, 0.90],
        [0.50, 0.15], [0.90, 1.50], [0.20, 0.50], [1.50, 1.20]]
    @test unique_claim_protocol_ics() == BioDynaX._unknown_edge_ics()

    @test unique_claim_is_protocol()
    @test unique_claim_is_protocol(; smoke = true) == false
    @test unique_claim_is_protocol(; seed = 7) == false
    @test unique_claim_is_protocol(; n_ics = 1) == false
    @test unique_claim_is_protocol(; n_points = 8) == false
    @test unique_claim_is_protocol(; adam_iterations = 2) == false
    @test unique_claim_is_protocol(; bfgs_iterations = 0) == false
    @test unique_claim_is_protocol(;
        seed = 103, n_ics = 9, n_points = 50,
        adam_iterations = 100, bfgs_iterations = 50)

    repro = unique_claim_reproduction()
    @test repro.seed == 103
    @test repro.n_ics == 9
    @test repro.n_points == 50
    @test repro.adam_iters == 100
    @test repro.bfgs_iters == 50
    @test repro.bootstrap == 8
    @test repro.discovery_seed == 3
    @test repro.protocol_kind === :protocol
    @test repro.smoke == false
    @test repro.is_protocol

    smoke = unique_claim_reproduction(; smoke = true)
    @test smoke.n_ics == 1
    @test smoke.n_points == 8
    @test smoke.bfgs_iters == 0
    @test smoke.bootstrap === nothing
    @test smoke.discovery_seed === nothing
    @test smoke.protocol_kind === :smoke
    @test smoke.is_protocol == false

    budget = unique_claim_training_budget()
    @test budget.tspan == (0.0, 8.0)
    @test budget.observation_noise == 0.0
    @test budget.is_protocol
    smoke_budget = unique_claim_training_budget(; smoke = true)
    @test smoke_budget.n_ics == 1
    @test smoke_budget.n_points == 8
    @test smoke_budget.bfgs_iterations == 0
    @test smoke_budget.is_protocol == false
    @test smoke_budget.adam_iterations == 100
end

@testset "golden-path example source follows the protocol helpers" begin
    violations = unique_claim_example_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    src = read(BioDynaX.unique_claim_example_path(), String)
    @test occursin("unique_claim_protocol_ics(; smoke)", src)
    @test occursin("unique_claim_protocol_n_points(; smoke)", src)
    @test occursin("unique_claim_discovery_extras", src)
    @test occursin("count_unknown_destructions(model)", src)
    @test occursin("unique_claim_protocol_kind(; smoke)", src)
    @test !occursin("extras = (\"1\", \"r\")", src)
    @test !occursin("build_hill_recovery_network", src)
    @test !occursin("Note:", src)
    @test occursin("assert_unique_claim_residual", src)
    @test occursin("assert_single_unknown_destruction", src)
    cfg = unique_claim_discovery_config()
    @test cfg.seed == UNIQUE_CLAIM_PROTOCOL.discovery_seed
    @test cfg.backend.bootstrap_samples == UNIQUE_CLAIM_PROTOCOL.bootstrap
end

@testset "formatter lock stays SciML and does not rewrite the tree" begin
    @test isfile(BioDynaX.julia_formatter_toml_path())
    @test julia_formatter_toml_holds()
    lock = BioDynaX.julia_formatter_lock()
    @test lock.style == "sciml"
    @test lock.margin == 92
    @test lock.indent == 4
    @test lock.whitespace_in_kwargs
    @test lock.remove_extra_newlines
    text = read(BioDynaX.julia_formatter_toml_path(), String)
    @test !occursin("overwrite = true", text)
    @test occursin("style = \"sciml\"", text)
end

@testset "UDE F1 attempt script stays an attempt, not the protocol" begin
    path = joinpath(pkgdir(BioDynaX), "benchmark", "ude_f1_attempt.jl")
    src = read(path, String)
    @test occursin("same library", src)
    @test occursin("No new atoms", src)
    @test occursin("unique_claim_discovery_extras", src)
    @test occursin("support_f1_clean", src)
    @test occursin("extras remain", lowercase(src)) ||
          occursin("RESULT:", src)
    @test !occursin("canonical Hill from a trained NN is open", src)
    @test !occursin("support_f1_ude = 0.99", src)
    @test occursin("RECOVERY_THRESHOLDS", src)
    verdict = unique_claim_f1_attempt_verdict(;
        extras = ["1", "r"], reaches_clean = false)
    @test verdict === :extras_remain_claim_stays_recall_plus_residual
end
