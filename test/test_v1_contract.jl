@testset "v1.0 scientific contract file exists" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "design", "v1_contract.md")
    @test isfile(path)
    text = read(path, String)
    @test occursin("BioDynaX v1.0 scientific contract", text)
    @test !occursin("HTTP 200", text)
    @test !occursin("TagBot ran", text)
    @test !occursin("]add BioDynaX", text)
    @test !occursin("mertebe", text)
    @test !occursin("support_f1_ude = 0.99", text)
end

@testset "contract states the implemented P-D·u form" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "design", "v1_contract.md")
    text = read(path, String)
    compiler = read(joinpath(pkgdir(BioDynaX), "src", "MechanismCompiler.jl"),
        String)
    @test occursin(
        "cache.production[i] - cache.destruction[i] * x[i]", compiler)
    @test occursin("P_i(u,p,t) - D_i(u,p,t)", text)
    @test occursin("The vision form is not the v1.0 product.", text)
    @test occursin("f_{\\mathrm{known}}", text)
end

@testset "Q1–Q7 stay conceptually separate; later layers are not implied" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "design", "v1_contract.md")
    text = read(path, String)
    @test !occursin("full distinction", lowercase(text))
    @test occursin("conceptual distinction", lowercase(text))
    @test occursin("Trajectory fit is not proof of mechanism recovery.", text)
    @test occursin("Q3 must remain a practical scale/parameter warning.", text)
    @test occursin(
        "Q5 support recall/F1 is not canonical Hill recovery.", text)
    @test occursin("Q4 is not a formal identifiability certificate.", text)
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7")
        @test occursin(q, text)
    end
    q4_at = findfirst("### Q4", text)
    q7_at = findfirst("### Q7", text)
    @test q4_at !== nothing
    @test q7_at !== nothing
    q4_body = text[first(q4_at):min(lastindex(text), first(q4_at) + 2000)]
    q7_body = text[first(q7_at):min(lastindex(text), first(q7_at) + 900)]
    @test occursin("implemented as a practical diagnostic, not a gate",
        lowercase(q4_body))
    @test occursin("q4 is not a formal identifiability certificate.",
        lowercase(q4_body))
    @test !occursin("not implemented", lowercase(q4_body))
    @test occursin("reported held-out generalization evidence", lowercase(q7_body))
    @test occursin("not an additional success gate", lowercase(q7_body))
    @test !occursin("not implemented", lowercase(q7_body))
end

@testset "current unique-claim hold is Q3 + Q1 residual + Q5 recall" begin
    @test UNIQUE_CLAIM_KPI_FIELDS ==
          (:unidentifiable_edge, :data_residual, :support_recall)
    @test :support_f1 ∉ UNIQUE_CLAIM_KPI_FIELDS
    @test :nn_correlation ∉ UNIQUE_CLAIM_KPI_FIELDS
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    hold = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.99,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpis_hold(hold)
    miss_q3 = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = false)))
    @test unique_claim_kpis_hold(miss_q3) == false
    @test unique_claim_kpi_failures(miss_q3) == [:unidentifiable_edge]
    miss_q1 = locked_ude_kpis((;
        data_residual = 0.31,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpis_hold(miss_q1) == false
    @test unique_claim_kpi_failures(miss_q1) == [:data_residual]
    miss_q5 = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 0.50,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpis_hold(miss_q5) == false
    @test unique_claim_kpi_failures(miss_q5) == [:support_recall]
end

@testset "Q3 flag is condition-number OR cosine; not structural" begin
    @test BioDynaX.unidentifiable_edge_from_fisher(;
        condition_number = 1.0e7, collinearity = 0.10)
    @test BioDynaX.unidentifiable_edge_from_fisher(;
        condition_number = 10.0, collinearity = 0.96)
    @test BioDynaX.unidentifiable_edge_from_fisher(;
        condition_number = 10.0, collinearity = 0.10) == false
    ident = (; unidentifiable_edge = true, production_param = :k_prod)
    @test coefficients_are_biological_constants(ident) == false
    product = identifiability_product(ident)
    @test product.practical_not_structural
    @test product.unidentifiable_edge
    @test product.coefficients_are_biological_constants == false
end

@testset "Q5 protocol result keeps canonical Hill closed" begin
    result = build_protocol_result((;
        data_residual = 0.003,
        support_recall = 1.0,
        support_f1 = 0.57,
        extras = ["1", "r"],
        identifiability = (; unidentifiable_edge = true)))
    @test result.canonical_hill_from_nn === false
    @test result.claim === :recall_plus_data_residual
    @test result.unidentifiable_edge
end

@testset "Q4 is not attached to existing unique-claim types; Q7 does not mutate ExperimentSet" begin
    report_fields = fieldnames(BioDynaX.IdentifiabilityReport)
    @test :fisher_information in report_fields
    @test :condition_number in report_fields
    @test :independently_trained_D ∉ report_fields
    @test :functional_identifiability ∉ report_fields
    @test hasfield(ExperimentSet, :experiments)
    @test !hasfield(ExperimentSet, :holdout)
    @test !hasfield(ExperimentSet, :train)
    @test UNIQUE_CLAIM_PROTOCOL.n_ics == 9
end

@testset "T-H-CERT contract names a practical diagnostic, not a certificate" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "design", "v1_contract.md")
    text = read(path, String)
    @test occursin("Q4 is not a formal identifiability certificate.", text)
    @test occursin("implemented as a practical diagnostic, not a gate", text)
    @test occursin("not a structural identifiability certificate", lowercase(text)) ||
          occursin("not a formal identifiability certificate", text)
    @test occursin("Do not call a Q4", text)
    @test occursin("functionally identifiable", text)
    warn_at = findfirst("Do not call a Q4", text)
    @test warn_at !== nothing
    warning = lowercase(text[first(warn_at):min(lastindex(text),
        first(warn_at) + 160)])
    @test occursin("functionally identifiable", warning)
    q4_at = findfirst("### Q4", text)
    q4_body = text[first(q4_at):min(lastindex(text), first(q4_at) + 2000)]
    @test !occursin("Q4 remains not implemented", q4_body)
    @test !occursin("When implemented", q4_body)
    @test occursin("(201, 202, 203, 204, 205)", q4_body)
end

@testset "T-H-GATE Q4 is not a hold conjunct and not a success gate" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "design", "v1_contract.md")
    text = read(path, String)
    @test occursin("not a success gate", lowercase(text))
    @test occursin("implemented as a practical diagnostic, not a gate", text)
    @test occursin("The hold is not Q2 uniqueness, not Q4, not Q7", text)
    @test occursin("not an input to `unique_claim_kpis_hold`", text)
    @test UNIQUE_CLAIM_KPI_FIELDS ==
          (:unidentifiable_edge, :data_residual, :support_recall)
    @test :function_disagree ∉ UNIQUE_CLAIM_KPI_FIELDS
    @test :status ∉ UNIQUE_CLAIM_KPI_FIELDS
    hold = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = true)))
    @test unique_claim_kpis_hold(hold)
    with_q4 = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = true),
        extras = (; function_disagree = true, status = :function_agree)))
    @test unique_claim_kpis_hold(with_q4)
    miss_q3 = locked_ude_kpis((;
        data_residual = 0.003,
        support_recall = 1.0,
        identifiability = (; unidentifiable_edge = false),
        extras = (; function_disagree = false)))
    @test unique_claim_kpis_hold(miss_q3) == false
    src = read(joinpath(pkgdir(BioDynaX), "src", "Recovery.jl"), String)
    start = findfirst("function unique_claim_kpis_hold", src)
    @test start !== nothing
    body = src[first(start):min(lastindex(src), first(start) + 800)]
    @test occursin("unidentifiable_edge", body)
    @test occursin("data_residual", body)
    @test occursin("support_recall", body)
    @test !occursin("function_disagree", body)
    @test !occursin("assess_functional_identifiability", body)
end

@testset "T-H-M2 Q4 is not a MechanismRecoveryResult field" begin
    @test :functional_identifiability ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :function_disagree ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :independently_trained_D ∉ fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :split in fieldnames(BioDynaX.MechanismRecoveryResult)
    @test :holdout in fieldnames(BioDynaX.MechanismRecoveryResult)
end

@testset "T-H-NAMES Q4 types stay unexported" begin
    @test :FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)
    @test :assess_functional_identifiability ∉ names(BioDynaX)
    @test :FUNCTIONAL_ID_RESTART_SEEDS ∉ names(BioDynaX)
    @test :FUNCTIONAL_ID_REPORTING_CUTOFFS ∉ names(BioDynaX)
    holdout = read(joinpath(pkgdir(BioDynaX), "test", "test_holdout.jl"), String)
    pipeline = read(joinpath(pkgdir(BioDynaX), "test",
        "test_recovery_pipeline.jl"), String)
    @test occursin(":FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)",
        holdout)
    @test occursin(":FunctionalIdentifiabilityDiagnostic ∉ names(BioDynaX)",
        pipeline)
    @test !occursin(
        "!isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)", holdout)
    @test !occursin(
        "!isdefined(BioDynaX, :FunctionalIdentifiabilityDiagnostic)", pipeline)
    @test public_export_list_holds()
    @test BioDynaX.FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
    @test BioDynaX.FUNCTIONAL_ID_REPORTING_CUTOFFS === (
        min_successful_restarts = 3,
        n_attempted_restarts = 5,
        traj_agree_rel_rmse = 0.05,
        d_disagree_scale_norm_rel_rmse = 0.20)
    @test recovery_thresholds_hold()
end

@testset "T-H-Q7 remains reported, not a gate, and not unimplemented" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "design", "v1_contract.md")
    text = read(path, String)
    q7_at = findfirst("### Q7", text)
    @test q7_at !== nothing
    q7_body = text[first(q7_at):min(lastindex(text), first(q7_at) + 1200)]
    @test occursin("**Reported, not a gate.**", q7_body)
    @test occursin("reported held-out generalization evidence", lowercase(q7_body))
    @test occursin("not an additional success gate", lowercase(q7_body))
    @test !occursin("not implemented", lowercase(q7_body))
    @test occursin("Q7 is reported held-out generalization evidence, not an additional success gate.",
        text)
end

@testset "contract lists closed claims without claiming them as product" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "design", "v1_contract.md")
    text = read(path, String)
    required = (
        "structural identifiability certificates",
        "unknown topology discovery",
        "general CRN solving",
        "arbitrary multi-hole discovery",
        "canonical Hill recovery from a trained NN",
        "biological-constant parameter claims under scale non-identifiability",
        "general missing-state UDE training",
        "wet-lab decision making",
        "general experimental-design engine",
        "LLM integration",
        "GPU training stack",
        "broad SBML kinetic parsing")
    for phrase in required
        @test occursin(phrase, text)
    end
    @test occursin("Out of scope", text)
    @test !occursin("canonical Hill from a trained NN is open", text)
    @test !occursin("StructuralIdentifiability.jl proves", text)
    @test !occursin("we extracted Hill", text)
end

@testset "landing pages point at the contract; gates are unchanged" begin
    root = pkgdir(BioDynaX)
    contract_link = "design/v1_contract.md"
    readme = read(joinpath(root, "README.md"), String)
    unique_page = read(joinpath(root, "docs", "src", "unique-claim.md"), String)
    architecture = read(joinpath(root, "docs", "src", "architecture.md"),
        String)
    scope = read(joinpath(root, "docs", "src", "out-of-scope.md"), String)
    index = read(joinpath(root, "docs", "src", "index.md"), String)
    stability = read(joinpath(root, "docs", "src", "stability.md"), String)
    make = read(joinpath(root, "docs", "make.jl"), String)
    @test occursin("docs/src/design/v1_contract.md", readme)
    @test occursin(contract_link, unique_page)
    @test occursin(contract_link, architecture)
    @test occursin(contract_link, scope)
    @test occursin(contract_link, index)
    @test occursin("design/v1_contract.md", make)
    @test occursin("The vision form is not the v1.0 product", architecture) ||
          occursin("is not the v1.0 product", architecture)
    @test !occursin("The scientific claim stays recall + residual", stability)
    @test occursin("not mechanistic recovery", lowercase(readme))
    for landing in (readme, unique_page, architecture, scope, index, stability)
        @test occursin("practical", lowercase(landing))
        @test occursin("not a gate", lowercase(landing)) ||
              occursin("not a success gate", lowercase(landing))
        @test occursin("certificate", lowercase(landing))
        @test !occursin("q4 remains not implemented", lowercase(landing))
        @test !occursin("m3 pending", lowercase(landing))
        @test !occursin("m3 (practical functional identifiability) and m4",
            lowercase(landing))
    end
    @test recovery_thresholds_hold()
    @test public_export_list_holds()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test validate_network_stays_open_source()
end

@testset "M4-0 semantic boundary does not replace Q4, composer, or holdout" begin
    root = pkgdir(BioDynaX)
    contract = read(joinpath(root, "docs", "src", "design", "v1_contract.md"),
        String)
    scope = read(joinpath(root, "docs", "src", "out-of-scope.md"), String)
    plan = read(joinpath(root, "docs", "research", "V1-IMPLEMENTATION-PLAN.md"),
        String)
    for text in (contract, scope)
        @test occursin("functional_identifiability_domain` remains the approved M3 domain",
            text)
        @test occursin("Q4 is not occupancy-based", text)
        @test occursin("Q4 is not structural identifiability", text)
        @test occursin("Q4 does not use M4 trajectory occupancy", text)
        @test occursin("_evaluate_unknown_rate_recovery", text)
        @test occursin("_regulator_grid", text)
        @test occursin("Dummy-time discovery remains", text)
        @test occursin("M4 occupancy must not replace the composer", text)
        @test occursin("evaluate_holdout` remains four-scalar `HoldoutEvidence",
            text)
        @test occursin("Holdout is not a 0.30 gate", text)
        @test occursin("Occupancy is not added to `HoldoutEvidence`", text)
        @test occursin("additional sampling/evaluation context", text)
        @test occursin("not a replacement for Q4", text)
        @test occursin("UNIQUE_CLAIM_PROTOCOL.seed = 103", text)
        @test occursin("FUNCTIONAL_ID_RESTART_SEEDS = (201, 202, 203, 204, 205)",
            text)
        @test occursin("ROBUSTNESS_SEEDS = (103, 107, 111, 113, 127)", text)
        @test occursin("RECOVERY_THRESHOLDS", text)
        @test occursin("FUNCTIONAL_ID_REPORTING_CUTOFFS", text)
        @test occursin("LOCKED_PUBLIC_EXPORTS", text)
        @test occursin("canonical_hill_from_nn == false", text)
        @test occursin("unique_claim_kpis_hold", text)
    end
    @test occursin("üç liste, karışmaz", plan) ||
          occursin("üç tohum listesi", plan)
    @test occursin("ROBUSTNESS_SEEDS", plan)
    @test occursin("FUNCTIONAL_ID_RESTART_SEEDS", plan)
    @test occursin("UNIQUE_CLAIM_PROTOCOL.seed", plan)
    @test occursin("(103, 107, 111, 113, 127)", plan)
    @test occursin("(201, 202, 203, 204, 205)", plan)
    @test UNIQUE_CLAIM_PROTOCOL.seed == 103
    @test BioDynaX.FUNCTIONAL_ID_RESTART_SEEDS === (201, 202, 203, 204, 205)
    @test recovery_thresholds_hold()
    @test public_export_list_holds()
    @test :ROBUSTNESS_SEEDS ∉ names(BioDynaX)
    @test :ROBUSTNESS_SEEDS ∉ LOCKED_PUBLIC_EXPORTS
end

@testset "M4-A2 live separation wording stays distinct from A1/B/C" begin
    root = pkgdir(BioDynaX)
    contract = read(joinpath(root, "docs", "src", "design", "v1_contract.md"),
        String)
    scope = read(joinpath(root, "docs", "src", "out-of-scope.md"), String)
    plan = read(joinpath(root, "docs", "research", "V1-IMPLEMENTATION-PLAN.md"),
        String)
    for text in (contract, scope)
        @test occursin("M4-A1 occupancy runtime exists", text)
        @test occursin("M4-A2 is live separation/contract tests", text)
        @test occursin("M4-B remains pending", text)
        @test occursin("M4-C remains pending", text)
        @test occursin("occupancy != Q4 domain.z", text)
        @test occursin("occupancy != M1 discovery grid", text)
        @test occursin("occupancy != M2 holdout evaluator", text)
        @test occursin(
            "Occupancy is not part of the recovery result, holdout result, or Q4 diagnostic",
            text)
        @test !occursin("M4-A runtime code is not present", text)
        @test !occursin("M4-A/B/C runtime code is not present", text)
    end
    @test occursin("M4-A1: implemented runtime", plan)
    @test occursin("M4-A2: live separation/contract tests", plan)
    @test occursin("M4-B: pending", plan)
    @test occursin("M4-C: pending", plan)
    @test occursin("occupancy ≠ M1 discovery grid", plan)
    @test occursin("occupancy ≠ M2 holdout evaluator", plan)
    @test occursin("occupancy ≠ M3 Q4 domain", plan)
    @test occursin("test/test_m4_a2_separation.jl", plan)
    @test !occursin("testler henüz uygulanmaz", plan)
end
