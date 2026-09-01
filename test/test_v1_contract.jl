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
    q4_body = text[first(q4_at):min(lastindex(text), first(q4_at) + 800)]
    q7_body = text[first(q7_at):min(lastindex(text), first(q7_at) + 900)]
    @test occursin("not implemented", lowercase(q4_body))
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

@testset "Q4 and Q7 have no implemented objects on existing types" begin
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
    @test recovery_thresholds_hold()
    @test public_export_list_holds()
    @test RECOVERY_THRESHOLDS.data_residual == 0.30
    @test RECOVERY_THRESHOLDS.support_recall == 0.99
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
    @test validate_network_stays_open_source()
end
