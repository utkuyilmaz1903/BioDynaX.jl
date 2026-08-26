@testset "landing docs drop HTTP 200 / TagBot lab notes" begin
    hits = unique_claim_docs_forbidden_hits()
    @test isempty(hits)
    for path in BioDynaX.unique_claim_user_doc_paths()
        @test isfile(path)
        text = read(path, String)
        @test !occursin("HTTP 200", text)
        @test !occursin("TagBot ran", text)
        @test !occursin("]add BioDynaX", text)
        @test !occursin("mertebe", text)
        @test !occursin("coverage badge", lowercase(text)) ||
              occursin("no coverage badge", lowercase(text)) ||
              occursin("without a fake", lowercase(text))
    end
end

@testset "unique-claim page states the product block and closed doors" begin
    path = joinpath(pkgdir(BioDynaX), "docs", "src", "unique-claim.md")
    @test isfile(path)
    text = read(path, String)
    sentences = unique_claim_locked_sentences()
    @test occursin("IDENTIFIABILITY", text)
    @test occursin("FIT", text)
    @test occursin("DISCOVERY", text)
    @test occursin("REPRODUCTION", text)
    @test occursin("unidentifiable_edge", text)
    @test occursin("coefficients_are_biological_constants", text)
    @test occursin("canonical_hill_from_nn", text)
    @test occursin("UNIQUE_CLAIM_PROTOCOL", text)
    @test occursin("seed", text) && occursin("103", text)
    @test occursin("9", text)
    @test occursin("BIODYNAX_SMOKE", text) || occursin("smoke", lowercase(text))
    @test occursin("validate_network", text)
    @test occursin("assert_single_unknown_destruction", text)
    @test occursin("0.50", text)
    @test !occursin("HTTP 200", text)
    @test !occursin("TagBot ran", text)
    @test !occursin("]add BioDynaX", text)
    @test occursin("not v1.0", lowercase(text)) || occursin("Not v1.0", text)
    @test occursin("not in General", text) || occursin("Not in General", text)
    @test occursin("skeleton", lowercase(text))
    for label in ("claim", "f1", "hill", "coefficients", "smoke", "preview")
        @test haskey(sentences, Symbol(label))
        @test !isempty(sentences[Symbol(label)])
    end
end

@testset "tutorial and howto keep one product block, not a mertebe table" begin
    tutorial = read(joinpath(pkgdir(BioDynaX), "docs", "src", "tutorial.md"), String)
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    index = read(joinpath(pkgdir(BioDynaX), "docs", "src", "index.md"), String)
    readme = read(joinpath(pkgdir(BioDynaX), "README.md"), String)
    for text in (tutorial, howto, index, readme)
        @test occursin("unidentifiable_edge", text) || text === howto
        @test !occursin("HTTP 200", text)
        @test !occursin("TagBot ran", text)
        @test !occursin("mertebe", text)
        @test !occursin("]add BioDynaX", text)
    end
    @test occursin("typical value", tutorial) || occursin("typical value", readme)
    @test occursin("format_protocol_result", tutorial)
    @test occursin("format_protocol_result", howto)
    @test occursin("unique-claim", lowercase(index)) ||
          occursin("Unique claim", index) ||
          occursin("unique claim", lowercase(index))
    @test occursin("seed 103", tutorial) || occursin("seed 103", readme)
    @test occursin("9", tutorial)
    @test occursin("BIODYNAX_SMOKE", howto) || occursin("smoke", lowercase(howto))
    @test occursin("assert_single_unknown_destruction", howto)
end

@testset "stability and contributing drop Pages plumbing" begin
    stability = read(joinpath(pkgdir(BioDynaX), "docs", "src", "stability.md"), String)
    contributing = read(joinpath(pkgdir(BioDynaX), "CONTRIBUTING.md"), String)
    @test !occursin("HTTP 200", stability)
    @test !occursin("HTTP 200", contributing)
    @test !occursin("Settings → Pages", contributing)
    @test !occursin("Settings -> Pages", contributing)
    @test !occursin("Pages source is `gh-pages`", stability)
    @test occursin("not yet in General", lowercase(stability)) ||
          occursin("not in General", stability)
    @test occursin("TagBot", contributing)
    @test !occursin("TagBot ran", stability)
    @test !occursin("TagBot ran", contributing)
    @test occursin("research preview", lowercase(stability))
    @test occursin("RECOVERY_THRESHOLDS", contributing)
end

@testset "public export list is frozen" begin
    @test public_export_list_holds()
    @test issetequal(names(BioDynaX), collect(locked_public_names()))
    @test :BioDynaX in locked_public_names()
    @test :validate_network in LOCKED_PUBLIC_EXPORTS
    @test :RECOVERY_THRESHOLDS in LOCKED_PUBLIC_EXPORTS
    @test !(:UNIQUE_CLAIM_PROTOCOL in LOCKED_PUBLIC_EXPORTS)
    @test !(:format_protocol_result in LOCKED_PUBLIC_EXPORTS)
    @test !(:build_protocol_result in LOCKED_PUBLIC_EXPORTS)
    @test !(:assert_single_unknown_destruction in LOCKED_PUBLIC_EXPORTS)
    @test !(:unique_claim_kpis_hold in LOCKED_PUBLIC_EXPORTS)
    @test !(:discovered_support_extras in LOCKED_PUBLIC_EXPORTS)
    @test !(:count_unknown_destructions in LOCKED_PUBLIC_EXPORTS)
    @test !(:identifiability_product in LOCKED_PUBLIC_EXPORTS)
    @test length(LOCKED_PUBLIC_EXPORTS) == length(unique(LOCKED_PUBLIC_EXPORTS))
end

@testset "docs make.jl lists the unique-claim page" begin
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    @test occursin("unique-claim.md", make)
    index = read(joinpath(pkgdir(BioDynaX), "docs", "src", "index.md"), String)
    @test occursin("unique-claim.md", index)
    news = read(joinpath(pkgdir(BioDynaX), "NEWS.md"), String)
    @test occursin("Unreleased", news)
    @test occursin("UNIQUE_CLAIM_PROTOCOL", news) || occursin("unique-claim", news)
end
