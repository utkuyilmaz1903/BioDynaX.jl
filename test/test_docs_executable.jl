@testset "docs executable helpers are not exported" begin
    @test !(:docs_executable_join_row in names(BioDynaX))
    @test !(:DocsExecutableRow in names(BioDynaX))
    @test !(:leftover_contradiction_hits in names(BioDynaX))
    @test !(:live_hl_kinetic_join_row in names(BioDynaX))
    @test public_export_list_holds()
    @test recovery_thresholds_hold()
    @test validate_network_stays_open_source()
    @test RECOVERY_THRESHOLDS.support_f1_ude == 0.50
    @test RECOVERY_THRESHOLDS.support_f1_clean == 0.99
end

@testset "source and landing contracts stay locked" begin
    @test BioDynaX.docs_executable_source_holds()
    @test BioDynaX.docs_executable_docs_hold()
    @test BioDynaX.docs_executable_landing_docs_hold()
    @test BioDynaX.docs_executable_docs_mention_helpers()
    @test BioDynaX.docs_executable_example_source_holds()
    @test BioDynaX.docs_executable_index_holds()
    @test BioDynaX.docs_executable_contract() ==
          BioDynaX.docs_executable_locked_sentences().join
    violations = BioDynaX.docs_executable_source_violations()
    @test isempty(violations.missing)
    @test isempty(violations.forbidden)
    @test BioDynaX.docs_executable_contract_holds()
end

@testset "H-L join and leftover scanners" begin
    join = BioDynaX.docs_executable_join_row()
    @test join.holds
    @test join.n == 5
    leftover = BioDynaX.leftover_contradiction_row()
    @test leftover.holds
    @test isempty(leftover.hits)
    @test BioDynaX.tutorial_mentions_hl_row().holds
    @test BioDynaX.howto_links_hl_row().holds
    @test BioDynaX.sciml_carries_hl_sentences_row().holds
    @test BioDynaX.architecture_carries_hl_links_row().holds
    @test BioDynaX.no_restated_ag_pages_row().holds
    @test BioDynaX.unique_claim_paths_include_hl_row().holds
    @test BioDynaX.make_lists_hl_and_exec_row().holds
    @test BioDynaX.leftover_catalog_holds()
    @test BioDynaX.hl_pages_exist_row().holds
end

@testset "live H-L executable joins stay compile-honest" begin
    kinetic = BioDynaX.live_hl_kinetic_join_row()
    @test kinetic.holds
    @test kinetic.kinetic_custom
    remap = BioDynaX.live_hl_remapped_join_row()
    @test remap.holds
    @test remap.nn_heads == 2
    @test BioDynaX.live_hl_linear_join_row().holds
    @test BioDynaX.live_hl_default_join_row().holds
    smoke = BioDynaX.live_hl_smoke_protocol_row()
    @test smoke.holds
    @test smoke.proto_ics == 9
    @test BioDynaX.live_hl_thresholds_row().holds
    @test BioDynaX.live_hl_exports_row().holds
    @test BioDynaX.executable_snippets_row().holds
    typed = BioDynaX.docs_executable_typed_matrix()
    @test typed.holds
    @test typed.named.n_surfaces == 5
    ics = BioDynaX.unique_claim_not_faster_docs_row()
    @test ics.holds
    @test ics.n_ics == 9
    @test BioDynaX.hl_own_page_sentences_row().holds
    @test BioDynaX.ag_pages_still_listed_row().holds
    @test BioDynaX.extra_user_doc_leftover_row().holds
    @test BioDynaX.live_hl_hill_join_row().holds
    @test BioDynaX.live_hl_mm_join_row().holds
    @test BioDynaX.live_hl_six_join_row().holds
    @test BioDynaX.live_hl_known_hill_join_row().holds
    @test BioDynaX.user_doc_inventory_holds()
    @test BioDynaX.required_tutorial_tokens_row().holds
    @test BioDynaX.required_howto_tokens_row().holds
    @test BioDynaX.required_sciml_tokens_row().holds
    @test BioDynaX.docs_executable_report_holds()
    @test BioDynaX.ag_page_inventory_row().holds
    @test BioDynaX.hl_page_inventory_row().holds
    @test BioDynaX.readme_leftover_row().holds
    @test BioDynaX.changelog_does_not_paint_f1_row().holds
    @test BioDynaX.hl_surface_callables_hold()
    @test BioDynaX.docs_executable_not_faster_protocol_row().holds
    @test BioDynaX.combined_f1_not_docs_kpi_row().holds
    @test BioDynaX.news_does_not_paint_f1_row().holds
    @test BioDynaX.index_page_leftover_row().holds
end

@testset "module include and docs page exist" begin
    src = read(joinpath(@__DIR__, "..", "src", "BioDynaX.jl"), String)
    @test occursin("include(\"DocsExecutable.jl\")", src)
    @test isfile(joinpath(@__DIR__, "..", "docs", "src", "docs-executable.md"))
    @test isfile(joinpath(@__DIR__, "..", "src", "DocsExecutable.jl"))
    make = read(joinpath(@__DIR__, "..", "docs", "make.jl"), String)
    @test occursin("docs-executable.md", make)
    howto = read(joinpath(@__DIR__, "..", "docs", "src", "howto.md"), String)
    @test occursin("docs-executable", howto)
    @test occursin("docs_executable_join_row", howto)
    sciml = read(joinpath(@__DIR__, "..", "docs", "src", "sciml.md"), String)
    @test occursin(BioDynaX.docs_executable_contract(), sciml)
    tutorial = read(joinpath(@__DIR__, "..", "docs", "src", "tutorial.md"), String)
    @test occursin("docs-executable", tutorial)
    @test occursin("parameter-schema-pack", tutorial)
    @test occursin("docs-executable",
        join(BioDynaX.unique_claim_user_doc_paths(), " "))
end
