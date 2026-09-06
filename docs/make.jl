using BioDynaX
using Documenter
using OrdinaryDiffEq
using Random
using SciMLBase

DocMeta.setdocmeta!(
    BioDynaX, :DocTestSetup,
    :(using BioDynaX, SciMLBase, OrdinaryDiffEq, Random);
    recursive = true)

# The changelog page is generated from the repository CHANGELOG.md so that
# there is a single source of truth.
let changelog = read(joinpath(@__DIR__, "..", "CHANGELOG.md"), String)
    write(joinpath(@__DIR__, "src", "changelog.md"), changelog)
end

makedocs(
    modules = [BioDynaX],
    sitename = "BioDynaX.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main"),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Tutorial" => "tutorial.md",
        "Concepts" => "concepts.md",
        "How-to recipes" => "howto.md",
        "Benchmarks" => "benchmarks.md",
        "Case study: laccase/ABTS (measured data)" => "case-study-laccase.md",
        "Case study: p53–Mdm2 (partially observed)" => "case-study-p53.md",
        "API reference" => "api.md",
        "Extensions" => "extensions.md",
        "Scope and limitations" => "limitations.md",
        "Changelog" => "changelog.md"
    ],
    checkdocs = :exports,
    doctest = true
)

# Documenter decides when to deploy: pushes to `main` update `dev/`, pushes
# of a `v*` tag create the version directory and move `stable`. Pull
# requests and other refs only build.
if get(ENV, "CI", "false") == "true"
    deploydocs(
        repo = "github.com/utkuyilmaz1903/BioDynaX.jl.git",
        devbranch = "main",
        push_preview = false
    )
end
