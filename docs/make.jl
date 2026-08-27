using BioDynaX
using Documenter
using OrdinaryDiffEq
using Random
using SciMLBase

DocMeta.setdocmeta!(
    BioDynaX, :DocTestSetup,
    :(using BioDynaX, SciMLBase, OrdinaryDiffEq, Random);
    recursive = true)

makedocs(
    modules = [BioDynaX],
    sitename = "BioDynaX.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "How-to" => "howto.md",
        "Unique claim" => "unique-claim.md",
        "Compiled experiment path" => "compiled-path.md",
        "Discovery streaming" => "discovery-streaming.md",
        "SciML Integration" => "sciml.md",
        "Metadata" => "metadata.md",
        "Architecture" => "architecture.md",
        "Recovery benchmarks" => "benchmarks.md",
        "API" => "api.md",
        "Experimental" => "experimental.md",
        "API stability" => "stability.md",
    ],
    checkdocs = :exports,
    doctest = true,
)

if get(ENV, "CI", "false") == "true" &&
        get(ENV, "GITHUB_REF", "") == "refs/heads/main"
    deploydocs(
        repo = "github.com/utkuyilmaz1903/BioDynaX.jl.git",
        devbranch = "main",
        push_preview = false,
    )
end
