using BioDynaX
using Documenter
using SciMLBase

makedocs(
    modules = [BioDynaX],
    sitename = "BioDynaX.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "How-to" => "howto.md",
        "SciML Integration" => "sciml.md",
        "Metadata" => "metadata.md",
        "Architecture" => "architecture.md",
        "Recovery benchmarks" => "benchmarks.md",
        "API" => "api.md",
        "Experimental" => "experimental.md",
        "API stability" => "stability.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)
