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
        "SciML Integration" => "sciml.md",
        "Metadata" => "metadata.md",
        "Architecture" => "architecture.md",
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)
