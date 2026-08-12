using BioDynaX
using Documenter

makedocs(
    modules = [BioDynaX],
    sitename = "BioDynaX.jl",
    pages = [
        "Home" => "index.md",
        "SciML Integration" => "sciml.md",
        "Metadata" => "metadata.md",
        "Architecture" => "architecture.md",
    ],
    checkdocs = :exports,
)
