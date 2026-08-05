using BioDynaX
using Documenter

makedocs(
    modules = [BioDynaX],
    sitename = "BioDynaX.jl",
    pages = [
        "Home" => "index.md",
        "Architecture" => "architecture.md",
    ],
    checkdocs = :exports,
)
