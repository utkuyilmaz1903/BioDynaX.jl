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
        "Training reuse" => "training-reuse.md",
        "SciML solve surface" => "sciml-solve-surface.md",
        "Recovery suite skip" => "recovery-suite-skip.md",
        "Experiment fingerprint and checkpoint" => "experiment-checkpoint.md",
        "Failure modes" => "failure-modes.md",
        "Hybrid compose path" => "hybrid-compose.md",
        "Hybrid residual versus solver" => "hybrid-residual.md",
        "Identifiability product rows" => "identifiability-product.md",
        "Graph-local library and ablation" => "graph-local-library.md",
        "Denominator and domain safety" => "denominator-domain.md",
        "Parameter schema and pack" => "parameter-schema-pack.md",
        "Docs executable path" => "docs-executable.md",
        "Allocation and type-stability gates" => "allocation-gates.md",
        "Claim metric honesty" => "claim-metric-honesty.md",
        "Out of scope" => "out-of-scope.md",
        "Internal workspaces" => "internal-workspaces.md",
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
