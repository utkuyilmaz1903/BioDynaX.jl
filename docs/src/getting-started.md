# Getting started

## Installation

BioDynaX requires Julia 1.10 or newer. It is not yet in the General
registry, so install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/utkuyilmaz1903/BioDynaX.jl")
```

To work on the package itself, clone the repository and instantiate its
environment:

```bash
git clone https://github.com/utkuyilmaz1903/BioDynaX.jl.git
cd BioDynaX.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The first `using BioDynaX` precompiles the SciML dependencies and can take
several minutes.

## A first model

A network is a list of nodes plus either reactions (with stoichiometry and
regulators) or edges. Each reaction carries typed kinetic metadata that names
its rate parameters. The example below is the network used throughout the
documentation: `S` is produced in proportion to `R` and degraded by a
Hill-type mechanism driven by `R`; `R` is produced from `S` and decays
linearly. Passing `known = false` marks the Hill degradation as the one
unknown term.

```@example gs
using BioDynaX, Random

function network(; known::Bool)
    nodes = [NodeSpec(name = :S), NodeSpec(name = :R)]
    reactions = [
        ReactionSpec(name = :produce_s, stoichiometry = Dict(1 => 1.0), regulators = [2],
            metadata = MassActionMetadata(rate_param = :k_prod)),
        ReactionSpec(name = :degrade_s, stoichiometry = Dict(1 => -1.0), regulators = [2],
            known = known, family = HILL,
            metadata = HillMetadata(vmax_param = :vmax, k_param = :K, hill_order = 2)),
        ReactionSpec(name = :produce_r, stoichiometry = Dict(2 => 1.0), regulators = [1],
            metadata = MassActionMetadata(rate_param = :k_rs)),
        ReactionSpec(name = :decay_r, stoichiometry = Dict(2 => -1.0), regulators = Int[],
            metadata = LinearDecayMetadata(rate_param = :k_r))]
    return BiologicalNetwork(nodes, EdgeSpec[]; reactions = reactions)
end

model, p0 = build_ude_model(MersenneTwister(1), network(known = false))
parameter_schema(model).phys_names
```

The compiled model has three physical parameters (the Hill parameters `vmax`
and `K` belong to the unknown term and are replaced by network weights in
`p0.nn`). The model is an ordinary `ODEProblem` and can be solved with any
OrdinaryDiffEq solver:

```@example gs
using SciMLBase, OrdinaryDiffEq
p = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), p0.nn)
prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), p)
sol = solve(prob, Tsit5(); saveat = 0:2.0:10.0)
round.(Array(sol); digits = 3)
```

## A complete fit

The block below generates synthetic data from the fully known version of the
network, trains the hybrid model on three initial conditions, prints the
identifiability warning, and discovers a symbolic rate. It runs in about two
minutes and is not executed in the documentation build.

```julia
rng = MersenneTwister(1)
tspan = (0.0, 10.0)
truth = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
data = generate_experiment_set(rng; network = network(known = true), truth_params = truth,
    initial_conditions = [[0.2, 0.1], [1.0, 0.5], [0.5, 1.2]], tspan = tspan,
    n_points = 40, noise_σ = 0.0)

model, p0 = build_ude_model(rng, network(known = false))
p_init = pack_parameters((k_prod = 0.8, k_rs = 0.8, k_r = 0.8), p0.nn)
trained = train_experiments(p_init, data, model;
    config = TrainingConfig(adam_iterations = 100, bfgs_iterations = 20), verbose = false)

e = data.experiments[1]
ident = BioDynaX.report_production_destruction_tradeoff(
    model, trained.params, e.observations, e.times, e.u0, tspan; verbose = true)
println("scale warning raised: ", ident.unidentifiable_edge)

X = hcat((predict_ude(trained.params, ex.u0, tspan, ex.times, model) for ex in data.experiments)...)
R, D, term = sample_unknown_destruction(model, trained.params, X)
found = discover_unknown_rate(R, 1:size(R, 2), D; verbose = false)
println(found.equations)
```

A run of this block in September 2026 printed a raised scale warning (cosine
0.996) and the rational rate

```text
dx[1]/dt = (0.27827*1 + -0.71365*x[1] + 2.3223*x[1]^2) / (1 + -1.5762*x[1] + 1.9558*x[1]^2)
```

where `x[1]` is the regulator `R`. The `x^2 / (K + x^2)` structure of the Hill
term is present; the constant and linear terms are nuisance terms that the
sparse regression did not remove. The [Concepts](concepts.md) page explains
why the coefficients are not biological constants, and the
[Tutorial](tutorial.md) runs the reference protocol with nine initial
conditions.

## Next steps

- `julia --project=. examples/unknown_inhibition.jl` runs the reference
  protocol and prints the full four-section report (about 10 to 15 minutes).
- `BIODYNAX_SMOKE=1 ADAM_ITERS=2 BFGS_ITERS=0 julia --project=. examples/unknown_inhibition.jl`
  runs the same script with one initial condition and two optimizer steps as
  a two-minute check that everything is installed.
