# BioDynaX.jl

**Graph-guided Universal Differential Equations for biological networks.**

BioDynaX.jl is a [SciML](https://sciml.ai/) Julia package that turns a biological interaction network into a trainable dynamical model. Known mechanisms are compiled from network metadata; unknown interactions are represented by a small Lux neural network. After fitting to time-series data, an implicit sparse-regression backend can recover rational kinetic forms on the graph.

Requires **Julia ≥ 1.10**.

---

## What it does

1. **Define a network** — nodes, edges, and stoichiometric reactions with kinetic metadata (mass action, Hill, competitive inhibition, or neural unknowns).
2. **Compile a UDE** — a production–destruction RHS `duᵢ = Pᵢ(u,p) − Dᵢ(u,p)·uᵢ` with non-negative production and destruction rates.
3. **Train** — fit physical and neural parameters with Adam (optional BFGS refinement), SciMLSensitivity adjoints, and differentiable soft constraints.
4. **Discover** — recover graph-local rational equations from the trained dynamics via implicit SINDy-PI (STLSQ on a monomial basis).

The default example is the **p53–Mdm2** feedback loop (`build_network()`). A second fully known test topology is available via `build_linear_test_network()`.

---

## Design highlights

| Topic | Implementation |
|--------|----------------|
| **Dynamics** | Compiled `MechanismCompiler` IR → `ude_system` (Zygote-safe out-of-place) or `ude_rhs!` (preallocated in-place) |
| **Unknown biology** | `NeuralDestructionTerm` with softplus-headed Lux MLP |
| **Positivity** | States read through `max(0, x)`; optional **augmented Lagrangian** soft penalties on trajectories |
| **Parameters** | `ComponentVector` with typed `phys` / `nn` axes; compile-time `ParameterSchema` |
| **Training** | `train_ude`, multi-experiment `train_experiments`, versioned checkpoints + `resume_training` |
| **Discovery** | **`ImplicitSINDyPI`** (default rational backend) and **`ExplicitSTLSQ`** (polynomial explicit backend); bootstrap support and validation hold-out on implicit path |
| **Execution** | Serial, threaded, or distributed experiment runners; optional **CUDA** extension for device arrays |

## SciML integration

```julia
using BioDynaX, SciMLBase, OrdinaryDiffEq

model, p = build_ude_model(MersenneTwister(0))
prob = ODEProblem(model, [0.2, 0.1], (0.0, 10.0), p)
sol = solve(prob, Tsit5(); saveat = 0:0.5:10.0)
```

Training uses **`ZygoteAD`** by default (`SolverConfig(ad_policy = ZygoteAD())`). A **`ProductionAD`** policy selects the in-place RHS path for forward integration; adjoint settings remain Zygote-based today.

---

## Repository layout

```
BioDynaX.jl/
├── src/
│   ├── Network.jl           # BiologicalNetwork, edges, reactions
│   ├── MechanismCompiler.jl # compile_mechanism, UDEModel, ude_system / ude_rhs!
│   ├── SciMLInterface.jl    # ODEProblem, build_ude_function, auto_sensealg
│   ├── Metadata.jl          # typed kinetic metadata structs
│   ├── UDE.jl               # Lux NN builder, pack_parameters
│   ├── ModelCache.jl        # preallocated RHS workspace
│   ├── ParameterSchema.jl   # compile-time parameter names & defaults
│   ├── DataGen.jl           # synthetic data (Hill ground truth or compiled model)
│   ├── Training.jl          # predict_ude, train_ude, checkpoints
│   ├── BasisFactory.jl      # graph-local monomial libraries
│   ├── Discovery.jl         # implicit SINDy-PI discovery
│   ├── Execution.jl         # serial / threads / distributed backends
│   ├── Experiments.jl       # Experiment, ExperimentSet
│   └── Config.jl            # training, solver, discovery, execution configs
├── ext/
│   ├── BioDynaXCUDAExt.jl   # optional GPU device transfer (requires CUDA.jl)
│   └── BioDynaXPlotsExt.jl  # optional plotting helpers (requires Plots.jl)
├── scripts/run_discovery.jl # end-to-end p53 example (needs Plots.jl)
├── test/                    # regression + quality gates
└── benchmark/allocation_gate.jl
```

---

## Installation

Clone the repository and instantiate the project environment:

```bash
git clone https://github.com/<your-org>/BioDynaX.jl.git
cd BioDynaX.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Optional extensions load when their weak dependencies are available:

```julia
using BioDynaX
using CUDA      # BioDynaXCUDAExt
using Plots     # BioDynaXPlotsExt
```

---

## Quick start (REPL)

```julia
using BioDynaX, Random

rng = MersenneTwister(42)
model, params = build_ude_model(rng)          # default p53–Mdm2 network

t_data, _, noisy, _ = generate_data(rng; noise_σ = 0.05)

result = train_ude(
    params, noisy, t_data, [0.2, 0.1], (0.0, 20.0), model;
    adam_iters = 100, bfgs_iters = 0, verbose = true)

discovery = discover_equations(result.params, model; verbose = true)
println(discovery.equations)
```

### End-to-end script

The bundled script trains on synthetic p53 data, runs discovery, and saves a verification plot (requires **Plots.jl**):

```bash
julia --project=. scripts/run_discovery.jl
```

---

## Custom networks

Build a network from `NodeSpec`, `EdgeSpec`, and `ReactionSpec`, then compile and train:

```julia
network = build_linear_test_network()   # or your own BiologicalNetwork(...)
model, params = build_ude_model(rng, network)
# ... generate_data(rng; network = network) or your own observations
```

Supported known kinetic families in the compiler include **mass action**, **Hill**, **competitive inhibition**, and **input-driven production**. Unknown edges map to **neural destruction** terms.

---

## Development

Run the test suite:

```bash
julia --project=. test/runtests.jl
```

CI (GitHub Actions) runs tests on **Windows and Linux** (Julia 1.10 and latest), **Aqua/JET** quality checks, documentation build, and an **allocation regression gate** on the linear test network RHS.

Build docs locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path=".")'
julia --project=docs docs/make.jl
```

---

## Limitations (current)

- Primary discovery backend is **ImplicitSINDyPI** only; full multi-trajectory discovery orchestration is partially exposed (`_collect_multi_trajectory_data` exists, public API is still largely single-trajectory oriented).
- **GPU** support transfers experiments to device arrays; it is not a fully batched training stack.
- **Zero-allocation** RHS is measured and gated for NN-free linear networks; Lux forward passes on the default p53 model still allocate on the hot path.
- Synthetic data for the default network uses a **Hill-kinetics ground truth** that differs from the UDE’s neural degradation term — by design for discovery benchmarks.

---

## License

See [LICENSE](LICENSE).
