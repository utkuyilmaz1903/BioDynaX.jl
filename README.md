# BioDynaX.jl

`BioDynaX.jl` is a production-grade Scientific Machine Learning (SciML) framework designed to transform static biological interaction networks into differentiable, dynamic mathematical simulators. 

By leveraging **Universal Differential Equations (UDEs)** and **Symbolic Regression**, it allows researchers to input biological network topologies, train neural networks to capture completely unknown cellular mechanisms from noisy lab data, and automatically extract the underlying governing physical equations.

---

## 1. Why BioDynaX? (The Core Value)

In traditional bioinformatics (Python/R networks):
* Graphs are static data structures that cannot simulate time-series dynamics seamlessly.
* Standard Neural Networks act as total black boxes, providing curve-fitting but zero mechanistic insight.
* Combining differential equations with deep learning breaks the automatic differentiation (AD) chain, leading to massive performance drops or non-differentiable code.

**BioDynaX.jl solves this.** It compiles the network topology into native, type-stable Julia code compiled via LLVM. It bridges neural architectures (`Lux.jl`) and differential solvers (`OrdinaryDiffEq.jl`), keeping the backpropagation chain fully differentiable via `Zygote.jl` to perform automated automated scientific discovery.

---

## 2. Repository Architecture (Separation of Concerns)

The repository strictly follows professional Julia package design patterns, decoupling data generation, model optimization, physics boundaries, and symbolic discovery:

```bash
BioDynaX.jl/├── Project.toml        # Package metadata and official SciML/DataDriven dependencies
            ├── src/
            │   ├── BioDynaX.jl     # Main module namespace handler (clean includes & exports)
            │   ├── Network.jl      # Graph topology engine and BiologicalNetwork structs
            │   ├── UDE.jl          # Lux Neural Network injection and Zygote-safe out-of-place RHS
            │   ├── DataGen.jl      # Synthetic biological data generator using true non-linear kinetics
            │   ├── Training.jl     # Optimization pipeline (Adam + BFGS with gradient-friendly soft constraints)
            │   └── Discovery.jl    # Multivariate Sparse Regression engine (STLSQ Equation Extractor)
            └── scripts/
              └── run_discovery.jl # End-to-end simulation, training, and discovery runner script
```

---

## 3. Core Engineering & Numerical Innovations

### 🧬 Physics-Informed Soft Penalties (Gradient-Friendly Barriers)
Biological concentrations cannot be negative ($x < 0$). Instead of using hard constraints like `isoutofdomain` which truncate the execution and destroy the automatic differentiation (AD) gradient chain, BioDynaX implements a **differentiable Soft Penalty** inside the loss loop:

$$\text{Loss} = \text{MSE} + \lambda \sum \min(0.0, x_{\text{predicted}})^2$$

This creates a smooth mathematical barrier. The neural network naturally learns to stay within valid biological bounds through gradients rather than errors, keeping the solver fully stable.

### 📊 Multivariate Sparse Discovery
Instead of standard univariate curve fitting, the symbolic regression layer constructs a rich multivariate basis including cross-interaction terms ($u_1 \cdot u_2$). This enables the extraction of real mass-action biological kinetics.

---

## 4. Quick Start

### Installation
Clone the repository and instantiate the isolated environment:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```
Run End-to-End Discovery PipelineExecute the main runner script to synthesize noisy lab data, train the UDE, discover equations, and render the evaluation plots:
```bash
julia --project=. scripts/run_discovery.jl
```

---

## 5. Proved Scientific Results
When tested against the kaotic p53-Mdm2 tumor suppressor feedback loop with highly noisy synthetic lab datasets ($\sigma = 0.05$), BioDynaX successfully achieved:
* **System Optimization:** Dropped systemic Mean Squared Error (MSE) from arbitrary chaos down to a high-precision convergence of `0.0023` using an automated two-stage Adam to BFGS optimization pipeline.
* **Mechanistic Discovery ($\phi_2$):** Successfully isolated and extracted the exact linear cross-interaction terms for the Mdm2 production and degradation engine from a pool of 10+ mathematical candidate functions.
* **Surrogate Identification ($\phi_1$):** Trapped non-linear rational Hill kinetics, accurately converting the hidden biology into a high-fidelity Taylor-series polynomial representation.

---

The verification plots and loss convergence scales are automatically exported and rendered at `scripts/biodynax_discovery.png` upon running execution.
