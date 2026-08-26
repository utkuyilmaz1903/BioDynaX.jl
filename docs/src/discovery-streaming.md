# Discovery streaming and library workspaces

Research preview. Not v1.0. Not in General.
Helpers on this page are **not exported**. Call them as `BioDynaX.foo`.
They do not grow the public freeze list.

Blocked STLSQ reuses one grow-only Gram workspace across bootstrap draws.
evaluate_library! writes monomials in place; it does not allocate a per-term vector.
Implicit design can be filled in row chunks without materialising the full n × p matrix.
Reusable library chunks overwrite one buffer; they do not allocate a new block per iterate.
ImplicitSINDyPI.chunk_size is the blocked-STLSQ row width; ExplicitSTLSQ uses the same helper default.

This page is the library-evaluation contract. It does not change
`RECOVERY_THRESHOLDS`, the unique-claim protocol (seed 103 / 9 ICs), or
`validate_network`. Combined F1 stays a skeleton floor (0.50). Canonical
Hill from a trained neural rate stays closed.

## What was wasted

`discover_unknown_rate` and `discover_equations` call implicit SINDy-PI
with a bootstrap. Each draw used to:

1. Allocate a full `n × (n_num + n_den)` implicit design.
2. Allocate a new Gram matrix and right-hand side on every STLSQ
   iteration.
3. Call `evaluate_term` from `evaluate_library!`, which built a fresh
   length-`n` vector per monomial.

Those allocations do not change the recovered support. They do change how
many times the same library is rebuilt when `bootstrap_samples > 0`.
`ImplicitSINDyPI.chunk_size` (default 256; unique-claim discovery uses 32)
is the row width of that blocked path.

## Workspace types

`STLSQWorkspace` holds scales, the active set, a `p × p` Gram, and one
row-chunk. `_stlsq_blocked!` writes the last coefficient vector into
`workspace.scaled` (invalidated by the next call). `resize_count`
increments only when a buffer grows.

`StreamingImplicitWorkspace` keeps chunk-sized numerator / denominator /
design blocks plus full-length prediction buffers for
`evaluate_candidate!`. `_fit_implicit` now calls `_fit_implicit_stream`,
which never stores the full design. The STLSQ right-hand side stays the
observed derivative on both stages; a rebuilt prediction is used only to
rebuild the implicit columns `−D(z) ẏ`.

`LibraryChunkWorkspace` is the buffer behind
`each_reusable_library_chunk`. The older `each_library_chunk` iterator
still allocates a block per iterate and remains the compatibility path.

```@example discovery-workspace-types
using BioDynaX
ws = BioDynaX.allocate_stlsq_workspace(Float64, 80, 6, 16)
(ws.n, ws.p, ws.chunk_size, ws.resize_count,
 size(ws.gram), size(ws.design_chunk))
```

## Blocked STLSQ agreement

`_stlsq` (dense QR on an augmented ridge system) is the numerical oracle.
`_stlsq_blocked` and `_stlsq_blocked!` accumulate `A'A` in row chunks.
Those two blocked paths must match. Dense QR may differ at the scale of
normal equations versus augmented QR; that difference is not a discovery
retcode.

```@example discovery-stlsq-agree
using BioDynaX, Random
rng = MersenneTwister(11)
A = randn(rng, 90, 5)
ξ = [1.0, 0.0, 0.4, 0.0, 0.2]
y = A * ξ
report = BioDynaX.stlsq_path_agreement(A, y, 1e-2; chunk_size = 16)
(report.holds, report.blocked_matches_workspace, report.workspace_stable)
```

A second `_stlsq_blocked!` on the same workspace must not reallocate the
Gram or the design chunk.

```@repl discovery-stlsq-reuse
using BioDynaX, Random
A = randn(MersenneTwister(2), 60, 4)
y = A * [0.8, 0.0, 0.3, 0.0]
ws = BioDynaX.allocate_stlsq_workspace(Float64, 60, 4, 12)
ptr = pointer(ws.gram)
BioDynaX._stlsq_blocked!(ws, A, y, 1e-2)
pointer(ws.gram) == ptr && ws.resize_count == 1
```

## In-place library evaluation

`evaluate_library!(output, terms, X)` writes each monomial through
`evaluate_term!` into a view of `output`. It must not call
`evaluate_term` (the allocating wrapper) on the hot path.

```@example discovery-library-inplace
using BioDynaX
net = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
spec = local_basis(net, 1; degree = 2, include_interactions = false)
X = reshape(collect(range(0.2, 1.5; length = 40)), 1, :)
out = Matrix{Float64}(undef, 40, length(spec.numerator))
BioDynaX.evaluate_library!(out, spec.numerator, X)
full = BioDynaX.evaluate_library(spec.numerator, X)
(out ≈ full, BioDynaX.basis_factory_evaluates_in_place())
```

Reusable chunks overwrite one `LibraryChunkWorkspace` buffer. Materialising
those chunks must match `evaluate_library`.

```@example discovery-library-chunks
using BioDynaX
net = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
spec = local_basis(net, 1; degree = 2, include_interactions = false)
X = reshape(collect(range(0.1, 1.2; length = 50)), 1, :)
report = BioDynaX.library_chunk_agreement(spec.numerator, X; chunk_size = 13)
(report.holds, report.matches_full)
```

## Implicit stream versus a materialised design

`_fit_implicit_materialised` rebuilds the historical `n × p` design and
is the agreement oracle for `_fit_implicit_stream`. A Michaelis–Menten
rate on one regulator must recover on both paths.

```@example discovery-implicit-stream
using BioDynaX
net = BiologicalNetwork([NodeSpec(name = :substrate)], EdgeSpec[])
x = collect(range(0.1, 2.0; length = 80))
X = reshape(x, 1, :)
d = 1.7 .* x ./ (0.45 .+ x)
spec = local_basis(net, 1; degree = 1, include_interactions = false,
    X, derivative = d, max_variables = 1)
report = BioDynaX.implicit_stream_agreement(
    spec, X, d, collect(eachindex(x)), 1e-7; chunk_size = 16)
(report.holds, report.num_stream, report.den_stream)
```

`evaluate_candidate!` must match `_evaluate_candidate` on the same
coefficients.

```@repl discovery-eval-candidate
using BioDynaX
net = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
X = reshape(collect(range(0.2, 1.4; length = 30)), 1, :)
spec = local_basis(net, 1; degree = 1, include_interactions = false)
num = zeros(length(spec.numerator)); num[2] = 1.5
den = zeros(length(spec.denominator)); den[1] = 0.6
BioDynaX.evaluate_candidate_agreement(spec, num, den, X).holds
```

## Bootstrap reuse

`_bootstrap_frequency` allocates one `StreamingImplicitWorkspace` and
passes it to every draw. `bootstrap_workspace_reuse_report` records
`resize_count` after the first allocation; further draws must not grow
the buffers.

```@example discovery-bootstrap-reuse
using BioDynaX
net = BiologicalNetwork([NodeSpec(name = :r)], EdgeSpec[])
r = collect(range(0.2, 1.6; length = 60))
X = reshape(r, 1, :)
D = 1.4 .* r .^ 2 ./ (0.5^2 .+ r .^ 2)
spec = local_basis(net, 1; degree = 2, include_interactions = false,
    X, derivative = D)
report = BioDynaX.bootstrap_workspace_reuse_report(
    spec, X, D, collect(1:48), 1e-3;
    bootstrap_samples = 4, chunk_size = 16, seed = 7)
(report.reused, report.workspace_resizes, report.stlsq_resizes)
```

## Chunk-size helper

`_backend_chunk_size` reads `ImplicitSINDyPI.chunk_size`. Explicit STLSQ
and DataDrivenSparse do not store a width; they use 256. Explicit
discovery no longer hard-codes that number at the call site.

```@repl discovery-chunk-helper
using BioDynaX
BioDynaX._backend_chunk_size(ImplicitSINDyPI(chunk_size = 32))
```

```@repl discovery-chunk-explicit
using BioDynaX
BioDynaX._backend_chunk_size(ExplicitSTLSQ())
```

## Contract locks

`discovery_workspace_contract_holds()` is the source + docs + export +
threshold lock for this layer. It does not train a UDE.

```@repl discovery-contract
using BioDynaX
BioDynaX.discovery_workspace_source_holds()
```

`InsufficientSamples` is still the retcode when a raw trajectory has
fewer than 20 finite columns. Streaming does not lower that floor.

```@repl discovery-insufficient
using BioDynaX
net = BiologicalNetwork([NodeSpec(name = :x)], EdgeSpec[])
X = reshape(collect(0.1:0.1:0.8), 1, :)
t = collect(range(0.0, 1.0; length = 8))
discover_equations(X, t, net; verbose = false).retcode
```

## What this page does not claim

- Coefficients are not biological constants when the edge is
  unidentifiable.
- Combined support F1 is not raised to 0.99.
- Hill-from-NN is not opened.
- The protocol is not made faster by dropping ICs, points, or seeds.
- `validate_network` does not gain a single-hole gate.
- The public export list is unchanged.
