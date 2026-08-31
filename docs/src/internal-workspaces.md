# Internal workspaces

These files are **not exported**. They are lock surfaces and runtime
wiring, not a second public API. Do not thin them by deleting
`MUST_CONTAIN` tests. `RECOVERY_THRESHOLDS` and `names(BioDynaX)` stay
locked.

## Runtime wiring (called from Recovery / Training)

| name | file | role |
|------|------|------|
| `train_experiments_with_warmup` | `src/TrainingReuse.jl` | First-IC Adam state is reused on unique-claim training (ICs 1–7 after a 9-IC generate) |
| `TrainingSolveSession` | `src/TrainingReuse.jl` | One `ODEProblem` remade across ICs |
| `denominator_split_counts` | `src/DenominatorDomain.jl` | Train / validation / orthant violation split |
| `compose_hybrid_rhs` | public + `src/HybridCompose.jl` | Neural destruction term recovers `ude_system` |
| `hybrid_data_residual` | public + `src/HybridResidual.jl` | Residual versus `SciMLBase.solve` |
| `admit_recovery_suite_network` | `src/RecoveryAdmission.jl` | Unique-claim sections fail closed on 0/2 holes |
| `recovery_suite_plan` | `src/RecoverySuiteSkip.jl` | Unused sections do not call `_train_unknown_edge` |

## Unexported lock surfaces (tests + docs, not the freeze list)

| file | lock |
|------|------|
| `src/DiscoveryWorkspace.jl` | Grow-only STLSQ / implicit buffers (#16) |
| `src/SciMLSolveSurface.jl` | `ude_system` / `ODEProblem` / remake / solve agreement |
| `src/ExperimentCheckpoint.jl` | Fingerprint + resume on one compiled tree |
| `src/FailureModes.jl` | `DiscoveryRetcode` catalog; `validate_network` stays open |
| `src/IdentifiabilityProduct.jl` | Tradeoff joins `UniqueClaimProtocolRow` |
| `src/GraphLocalLibrary.jl` | `scope=:graph` vs `:global`; parent-membership gates |
| `src/ParameterSchemaPack.jl` | `:k_custom` in `parameter_schema` |
| `src/DocsExecutable.jl` | H–L join + leftover scanners |
| `src/AllocationGates.jl` | Measured `@allocated` ceilings |
| `src/ClaimMetricHonesty.jl` | F1 extras, Fisher flag, MM-not-Hill |
| `src/ClaimScopeHonesty.jl` | Synthetic CSV, partial obs, out of scope |

The golden path is `examples/unknown_inhibition.jl`.
`scripts/run_discovery.jl` is a debug runner, not the product.
