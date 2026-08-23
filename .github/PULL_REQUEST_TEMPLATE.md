## Summary

## Contract

- [ ] `RECOVERY_THRESHOLDS` is unchanged (loosening is breaking)
- [ ] Public `export` list is unchanged
- [ ] Experimental GPU / SBML / MTK / Fisher stays unexported
- [ ] Every new public sentence matches a CI gate or is labeled experimental

## Test plan

- [ ] `julia --project=. test/runtests.jl`
- [ ] Recovery job still required for unknown-edge changes
