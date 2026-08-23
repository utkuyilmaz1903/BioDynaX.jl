---
name: Bug report
about: A failure that should be able to go red in CI
---

## What broke

## Contract

- [ ] I am not asking to loosen `RECOVERY_THRESHOLDS`
- [ ] I am not asking to grow the public `export` list
- [ ] This is not an experimental GPU / SBML / MTK product request

## How to reproduce

```bash
julia --project=. test/runtests.jl
```
