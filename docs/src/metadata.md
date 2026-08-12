# Typed kinetic metadata

Edges and reactions accept typed metadata structs instead of untyped dictionaries.
Dict metadata remains supported for backward compatibility during the 0.x migration.

## Example

```julia
ReactionSpec(
    name = :Mdm2_linear_decay,
    stoichiometry = Dict(3 => -1.0),
    regulators = Int[],
    metadata = LinearDecayMetadata(rate_param = :γ_mdm2),
)

ReactionSpec(
    name = :input_drives_p53,
    stoichiometry = Dict(2 => 1.0),
    metadata = InputDriveMetadata(
        rate_param = :α_p53,
        input_param = :signal,
        input_node = 1,
    ),
)
```

## Available metadata types

```@docs
KineticMetadata
EmptyMetadata
InputDriveMetadata
MassActionMetadata
HillMetadata
CompetitiveMetadata
LinearDecayMetadata
SaturationMetadata
CustomKineticMetadata
metadata_summary
```
