###############################################################################
# Deprecated names, removed in 0.13. Each forwards to its replacement and
# warns once per session.
###############################################################################

Base.@deprecate_binding UNIQUE_CLAIM_PROTOCOL REFERENCE_PROTOCOL false

Base.@deprecate unique_claim_experiment_set(args...; kwargs...) reference_protocol_experiment_set(
    args...; kwargs...) false

Base.@deprecate unique_claim_discovery_config(; kwargs...) reference_protocol_discovery_config(;
    kwargs...) false
