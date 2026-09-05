"""
    RunMetadata

Reproducibility envelope attached to training and discovery artifacts.
"""
struct RunMetadata
    seed::UInt64
    created_at::DateTime
    julia_version::VersionNumber
    package_version::VersionNumber
    data_hash::String
    config::Any
end

function RunMetadata(; seed::Integer = 0,
        package_version::VersionNumber = PACKAGE_VERSION,
        data_hash::AbstractString = "",
        config = (;))
    return RunMetadata(UInt64(seed), now(UTC), VERSION, package_version,
        String(data_hash), config)
end

"""
    TrainingResult

Stable, typed output for optimization runs.
"""
struct TrainingResult{P, T, H, M, D, R}
    params::P
    history::H
    initial_loss::T
    final_loss::T
    metadata::M
    diagnostics::D
    converged::Bool
    retcode::R
end

"""
    DiscoveryRetcode

Outcome of `discover_equations`. Failures are never silent: `DiscoveryResult.success`
is false and `retcode` names the failure class.
"""
@enum DiscoveryRetcode begin
    DiscoverySuccess
    InsufficientSamples
    DenominatorUnsafe
    EmptySupport
    SingularLibrary
    DiscoveryFailed
end

@doc "Support recovered; a hybrid RHS is allowed." DiscoverySuccess
@doc "Not enough finite trajectory samples for STLSQ." InsufficientSamples
@doc "Discovered denominator changed sign or hit the floor." DenominatorUnsafe
@doc "STLSQ wiped every library term." EmptySupport
@doc "Design matrix was singular." SingularLibrary
@doc "Discovery failed for any other reason." DiscoveryFailed

"""
    DiscoveryResult

Stable output type shared by explicit and implicit discovery backends.
`retcode` is a `DiscoveryRetcode`; `success` is `retcode === DiscoverySuccess`.
"""
struct DiscoveryResult{E, B, S, C, M, R}
    success::Bool
    message::String
    equations::E
    basis::B
    solution::S
    candidates::C
    metadata::M
    retcode::R
end

function DiscoveryResult(success::Bool, message, equations, basis, solution,
        candidates, metadata)
    retcode = success ? DiscoverySuccess : DiscoveryFailed
    return DiscoveryResult{typeof(equations), typeof(basis), typeof(solution),
        typeof(candidates), typeof(metadata),
        DiscoveryRetcode}(
        success, String(message), equations, basis, solution, candidates,
        metadata, retcode)
end

function Base.getproperty(r::DiscoveryResult, name::Symbol)
    name === :equation ? getfield(r, :equations) : getfield(r, name)
end

"""
    Checkpoint

Versioned, backend-neutral training checkpoint.
"""
struct Checkpoint{P, O, M}
    schema_version::VersionNumber
    params::P
    optimizer_state::O
    iteration::Int
    metadata::M
end

const CHECKPOINT_SCHEMA_VERSION = v"1.0.0"

struct ArtifactEnvelope{T}
    schema_version::VersionNumber
    kind::Symbol
    payload::T
end

const ARTIFACT_SCHEMA_VERSION = v"1.0.0"

function save_result(path::AbstractString, result)
    directory = dirname(abspath(path))
    isdir(directory) ||
        throw(ArgumentError("artifact directory does not exist: $directory"))
    kind = result isa TrainingResult ? :training :
           result isa DiscoveryResult ? :discovery : :generic
    envelope = ArtifactEnvelope(ARTIFACT_SCHEMA_VERSION, kind, result)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        serialize(io, envelope)
    end
    mv(temporary, path; force = true)
    return path
end

function load_result(path::AbstractString)
    envelope = open(deserialize, path)
    envelope isa ArtifactEnvelope ||
        throw(ArgumentError("file is not a BioDynaX result artifact"))
    envelope.schema_version.major == ARTIFACT_SCHEMA_VERSION.major ||
        throw(ArgumentError("incompatible result artifact schema"))
    return envelope.payload
end

function data_fingerprint(arrays...)
    ctx = SHA.SHA256_CTX()
    for value in arrays
        buffer = IOBuffer()
        serialize(buffer, value)
        SHA.update!(ctx, take!(buffer))
    end
    return bytes2hex(SHA.digest!(ctx))
end
