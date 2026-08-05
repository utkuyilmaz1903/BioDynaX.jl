"""
    execute_experiments(f, experiments; config)

Backend-neutral experiment executor. Results preserve input order for all
backends. GPU execution is provided by an optional package extension.
"""
abstract type ExecutionBackend end
struct SerialBackend <: ExecutionBackend end
struct ThreadsBackend <: ExecutionBackend end
struct DistributedBackend <: ExecutionBackend end
struct GPUBackend <: ExecutionBackend end

_resolve_backend(::Val{:serial}) = SerialBackend()
_resolve_backend(::Val{:threads}) = ThreadsBackend()
_resolve_backend(::Val{:distributed}) = DistributedBackend()
_resolve_backend(::Val{:gpu}) = GPUBackend()

function _resolve_backend(symbol::Symbol)
    symbol == :serial && return SerialBackend()
    symbol == :threads && return ThreadsBackend()
    symbol == :distributed && return DistributedBackend()
    symbol == :gpu && return GPUBackend()
    throw(ArgumentError("unknown execution backend $symbol"))
end

to_device(value, ::Val{:cpu}) = value

function _cuda_extension()
    extension = Base.get_extension(@__MODULE__, :BioDynaXCUDAExt)
    extension === nothing &&
        throw(ArgumentError("GPU support requires loading CUDA.jl"))
    return extension
end

gpu_available() = try
    _cuda_extension().functional()
catch
    false
end

to_device(value, ::Val{:gpu}) = _cuda_extension().to_device(value)
gpu_execute(f, experiments, config) =
    _cuda_extension().gpu_execute(f, experiments, config)

function execute_experiments(f, experiments::ExperimentSet;
                             config::ExecutionConfig = ExecutionConfig())
    backend = _resolve_backend(config.backend)
    return execute_experiments(f, experiments, backend; config = config)
end

function execute_experiments(f, experiments::ExperimentSet,
                             backend::SerialBackend; config = ExecutionConfig())
    return map(f, experiments.experiments)
end

function execute_experiments(f, experiments::ExperimentSet,
                             backend::ThreadsBackend; config = ExecutionConfig())
    outputs = Vector{Any}(undef, length(experiments))
    Threads.@threads for index in eachindex(experiments.experiments)
        outputs[index] = f(experiments[index])
    end
    return outputs
end

function execute_experiments(f, experiments::ExperimentSet,
                             backend::DistributedBackend;
                             config::ExecutionConfig = ExecutionConfig())
    nworkers() > 1 ||
        throw(ArgumentError("distributed backend requires worker processes"))
    return pmap(f, experiments.experiments; batch_size = config.batch_size)
end

function execute_experiments(f, experiments::ExperimentSet,
                             backend::GPUBackend;
                             config::ExecutionConfig = ExecutionConfig())
    return gpu_execute(f, experiments, config)
end
