module BioDynaXCUDAExt

using BioDynaX
using CUDA

functional() = CUDA.functional()
to_device(value::AbstractArray) = cu(value)
to_device(value::Number) = value

function to_device(experiment::BioDynaX.Experiment)
    return BioDynaX.DeviceExperiment(
        experiment.name,
        cu(experiment.times),
        cu(experiment.observations),
        cu(experiment.mask),
        cu(experiment.u0),
        copy(experiment.metadata))
end

function gpu_execute(f, experiments::BioDynaX.ExperimentSet, config)
    CUDA.functional() ||
        throw(ErrorException("CUDA is loaded but no functional GPU is available"))
    # Each experiment remains a dense GPU array copy. This is not a batched
    # GPU ODE or training stack; kernels inside `f` are the caller's job.
    return map(experiments.experiments) do experiment
        f(to_device(experiment))
    end
end

end
