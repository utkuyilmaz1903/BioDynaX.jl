function plot_training(args...; kwargs...)
    extension = Base.get_extension(@__MODULE__, :BioDynaXPlotsExt)
    extension === nothing &&
        throw(ArgumentError("plot_training requires loading Plots.jl"))
    return extension.plot_training(args...; kwargs...)
end
