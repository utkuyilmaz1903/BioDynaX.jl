module BioDynaXPlotsExt

using BioDynaX
using Plots

function plot_training(training::BioDynaX.TrainingResult, times, observations,
        dense_times, truth, prediction;
        state_names = ["x1", "x2"])
    state_count = size(observations, 1)
    figure = plot(
        layout = (state_count + 1, 1),
        size = (900, 280 * (state_count + 1)),
        legend = :topright)
    for state in 1:state_count
        plot!(figure[state], dense_times, truth[state, :];
            label = "ground truth", linewidth = 2)
        scatter!(figure[state], times, observations[state, :];
            label = "observations", markersize = 3, alpha = 0.7)
        plot!(figure[state], dense_times, prediction[state, :];
            label = "UDE", linewidth = 2, linestyle = :dash)
        ylabel!(figure[state], state_names[state])
    end
    plot!(figure[end], eachindex(training.history), training.history;
        yscale = :log10, label = "objective", linewidth = 2)
    xlabel!(figure[end], "iteration")
    ylabel!(figure[end], "objective")
    return figure
end

end
