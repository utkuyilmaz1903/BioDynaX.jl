#!/usr/bin/env julia
###############################################################################
# BioDynaX — end-to-end discovery pipeline
#
#   1) Activate the project environment.
#   2) Build the biological regulatory network.
#   3) Generate synthetic noisy data from the Hill-kinetics ground truth.
#   4) Build the UDE (Lux NN inside the ODE).
#   5) Train the UDE  (Adam → BFGS) against the noisy data.
#   6) Run sparse symbolic regression to recover a closed-form expression
#      for the unknown Mdm2 → p53 degradation kinetics.
#   7) Save a 3-panel verification plot next to this script.
###############################################################################

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BioDynaX
using Random
using Plots
using OrdinaryDiffEq

const SCRIPT_DIR = @__DIR__

# ---------------------------------------------------------------------------
# Tiny visual helper for section headers — no global state, no side effects.
# ---------------------------------------------------------------------------
function section(title::AbstractString)
    bar = "─"^78
    println("\n", bar)
    println("  ", title)
    println(bar)
end

function main(; seed::Int       = 42,
                adam_iters::Int = 300,
                adam_lr::Float64 = 0.01,
                bfgs_iters::Int = 100,
                noise_σ::Float64 = 0.05,
                poly_degree::Int = 3,
                sparsity_threshold::Real = 1e-2)

    rng = Random.default_rng()
    Random.seed!(rng, seed)

    # 1) Network ------------------------------------------------------------
    section("1) Build biological network")
    net = build_network()
    describe_network(net)

    # 2) Synthetic data -----------------------------------------------------
    section("2) Generate synthetic noisy data")
    u0    = [0.2, 0.1]
    tspan = (0.0, 20.0)
    t_data, clean_data, noisy_data, truth_params =
        generate_data(rng; u0 = u0, tspan = tspan,
                      n_points = 40, noise_σ = noise_σ)
    println("  → ", length(t_data), " time-points,  noise σ = ", noise_σ)

    # 3) UDE construction ---------------------------------------------------
    section("3) Build UDE (Lux neural network)")
    nn, nn_ps, nn_st = build_ude_nn(rng)
    phys_init = (α_p53 = 0.9, β_mdm2 = 1.1, γ_mdm2 = 1.5, signal = 1.0)
    p_init    = pack_parameters(phys_init, nn_ps)
    println("  → trainable parameters : ", length(p_init),
            "  (phys = ", length(p_init.phys),
            ", nn = ", length(p_init.nn), ")")

    # 4) Train --------------------------------------------------------------
    section("4) Train UDE (Adam → BFGS)")
    tr = train_ude(p_init, noisy_data, t_data, u0, tspan, nn, nn_st;
                   adam_iters = adam_iters, adam_lr = adam_lr,
                   bfgs_iters = bfgs_iters, verbose = true)

    # 5) Symbolic regression -----------------------------------------------
    section("5) Symbolic regression (Discovery)")
    discovery = discover_equations(tr.params, nn, nn_st;
                                   network = net,
                                   polynomial_degree  = poly_degree,
                                   sparsity_threshold = sparsity_threshold,
                                   verbose = true)
    if !discovery.success
        @warn "Discovery did not return a closed-form expression: $(discovery.message)"
    end

    # 6) Plot ---------------------------------------------------------------
    section("6) Render verification plot")
    t_dense    = collect(range(tspan[1], tspan[2]; length = 300))
    pred       = predict_ude(tr.params, u0, tspan, t_dense, nn, nn_st)

    truth_prob  = ODEProblem(ground_truth!, u0, tspan, truth_params)
    truth_dense = Array(solve(truth_prob, Tsit5();
                              saveat = t_dense, abstol = 1e-9, reltol = 1e-9))

    plt = plot(layout = (3, 1), size = (900, 850), legend = :topright,
               left_margin = 10 * Plots.mm, bottom_margin = 6 * Plots.mm)

    plot!(plt[1], t_dense, truth_dense[1, :];
          label = "Ground truth p53", lw = 2, color = :steelblue)
    scatter!(plt[1], t_data, noisy_data[1, :];
             label = "Noisy lab data", ms = 4, color = :firebrick, alpha = 0.7)
    plot!(plt[1], t_dense, pred[1, :];
          label = "UDE prediction", lw = 2, ls = :dash, color = :seagreen)
    title!(plt[1], "p53 dynamics");  ylabel!(plt[1], "[p53]")

    plot!(plt[2], t_dense, truth_dense[2, :];
          label = "Ground truth Mdm2", lw = 2, color = :steelblue)
    scatter!(plt[2], t_data, noisy_data[2, :];
             label = "Noisy lab data", ms = 4, color = :firebrick, alpha = 0.7)
    plot!(plt[2], t_dense, pred[2, :];
          label = "UDE prediction", lw = 2, ls = :dash, color = :seagreen)
    title!(plt[2], "Mdm2 dynamics"); ylabel!(plt[2], "[Mdm2]")

    plot!(plt[3], 1:length(tr.history), tr.history;
          yscale = :log10, lw = 2, color = :purple, label = "loss")
    title!(plt[3], "Training loss (log scale)")
    xlabel!(plt[3], "iteration"); ylabel!(plt[3], "objective")

    out = joinpath(SCRIPT_DIR, "biodynax_discovery.png")
    savefig(plt, out)
    println("  → saved figure to: ", out)

    # 7) Final summary ------------------------------------------------------
    section("Summary")
    println("  initial loss : ", round(tr.initial_loss; digits = 6))
    println("  final loss   : ", round(tr.final_loss;   digits = 6))
    println("  discovery    : ", discovery.success ? "SUCCESS" : "FAILED")
    if discovery.success
        println("  equation     :\n", discovery.equation)
    else
        println("  reason       : ", discovery.message)
    end

    return (; network  = net,
              training = tr,
              discovery = discovery,
              plot_path = out)
end

# Only auto-run when executed as a script (so `include(...)` from the REPL
# leaves `main` defined for interactive experimentation).
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
