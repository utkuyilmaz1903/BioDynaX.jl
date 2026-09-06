# One-call entry point: discover_unknown_term must run the same functions as
# the chained calls of examples/unknown_inhibition.jl, in the same order and
# with the same defaults, so its result matches the chain field by field.

const _DUT_TRUTH = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
const _DUT_CONFIG = TrainingConfig(adam_iterations = 2, bfgs_iterations = 0,
    log_every = 10^6)

function _dut_fixture()
    truth_net = BioDynaX.build_hill_recovery_network(; known = true, hill_order = 2)
    ude_net = BioDynaX.build_hill_recovery_network(; known = false, hill_order = 2)
    set = BioDynaX.reference_protocol_experiment_set(
        MersenneTwister(103), truth_net; smoke = true, truth_params = _DUT_TRUTH,
        initial_conditions = [[0.25, 0.20], [0.80, 0.35], [0.40, 1.10]])
    return ude_net, set
end

function _dut_chain(ude_net, set; rng_seed = 7)
    rng = MersenneTwister(rng_seed)
    model, params = build_ude_model(rng, ude_net)
    phys_names = Tuple(parameter_schema(model).phys_names)
    guess = NamedTuple{phys_names}(ntuple(_ -> 0.8, length(phys_names)))
    ude_init = pack_parameters(guess, params.nn)
    first_exp = first(set.experiments)
    tspan = (first(first_exp.times), last(first_exp.times))
    warm = train_ude(ude_init, first_exp.observations, first_exp.times, first_exp.u0,
        tspan, model;
        config = TrainingConfig(adam_iterations = _DUT_CONFIG.adam_iterations,
            bfgs_iterations = 0,
            horizon_schedule = HorizonCurriculum(fractions = [0.35, 0.7, 1.0]),
            log_every = 10^6),
        verbose = false)
    trained = train_experiments(warm.params, set, model; config = _DUT_CONFIG,
        verbose = false)
    term = only(BioDynaX.neural_destruction_terms(model))
    r_range = BioDynaX._regulator_grid(set, term)
    R, D, term = BioDynaX.sample_unknown_destruction_grid(model, trained.params, term;
        r_range = r_range)
    times_grid = collect(range(0.0, 1.0; length = size(R, 2)))
    discovery = discover_unknown_rate(R, times_grid, D;
        config = BioDynaX.reference_protocol_discovery_config(), verbose = false,
        strict = false)
    ident = BioDynaX.report_production_destruction_tradeoff(
        model, trained.params, first_exp.observations, first_exp.times,
        first_exp.u0, tspan; term = term, verbose = false)
    residual = Inf
    if discovery.success && !isempty(discovery.candidates)
        rate_fn = equation_to_function(discovery.candidates[1])
        residual = hybrid_data_residual(model, trained.params, term, rate_fn,
            first_exp.u0, tspan, first_exp.times, first_exp.observations)
    end
    return (; model, trained, term, R, D, discovery, ident, residual)
end

@testset "discover_unknown_term" begin
    ude_net, set = _dut_fixture()

    @testset "matches the chained calls field by field" begin
        chain = _dut_chain(ude_net, set)
        result = discover_unknown_term(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), verbose = false,
            known_support = BioDynaX.hill_rate_support(2))
        @test result isa UnknownTermResult
        @test BioDynaX.nn_parameter_fingerprint(result.params.nn) ==
              BioDynaX.nn_parameter_fingerprint(chain.trained.params.nn)
        @test collect(result.params.phys) == collect(chain.trained.params.phys)
        @test result.training.final_loss == chain.trained.final_loss
        @test result.term === result.term
        @test result.samples.R == chain.R
        @test result.samples.D == chain.D
        @test result.discovery.success == chain.discovery.success
        @test result.discovery.equations == chain.discovery.equations
        @test result.discovery.retcode == chain.discovery.retcode
        @test result.identifiability.unidentifiable_edge ==
              chain.ident.unidentifiable_edge
        @test result.identifiability.collinearity == chain.ident.collinearity
        @test isequal(result.residuals.data_residual, chain.residual)
        @test isnan(result.residuals.data_residual_holdout)
        @test result.training_indices == 1:3
        @test isempty(result.holdout_indices)
        if chain.discovery.success
            @test result.extras ==
                  BioDynaX.reference_protocol_discovery_extras(chain.discovery.candidates[1])
            @test isfinite(result.residuals.data_residual_train)
        end
        @test result.settings.n_ics == 3
        @test result.settings.n_points == size(first(set.experiments).observations, 2)
        @test result.settings.adam_iters == 2
        @test result.settings.bfgs_iters == 0
        @test result.settings.bootstrap == BioDynaX.REFERENCE_PROTOCOL.bootstrap
        @test result.settings.discovery_seed ==
              BioDynaX.REFERENCE_PROTOCOL.discovery_seed
        @test result.settings.unknown_holes == 1
    end

    @testset "report and show" begin
        result = discover_unknown_term(ude_net, set; training = _DUT_CONFIG,
            holdout = 1, rng = MersenneTwister(7), verbose = false, seed = 103)
        text = report_unknown_term(result)
        for section in ("IDENTIFIABILITY", "FIT", "DISCOVERY", "REPRODUCTION")
            @test occursin("\n" * section * "\n", "\n" * text)
        end
        @test occursin("hybrid_data_residual_train:", text)
        @test occursin("hybrid_data_residual_holdout:", text)
        @test occursin("seed: 103", text)
        @test occursin("n_ics: 3", text)
        @test occursin("extras: NA", text)
        @test sprint(show, MIME("text/plain"), result) == text
        summary = sprint(show, result)
        @test startswith(summary, "UnknownTermResult(")
        @test occursin("held out = 1", summary)
        @test result.training_indices == 1:2
        @test result.holdout_indices == [3]
        # The default report carries no held-out lines, so existing output is unchanged.
        plain = BioDynaX.format_protocol_result(result.identifiability)
        @test !occursin("hybrid_data_residual_train", plain)
        @test !occursin("hybrid_data_residual_holdout", plain)
        captured_path = joinpath(mktempdir(), "verbose.txt")
        open(captured_path, "w") do io
            redirect_stdout(io) do
                discover_unknown_term(ude_net, set; training = _DUT_CONFIG,
                    holdout = 1, rng = MersenneTwister(7), verbose = true)
            end
        end
        @test occursin("REPRODUCTION", read(captured_path, String))
    end

    @testset "arguments" begin
        @test_throws ArgumentError discover_unknown_term(ude_net, set;
            training = _DUT_CONFIG, holdout = 3, verbose = false)
        @test_throws ArgumentError discover_unknown_term(ude_net, set;
            training = _DUT_CONFIG, holdout = -1, verbose = false)
        known = BioDynaX.build_hill_recovery_network(; known = true, hill_order = 2)
        @test_throws ErrorException discover_unknown_term(known, set;
            training = _DUT_CONFIG, verbose = false)
        no_warm = discover_unknown_term(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), warmup = false, verbose = false)
        warm = discover_unknown_term(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), verbose = false)
        @test no_warm.settings.warmup == false
        @test BioDynaX.nn_parameter_fingerprint(no_warm.params.nn) !=
              BioDynaX.nn_parameter_fingerprint(warm.params.nn)
        indexed = discover_unknown_term(ude_net, set; training = _DUT_CONFIG,
            holdout = 0, rng = MersenneTwister(7), term = 1, verbose = false)
        @test indexed.discovery.equations == warm.discovery.equations
        @test_throws ArgumentError discover_unknown_term(ude_net, set;
            training = _DUT_CONFIG, holdout = 0, term = "first", verbose = false)
    end
end
