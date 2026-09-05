# Stability-selection pruning stage of implicit discovery. Off by default;
# with the option off the discovery output is unchanged, with it on terms are
# only ever removed and every library term reports its selection frequency.

function _ss_fixture(; nuisance = true, seed = 103, bootstrap = 0)
    r = collect(range(0.1, 2.0; length = 180))
    D = hill_rate_truth(r; vmax = 1.7, K = 0.6, n = 2)
    y = nuisance ? D .+ 0.04 .+ 0.04 .* r : D
    times = collect(range(0.0, 1.0; length = length(r)))
    config = rate_discovery_config(bootstrap = bootstrap, seed = seed)
    return reshape(r, 1, :), times, reshape(y, 1, :), config
end

function _ss_fields(candidate)
    return (candidate.numerator_coefficients, candidate.denominator_coefficients,
        candidate.selection_frequency, candidate.validation_error,
        candidate.denominator_minimum)
end

@testset "stability selection" begin
    @testset "constructor" begin
        selection = StabilitySelection()
        @test selection.n_boot == 100
        @test selection.τ == 0.8
        @test selection.seed == UInt64(7)
        @test StabilitySelection(tau = 0.6).τ == 0.6
        @test StabilitySelection(n_boot = 5, τ = 1.0, seed = 3).n_boot == 5
        @test_throws ArgumentError StabilitySelection(n_boot = 0)
        @test_throws ArgumentError StabilitySelection(τ = 0.0)
        @test_throws ArgumentError StabilitySelection(τ = 1.5)
    end

    @testset "off by default: output identical with and without the keyword" begin
        for nuisance in (false, true), bootstrap in (0, 8)
            R, times, D, config = _ss_fixture(; nuisance, bootstrap)
            plain = discover_unknown_rate(R, times, D; config, verbose = false,
                strict = true)
            explicit = discover_unknown_rate(R, times, D; config, verbose = false,
                strict = true, stability_selection = nothing)
            @test plain.success && explicit.success
            @test _ss_fields(plain.candidates[1]) == _ss_fields(explicit.candidates[1])
            @test plain.equations == explicit.equations
            @test plain.metadata.config == explicit.metadata.config
            @test stability_selection_report(plain) === nothing
            @test format_stability_selection(plain) == "stability selection: off"
        end
    end

    @testset "on: prunes only, reports every term, deterministic" begin
        R, times, D, config = _ss_fixture(; nuisance = true, bootstrap = 0)
        base = discover_unknown_rate(R, times, D; config, verbose = false,
            strict = true)
        selection = StabilitySelection(n_boot = 40, τ = 0.8, seed = 11)
        pruned = discover_unknown_rate(R, times, D; config, verbose = false,
            strict = true, stability_selection = selection)
        @test pruned.success
        report = stability_selection_report(pruned)
        @test report !== nothing
        @test report.n_boot == 40
        @test report.τ == 0.8
        @test length(report.reports) == 1
        entry = report.reports[1]
        @test entry.target == 1
        base_candidate = base.candidates[1]
        candidate = pruned.candidates[1]
        n_terms = length(base_candidate.specification.numerator) +
                  length(base_candidate.specification.denominator)
        @test length(entry.terms) == n_terms
        @test all(0 <= term.frequency <= 1 for term in entry.terms)
        @test candidate.selection_frequency == [term.frequency for term in entry.terms]
        base_active = abs.(vcat(base_candidate.numerator_coefficients,
            base_candidate.denominator_coefficients)) .> 1e-8
        @test [term.in_candidate for term in entry.terms] == base_active
        active = abs.(vcat(candidate.numerator_coefficients,
            candidate.denominator_coefficients)) .> 1e-8
        @test [term.kept for term in entry.terms] == active
        # Never adds a term.
        @test all(active .<= base_active)
        if entry.applied
            @test any(base_active .& .!active)
            @test all(term.frequency >= 0.8 for term in entry.terms if term.kept)
        else
            @test active == base_active
            @test !isempty(entry.reason)
        end
        @test candidate.denominator_minimum >= config.backend.denominator_floor
        @test isfinite(candidate.validation_error)
        text = format_stability_selection(pruned)
        @test occursin("n_boot = 40", text)
        @test occursin("frequency", text)
        again = discover_unknown_rate(R, times, D; config, verbose = false,
            strict = true, stability_selection = selection)
        @test _ss_fields(again.candidates[1]) == _ss_fields(candidate)
        @test stability_selection_report(again) == report
    end

    @testset "τ at zero frequency keeps the candidate" begin
        R, times, D, config = _ss_fixture(; nuisance = false, bootstrap = 0)
        base = discover_unknown_rate(R, times, D; config, verbose = false,
            strict = true)
        # τ = 1 requires selection in every resample; the clean Hill fixture
        # is recovered with two terms and either both survive or the stage
        # reports why it was not applied.
        pruned = discover_unknown_rate(R, times, D; config, verbose = false,
            strict = true, stability_selection = StabilitySelection(
                n_boot = 20, τ = 1.0, seed = 5))
        entry = stability_selection_report(pruned).reports[1]
        base_active = abs.(vcat(base.candidates[1].numerator_coefficients,
            base.candidates[1].denominator_coefficients)) .> 1e-8
        active = abs.(vcat(pruned.candidates[1].numerator_coefficients,
            pruned.candidates[1].denominator_coefficients)) .> 1e-8
        @test all(active .<= base_active)
        @test any(active[1:length(base.candidates[1].specification.numerator)])
        entry.applied || @test !isempty(entry.reason)
    end

    @testset "explicit backend rejects the option" begin
        R, times, D, config = _ss_fixture(; nuisance = false)
        explicit = DiscoveryConfig(backend = ExplicitSTLSQ(), include_interactions = false)
        @test_throws ArgumentError discover_unknown_rate(R, times, D;
            config = explicit, verbose = false, strict = true,
            stability_selection = StabilitySelection(n_boot = 2))
        lenient = discover_unknown_rate(R, times, D; config = explicit,
            verbose = false, strict = false,
            stability_selection = StabilitySelection(n_boot = 2))
        @test !lenient.success
    end
end
