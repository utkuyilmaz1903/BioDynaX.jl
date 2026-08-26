###############################################################################
# SciML solve surface: ODEFunction, ODEProblem, remake, inplace, sensealg.
#
# CompiledPath already compares generate vs ODEProblem. This file locks the
# rest of the solve surface: ude_system vs ODEFunction, inplace cache reuse,
# remake of u0 / tspan / p, SciMLBase.solve(model), predict_ude, and the
# recommend_sensealg observation-count boundary (64 vs 65). No new solvers.
# Does not drop protocol ICs, points, or seeds. Does not grow exports.
###############################################################################

"""Source strings that prove the solve-surface helpers stay wired."""
const SCIML_SOLVE_SURFACE_MUST_CONTAIN = (
    "struct SolveSurfaceRow",
    "function solve_surface_row",
    "function sensealg_boundary_row",
    "function cache_reuse_row",
    "function remake_field_row",
    "function odefunction_rhs_row")

const SCIML_SOLVE_SURFACE_MUST_NOT_CONTAIN = (
    "support_f1_ude = 0.99",
    "function validate_network",
    "Rodas5",
    "KenCarp4",
    "GPU")

function sciml_solve_surface_locked_sentences()
    return (;
        surface = "The SciML solve surface agrees ude_system, ODEFunction, ODEProblem, remake, inplace cache, SciMLBase.solve, and predict_ude.",
        boundary = "Mechanistic models switch from BacksolveAdjoint to InterpolatingAdjoint when n_observations exceeds 64.",
        cache = "An in-place ODEProblem reuses one UDEModelCache across remakes; allocate_cache is not called per IC.",
        nosolver = "This surface does not add an OrdinaryDiffEq algorithm; SolverConfig.algorithm stays Tsit5.")
end

function sciml_solve_surface_source_path()
    joinpath(pkgdir(BioDynaX), "src", "SciMLSolveSurface.jl")
end

function sciml_interface_source_path()
    joinpath(pkgdir(BioDynaX), "src", "SciMLInterface.jl")
end

# -- Tight forward solver (no new algorithm) ----------------------------------

function solve_surface_solver(;
        ad_policy::AbstractADPolicy = ZygoteAD(),
        sensealg = nothing,
        abstol = 1e-9,
        reltol = 1e-9)
    return SolverConfig(
        algorithm = Tsit5(),
        ad_policy = ad_policy,
        sensealg = sensealg,
        abstol = abstol,
        reltol = reltol)
end

function _surface_times(tspan, n_points::Int)
    return collect(range(first(tspan), last(tspan); length = n_points))
end

function _surface_u0(model::UDEModel, u0)
    n = model.compiled.nstates
    return u0 === nothing ? fill(0.22, n) : Float64.(u0)
end

function _surface_compile_count(f)
    return with_compile_network_counter() do counter
        f()
        counter[]
    end
end

# -- ODEFunction / RHS --------------------------------------------------------

"""
    odefunction_rhs_row(model, params, u; t=0.0)

`build_ude_function` out-of-place equals `ude_system`. In-place
`ude_rhs!` equals that vector. `SciMLBase.isinplace` matches the flag.
"""
function odefunction_rhs_row(model::UDEModel, params, u; t = 0.0)
    uvec = Float64.(u)
    direct = ude_system(uvec, params, t, model)
    f_oop = build_ude_function(model)
    from_fun = f_oop(uvec, params, t)
    cache = allocate_cache(model, Float64)
    f_ip = build_ude_function(model; inplace = true, cache = cache)
    du = similar(uvec)
    f_ip(du, uvec, params, t)
    du2 = similar(uvec)
    ude_rhs!(du2, uvec, params, t, model, cache)
    compiled = _surface_compile_count() do
        f_oop(uvec, params, t)
        f_ip(du, uvec, params, t)
    end
    return (;
        direct,
        from_fun,
        inplace = du,
        rhs = du2,
        oop_inplace = SciMLBase.isinplace(f_oop),
        ip_inplace = SciMLBase.isinplace(f_ip),
        matches_function = direct ≈ from_fun,
        matches_inplace = direct ≈ du,
        matches_rhs = direct ≈ du2,
        no_compile = compiled == 0,
        holds = direct ≈ from_fun && direct ≈ du && direct ≈ du2 &&
                SciMLBase.isinplace(f_oop) == false &&
                SciMLBase.isinplace(f_ip) &&
                compiled == 0)
end

# -- Cache reuse --------------------------------------------------------------

"""
    cache_reuse_row(model, params, u0; tspan, n_points)

One `UDEModelCache` is allocated and reused for two in-place solves
(original IC and a remade IC). The cache pointer is stable.
"""
function cache_reuse_row(model::UDEModel, params, u0;
        tspan = (0.0, 0.6), n_points::Int = 6)
    times = _surface_times(tspan, n_points)
    cache = allocate_cache(model, Float64)
    ptr = pointer(cache.du)
    prob = SciMLBase.ODEProblem(
        model, Float64.(u0), tspan, params; inplace = true, cache = cache)
    sol1 = solve(prob, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9,
        sensealg = nothing)
    u1 = last(Float64.(u0)) == 0 ? Float64.(u0) .+ 0.04 : Float64.(u0) .* 1.15
    remade = SciMLBase.remake(prob; u0 = u1)
    sol2 = solve(remade, Tsit5(); saveat = times, abstol = 1e-9, reltol = 1e-9,
        sensealg = nothing)
    cache2 = allocate_cache(model, Float64)
    compiled = _surface_compile_count() do
        solve(SciMLBase.remake(prob; u0 = u1), Tsit5();
            saveat = times, abstol = 1e-9, reltol = 1e-9, sensealg = nothing)
    end
    return (;
        cache_ptr = ptr,
        same_du = pointer(cache.du) === ptr,
        heads = neural_cache_matches_heads(model, cache),
        heads2 = neural_cache_matches_heads(model, cache2),
        finite = all(isfinite, Array(sol1)) && all(isfinite, Array(sol2)),
        nstates = length(cache.du),
        no_compile = compiled == 0,
        holds = pointer(cache.du) === ptr &&
                neural_cache_matches_heads(model, cache) &&
                neural_cache_matches_heads(model, cache2) &&
                all(isfinite, Array(sol1)) && all(isfinite, Array(sol2)) &&
                length(cache.du) == model.compiled.nstates &&
                compiled == 0)
end

# -- Remake of individual fields ----------------------------------------------

"""
    remake_field_row(model, params, u0; tspan, n_points)

`SciMLBase.remake` of `p`, `u0`, and `tspan` each matches a freshly
constructed `ODEProblem` with that field. No compile.
"""
function remake_field_row(model::UDEModel, params, u0;
        tspan = (0.0, 0.6), n_points::Int = 6)
    times = _surface_times(tspan, n_points)
    u = Float64.(u0)
    template = SciMLBase.ODEProblem(model, u, tspan, params)
    sol0 = Array(solve(template, Tsit5(); saveat = times, abstol = 1e-9,
        reltol = 1e-9, sensealg = nothing))
    remade_p = SciMLBase.remake(template; p = params)
    sol_p = Array(solve(remade_p, Tsit5(); saveat = times, abstol = 1e-9,
        reltol = 1e-9, sensealg = nothing))
    fresh_p = Array(solve(
        SciMLBase.ODEProblem(model, u, tspan, params), Tsit5();
        saveat = times, abstol = 1e-9, reltol = 1e-9, sensealg = nothing))
    u2 = u .* 1.1
    remade_u = SciMLBase.remake(template; u0 = u2)
    sol_u = Array(solve(remade_u, Tsit5(); saveat = times, abstol = 1e-9,
        reltol = 1e-9, sensealg = nothing))
    fresh_u = Array(solve(
        SciMLBase.ODEProblem(model, u2, tspan, params), Tsit5();
        saveat = times, abstol = 1e-9, reltol = 1e-9, sensealg = nothing))
    tspan2 = (first(tspan), last(tspan) * 0.8)
    times2 = _surface_times(tspan2, n_points)
    remade_t = SciMLBase.remake(template; tspan = tspan2)
    sol_t = Array(solve(remade_t, Tsit5(); saveat = times2, abstol = 1e-9,
        reltol = 1e-9, sensealg = nothing))
    fresh_t = Array(solve(
        SciMLBase.ODEProblem(model, u, tspan2, params), Tsit5();
        saveat = times2, abstol = 1e-9, reltol = 1e-9, sensealg = nothing))
    compiled = _surface_compile_count() do
        SciMLBase.remake(template; p = params, u0 = u2, tspan = tspan2)
    end
    return (;
        matches_p = sol_p ≈ sol0 && sol_p ≈ fresh_p,
        matches_u0 = sol_u ≈ fresh_u,
        matches_tspan = sol_t ≈ fresh_t,
        remake_count = 3,
        no_compile = compiled == 0,
        holds = sol_p ≈ sol0 && sol_p ≈ fresh_p && sol_u ≈ fresh_u &&
                sol_t ≈ fresh_t && compiled == 0)
end

# -- Full solve-surface row ---------------------------------------------------

"""
    SolveSurfaceRow

One compiled model compared across the SciML entries the package owns.
"""
struct SolveSurfaceRow
    nstates::Int
    n_heads::Int
    dense::Bool
    oop_inplace::Bool
    ip_inplace::Bool
    matches_odefunction::Bool
    matches_inplace::Bool
    matches_remake::Bool
    matches_sciml_solve::Bool
    matches_predict::Bool
    matches_session::Bool
    matches_generate::Bool
    no_compile::Bool
    finite::Bool
    holds::Bool
end

function solve_surface_row(model::UDEModel, params, u0;
        tspan = (0.0, 0.6), n_points::Int = 6)
    u = _surface_u0(model, u0)
    times = _surface_times(tspan, n_points)
    rhs = odefunction_rhs_row(model, params, u)
    cache = allocate_cache(model, Float64)
    prob = SciMLBase.ODEProblem(model, u, tspan, params)
    @assert !SciMLBase.isinplace(prob.f)
    sol_oop = Array(solve(prob, Tsit5(); saveat = times, abstol = 1e-9,
        reltol = 1e-9, sensealg = nothing))
    prob_ip = SciMLBase.ODEProblem(
        model, u, tspan, params; inplace = true, cache = cache)
    @assert SciMLBase.isinplace(prob_ip.f)
    sol_ip = Array(solve(prob_ip, Tsit5(); saveat = times, abstol = 1e-9,
        reltol = 1e-9, sensealg = nothing))
    remade = SciMLBase.remake(prob; p = params, u0 = u, tspan = tspan)
    sol_remade = Array(solve(remade, Tsit5(); saveat = times, abstol = 1e-9,
        reltol = 1e-9, sensealg = nothing))
    sciml = Array(SciMLBase.solve(
        model, u, tspan, params;
        saveat = times,
        solver_config = solve_surface_solver()))
    pred = predict_ude(
        params, u, tspan, times, model;
        solver_config = solve_surface_solver())
    session = training_solve_session(
        model, u, tspan, params;
        solver = lock_training_solver(model, solve_surface_solver(
            ad_policy = ZygoteAD(),
            sensealg = auto_sensealg(model; n_observations = 100))))
    sess = predict_ude_session(session, params, u, tspan, times)
    times_g, clean, _, used = generate_from_compiled_model(
        model, params, MersenneTwister(0);
        u0 = u, tspan = tspan, n_points = n_points, noise_σ = 0.0)
    gen_pred = predict_ude(
        used, u, tspan, times_g, model;
        solver_config = solve_surface_solver())
    compiled = _surface_compile_count() do
        predict_ude(params, u, tspan, times, model;
            solver_config = solve_surface_solver())
        SciMLBase.solve(model, u, tspan, params;
            saveat = times, solver_config = solve_surface_solver())
    end
    finite = all(isfinite, sol_oop) && all(isfinite, sol_ip) &&
             all(isfinite, sol_remade) && all(isfinite, sciml) &&
             all(isfinite, pred) && all(isfinite, sess) && all(isfinite, clean)
    matches_ip = sol_oop ≈ sol_ip
    matches_remake = sol_oop ≈ sol_remade
    matches_sciml = sol_oop ≈ sciml
    matches_pred = sol_oop ≈ pred
    matches_sess = sol_oop ≈ sess
    matches_gen = clean ≈ gen_pred
    arch = compiled_nn_architecture(model, params)
    holds = rhs.holds && finite && matches_ip && matches_remake &&
            matches_sciml && matches_pred && matches_sess && matches_gen &&
            compiled == 0 && arch.dense
    return SolveSurfaceRow(
        model.compiled.nstates,
        neural_head_count(model),
        arch.dense,
        rhs.oop_inplace,
        rhs.ip_inplace,
        rhs.holds,
        matches_ip,
        matches_remake,
        matches_sciml,
        matches_pred,
        matches_sess,
        matches_gen,
        compiled == 0,
        finite,
        holds)
end

function solve_surface_row_namedtuple(row::SolveSurfaceRow)
    return (;
        nstates = row.nstates,
        n_heads = row.n_heads,
        dense = row.dense,
        matches_odefunction = row.matches_odefunction,
        matches_inplace = row.matches_inplace,
        matches_remake = row.matches_remake,
        matches_sciml_solve = row.matches_sciml_solve,
        matches_predict = row.matches_predict,
        matches_session = row.matches_session,
        matches_generate = row.matches_generate,
        no_compile = row.no_compile,
        holds = row.holds)
end

# -- Sensealg observation-count boundary --------------------------------------

"""
    sensealg_boundary_row(model; n_observations)

Named recommendation at one observation count, plus ProductionAD.
"""
function sensealg_boundary_row(model::UDEModel; n_observations::Int = 64)
    zy = recommend_sensealg(
        model; policy = ZygoteAD(), n_observations = n_observations)
    prod = recommend_sensealg(
        model; policy = ProductionAD(), n_observations = n_observations)
    auto = auto_sensealg(
        model; policy = ZygoteAD(), n_observations = n_observations)
    cfg = default_solver_config(
        model; ad_policy = ZygoteAD(), n_observations = n_observations)
    neural = neural_head_count(model) > 0
    nstates = model.compiled.nstates
    expected = if neural
        :interpolating_neural
    elseif nstates ≤ 8 && n_observations ≤ 64
        :backsolve_mechanistic
    else
        :interpolating_default
    end
    kind = training_sensealg_kind(zy.sensealg)
    locked100 = lock_training_solver(model, SolverConfig())
    return (;
        n_observations,
        neural,
        nstates,
        zygote_name = zy.name,
        production_name = prod.name,
        expected,
        kind,
        auto_matches = training_sensealg_kind(auto) === kind,
        default_matches = training_sensealg_kind(cfg) === kind,
        production_is_interpolating = prod.name === :interpolating_production,
        lock_at_100 = training_sensealg_kind(locked100),
        holds = zy.name === expected &&
                prod.name === :interpolating_production &&
                training_sensealg_kind(auto) === kind &&
                training_sensealg_kind(cfg) === kind)
end

function sensealg_boundary_grid(model::UDEModel)
    n20 = sensealg_boundary_row(model; n_observations = 20)
    n64 = sensealg_boundary_row(model; n_observations = 64)
    n65 = sensealg_boundary_row(model; n_observations = 65)
    n100 = sensealg_boundary_row(model; n_observations = 100)
    neural = neural_head_count(model) > 0
    mechanistic = !neural && model.compiled.nstates ≤ 8
    crossing = mechanistic ?
        (n64.zygote_name === :backsolve_mechanistic &&
         n65.zygote_name === :interpolating_default) :
        (n64.zygote_name === n65.zygote_name)
    return (;
        n20, n64, n65, n100, neural, mechanistic, crossing,
        holds = n20.holds && n64.holds && n65.holds && n100.holds && crossing)
end

function sensealg_boundary_matrix()
    linear = sensealg_boundary_grid(
        build_ude_model(MersenneTwister(7), build_linear_test_network())[1])
    hill = sensealg_boundary_grid(
        build_ude_model(MersenneTwister(11),
            build_hill_recovery_network(; known = false, hill_order = 2))[1])
    remap = sensealg_boundary_grid(
        build_ude_model(MersenneTwister(13),
            build_remapped_two_regulator_network())[1])
    six = sensealg_boundary_grid(
        build_ude_model(MersenneTwister(41),
            build_six_state_unknown_network(; known = false))[1])
    competitive = sensealg_boundary_grid(
        build_ude_model(MersenneTwister(47),
            build_competitive_test_network(; known = true))[1])
    zero = sensealg_boundary_grid(
        build_ude_model(MersenneTwister(7),
            build_zero_unknown_linear_network())[1])
    dual = sensealg_boundary_grid(
        build_ude_model(MersenneTwister(21), build_dual_unknown_network())[1])
    return (;
        linear, hill, remap, six, competitive, zero, dual,
        holds = linear.holds && hill.holds && remap.holds && six.holds &&
                competitive.holds && zero.holds && dual.holds &&
                linear.mechanistic && linear.crossing &&
                competitive.mechanistic && competitive.crossing &&
                zero.mechanistic &&
                hill.neural && remap.neural && six.neural && dual.neural &&
                hill.n64.zygote_name === :interpolating_neural &&
                hill.n65.zygote_name === :interpolating_neural)
end

# -- Failed / invalid solves --------------------------------------------------

"""
    failed_solve_row(model, params, u0)

NaN initial conditions must not produce a silent success. `maxiters = 1`
on a long horizon records the SciML retcode instead of inventing data.
"""
function failed_solve_row(model::UDEModel, params, u0;
        tspan = (0.0, 8.0), n_points::Int = 20)
    times = _surface_times(tspan, n_points)
    nan_u0 = fill(NaN, length(u0))
    nan_threw = try
        predict_ude(
            params, nan_u0, tspan, times, model;
            solver_config = solve_surface_solver())
        false
    catch
        true
    end
    tight = SolverConfig(
        algorithm = Tsit5(),
        sensealg = nothing,
        abstol = 1e-9,
        reltol = 1e-9,
        maxiters = 1)
    sol = SciMLBase.solve(
        model, Float64.(u0), tspan, params;
        saveat = times, solver_config = tight)
    retcode = sol.retcode
    success = SciMLBase.successful_retcode(sol)
    return (;
        nan_threw,
        retcode,
        success,
        maxiters = 1,
        holds = nan_threw)
end

"""SciMLBase.solve(model, ...) disables dense output; saveat is the record."""
function dense_disabled_row(model::UDEModel, params, u0;
        tspan = (0.0, 0.5), n_points::Int = 6)
    times = _surface_times(tspan, n_points)
    sol = SciMLBase.solve(
        model, Float64.(u0), tspan, params;
        saveat = times, solver_config = solve_surface_solver())
    src = read(sciml_interface_source_path(), String)
    start = findfirst("function SciMLBase.solve(model::UDEModel", src)
    body = if start === nothing
        ""
    else
        rest = src[first(start):end]
        nxt = findnext(r"\nfunction ", rest, 2)
        nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    end
    return (;
        dense = sol.dense,
        n_saved = length(sol.t),
        source_disables_dense = occursin("dense = false", body),
        source_skips_everystep = occursin("save_everystep = false", body),
        holds = sol.dense == false && length(sol.t) == n_points &&
                occursin("dense = false", body) &&
                occursin("save_everystep = false", body))
end

function saveat_length_row(model::UDEModel, params, u0;
        tspan = (0.0, 0.5), n_points::Int = 7)
    times = _surface_times(tspan, n_points)
    pred = predict_ude(
        params, Float64.(u0), tspan, times, model;
        solver_config = solve_surface_solver())
    sol = SciMLBase.solve(
        model, Float64.(u0), tspan, params;
        saveat = times, solver_config = solve_surface_solver())
    return (;
        requested = n_points,
        predicted = size(pred, 2),
        solved = length(sol.t),
        states = size(pred, 1),
        holds = size(pred, 2) == n_points &&
                length(sol.t) == n_points &&
                size(pred, 1) == model.compiled.nstates)
end

# -- Multi-IC remake of one template ------------------------------------------

"""
    multi_ic_remake_row(model, params, ics; tspan, n_points)

One `ODEProblem` remade per IC versus a fresh problem per IC. Trajectories
match. `compile_network` stays at zero.
"""
function multi_ic_remake_row(model::UDEModel, params, ics;
        tspan = (0.0, 0.5), n_points::Int = 6)
    times = _surface_times(tspan, n_points)
    first_u = Float64.(first(ics))
    template = SciMLBase.ODEProblem(model, first_u, tspan, params)
    remade = Vector{Matrix{Float64}}(undef, length(ics))
    fresh = Vector{Matrix{Float64}}(undef, length(ics))
    compiled = _surface_compile_count() do
        for (i, u) in pairs(ics)
            uvec = Float64.(u)
            remade[i] = Array(solve(
                SciMLBase.remake(template; u0 = uvec), Tsit5();
                saveat = times, abstol = 1e-9, reltol = 1e-9,
                sensealg = nothing))
            fresh[i] = Array(solve(
                SciMLBase.ODEProblem(model, uvec, tspan, params), Tsit5();
                saveat = times, abstol = 1e-9, reltol = 1e-9,
                sensealg = nothing))
        end
    end
    matches = all(i -> remade[i] ≈ fresh[i], eachindex(ics))
    finite = all(A -> all(isfinite, A), remade)
    return (;
        n_ics = length(ics),
        matches,
        finite,
        no_compile = compiled == 0,
        holds = matches && finite && compiled == 0 && length(ics) ≥ 2)
end

# -- default_solver_config vs lock vs auto ------------------------------------

function solver_config_agreement_row(model::UDEModel)
    auto = auto_sensealg(model; n_observations = 100)
    rec = recommend_sensealg(model; n_observations = 100)
    default = default_solver_config(model; n_observations = 100)
    locked = lock_training_solver(model)
    zy_locked = lock_training_solver(model, SolverConfig())
    prod_nothing = lock_training_solver(
        model, SolverConfig(ad_policy = ProductionAD(), sensealg = nothing))
    return (;
        auto_kind = training_sensealg_kind(auto),
        rec_name = rec.name,
        default_kind = training_sensealg_kind(default),
        locked_kind = training_sensealg_kind(locked),
        zy_locked_kind = training_sensealg_kind(zy_locked),
        prod_nothing_unlocked = prod_nothing.sensealg === nothing,
        algorithm_is_tsit5 = nameof(typeof(default.algorithm)) === :Tsit5,
        locked_algorithm_is_tsit5 = nameof(typeof(locked.algorithm)) === :Tsit5,
        holds = training_sensealg_kind(auto) ===
                training_sensealg_kind(rec.sensealg) &&
                training_sensealg_kind(default) ===
                training_sensealg_kind(locked) &&
                training_sensealg_kind(locked) ===
                training_sensealg_kind(zy_locked) &&
                prod_nothing.sensealg === nothing &&
                nameof(typeof(default.algorithm)) === :Tsit5)
end

# -- Fixture paths ------------------------------------------------------------

function _compiled_fixture(rng::AbstractRNG, network::BiologicalNetwork;
        truth_params = nothing)
    model, params = build_ude_model(rng, network)
    packed = truth_params === nothing ? params :
        pack_parameters(truth_params, params.nn)
    u0 = fill(0.22, model.compiled.nstates)
    return model, packed, u0
end

function linear_solve_surface_path()
    model, params, u0 = _compiled_fixture(
        MersenneTwister(7), build_linear_test_network();
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    row = solve_surface_row(model, params, [0.22, 0.14])
    remake = remake_field_row(model, params, [0.22, 0.14])
    cache = cache_reuse_row(model, params, [0.22, 0.14])
    multi = multi_ic_remake_row(
        model, params, [[0.22, 0.14], [0.30, 0.18], [0.18, 0.12]])
    boundary = sensealg_boundary_grid(model)
    cfg = solver_config_agreement_row(model)
    failed = failed_solve_row(model, params, [0.22, 0.14])
    saveat = saveat_length_row(model, params, [0.22, 0.14])
    return (;
        row, remake, cache, multi, boundary, cfg, failed, saveat,
        holds = row.holds && remake.holds && cache.holds && multi.holds &&
                boundary.holds && cfg.holds && failed.holds && saveat.holds &&
                row.n_heads == 0 && boundary.mechanistic)
end

function hill_ude_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(11),
        build_hill_recovery_network(; known = false, hill_order = 2);
        truth_params = (k_prod = 0.9, k_rs = 1.0, k_r = 0.6))
    row = solve_surface_row(model, params, [0.30, 0.25])
    boundary = sensealg_boundary_grid(model)
    cfg = solver_config_agreement_row(model)
    return (;
        row, boundary, cfg,
        holds = row.holds && boundary.holds && cfg.holds &&
                row.n_heads == 1 && boundary.neural)
end

function remapped_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(13), build_remapped_two_regulator_network();
        truth_params = remapped_two_regulator_phys_truth())
    u0 = remapped_two_regulator_state()
    row = solve_surface_row(model, params, u0)
    remake = remake_field_row(model, params, u0)
    cache = cache_reuse_row(model, params, u0)
    multi = multi_ic_remake_row(
        model, params, [u0, [0.18, 0.16, 0.22, 0.12]])
    boundary = sensealg_boundary_grid(model)
    return (;
        row, remake, cache, multi, boundary,
        holds = row.holds && remake.holds && cache.holds && multi.holds &&
                boundary.holds && row.n_heads == 2 && row.dense)
end

function two_regulator_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(19), build_two_regulator_unknown_network();
        truth_params = (k_es = 0.8, k_i = 0.5, k_e = 0.4))
    row = solve_surface_row(model, params, [0.25, 0.20, 0.15])
    boundary = sensealg_boundary_grid(model)
    return (;
        row, boundary,
        holds = row.holds && boundary.holds && row.n_heads == 1 &&
                compiled_nn_architecture(model, params).arities == [2])
end

function six_state_solve_surface_path()
    model, params, u0 = _compiled_fixture(
        MersenneTwister(41),
        build_six_state_unknown_network(; known = false))
    row = solve_surface_row(model, params, u0; tspan = (0.0, 0.4), n_points = 5)
    cache = cache_reuse_row(model, params, u0; tspan = (0.0, 0.4), n_points = 5)
    boundary = sensealg_boundary_grid(model)
    return (;
        row, cache, boundary,
        holds = row.holds && cache.holds && boundary.holds &&
                row.nstates == 6 && row.n_heads == 1)
end

function competitive_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(47), build_competitive_test_network(; known = true);
        truth_params = (k_in = 0.9, vmax = 1.5, km = 0.4, ki = 0.6,
                        k_s = 0.8, k_i = 0.5))
    row = solve_surface_row(model, params, [0.25, 0.45, 0.20])
    boundary = sensealg_boundary_grid(model)
    cfg = solver_config_agreement_row(model)
    return (;
        row, boundary, cfg,
        holds = row.holds && boundary.holds && cfg.holds &&
                row.n_heads == 0 && boundary.mechanistic)
end

function mm_unknown_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(43), build_mm_recovery_network(; known = false);
        truth_params = (k_prod = 0.9, k_rs = 1.0, k_r = 0.6))
    row = solve_surface_row(model, params, [0.30, 0.25])
    boundary = sensealg_boundary_grid(model)
    return (;
        row, boundary,
        holds = row.holds && boundary.holds && row.n_heads == 1 &&
                unique_claim_recovery_admits(
                    build_mm_recovery_network(; known = false)))
end

function dual_unknown_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(21), build_dual_unknown_network();
        truth_params = (k_ca = 0.8, k_cb = 0.9, k_c = 0.5))
    row = solve_surface_row(model, params, [0.22, 0.18, 0.16])
    multi = multi_ic_remake_row(
        model, params, [[0.22, 0.18, 0.16], [0.30, 0.24, 0.20]])
    boundary = sensealg_boundary_grid(model)
    return (;
        row, multi, boundary,
        holds = row.holds && multi.holds && boundary.holds &&
                row.n_heads == 2 &&
                unique_claim_recovery_admits(build_dual_unknown_network()) == false)
end

function zero_hole_solve_surface_path()
    net = build_zero_unknown_linear_network()
    model, params, _ = _compiled_fixture(
        MersenneTwister(7), net;
        truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))
    row = solve_surface_row(model, params, [0.22, 0.14])
    boundary = sensealg_boundary_grid(model)
    return (;
        row, boundary,
        validate_open = validate_network(net) === net,
        holds = row.holds && boundary.holds && row.n_heads == 0 &&
                validate_network(net) === net)
end

function skipped_duplicate_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(13), build_skipped_duplicate_unknown_network();
        truth_params = (k_ca = 0.8, k_b = 0.5, k_c = 0.4))
    row = solve_surface_row(model, params, [0.2, 0.3, 0.4])
    rhs = odefunction_rhs_row(model, params, [0.2, 0.3, 0.4])
    return (;
        row, rhs,
        holds = row.holds && rhs.holds && row.dense && row.n_heads == 2)
end

function skipped_middle_solve_surface_path()
    model, params, u0 = _compiled_fixture(
        MersenneTwister(17), build_skipped_middle_unknown_network())
    row = solve_surface_row(model, params, u0; tspan = (0.0, 0.4), n_points = 5)
    rhs = odefunction_rhs_row(model, params, u0)
    arch = compiled_nn_architecture(model, params)
    return (;
        row, rhs, arch,
        holds = row.holds && rhs.holds && row.dense &&
                neural_index_is_dense(model) && arch.n_heads ≥ 2)
end

function unique_claim_solve_surface_path(; smoke::Bool = true)
    net = build_hill_recovery_network(; known = true, hill_order = 2)
    ude = build_hill_recovery_network(; known = false, hill_order = 2)
    truth_params = (k_prod = 0.9, vmax = 1.8, K = 0.55, k_rs = 1.0, k_r = 0.6)
    set = unique_claim_experiment_set(
        MersenneTwister(103), net; smoke = smoke, truth_params = truth_params)
    model, p0 = build_ude_model(MersenneTwister(11), ude)
    packed = pack_parameters((k_prod = 0.9, k_rs = 1.0, k_r = 0.6), p0.nn)
    exp = first(set.experiments)
    row = solve_surface_row(
        model, packed, exp.u0;
        tspan = (first(exp.times), last(exp.times)),
        n_points = length(exp.times))
    return (;
        row,
        compiled_once = experiment_set_is_compiled_once(set),
        n_ics = length(set.experiments),
        holds = row.holds && experiment_set_is_compiled_once(set) &&
                row.n_heads == 1)
end

"""
    production_inplace_agreement(model, params, u0; tspan, n_points)

`ProductionAD` + `sensealg === nothing` (in-place) versus Zygote
out-of-place `predict_ude` at the same tolerances. No compile.
"""
function production_inplace_agreement(model::UDEModel, params, u0;
        tspan = (0.0, 0.5), n_points::Int = 6)
    times = _surface_times(tspan, n_points)
    zy = SolverConfig(ad_policy = ZygoteAD(), abstol = 1e-9, reltol = 1e-9)
    prod = SolverConfig(
        ad_policy = ProductionAD(), sensealg = nothing,
        abstol = 1e-9, reltol = 1e-9)
    cache = allocate_cache(model, Float64)
    oop = predict_ude(
        params, Float64.(u0), tspan, times, model; solver_config = zy)
    ip = predict_ude(
        params, Float64.(u0), tspan, times, model;
        solver_config = prod, cache = cache)
    compiled = _surface_compile_count() do
        predict_ude(
            params, Float64.(u0), tspan, times, model;
            solver_config = prod, cache = cache)
    end
    return (;
        matches = oop ≈ ip,
        inplace = _forward_inplace(prod),
        oop_inplace = _forward_inplace(zy),
        finite = all(isfinite, oop) && all(isfinite, ip),
        no_compile = compiled == 0,
        holds = oop ≈ ip && _forward_inplace(prod) &&
                !_forward_inplace(zy) && compiled == 0)
end

function irregular_saveat_row(model::UDEModel, params, u0;
        tspan = (0.0, 1.0))
    times = [0.0, 0.12, 0.37, 0.61, 0.88, 1.0]
    pred = predict_ude(
        params, Float64.(u0), tspan, times, model;
        solver_config = solve_surface_solver())
    sol = SciMLBase.solve(
        model, Float64.(u0), tspan, params;
        saveat = times, solver_config = solve_surface_solver())
    arr = Array(sol)
    return (;
        requested = length(times),
        predicted = size(pred, 2),
        solved = length(sol.t),
        matches = pred ≈ arr,
        times_match = collect(sol.t) ≈ times,
        holds = size(pred, 2) == length(times) &&
                length(sol.t) == length(times) &&
                pred ≈ arr && collect(sol.t) ≈ times)
end

function zygote_gradient_finite_row(model::UDEModel, params, u0;
        tspan = (0.0, 0.4), n_points::Int = 5)
    times = _surface_times(tspan, n_points)
    data = predict_ude(
        params, Float64.(u0), tspan, times, model;
        solver_config = SolverConfig(ad_policy = ZygoteAD()))
    loss = p -> loss_mse(
        p, data, times, Float64.(u0), tspan, model;
        solver_config = SolverConfig(ad_policy = ZygoteAD()))
    grad = Zygote.gradient(loss, params)[1]
    compiled = _surface_compile_count() do
        Zygote.gradient(loss, params)
    end
    return (;
        finite = all(isfinite, grad),
        no_compile = compiled == 0,
        holds = all(isfinite, grad) && compiled == 0)
end

function default_example_solve_surface_path()
    net = DEFAULT_EXAMPLE_NETWORK
    model, params = build_ude_model(MersenneTwister(0), net)
    u0 = [0.20, 0.10]
    row = solve_surface_row(model, params, u0; tspan = (0.0, 0.4), n_points = 5)
    rhs = odefunction_rhs_row(model, params, u0)
    prod = production_inplace_agreement(
        model, params, u0; tspan = (0.0, 0.4), n_points = 5)
    return (;
        row, rhs, prod,
        holes = count_unknown_destructions(net),
        duplicate = default_example_has_duplicate_unknown_declaration(),
        holds = row.holds && rhs.holds && prod.holds &&
                default_example_has_duplicate_unknown_declaration() &&
                row.dense)
end

const SENSEALG_RATIONALE = (
    interpolating_production = "ProductionAD pairs in-place forward with checkpointed InterpolatingAdjoint.",
    backsolve_mechanistic = "No neural terms and modest state/observation count; BacksolveAdjoint is preferred.",
    interpolating_neural = "Neural destruction terms require reverse-mode VJP adjoints.",
    interpolating_default = "Default checkpointed InterpolatingAdjoint for general UDE models.")

function recommend_sensealg_rationale_row(model::UDEModel; n_observations::Int = 64)
    rec = recommend_sensealg(model; n_observations = n_observations)
    expected = getfield(SENSEALG_RATIONALE, rec.name)
    return (;
        name = rec.name,
        rationale = rec.rationale,
        expected,
        holds = rec.rationale == expected)
end

function recommend_sensealg_rationale_matrix()
    linear20 = recommend_sensealg_rationale_row(
        build_ude_model(MersenneTwister(7), build_linear_test_network())[1];
        n_observations = 20)
    linear100 = recommend_sensealg_rationale_row(
        build_ude_model(MersenneTwister(7), build_linear_test_network())[1];
        n_observations = 100)
    hill = recommend_sensealg_rationale_row(
        build_ude_model(MersenneTwister(11),
            build_hill_recovery_network(; known = false))[1];
        n_observations = 20)
    prod = recommend_sensealg(
        build_ude_model(MersenneTwister(7), build_linear_test_network())[1];
        policy = ProductionAD(), n_observations = 20)
    return (;
        linear20,
        linear100,
        hill,
        production_name = prod.name,
        production_rationale = prod.rationale,
        holds = linear20.holds && linear100.holds && hill.holds &&
                linear20.name === :backsolve_mechanistic &&
                linear100.name === :interpolating_default &&
                hill.name === :interpolating_neural &&
                prod.name === :interpolating_production &&
                prod.rationale == SENSEALG_RATIONALE.interpolating_production)
end

function mm_known_solve_surface_path()
    model, params, _ = _compiled_fixture(
        MersenneTwister(29), build_mm_test_network();
        truth_params = (vmax = 1.6, km = 0.45, k_s = 0.7, k_se = 0.9, k_e = 0.55))
    row = solve_surface_row(model, params, [0.40, 0.30])
    boundary = sensealg_boundary_grid(model)
    prod = production_inplace_agreement(model, params, [0.40, 0.30])
    return (;
        row, boundary, prod,
        holds = row.holds && boundary.holds && prod.holds &&
                row.n_heads == 0 && boundary.mechanistic)
end

function repressilator_solve_surface_path()
    net = build_repressilator_network()
    model, params = build_ude_model(MersenneTwister(5), net)
    u0 = [0.30, 0.10, 0.20]
    row = solve_surface_row(model, params, u0; tspan = (0.0, 0.3), n_points = 5)
    rhs = odefunction_rhs_row(model, params, u0)
    boundary = sensealg_boundary_grid(model)
    return (;
        row, rhs, boundary,
        nstates = model.compiled.nstates,
        holds = row.holds && rhs.holds && boundary.holds &&
                model.compiled.nstates == 3 && row.n_heads == 0 &&
                boundary.mechanistic)
end

function production_inplace_matrix()
    linear = production_inplace_agreement(
        _compiled_fixture(
            MersenneTwister(7), build_linear_test_network();
            truth_params = (k_ba = 0.8, k_a = 1.2, k_b = 0.5))...)
    hill = let
        model, params, _ = _compiled_fixture(
            MersenneTwister(11),
            build_hill_recovery_network(; known = false);
            truth_params = (k_prod = 0.9, k_rs = 1.0, k_r = 0.6))
        production_inplace_agreement(model, params, [0.30, 0.25])
    end
    remap = let
        model, params, _ = _compiled_fixture(
            MersenneTwister(13), build_remapped_two_regulator_network();
            truth_params = remapped_two_regulator_phys_truth())
        production_inplace_agreement(
            model, params, remapped_two_regulator_state())
    end
    return (;
        linear, hill, remap,
        holds = linear.holds && hill.holds && remap.holds)
end

function solve_surface_fixture_matrix()
    linear = linear_solve_surface_path()
    hill = hill_ude_solve_surface_path()
    remap = remapped_solve_surface_path()
    two = two_regulator_solve_surface_path()
    six = six_state_solve_surface_path()
    competitive = competitive_solve_surface_path()
    mm = mm_unknown_solve_surface_path()
    dual = dual_unknown_solve_surface_path()
    zero = zero_hole_solve_surface_path()
    skipped = skipped_duplicate_solve_surface_path()
    middle = skipped_middle_solve_surface_path()
    claim = unique_claim_solve_surface_path()
    boundary = sensealg_boundary_matrix()
    example = default_example_solve_surface_path()
    prod = production_inplace_matrix()
    mm_known = mm_known_solve_surface_path()
    repressilator = repressilator_solve_surface_path()
    rationale = recommend_sensealg_rationale_matrix()
    return (;
        linear, hill, remap, two, six, competitive, mm, dual, zero,
        skipped, middle, claim, boundary, example, prod, mm_known,
        repressilator, rationale,
        holds = linear.holds && hill.holds && remap.holds && two.holds &&
                six.holds && competitive.holds && mm.holds && dual.holds &&
                zero.holds && skipped.holds && middle.holds && claim.holds &&
                boundary.holds && example.holds && prod.holds &&
                mm_known.holds && repressilator.holds && rationale.holds)
end

# -- Source / docs locks ------------------------------------------------------

function sciml_recommend_sensealg_source_holds()
    src = read(sciml_interface_source_path(), String)
    start = findfirst("function recommend_sensealg", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("n_observations ≤ 64", body) &&
           occursin(":backsolve_mechanistic", body) &&
           occursin(":interpolating_neural", body) &&
           occursin(":interpolating_default", body) &&
           occursin(":interpolating_production", body) &&
           occursin(SENSEALG_RATIONALE.backsolve_mechanistic, body) &&
           occursin(SENSEALG_RATIONALE.interpolating_neural, body) &&
           occursin(SENSEALG_RATIONALE.interpolating_default, body) &&
           occursin(SENSEALG_RATIONALE.interpolating_production, body) &&
           !occursin("Rodas5", body) &&
           !occursin("KenCarp4", body) &&
           !occursin("TRBDF2", body)
end

function sciml_interface_adds_no_solver()
    src = read(sciml_interface_source_path(), String)
    return occursin("Tsit5", src) &&
           !occursin("Rodas5", src) &&
           !occursin("KenCarp4", src) &&
           !occursin("QNDF", src) &&
           !occursin("Vern7", src)
end

function sciml_odeproblem_uses_build_ude_function()
    src = read(sciml_interface_source_path(), String)
    start = findfirst("function SciMLBase.ODEProblem(model::UDEModel", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("build_ude_function", body) &&
           occursin("inplace", body)
end

function sciml_solve_uses_odeproblem()
    src = read(sciml_interface_source_path(), String)
    start = findfirst("function SciMLBase.solve(model::UDEModel", src)
    start === nothing && return false
    rest = src[first(start):end]
    nxt = findnext(r"\nfunction ", rest, 2)
    body = nxt === nothing ? rest : rest[1:(first(nxt) - 1)]
    return occursin("SciMLBase.ODEProblem", body) &&
           occursin("solver_config.algorithm", body) &&
           !occursin("KenCarp4", body)
end

function sciml_solve_surface_source_holds()
    src = read(sciml_solve_surface_source_path(), String)
    return all(occursin(needle, src) for needle in SCIML_SOLVE_SURFACE_MUST_CONTAIN) &&
           !any(occursin(needle, src) for needle in SCIML_SOLVE_SURFACE_MUST_NOT_CONTAIN)
end

function sciml_solve_surface_docs_path()
    joinpath(pkgdir(BioDynaX), "docs", "src", "sciml-solve-surface.md")
end

function sciml_solve_surface_docs_hold()
    path = sciml_solve_surface_docs_path()
    isfile(path) || return false
    text = read(path, String)
    for sentence in values(sciml_solve_surface_locked_sentences())
        occursin(sentence, text) || return false
    end
    make = read(joinpath(pkgdir(BioDynaX), "docs", "make.jl"), String)
    occursin("sciml-solve-surface.md", make) || return false
    return !occursin("HTTP 200", text) && !occursin("]add BioDynaX", text) &&
           !occursin("TagBot ran", text)
end

function sciml_solve_surface_landing_docs_hold()
    sciml = read(joinpath(pkgdir(BioDynaX), "docs", "src", "sciml.md"), String)
    howto = read(joinpath(pkgdir(BioDynaX), "docs", "src", "howto.md"), String)
    sentences = sciml_solve_surface_locked_sentences()
    return occursin("sciml-solve-surface", sciml) &&
           occursin("SolveSurfaceRow", howto) &&
           occursin(sentences.surface, sciml)
end

function sciml_solve_surface_source_violations()
    src = read(sciml_solve_surface_source_path(), String)
    missing = [s for s in SCIML_SOLVE_SURFACE_MUST_CONTAIN if !occursin(s, src)]
    forbidden = [s for s in SCIML_SOLVE_SURFACE_MUST_NOT_CONTAIN if occursin(s, src)]
    return (; missing, forbidden)
end

function sciml_solve_surface_contract_holds()
    return sciml_solve_surface_source_holds() &&
           sciml_recommend_sensealg_source_holds() &&
           sciml_odeproblem_uses_build_ude_function() &&
           sciml_solve_uses_odeproblem() &&
           sciml_interface_adds_no_solver() &&
           sciml_solve_surface_docs_hold() &&
           public_export_list_holds() &&
           recovery_thresholds_hold() &&
           validate_network_stays_open_source()
end
