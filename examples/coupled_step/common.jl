# Shared pieces of the coupled-step examples: a current-carrying column in a pure toroidal
# field, toroidal loops around it, the plasma flux each loop links, and the lumped circuit
# model the plots compare against. `include("common.jl")` from a script in this directory.
#
# A step refreshes the collision rates at its end, before `callback_after_step` runs. A
# callback that changes n, T or u must refresh them itself, or the next step uses the rates
# of the old state: `RAPID2D.update_transport_quantities!(RP)`.

include(joinpath(@__DIR__, "..", "common.jl"))

# A uniform column in a pure toroidal field (Eϕ = E0·R̄/R) with Coulomb drag as its only
# friction: the column of current_diffusion.jl, at Te = 1 eV so that its L/R time is about
# 0.2 ms. Ampère and the inductive E∥ are on; Ampère runs once |I_p| ≥ `threshold`
# (0: from the first step). No sources and fixed temperatures. `moving = true` turns on the
# mean E×B drift and the advection of n and u∥ by it, so that a script can push the column.
function column(
        name; E0 = 0.3, Te = 1.0, n0 = 1.0e16, cenR = 1.5, cenZ = 0.0, radius = 0.3,
        threshold = 0.0, dt = 5.0e-6, t_end = 1.0e-3, moving = false,
        manual = pure_toroidal(E0), wall_R = Float64[], wall_Z = Float64[],
    )
    config = SimulationConfig{Float64}(;
        device_Name = "manual", manual, wall_R, wall_Z, NR = 30, NZ = 50, R0B0 = 3.0,
        prefilled_gas_pressure = 0.0,          # vacuum: no neutrals
        dt, t_end_s = t_end, snap0D_Δt_s = 10dt, snap2D_Δt_s = t_end,
        Output_path = output_dir(name),
    )
    RP = setup(
        config;
        Ampere = true, Ampere_Itor_threshold = threshold, E_para_self_EM = true,
        ud_evolve = true, Coulomb_Collision = true,
        Atomic_Collision = false, src = false, convec = moving, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = moving, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        E_para_self_ES = false, mean_ExB = moving, turb_ExB_mixing = false,
        FLF_nstep = 100_000,                   # no field-line tracing
    )
    set_column!(RP, tophat(RP.G; cenR, cenZ, radius, n0); Te_eV = Te)
    return RP
end

# A toroidal loop of minor radius `a` at (r, z), L = μ0 r (ln(8r/a) − 7/4); returns L.
# R = 1e-12 Ω makes it superconducting on these time scales. A nonzero V powers it. Call
# initialize_coil_system! after the last one.
function add_loop!(RP, r, z; a = 0.05, R = 1.0e-12, V = 0.0, I0 = 0.0, name = "loop")
    L = RP.config.constants.μ0 * r * (log(8r / a) - 7 / 4)
    coil = Coil{Float64}(;
        location = (r = r, z = z), area = π * a^2, resistance = R, self_inductance = L,
        is_powered = V != 0, is_controllable = false, name, current = I0, voltage_ext = V,
    )
    add_coil!(RP.coil_system, coil)
    return L
end

# Toroidal current density of the present state, electrons and ions.
function current_density(RP)
    p, c = RP.plasma, RP.config.constants
    Zi = Float64(RAPID2D.bulk_ion_charge(RP))
    return @. (c.qe * p.ne * p.ue_para + p.ni * (c.ee * Zi) * p.ui_para) * RP.fields.bϕ
end
plasma_current(RP, J) = sum(J) * RP.G.dR * RP.G.dZ

# Plasma flux through each loop, 2π Σ G(r_loop; r) J dA.
flux_at_coils(RP, J) = 2π .* (RP.coil_system.Green_grid2coils * vec(J)) .* (RP.G.dR * RP.G.dZ)

# ── the lumped model: the column as one plasma loop ────────────────────────────────
# One loop carrying a uniform current over the cells the column fills (area S, a = √(S/π)):
# L_p = μ0 R (ln(8R/a) − 7/4); the electrons' inertia L_kin = mₑ 2πR / (n e² S) at the present
# density; and M, the flux through each coil per ampere of that current.
function lumped_column(RP; R0 = 1.5)
    G, pla, c = RP.G, RP.plasma, RP.config.constants
    col = findall(>(0), pla.ne)
    S = length(col) * G.dR * G.dZ
    J = zeros(size(pla.ne))
    J[col] .= 1.0
    return (;
        L_p = c.μ0 * R0 * (log(8R0 / sqrt(S / π)) - 7 / 4),
        L_kin = c.me * 2π * R0 / (mean(pla.ne[col]) * c.ee^2 * S),
        M = RP.coil_system.n_total > 0 ? flux_at_coils(RP, J) ./ plasma_current(RP, J) : Float64[],
        S,
    )
end

# Resistance of the column to a uniform loop voltage, 1/R_p = Σ σ dA / (2πR), with the Coulomb
# conductivity σ = n e² / (mₑ ν), which does not depend on n (ν ∝ n).
function column_resistance(RP)
    G, pla, c = RP.G, RP.plasma, RP.config.constants
    col = findall(>(0), pla.ne)
    return 1 / (sum(@. c.ee^2 * pla.ne[col] / (c.me * pla.ν_ei_eff[col] * 2π * G.R2D[col])) * G.dR * G.dZ)
end

# Circuits in flux form, dΨ/dt = V − R I with Ψ = L I, by RK4 over the times `t` from
# I(t[1]) = I0; L(t) and R(t) are matrices given as functions of time. Where L jumps, Ψ stays
# continuous, as the circuit equations require.
function circuits(L, R, V, I0, t; substeps = 20)
    f(τ, Ψ) = V .- R(τ) * (L(τ) \ Ψ)
    Ψ = L(t[1]) * I0
    I = [I0]
    for k in 2:length(t)
        h = (t[k] - t[k - 1]) / substeps
        for j in 1:substeps
            τ = t[k - 1] + (j - 1) * h
            k1 = f(τ, Ψ)
            k2 = f(τ + h / 2, Ψ .+ h / 2 .* k1)
            k3 = f(τ + h / 2, Ψ .+ h / 2 .* k2)
            k4 = f(τ + h, Ψ .+ h .* k3)
            Ψ = Ψ .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
        push!(I, L(t[k]) \ Ψ)
    end
    return I
end

# Where things sit: the current density (scaled to its peak), the wall, and the loops.
function plot_layout(RP; J = current_density(RP), title = "layout")
    G = RP.G
    rc = [c.location.r for c in RP.coil_system.coils]
    zc = [c.location.z for c in RP.coil_system.coils]
    Jmax = maximum(abs, J)
    p = heatmap(
        G.R1D, G.Z1D, permutedims(Jmax > 0 ? J ./ Jmax : J);
        c = :balance, clims = (-1, 1), aspect_ratio = :equal, colorbar_title = "J / max|J|",
        xlims = (min(G.R1D[1], minimum(rc; init = Inf) - 0.1), max(G.R1D[end], maximum(rc; init = -Inf) + 0.1)),
        ylims = extrema(G.Z1D), xlabel = "R (m)", ylabel = "Z (m)", title, titlefontsize = 10,
        framestyle = :box,
    )
    plot!(p, vcat(RP.wall.R, RP.wall.R[1]), vcat(RP.wall.Z, RP.wall.Z[1]); c = :gray40, lw = 1, label = "wall")
    isempty(rc) || scatter!(p, rc, zc; c = :orange, ms = 4, msw = 0, label = "loops")
    return p
end

# ── the coupled solve's outer iteration ────────────────────────────────────────────────
# Within a step the coupled solve iterates on the boundary flux and the coil currents: solve
# u∥ and ψ inside the domain with the boundary flux held, then recompute the boundary flux
# (Green's functions) and the coil currents (circuits) from the new plasma current, and repeat.
# The default mixes the iterates by Anderson (memory 8, at most 20 block solves); the relaxed
# iteration (anderson_m = 0, boundary flux weighted by w = 0.5) was the default before.
# DIRECT_PICARD solves the same equations without iterating (method = :direct: the map's
# linear part assembled, then Newton's step on it), checked to 1e-10 of the step's field and of
# each coil's change with no floors: what the default should reproduce. A run whose direct solve
# did not meet that has no expected result.
const DIRECT_PICARD = (method = :direct, tolerance = 1.0e-10, E_floor = 0.0, I_floor = 0.0)

# The largest gap between a plasma current history and the expected one, step by step against
# the expected current of that step, floored at 1e-3 of its peak where it passes through zero.
# Exact agreement is no gap, also where the expected current is zero throughout.
function current_gap(I, I_ref)
    scale = max.(abs.(I_ref), 1.0e-3 * maximum(abs, I_ref))
    return maximum(ifelse(e == 0, 0.0, e / s) for (e, s) in zip(abs.(I .- I_ref), scale))
end

# Geometries. A KSTAR-like domain: the grid of the KSTAR field files (R 1.2–2.4 m, Z ±1.2 m)
# with the KSTAR first wall (KSTAR_First_Wall.dat), whose inboard side is 6 cm (1.5 cells at
# 30×50) inside the grid. A tight box: the default box wall one cell inside a 1.2 × 1.6 m grid.
const KSTAR_WALL_R = [1.26, 1.632, 1.992, 2.256, 2.256, 1.992, 1.632, 1.26, 1.26]   # closed
const KSTAR_WALL_Z = [1.13, 1.056, 0.732, 0.456, -0.456, -0.732, -1.056, -1.13, 1.13]
kstar_like(E0) = ManualSetup{Float64}(R = (1.2, 2.4), Z = (-1.2, 1.2), BR = 0.0, BZ = 0.0, Eϕ = E0)
tight_box(E0) = ManualSetup{Float64}(R = (1.0, 2.2), Z = (-0.8, 0.8), BR = 0.0, BZ = 0.0, Eϕ = E0, wall_margin_cells = 1)

# A shell of `nfil` copper filaments on a circle of radius `r` around (cenR, 0), each of the
# square cross-section that tiles the circle: a passive conducting structure inside the grid.
function filament_shell!(RP; cenR, r, nfil = 24)
    side = 2π * r / nfil
    for k in 1:nfil
        θ = 2π * (k - 0.5) / nfil + 0.05
        R, Z = cenR + r * cos(θ), r * sin(θ)
        add_loop!(RP, R, Z; a = side / sqrt(π), R = 1.68e-8 * 2π * R / side^2, name = "shell_$k")
    end
    initialize_coil_system!(RP)
    return RP
end

quiet(f) = redirect_stdout(() -> redirect_stderr(f, devnull), devnull)

# The column of `make()` run for `nsteps` steps twice, with the default Picard and with
# DIRECT_PICARD: the plasma current and the induced field after each step, and the Picard
# counters.
function default_vs_direct(make; nsteps)
    return map((nothing, DIRECT_PICARD)) do picard
        RP = make()
        isnothing(picard) || (RP.flags.ampere_picard = PicardSettings{Float64}(; picard...))
        RP.t_end_s = nsteps * RP.dt
        I, E = Float64[], Matrix{Float64}[]
        record(rp) = (push!(I, plasma_current(rp, current_density(rp))); push!(E, copy(rp.fields.Eϕ_self)))
        quiet(() -> run_simulation!(RP; callback_after_step = record))
        (; RP, I, E, stats = deepcopy(RP.diagnostics.ampere_picard))
    end
end

# The first step of the column of `make()`, solved from the same state and stopped after
# L = 1…Lmax block solves, against the step solved directly: the error of the induced field,
# max|Eϕ_L − Eϕ*| / max|Eϕ*|, for the relaxed iteration (anderson_m = 0, w = 0.5) and for the
# default solve. Both keep the solve's failure policy: a residual that grows a thousandfold restarts
# from the best iterate with half the mixing, so the relaxed iteration no longer runs away.
function picard_error_by_iteration(make; Lmax = 30)
    RP = make()
    pla, F, csys = RP.plasma, RP.fields, RP.coil_system
    # what run_simulation! does before its first step
    pla.ne[RP.G.nodes.on_out_wall_nids] .= 0.0
    pla.ni[RP.G.nodes.on_out_wall_nids] .= 0.0
    initialize_coupled_fields!(RP)
    RAPID2D.update_transport_quantities!(RP)
    prepare_timestep!(RP)
    saved = (
        u = copy(pla.ue_para), ψ = copy(F.ψ_self), E = copy(F.Eϕ_self), Ep = copy(F.Eϕ_self_prev),
        I = csys.n_total > 0 ? copy(get_all_currents(csys)) : Float64[],
        Φ = csys.n_total > 0 ? copy(csys.coils.ψ_pla) : Float64[], t = csys.time_s,
    )
    function trial(; kw...)
        pla.ue_para .= saved.u; F.ψ_self .= saved.ψ; F.Eϕ_self .= saved.E; F.Eϕ_self_prev .= saved.Ep
        if csys.n_total > 0
            set_all_currents!(csys, copy(saved.I)); csys.coils.ψ_pla = copy(saved.Φ); csys.time_s = saved.t
        end
        quiet(() -> RAPID2D.solve_combined_momentum_Ampere_equations_with_coils!(RP; kw...))
        return copy(F.Eϕ_self)
    end
    E_star = trial(; DIRECT_PICARD...)
    err(m, L) = maximum(abs, trial(; tolerance = 0.0, max_iter = L, anderson_m = m, E_floor = 0.0, I_floor = 0.0) .- E_star)
    scale = maximum(abs, E_star)
    return (relaxed = [err(0, L) for L in 1:Lmax] ./ scale, default = [err(RP.flags.ampere_picard.anderson_m, L) for L in 1:Lmax] ./ scale)
end

# One figure for a case of the coupled solve: where the column and the conductors sit, the
# plasma current step by step with the default and the direct solve, the first step's
# induced-field error against the number of block solves (relaxed iteration and default), and
# that error over the grid after the default solve. Passes when the default stays within 1 % of
# the direct solve's current at every step (`current_gap`).
function picard_case(make, name, title; nsteps = 20)
    def, direct = default_vs_direct(make; nsteps)
    errs = picard_error_by_iteration(make)
    t = (1:nsteps) .* def.RP.dt .* 1.0e6
    gap = current_gap(def.I, direct.I)
    pass = direct.stats.nunconverged == 0 && isfinite(gap) && gap <= 1.0e-2

    blowup = maximum(abs, def.I) > 100 * maximum(abs, direct.I)
    p1 = plot!(plot_layout(direct.RP; title); colorbar = false, titlefontsize = 9)
    p2 = plot(
        t, blowup ? abs.(direct.I) : direct.I; c = :black, lw = 3, label = "direct solve (expected)",
        xlabel = "t (µs)", ylabel = blowup ? "|I_p| (A)" : "I_p (A)", yscale = blowup ? :log10 : :identity,
        title = "plasma current", legend = :topleft,
    )
    plot!(p2, t, blowup ? max.(abs.(def.I), 1.0e-3) : def.I; c = :red3, ls = :dash, lw = 2, m = :circle, ms = 3, label = "default solve")
    p3 = plot(
        1:length(errs.relaxed), max.(errs.relaxed, 1.0e-16); yscale = :log10, c = :gray50, lw = 2, m = :circle, ms = 2,
        label = "relaxed iteration (w = 0.5, halved when it grows)", xlabel = "block solves in the first step", ylabel = "max|ΔEϕ| / max|Eϕ*|",
        title = "first step: error by iteration", ylims = (1.0e-12, 1.0e6),
    )
    plot!(p3, 1:length(errs.default), max.(errs.default, 1.0e-16); c = :red3, lw = 2, m = :circle, ms = 3, label = "default solve")
    vline!(p3, [def.RP.flags.ampere_picard.max_iter]; c = :gray, ls = :dash, label = "default limit")
    hline!(p3, [1.0e-3]; c = :green, ls = :dot, label = "tolerance (1e-3)")
    G = direct.RP.G
    ΔE = (def.E[1] .- direct.E[1]) ./ maximum(abs, direct.E[1])
    lim = max(maximum(abs, filter(isfinite, ΔE)), 1.0e-12)
    p4 = heatmap(
        G.R1D, G.Z1D, permutedims(ΔE); c = :balance, clims = (-lim, lim), aspect_ratio = :equal,
        xlabel = "R (m)", ylabel = "Z (m)", title = "step 1: (Eϕ − Eϕ*) / max|Eϕ*|", titlefontsize = 10,
        framestyle = :box, xlims = extrema(G.R1D), ylims = extrema(G.Z1D),
    )
    plot!(p4, vcat(direct.RP.wall.R, direct.RP.wall.R[1]), vcat(direct.RP.wall.Z, direct.RP.wall.Z[1]); c = :gray40, lw = 1, label = "")
    fig = plot(
        p1, p2, p3, p4; layout = (1, 4), size = (1800, 540), margin = 5Plots.mm, left_margin = 10Plots.mm,
        top_margin = 10Plots.mm, bottom_margin = 12Plots.mm,
    )
    detail = @sprintf(
        "at every step the default solve is within %.2g %% of that step's direct-solve current (floored at 1e-3 of the peak; passes under 1 %%); first step %.3g A vs %.3g A; %.1f block solves/step, %d unconverged (direct run: %.0f/step, %d unconverged)",
        100gap, def.I[1], direct.I[1], def.stats.niter / def.stats.nsolve, def.stats.nunconverged,
        direct.stats.niter / direct.stats.nsolve, direct.stats.nunconverged
    )
    save_with_verdict(fig, output_dir("coupled_step"), name, pass, detail)
    return (; def, direct, errs, gap, pass)
end
